import torch
import torch.nn as nn
import torchvision
import os
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms.functional import to_pil_image
import wandb

from talk3d_dataloader import Train_Dataset_nosync, Inference_Dataset, Train_Dataset_sync
from talk3d_helper import *
from vive3D.visualizer import *
from vive3D.eg3d_generator import *
from vive3D.landmark_detector import *
from vive3D.video_tool import *
from vive3D.segmenter import *
from vive3D.inset_pipeline import *
from vive3D.aligner import *
from vive3D.interfaceGAN_editor import *
from vive3D.config import *
from talk3d_helper import *
from Wav2Lip.models import SyncNet_color as SyncNet
from GFPGAN.gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from training.networks_diffusers import Deltaplane_Predictor
import torchvision


class Trainer:
    def __init__(self, opts, G, device, world_size):
        self.opts = opts
        self.keep_training = True
        short_configs = opts.short_configs
        self.device = device
        self.data_root = os.path.join(opts.data_root_dir, opts.personal_id)
        self.tune_switch = False
        if opts.checkpoint_dir=='None':
            self.load_checkpoint = False  
        else:
            self.load_checkpoint = True
        
        'Make saving directories'
        self.set_saving_dirs(opts)
        
        'Set modules that only needs for single time (single device)'
        if self.device == 0:
            parser2yaml(opts, f'{self.config_save_path}/{short_configs}', opts.batchsize*world_size)
            self.vid = VideoTool(os.path.join(opts.data_root_dir, opts.personal_id, 'GT_video_recon.mp4'))
            'Save inference wav'
            self.trim_and_save_inference_wav()
        
        'Setting up models and functions - do not need gradient'
        self.generator = G
        self.segmenter = Segmenter(device=self.device)
        self.landmark_detector = LandmarkDetector(device=self.device)
        self.align = Aligner(landmark_detector=self.landmark_detector, segmenter=self.segmenter, device=self.device)
        self.L2loss, self.L1loss, self.LPIPS, self.IDloss = makelossfunc(self.device)
        self.downsampler_128 = BicubicDownSample(factor=512//128, device=self.device).to(self.device)
        self.segmenter = Segmenter(device=self.device)
        
        if opts.use_sync_loss:
            self.syncnet = SyncNet().to(self.device)
            for p in self.syncnet.parameters():
                p.requires_grad = False
            syncnet_path = opts.syncnet_weight_path # lipsync_expert.pth"
            load_syncnet_checkpoint(syncnet_path, self.syncnet, None, reset_optimizer=True, overwrite_global_states=False)
        
        'Setting up models - need gradient'
        self.null_vector = torch.randn(1, opts.num_null_vector, 64) #null vector embedding
        self.null_vector.requires_grad = True
        
        self.optimizable_list = self.setup_models(opts, self.device, load_checkpoint=self.load_checkpoint)
        self.configure_optimizer(opts) # returns self.optimizer, self.scheduler
        
        w_person_dir = os.path.join(self.data_root, opts.w_id_dir)
        ID_triplane_dir = os.path.join(self.data_root, opts.ID_triplane_dir)
        self.w_person = torch.load(w_person_dir).to(self.device).unsqueeze(0)[:, :1, :]

        'save identity triplane for boosting the training'
        if not os.path.isfile(ID_triplane_dir):
            print('No ID_triplane exist!')
        print('Saving ID_triplane...')
        with torch.no_grad():
            synthesis_kwargs = {'noise_mode': 'const'}
            plane = self.generator.active_G.backbone.synthesis(self.w_person.repeat(1,14,1), update_emas=False, **synthesis_kwargs).cpu()
            torch.save(plane.repeat(16,1,1,1), ID_triplane_dir) # 16 is just for save -> doesnt effect on training
        self.ID_triplane = torch.load(ID_triplane_dir).to(self.device)
        
        'setup dataloader'
        self.outside_batchsize = 2
        self.dataloader_setup(opts, self.device, batchsize=opts.batchsize, outbatchsize=self.outside_batchsize) # returns self.train_dataloader, self.inference_dataloader
        self.total_bs = opts.batchsize*world_size
        
        # GFPGAN setup (if tuned checkpoint is given)
        self.G_tune_start = float('inf')
        if (self.opts.tuning_start_iter <= self.iter_idx) and self.tune_switch==False:
            self.tune_switch = True
            self.opts.num_iter_inf /= 20 
            self.initialize_G_tuning_stage(self.opts.checkpoint_dir, use_gfpgan=self.opts.use_GFPGAN)
            self.Loader = self.TrainLoader  
    
    def train_epoch(self):
        if (self.opts.tuning_start_iter <= self.iter_idx) and self.tune_switch==False:
            self.tune_switch = True
            self.opts.num_iter_inf /= 20 
            self.initialize_G_tuning_stage(self.opts.checkpoint_dir, use_gfpgan=self.opts.use_GFPGAN)
            self.Loader = self.TrainLoader  
        
        with tqdm(self.Loader) as pbar:
            for datas in pbar:
                torch.distributed.barrier()
                self.train_iter(datas)
                pbar.set_description(f'Epoch : {self.epoch_idx} Iteration :{self.iter_idx} | ' + self.loss_text)
                
                if self.opts.vis_opt_process and self.iter_idx%(10//self.total_bs) == 0 and self.device == 0: # and iter_idx != 0
                    with torch.no_grad():
                        to_pil_image(((self.out['image'][0].clone().detach().cpu()+1)*127.5).to(torch.uint8).clamp(min=0, max=255)).save('./test_synth.png')
                        to_pil_image(((self.target_image[0].clone().detach().cpu()+1)*127.5).to(torch.uint8).clamp(min=0, max=255)).save('./test_target.png') # save image
                
                torch.distributed.barrier()
                if self.iter_idx%(self.opts.num_iter_inf//self.total_bs) == 0 and self.opts.do_inference: # and self.iter_idx != 0
                    self.model2eval(self.tune_switch)
                    with torch.no_grad():
                        # self.inference_step(inferencetype='train')
                        if self.opts.do_inference_novel:
                            self.inference_step(inferencetype='novel')
                            if self.opts.inf_camera_type != 'GT':
                                self.inference_step(inferencetype='novel', cameratype=self.opts.inf_camera_type)
                        if self.opts.do_inference_OOD:
                            self.inference_step(inferencetype='OOD')

                    self.model2train(self.tune_switch)
    
                if self.iter_idx%(self.opts.num_iter_inf//self.total_bs)==0 and self.opts.use_wandb and self.device == 0:
                    self.logging()

                if self.iter_idx%(self.opts.num_iter_inf//self.total_bs)==0 and self.device == 0:
                    self.save_model_weights()
                torch.distributed.barrier()

            self.scheduler.step()
            self.epoch_idx += 1
        
    def train_iter(self, datas):
        
        if self.iter_idx - self.G_tune_start > self.opts.max_G_tuning_iter:
            self.keep_training = False
            return
        
        if self.opts.use_sync_loss and self.tune_switch==False:
            target_image, segmentation, mouthseg, torsoseg, yaw, pitch, aud_eo, eye, wav2lip_frames, wav2lip_bboxs, indiv_mels, mel, angle, landmark = datas
        else:
            target_image, segmentation, mouthseg, torsoseg, yaw, pitch, aud_eo, eye, angle, landmark = datas
        self.target_image = target_image.clone() # for logging

        target_image = target_image.to(self.device)
        segmentation = segmentation.to(self.device)
        mouthseg = mouthseg.to(self.device)
        torsoseg = torsoseg.to(self.device)
        yaw = yaw.to(self.device)
        pitch = pitch.to(self.device)
        aud_eo = aud_eo.to(self.device)
        eye = eye.to(self.device)
        angle = angle.to(self.device)
        landmark = landmark.to(self.device)
        null = self.null_vector.to(self.device)
        if self.opts.use_sync_loss and self.tune_switch==False:

            target_image = torch.cat([target_image[i, :] for i in range(self.opts.batchsize)], dim=0) # outbatch*batch , c , h, w
            segmentation = torch.cat([segmentation[i, :] for i in range(self.opts.batchsize)], dim=0)
            mouthseg = torch.cat([mouthseg[i, :] for i in range(self.opts.batchsize)], dim=0)
            torsoseg = torch.cat([torsoseg[i, :] for i in range(self.opts.batchsize)], dim=0)
            yaw = torch.cat([yaw[i, :] for i in range(self.opts.batchsize)], dim=0)
            pitch = torch.cat([pitch[i, :] for i in range(self.opts.batchsize)], dim=0)
            aud_eo = torch.cat([aud_eo[i, :] for i in range(self.opts.batchsize)], dim=0)
            eye = torch.cat([eye[i, :] for i in range(self.opts.batchsize)], dim=0)
            angle = torch.cat([angle[i, :] for i in range(self.opts.batchsize)], dim=0)
            landmark = torch.cat([landmark[i, :] for i in range(self.opts.batchsize)], dim=0)
            
            
            wav2lip_frames   = wav2lip_frames.to(self.device) # (outbatch, 3, 5, h, w)
            wav2lip_bboxs    = wav2lip_bboxs.to(self.device) # (outbatch, 5, 4)
            indiv_mels   = indiv_mels.to(self.device)
            mel          = mel.to(self.device) # (outbatch, 1, 80, 16)
        else:
            wav2lip_frames   = None#wav2lip_frames.to(self.device) # (outbatch, 3, 5, h, w)
            wav2lip_bboxs    = None#wav2lip_bboxs.to(self.device) # (outbatch, 5, 4)
            indiv_mels   = None#indiv_mels.to(self.device)
            mel          = None#mel.to(self.device) # (outbatch, 1, 80, 16)

        cond_feat = []
        for i in range(aud_eo.shape[0]):
            enc_a = self.audio_net(aud_eo[i]) #B, window, 44, 16 -> T, 64
            cond_feat.append(self.audio_att_net(enc_a.unsqueeze(0))) #T, 64 -> 1, 64
            
        cond_feat = torch.stack(cond_feat, dim=0) #B, 1, 64 or B, 8, 64

        pred_w = self.w_person.repeat(cond_feat.shape[0], 14, 1)

        delta_plane = self.predict_delta_plane(cond_feat, eye, null, zero_plane= not self.opts.predict_deltaplane,angle = angle, landmark =landmark)
        out = self.generator.generate(pred_w, triplane_offset=delta_plane, yaw=yaw, pitch=pitch, ID_triplane_dir=os.path.join(self.data_root, self.opts.ID_triplane_dir), output_all=True, grad=True)
        out_128 = out['image_raw']
        if self.tune_switch and self.opts.use_GFPGAN:
            out_512 = self.GFPGAN.enhance_trainable(out_128, has_aligned=True, paste_back=True, weight=0.5)
        else:
            out_512 = out['image']
        
        loss = self.calc_loss(out_128, out_512, target_image, segmentation, mouthseg, torsoseg, wav2lip_frames, wav2lip_bboxs, mel)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.iter_idx += 1


    def predict_delta_plane(self, cond_feat, eye, null, zero_plane=False, angle = None, landmark = None):
        if zero_plane:
            return torch.zeros(self.opts.batchsize, 96, 256, 256).to(self.device) # dont use deltaplane!
        
        eye_feat = self.blink_encoder(eye)
        delta_triplane_condition = torch.cat([cond_feat, eye_feat, null.repeat(cond_feat.shape[0],1,1)], dim=-2)
        
        landmark = torch.flatten(landmark, start_dim = 1)        
        angle, landmark= self.camera_encoder(angle, landmark)     
        delta_triplane_condition = torch.cat([delta_triplane_condition, angle, landmark],dim=-2)     
            
        
        ID_triplane = self.ID_triplane[:cond_feat.shape[0]]
        delta_plane, scoremaps = self.deltaplane_predictor(ID_triplane, delta_triplane_condition)
        
        if self.opts.vis_score_map:
            if self.iter_idx % 200 == 0 and self.device == 0:
                color_coding(scoremaps, self.epoch_idx, self.iter_idx, f'{self.att_map_save_path}/{self.opts.short_configs}')

        return delta_plane

    def save_image(self, img):
        torchvision.utils.save_image((img+1)/2, './test.png')

    def calc_loss(self, synth_image_128, synth_image, target_image, segmentation, mouth_seg, torsoseg, wav2lip_frames, wav2lip_bboxs, mel):
        target_image_128 = self.downsampler_128(target_image)
        target_foreground = segmentation*target_image
        synth_image_foreground = segmentation * synth_image 
        # segmentation_128 = torch.round(self.downsampler_128(segmentation.repeat(1, 3, 1, 1))).to(torch.uint8).to(self.device)
        # synth_image_foreground_128 = segmentation_128 * synth_image_128
        # target_image_foreground_128 = segmentation_128 * target_image_128
        
        target_torso = torsoseg*target_image
        synth_image_torso = torsoseg * synth_image 
        # torso_segmentation_128 = torch.round(self.downsampler_128(torsoseg.repeat(1, 3, 1, 1))).to(torch.uint8).to(self.device)
        # synth_image_torso_128 = torso_segmentation_128 * synth_image_128
        # target_image_torso_128 = torso_segmentation_128 * target_image_128

        #Loss functions
        loss = 0
        self.loss_text = ''
        self.loss_dict = {}

        if self.tune_switch:
            loss_l1_foreground = self.L1loss(synth_image, target_image) * self.opts.image_loss_weight
            loss_perc_foreground = self.LPIPS(synth_image, target_image).sum() * self.opts.LPIPS_loss_weight
                
        else:
            loss_l1_foreground = self.L1loss(synth_image_128, target_image_128) * self.opts.image_loss_weight
            loss_perc_foreground = self.LPIPS(synth_image_128, target_image_128).sum() * self.opts.LPIPS_loss_weight
            loss_l1_foreground += self.L1loss(synth_image, target_image) * self.opts.image_loss_weight
            loss_perc_foreground += self.LPIPS(synth_image, target_image).sum() * self.opts.LPIPS_loss_weight
                
            
        loss += loss_l1_foreground
        loss += loss_perc_foreground
            
        self.loss_text += f'L1 = {round(loss_l1_foreground.item(), 2):.2f} '
        self.loss_text += f'LP = {round(loss_perc_foreground.item(), 2):.2f} '
        self.loss_dict['L1'] = loss_l1_foreground.item()
        self.loss_dict['LP'] = loss_perc_foreground.item()
        

        if self.opts.use_id_loss and self.tune_switch==False:
            loss_id = self.IDloss(synth_image_foreground, target_foreground) * self.opts.id_loss_weight

            loss += loss_id
            self.loss_text += f'ID = {round(loss_id.item(), 2):.2f} '
            self.loss_dict['ID'] = loss_id.item()
        
        if self.opts.use_mouth_loss and self.tune_switch==False:

            mouth_seg_128 = torch.round(self.downsampler_128(mouth_seg.repeat(1, 3, 1, 1))).to(torch.uint8).to(self.device)
            num_mouth_px = mouth_seg.sum()
            num_mouth_px_128 = mouth_seg_128.sum()
            target_mouth_128 = mouth_seg_128*target_image_128
            synth_mouth_128 = mouth_seg_128*synth_image_128
            
            loss_l2_mouth = self.L2loss(synth_mouth_128, target_mouth_128) / num_mouth_px_128 * self.opts.mouth_loss_weight
            loss_perc_mouth = self.LPIPS(synth_mouth_128, target_mouth_128).sum() * self.opts.mouth_loss_weight 

            loss += loss_l2_mouth
            loss += loss_perc_mouth
            self.loss_text += f'L2M = {round(loss_l2_mouth.item(), 2):.2f} '
            self.loss_text += f'LPM = {round(loss_perc_mouth.item(), 2):.2f} '
            self.loss_dict['L2M'] = loss_l2_mouth.item()
            self.loss_dict['LPM'] = loss_perc_mouth.item()
        
        if self.opts.use_sync_loss and self.tune_switch==False:
            # wav2lip_bbox: (b,5,4) : x1, y1, x2 y2
            # wav2lip_frames: torch.Size([b==1, 3, 5, 96, 96]) # (b, c, syncnet_T, h, w) # already resized in the dataloader.
            if self.opts.batchsize==3:
                idxrange = range(1,4)#lidx, ridx = 1, 4
            elif self.opts.batchsize == 2:
                idxrange = [1,3]#lidx, ridx = 1,3
            elif self.opts.batchsize==5:
                idxrange = range(0,5)
            elif self.opts.batchsize==1:
                idxrange = range(2,3)
                
            wav2lip_bboxs = wav2lip_bboxs/(512//128) # box coord scaling # (b, 5=syncnet_t, 4)
            cur_bboxs = wav2lip_bboxs[:,idxrange,:] # -> (outbatch, len(idrange), 4) 
            synth_image_crop_96 = []
            
            # (4, c, h, w)
            for ii in range(self.outside_batchsize):
                for jj in range(self.opts.batchsize):
                    curidx = self.opts.batchsize * ii + jj
                    x1, y1, x2, y2 = cur_bboxs[ii][jj]
                    synth_image_crop_96.append(  F.interpolate( synth_image_128[curidx:curidx+1, :, int(y1):int(y2), int(x1):int(x2)], (96,96) ) )
            
            synth_image_crop_96 = torch.cat(synth_image_crop_96).to(self.device) # (b=2*2,3=c,96,96)
            synth_image_crop_96 = (synth_image_crop_96[:,[2,1,0],:,:] +1)/2 # denormalize and convert RGB->BGR
            synth_image_crop_96 = synth_image_crop_96.reshape(self.outside_batchsize, self.opts.batchsize, 3, 96, 96)

            wav2lip_frames = wav2lip_frames.float()
            
            wav2lip_frames[:, :, idxrange, :, :] = synth_image_crop_96.permute(0,2,1,3,4).float()# (outbatch, c, num_frames(~batchsize), 96,96)
            
            sync_loss = get_sync_loss(self.syncnet, mel, wav2lip_frames.float(), self.device, use_ddp=False)

            if self.epoch_idx>20:
                sync_loss=sync_loss*self.opts.sync_loss_weight #* opts.sync_loss_weight
            else:
                sync_loss = sync_loss*0.
            loss += sync_loss
                
            self.loss_text += f'Sync = {round(sync_loss.item(), 3):.3f} ' 
            self.loss_dict['Sync'] = sync_loss.item()
            
            if self.opts.log_psnr:
                psnr = tensor2psnr(synth_image, target_image, self.device, m = 2)
                self.loss_text += f'PSNR : w/bg 512 = {round(psnr.item(), 2):.2f} '
                psnr = tensor2psnr(synth_image_foreground, target_foreground, self.device, m = 2)
                self.loss_text += f'body 512 = {round(psnr.item(), 2):.2f} '
                psnr = tensor2psnr(synth_image_torso, target_torso, self.device, m = 2)
                self.loss_text += f'torso 512 = {round(psnr.item(), 2):.2f} '

        return loss

    def inference_step(self, inferencetype='train', cameratype='GT'):

        images_gt = []
        face_segs = []
        body_segs = []

        smooth_cond_feat = None
        _lambda = 0.2

        if inferencetype == 'OOD':
            inferencedataloader = self.OODLoader
            type_ = 'OOD'
        elif inferencetype == 'novel':
            inferencedataloader = self.TestLoader
            type_ = 'novel'
        elif inferencetype == 'train':
            inferencedataloader = self.ValidLoader
            type_ = ''

        '''
        camera controller - set camera type in {'rotation', 'GT', radians}
        radians should be looks like : '20_30' (fixed in yaw 20 degrees, pitch 30 degrees)
        '''
        if cameratype == 'rotation':
            new_yaw, new_pitch = gen_interp_cam(len(inferencedataloader)*(self.opts.inf_batchsize+1), self.opts.max_rotation_yaw, self.opts.max_rotation_pitch)
        elif cameratype.startswith('fix'): # fixed camera
            max_angles = cameratype.lstrip('fix').split('_')
            new_yaw = torch.ones(1).repeat(len(inferencedataloader)*(self.opts.inf_batchsize+1)).to(torch.float32) * int(max_angles[0])*math.pi/180
            new_pitch = torch.ones(1).repeat(len(inferencedataloader)*(self.opts.inf_batchsize+1)).to(torch.float32) * int(max_angles[1])*math.pi/180
            cameratype = 'fix' + cameratype # for video naming
        else: # GT camera
            cameratype = '' # for video naming
        face_editeds = []
        face_editeds_128 = []

        for j, (image, face_seg, body_seg, _, yaw, pitch, aud_eo, eye, angle, landmark) in tqdm(enumerate(inferencedataloader), desc='inferencing audio...'):
            yaw = yaw.to(self.device) if cameratype == '' else new_yaw[j*yaw.shape[0]:j*yaw.shape[0] + yaw.shape[0]].to(self.device)
            pitch = pitch.to(self.device) if cameratype == '' else new_pitch[j*yaw.shape[0]:j*yaw.shape[0] + yaw.shape[0]].to(self.device)
            null = self.null_vector.to(self.device)
            
            cond_feats = []
            for i in range(aud_eo.shape[0]):
                enc_a = self.audio_net(aud_eo[i]) #B, window, 44, 16 -> T, 64
                cond_feat = self.audio_att_net(enc_a.unsqueeze(0))
                if self.opts.lip_smoothing and smooth_cond_feat is not None:
                    cond_feat = _lambda * smooth_cond_feat + (1 - _lambda) * cond_feat
                    
                cond_feats.append(cond_feat) #T, 64 -> 1, 64
                smooth_cond_feat = cond_feat.unsqueeze(0) if len(cond_feat.shape)<3 else cond_feat
            cond_feats = torch.stack(cond_feats, dim=0)

            pred_w = self.w_person.repeat(cond_feats.shape[0], 14, 1)
            delta_plane = self.predict_delta_plane(cond_feats, eye, null, zero_plane=not self.opts.predict_deltaplane, angle = angle, landmark =landmark)

            face_edited = self.generator.generate(pred_w, yaw=yaw, pitch=pitch, triplane_offset=delta_plane, ID_triplane_dir=os.path.join(self.data_root, self.opts.ID_triplane_dir), output_all=True, grad=False)
            
            
            if self.tune_switch:
                face_editeds.append(self.GFPGAN.enhance_trainable(face_edited['image_raw'], has_aligned=True, paste_back=True, weight=0.5).cpu())
            else:
                face_editeds.append(face_edited['image'].cpu())

            images_gt.append(image.cpu())
            face_segs.append(face_seg.cpu())
            body_segs.append(body_seg.cpu())
            

        face_editeds = torch.cat(face_editeds, dim=0)
        
        if inferencetype == 'train':
            inf_audio_dir = os.path.join(self.data_root, self.opts.inference_wav)
        elif inferencetype == 'novel':
            inf_audio_dir = os.path.join(self.data_root, self.opts.inference_wav_novel)
        elif inferencetype == 'OOD':
            inf_audio_dir = os.path.join(self.data_root, self.opts.inference_wav_OOD)
            
        if self.device == 0:    
            vidname_ = f'{self.video_output_path}/{self.opts.short_configs}/{self.epoch_idx}_{self.iter_idx}' #temp
            imageio.mimwrite(f'{vidname_}_{type_}_{cameratype}_.mp4', tensor_to_image(face_editeds.cpu()), fps=25,  quality=8, macro_block_size=1)
            cmd = f'ffmpeg -loglevel quiet -y -i {vidname_}_{type_}_{cameratype}_.mp4 -i {inf_audio_dir} -max_muxing_queue_size 9999 -c:v copy -c:a aac {vidname_}_{type_}_{cameratype}.mp4'
            os.system(cmd)
            cmd = f'rm {vidname_}_{type_}_{cameratype}_.mp4'
            os.system(cmd)

            if self.opts.do_evaluation:
                body_segs = torch.cat(body_segs, dim=0)
                face_segs = torch.cat(face_segs, dim=0)
                images_gt = torch.cat(images_gt, dim=0)
                synth_video_dir = f'{vidname_}_{type_}_{cameratype}.mp4'
                self.evaluation_step(face_editeds, images_gt, face_segs, body_segs, synth_video_dir, inferencetype=inferencetype)
                    
        print('done.')
        torch.cuda.empty_cache()
        
        
    def evaluation_step(self, synth_image, gt_image, face_mask, body_mask, synth_video_dir, inferencetype='novel'):
        self.eval_dict = {}
        if inferencetype == 'novel':
            if self.opts.recon_eval_type == 'all' or self.opts.recon_eval_type == 'image':
                metric_image_dict = self.image2recon_metrics(synth_image, gt_image, type_='image')
                self.eval_dict.update(metric_image_dict)
            
            if self.opts.recon_eval_type == 'all' or self.opts.recon_eval_type == 'face':
                metric_face_dict = self.image2recon_metrics(synth_image*face_mask, gt_image*face_mask, type_='face')
                self.eval_dict.update(metric_face_dict)
            
            if self.opts.recon_eval_type == 'all' or self.opts.recon_eval_type == 'body':
                metric_body_dict = self.image2recon_metrics(synth_image*body_mask, gt_image*body_mask, type_='body')
                self.eval_dict.update(metric_body_dict)
        

        if self.opts.eval_type != 'only_recon':
            cmd = f'sh do_evaluation.sh {self.opts.personal_id} {self.opts.short_configs} {self.opts.data_root_dir} {synth_video_dir} {self.opts.eval_type} {inferencetype}'
            os.system(cmd)
            with open('./eval/metric.json', 'r') as metric_json:
                json_data = json.load(metric_json)

            self.eval_dict.update(json_data)
                
        
    def tensor2lpips(self,x,y):
        lpips = self.LPIPS(x,y)
        return lpips
    
    def image2recon_metrics(self, x, y, type_=''):
        output_dict = {}
        x = torch.clamp(x, max=1, min=-1)
        y = torch.clamp(y, max=1, min=-1)

        psnr = tensor2psnr(x, y, self.device, m=2)
        ssim = tensor2ssim(x, y)
        lpips = self.tensor2lpips(x.to(self.device), y.to(self.device))
        output_dict[f'{type_}_PSNR'] = psnr
        output_dict[f'{type_}_SSIM'] = ssim
        output_dict[f'{type_}_LPIPS'] = lpips
        
        return output_dict
    
    def initialize_G_tuning_stage(self, checkpoint_dir, use_gfpgan=True):
        # wrap G with ddp
        print('Initializing SR tuning stage...')
        print(f'Current num of step : Epoch : {self.epoch_idx}, Iters : {self.iter_idx}')
        del self.syncnet # unused when tuning
        self.G_tune_start = self.iter_idx # iteration counter for G tuning
        if use_gfpgan == False:
            for name, params in self.generator.active_G.named_parameters():
                if not name.startswith("superresolution"):
                    params.requires_grad = False
                else : 
                    params.requires_grad = True

            self.generator.use_ddp = True
            self.generator.active_G = DDP(self.generator.active_G)  
        else:
            for params in self.GFPGAN.gfpgan.parameters():
                params.requires_grad=True
            self.GFPGAN.gfpgan = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.GFPGAN.gfpgan)
            self.GFPGAN.gfpgan = DDP(self.GFPGAN.gfpgan, device_ids=[self.device], find_unused_parameters=True)
            if '_tune.pt' in checkpoint_dir:
                checkpoint = torch.load(checkpoint_dir)
                self.GFPGAN.gfpgan.module.load_state_dict(checkpoint['superresolution'])
            else:
                weight = torch.load(self.opts.gfpgan_weights_dir)
                self.GFPGAN.gfpgan.module.load_state_dict(weight['params_ema'], strict=True)
        
        # re-initialize optimizer with G params
        self.configure_optimizer(self.opts, tune=self.tune_switch)

    def logging(self):
        # loss graphs
        for key in self.loss_dict.keys():
            wandb.log({f'{key}': self.loss_dict[key]}, step=self.iter_idx)
            
        # evaluation metrics
        for key in self.eval_dict.keys():
            wandb.log({f'{key}': self.eval_dict[key]}, step=self.iter_idx)


    def dataloader_setup(self, opts, device, batchsize=3, outbatchsize=None):
        print(f'*******************************************************************************')
        print('Setup Dataloader...')
        'train set'
        total_data_length = len(glob.glob(f'{os.path.join(self.data_root, opts.image_dir)}/*.png'))
        train_length = int(total_data_length * opts.traintest_split_rate)
        test_start_idx = train_length
            
        Dataset_sync = Train_Dataset_sync(batchsize, 
                                            self.data_root, 
                                            opts,
                                            max_length=train_length, 
                                            )
        
        Dataset = Train_Dataset_nosync(self.data_root, opts, max_length=train_length)
        outbatchsize = batchsize
        Dataset_sampler = torch.utils.data.distributed.DistributedSampler(Dataset, rank=device, shuffle=True)
        self.TrainLoader = torch.utils.data.DataLoader(
                                            Dataset,
                                            num_workers=8, 
                                            batch_size=outbatchsize,
                                            drop_last=True,
                                            sampler=Dataset_sampler)
        
        Dataset_sync_sampler = torch.utils.data.distributed.DistributedSampler(Dataset_sync, rank=device, shuffle=False)
        self.TrainLoader_sync = torch.utils.data.DataLoader(
                                            Dataset_sync,
                                            num_workers=8, 
                                            batch_size=outbatchsize,
                                            drop_last=True,
                                            sampler=Dataset_sync_sampler)
        if opts.use_sync_loss:
            self.Loader = self.TrainLoader_sync
        else:
            self.Loader = self.TrainLoader
        
        'inference set'
        ValidDataset = Inference_Dataset(self.data_root,
                                                opts,
                                                opts.eye_smoothing, 
                                                start_frame=0,
                                                set_max_length=opts.val_max_length,
                                                )
        
        self.ValidLoader = torch.utils.data.DataLoader(
                                                ValidDataset,
                                                num_workers=8, 
                                                batch_size=opts.inf_batchsize,
                                                shuffle=False,
                                                drop_last=True)
        
        TestDataset = Inference_Dataset(self.data_root,
                                               opts,
                                               opts.eye_smoothing, 
                                               start_frame=test_start_idx, 
                                               set_max_length=opts.val_max_length,
                                               )
        
        self.TestLoader = torch.utils.data.DataLoader(
                                                TestDataset,
                                                num_workers=8, 
                                                batch_size=opts.inf_batchsize,
                                                shuffle=False,
                                                drop_last=True)
        
        OODDataset = Inference_Dataset(self.data_root,
                                               opts,
                                               opts.eye_smoothing, 
                                               start_frame=0, 
                                               set_max_length=opts.val_max_length, 
                                               is_OOD=True,
                                               )
        
        self.OODLoader = torch.utils.data.DataLoader(
                                                OODDataset,
                                                num_workers=8, 
                                                batch_size=opts.inf_batchsize,
                                                shuffle=False,
                                                drop_last=True)
        
    def setup_models(self, opts, device, load_checkpoint=False):
        self.deltaplane_predictor = Deltaplane_Predictor(96, vis_score_map=self.opts.vis_score_map).to(device)
        self.audio_net = AudioNet(44,64).to(device)
        self.audio_att_net = AudioAttNet(64).to(device)
        
        # blink
        self.blink_encoder = Blinkencoder(16, 64, 3).to(device) # in_dim, out_dim, n_layers, type_='constant'
        self.camera_encoder = Cameraencoder(16, 64, 3).to(device)# in_dim, out_dim, n_layers, type_='constant'
        
        #init GFPGAN
        if opts.use_GFPGAN:
            model_path = self.opts.gfpgan_weights_dir
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.3'
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True)
            self.GFPGAN = GFPGANer(
                model_path=model_path,
                upscale=2,
                arch=arch,
                channel_multiplier=channel_multiplier,
                bg_upsampler=bg_upsampler,
                device = self.device)
        
        
        optimizable_list = self.setup_model_ddp(device)
        if load_checkpoint:
            self.load_chkpts(opts.checkpoint_dir)
        else:
            print('Train from scratch')
            
            self.epoch_idx = 0
            self.iter_idx = 0
            
        return optimizable_list
        
    def setup_model_ddp(self, device):
        optimizable_list = []

        'delta plane predictor'
        self.deltaplane_predictor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.deltaplane_predictor)
        self.deltaplane_predictor = DDP(self.deltaplane_predictor, device_ids=[device])
        self.blink_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.blink_encoder)
        self.blink_encoder = DDP(self.blink_encoder, device_ids=[device])
        self.camera_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.camera_encoder)
        self.camera_encoder = DDP(self.camera_encoder, device_ids=[device])

        optimizable_list.append(self.deltaplane_predictor)
        optimizable_list.append(self.blink_encoder)
        optimizable_list.append(self.camera_encoder)
            
        'radnerf audio feature extraction'
        self.audio_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.audio_net)
        self.audio_att_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.audio_att_net)
        self.audio_net = DDP(self.audio_net, device_ids=[device])
        self.audio_att_net = DDP(self.audio_att_net, device_ids=[device])

        optimizable_list.append(self.audio_net)
        optimizable_list.append(self.audio_att_net)

        'zeroconv init'
        if not self.load_checkpoint: # if loaded checkpoint, zero conv doesnt needed
            for i, params in self.deltaplane_predictor.module.named_parameters(): 
                if 'Upnet3' in i and 'conv1' in i: #in modulated_conv2d : x * affine style makes gradient 0 - conv_weight * affine_weight = 0 -> zero gradient
                    params.detach().zero_()
                params.requires_grad = True
        
        return optimizable_list

        
    def configure_optimizer(self, opts, tune=False):
        # Commencing tuning step : re-configure params and optimizers
        if tune:
            params_list = []
            if opts.use_GFPGAN:
                for params in self.GFPGAN.gfpgan.module.parameters():
                    params_list.append(params)
            else: # use EG3D SR module
                for name, module in self.generator.active_G.module.named_modules():
                    if 'superresolution' in name:
                        for p in module.parameters():
                            p.requires_grad = True
                            params_list.append(p)
                    else:
                        for p in module.parameters():
                            p.requires_grad = False

            self.optimizer = torch.optim.Adam(params_list, lr=opts.learning_rate_tune)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(opts.scheduler_step_size), gamma=float(opts.gamma))
        else:
            params_list = []
            for model in self.optimizable_list:
                params_list += list(model.module.parameters())
            
            self.optimizer = torch.optim.Adam(params_list + [self.null_vector], lr=opts.learning_rate)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(opts.scheduler_step_size), gamma=float(opts.gamma))


    def save_model_weights(self, tune=False):
        print('Saving checkpoint....')
        
        if self.tune_switch:
            params_path = f'{self.model_save_path}/{self.opts.short_configs}/{self.epoch_idx}_{self.iter_idx}_tune.pt'
            sr_dict = self.GFPGAN.gfpgan.module.state_dict() if self.opts.use_GFPGAN else self.generator.active_G.module.superresolution.state_dict()
            torch.save({
                'superresolution' : sr_dict,
                'deltaplane_predictor' : self.deltaplane_predictor.module.state_dict(),
                'blink_encoder' : self.blink_encoder.module.state_dict(),
                'camera_encoder' : self.camera_encoder.module.state_dict(),
                'audio_net': self.audio_net.module.state_dict(),
                'audio_att_net': self.audio_att_net.module.state_dict(),
                'null_vector': self.null_vector,
                'epoch': self.epoch_idx,
                'iter': self.iter_idx,
                'optimizer': self.optimizer,
                'scheduler': self.scheduler,
                }, params_path)
        else:
            params_path = f'{self.model_save_path}/{self.opts.short_configs}/{self.epoch_idx}_{self.iter_idx}.pt'
            torch.save({
                'deltaplane_predictor' : self.deltaplane_predictor.module.state_dict(),
                'blink_encoder' : self.blink_encoder.module.state_dict(),
                'camera_encoder' : self.camera_encoder.module.state_dict(),
                'audio_net': self.audio_net.module.state_dict(),
                'audio_att_net': self.audio_att_net.module.state_dict(),
                'null_vector': self.null_vector,
                'epoch': self.epoch_idx,
                'iter': self.iter_idx,
                'optimizer': self.optimizer,
                'scheduler': self.scheduler,
                }, params_path)
        print('Checkpoint saved.')

    def load_chkpts(self, checkpoint_dir):
        checkpoint = torch.load(checkpoint_dir)
        
        loaded_list = [i for i in checkpoint.keys()]
        failed_list = []
        for key in checkpoint.keys():
            if key == 'epoch':
                self.epoch_idx = checkpoint[key]
                continue
            if key == 'iter':
                self.iter_idx = checkpoint[key]
                continue
            if key == 'null_vector':
                self.null_vector = checkpoint[key]
                continue
            
            # if key == 'superresolution':
            #     if self.opts.use_GFPGAN:
            #         self.GFPGAN.gfpgan.module.load_state_dict(checkpoint[key])
            #     else:
            #         self.generator.active_G.superresolution.load_state_dict(checkpoint[key])
            #     continue
            
            if checkpoint[key] == None:
                failed_list.append(key)
                loaded_list.remove(key)
                continue
            try:
                eval(f'self.{key}').module.load_state_dict(checkpoint[key]) #eval(key) == model 변수
            except:
                try:
                    eval(f'self.{key}').load_state_dict(checkpoint[key])
                except:
                    failed_list.append(key)
                    loaded_list.remove(key)
                
        print('Checkpoint Loading Finished. details ----------------------')
        print(f'Load checkpoint from epoch :{self.epoch_idx}')
        print(f'Loaded :{loaded_list}')
        print(f'Failed :{failed_list}')
        print('-----------------------------------------------------------')
        
    def set_saving_dirs(self, opts):
        saveroot_path = opts.saveroot_path
        personal_id = opts.personal_id
        short_configs = opts.short_configs
        os.makedirs(saveroot_path, exist_ok=True)
        self.video_output_path = f'{saveroot_path}/video/{personal_id}'
        self.model_save_path = f'{saveroot_path}/weight/{personal_id}'
        self.config_save_path = f'{saveroot_path}/configs/{personal_id}'
        self.att_map_save_path = f'{saveroot_path}/scoremaps/{personal_id}'
        os.makedirs(f'{self.video_output_path}/{short_configs}', exist_ok=True)
        os.makedirs(f'{self.model_save_path}/{short_configs}', exist_ok=True)
        os.makedirs(f'{self.config_save_path}/{short_configs}', exist_ok=True)
        os.makedirs(f'{self.att_map_save_path}/{short_configs}', exist_ok=True)
        
    def trim_and_save_inference_wav(self, sr=44100, fps=25):
        '''
        automatically trim and save inference wav file (default=10/11)
        '''
        import librosa
        import soundfile as sf
        print('Trimming wav files...')
        wav_raw_dir = os.path.join(self.data_root, self.opts.inference_wav_raw)
        wav_OOD_raw_dir = os.path.join(self.data_root, self.opts.inference_wav_OOD_raw)

        wav_dir = os.path.join(self.data_root, self.opts.inference_wav)
        wav_novel_dir = os.path.join(self.data_root, self.opts.inference_wav_novel)
        wav_OOD_dir = os.path.join(self.data_root, self.opts.inference_wav_OOD)

        wav_raw, _ = librosa.load(wav_raw_dir, sr=sr)
        wav_OOD_raw, _ = librosa.load(wav_OOD_raw_dir, sr=sr)

        if self.opts.val_max_length < 0:
            time_duration = int(wav_raw.shape[0]*self.opts.traintest_split_rate)
        else:
            time_duration = (self.opts.val_max_length * sr) // fps
        

        wav_train = self.trim_wav(wav_raw, sr, split_rate=0, duration_second=time_duration)
        wav_novel = self.trim_wav(wav_raw, sr, split_rate = self.opts.traintest_split_rate, duration_second=time_duration, is_novel=True)
        wav_OOD = self.trim_wav(wav_OOD_raw, sr, split_rate=0, duration_second=time_duration)

        sf.write(wav_dir, wav_train, sr, 'PCM_24')
        sf.write(wav_novel_dir, wav_novel, sr, 'PCM_24')
        sf.write(wav_OOD_dir, wav_OOD, sr, 'PCM_24')

    def trim_wav(self, wav_np, sr, split_rate=0, duration_second=0, is_novel=False): # split rate == (n-1)/n
        total_length = wav_np.shape[0]
        total_seconds = total_length / sr

        duration = duration_second
        if self.opts.personal_id == 'Jae-in' and is_novel:
            split_rate += (1-split_rate)/2

        split_start_second = total_seconds * split_rate
        split_start = round(split_start_second * sr)
        return wav_np[split_start:split_start+duration]
        
    def model2eval(self, tune_switch):
        self.generator.active_G.eval()
        self.deltaplane_predictor.eval()
        self.blink_encoder.eval()
        self.camera_encoder.eval()
        self.audio_net.eval()
        self.audio_att_net.eval()
        if tune_switch:
            if self.opts.use_GFPGAN:
                self.GFPGAN.gfpgan.eval()
            else:
                self.generator.active_G.module.superresolution.eval()
            
    def model2train(self, tune_switch):
        if not tune_switch:
            self.generator.active_G.train()
            self.deltaplane_predictor.train()
            self.blink_encoder.train()
            self.camera_encoder.train()
            self.audio_net.train()
            self.audio_att_net.train()
        else:
            self.generator.active_G.eval()
            self.deltaplane_predictor.eval()
            self.blink_encoder.eval()
            self.camera_encoder.eval()
            self.audio_net.eval()
            self.audio_att_net.eval()
            if self.opts.use_GFPGAN:
                self.GFPGAN.gfpgan.train()
            else:
                self.generator.active_G.module.superresolution.train()
