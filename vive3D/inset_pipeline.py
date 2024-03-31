from vive3D import config
from tqdm import tqdm
from vive3D.segmenter import Segmenter
import torch
import lpips
import torch.nn.functional as F
import matplotlib.pyplot as plt
from vive3D.visualizer import *
from vive3D.aligner import *
from vive3D.util import *
from torchvision.transforms.functional import gaussian_blur
import cv2 as cv
from scipy.ndimage import gaussian_filter1d
from torchvision.utils import save_image

# porting Space Regularizer from PTI
class Space_Regularizer:
    def __init__(self, G, lpips_net, l2):
        self.generator = G
        self.morphing_regularizer_alpha = 0.1 
        self.regularizer_l2_lambda = 0.05 
        self.regularizer_lpips_lambda = 0.05 
        self.latent_ball_num_of_samples = 1
        self.lpips_loss = lpips_net
        self.l2_loss = l2

    def get_morphed_w_code(self, new_w_code, fixed_w):
        interpolation_direction = new_w_code - fixed_w
        interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
        direction_to_move = self.morphing_regularizer_alpha * interpolation_direction / interpolation_direction_norm
        result_w = fixed_w + direction_to_move
        self.morphing_regularizer_alpha * fixed_w + (1 - self.morphing_regularizer_alpha) * new_w_code

        return result_w

    def get_image_from_ws(self, w_codes, G):
        return torch.cat([G.synthesis(w_code, noise_mode='none', force_fp32=True) for w_code in w_codes])

    def ball_holder_loss_lazy(self, new_G, num_of_sampled_latents, w_batch):
        loss = 0.0
        
        z_samples = np.random.randn(len(w_batch), self.generator.G.z_dim)
        w_samples = self.generator.get_ws(torch.from_numpy(z_samples).to(self.generator.device), truncation_psi=0.5)
        territory_indicator_ws = [self.get_morphed_w_code(w_code.unsqueeze(0), w_b) for w_code, w_b in zip(w_samples, w_batch)]

        for w_code in territory_indicator_ws:
            self.generator.active_G = self.generator.G_tune
            new_img = self.generator.generate_original(w_code, grad=True)
            with torch.no_grad():
                self.generator.active_G = self.generator.G
                old_img = self.generator.generate_original(w_code, grad=True)

            if self.regularizer_l2_lambda > 0:
                l2_loss_val = self.l2_loss(old_img, new_img)
                loss += l2_loss_val * self.regularizer_l2_lambda

            if self.regularizer_lpips_lambda > 0:
                loss_lpips = self.lpips_loss(old_img, new_img)
                loss_lpips = torch.mean(torch.squeeze(loss_lpips))
                loss += loss_lpips * self.regularizer_lpips_lambda
            self.generator.active_G = self.generator.G_tune
        return loss / len(territory_indicator_ws)

    def space_regularizer_loss(self, new_G, w_batch):
        ret_val = self.ball_holder_loss_lazy(new_G, self.latent_ball_num_of_samples, w_batch)
        return ret_val
    
# ID loss only relevant when ID loss is chosen (not default)
class IDLoss(torch.nn.Module):
    def __init__(self):
        from model_irse import Backbone
        
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(f'{config.HOME}/Users/Anna/insetGAN/models/vive3D_se50.pth'))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats
    
    def forward(self, y_hat, y):
        y_feats = self.extract_feats(y).detach()  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        loss = 0
        count = 0
        for i in range(y.shape[0]):
            loss += (1 - y_hat_feats[i].dot(y_feats[i]))
            count += 1
        return loss / count

class BicubicDownSample(torch.nn.Module):
    def bicubic_kernel(self, x, a=-0.50):
        """
        This equation is exactly copied from the website below:
        https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic
        """
        abs_x = torch.abs(x)
        if abs_x <= 1.:
            return (a + 2.) * torch.pow(abs_x, 3.) - (a + 3.) * torch.pow(abs_x, 2.) + 1
        elif 1. < abs_x < 2.:
            return a * torch.pow(abs_x, 3) - 5. * a * torch.pow(abs_x, 2.) + 8. * a * abs_x - 4. * a
        else:
            return 0.0

    def __init__(self, factor=4, cuda=True, device='cuda', padding='reflect'):
        super().__init__()
        self.factor = factor
        size = factor * 4
        k = torch.tensor([self.bicubic_kernel((i - torch.floor(torch.tensor(size / 2)) + 0.5) / factor)
                          for i in range(size)], dtype=torch.float32).to(device)
        k = k / torch.sum(k)
        k1 = torch.reshape(k, shape=(1, 1, size, 1))
        self.k1 = torch.cat([k1, k1, k1], dim=0).to(device)
        k2 = torch.reshape(k, shape=(1, 1, 1, size))
        self.k2 = torch.cat([k2, k2, k2], dim=0).to(device)
        self.cuda = '.cuda' if cuda else ''
        self.device = device
        self.padding = padding
        for param in self.parameters():
            param = param.to(device)
            param.requires_grad = False

    def forward(self, x, nhwc=False, clip_round=False, byte_output=False):
        filter_height = self.factor * 4
        filter_width = self.factor * 4
        stride = self.factor

        pad_along_height = max(filter_height - stride, 0)
        pad_along_width = max(filter_width - stride, 0)
        filters1 = self.k1.to(self.device)
        filters2 = self.k2.to(self.device)

        # compute actual padding values for each side
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        # apply mirror padding
        if nhwc:
            x = torch.transpose(torch.transpose(
                x, 2, 3), 1, 2)   # NHWC to NCHW

        # downscaling performed by 1-d convolution
        x = F.pad(x, (0, 0, pad_top, pad_bottom), self.padding).to(self.device)
        x = F.conv2d(input=x, weight=filters1, stride=(stride, 1), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.)

        x = F.pad(x, (pad_left, pad_right, 0, 0), self.padding).to(self.device)
        x = F.conv2d(input=x, weight=filters2, stride=(1, stride), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.)

        if nhwc:
            x = torch.transpose(torch.transpose(x, 1, 3), 1, 2)
        if byte_output:
            return x.type('torch.ByteTensor'.format(self.cuda))
        else:
            return x
    
class Pipeline:
    def __init__(self, generator, segmenter=None, aligner=None, landmark_detector=None, device='cuda'):
        self.generator = generator
        self.device = device
        self.loss_L2 = torch.nn.MSELoss(reduction='sum').to(device) #
        self.loss_L1 = torch.nn.L1Loss(reduction='sum').to(device) #
        self.loss_percept = lpips.LPIPS(net='alex').to(device)
        
        self.locality_regularization_interval = 1
        self.use_ball_holder = True
        
        # Load VGG16 feature detector
        # download from 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        self.vgg16 = torch.jit.load('./models/vgg16.pt').eval().to(device)
        
        self.downsampler_128 = BicubicDownSample(factor=512//128, device=device).to(device)
        
        self.segmenter = Segmenter(device=device) if segmenter is None else segmenter
        self.aligner = Aligner(device=device) if aligner is None else aligner
        self.landmark_detector = landmark_detector
        self.input_image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
    def tune(self,
             w_latent, 
             w_latent_offsets,
             estimated_yaw, 
             estimated_pitch,
             input_images,
             input_segmentations=None,
             num_steps = 400,
             learning_rate = 0.001,
             weight_l1 = 1.0,
             weight_lpips = 1.0,
             plot_progress = True,
             reload_generator=True,
             output_progress_frames=False,
             image_log_step = 50,
             name=None,
             out_folder=None):
        
        if plot_progress:
            fig = plt.figure(figsize=(20, 20), dpi=100)
            hfig = display(fig, display_id=True)
            ax = plt.gca()
            if output_progress_frames:
                progress_images = []
                frames_per_rot = 60
                steps = np.linspace(0, 1, frames_per_rot)
                pitches_circle = 0.3 * np.sin(steps * 2 * np.pi)
                yaws_circle = 0.3 * np.cos(steps * 2 * np.pi)
        
        num_images = len(input_images)
        # os.makedirs(f'./tune/{name}', exist_ok=True)
        # clone Generator to optimize
        self.generator.tune(force=reload_generator)
    
        res = 512

        self.space_regularizer = Space_Regularizer(self.generator, self.loss_percept, self.loss_L2)
        
        w_latent.requires_grad = False
        w_latent_offsets.requires_grad = False
        estimated_yaw.requires_grad = False
        estimated_pitch.requires_grad = False

        optimizer = torch.optim.Adam(self.generator.G_tune.parameters(), lr=learning_rate)
    
        target_images = input_images.to(self.device)
        target_images = target_images.to(torch.float)
        target_images_128 = self.downsampler_128(target_images)
        
        use_segmentation = input_segmentations is not None
        
        if use_segmentation:
            target_segmentations = input_segmentations.float().to(self.device)
            target_segmentations_128 = torch.round(self.downsampler_128(target_segmentations.repeat(1, 3, 1, 1))).to(torch.uint8).to(self.device)
            target_foreground = target_segmentations*target_images
            
            num_foreground_px = target_segmentations.sum()
            target_foreground_128 = target_segmentations_128 * target_images_128
            target_background_128 = (1-target_segmentations_128) * target_images_128

        pbar = tqdm(range(num_steps))
        count = 0

        out_folder_tune = os.path.join(out_folder, 'tune')
        os.makedirs(out_folder_tune, exist_ok=True)
        for i in pbar:
            loss_text = ''

            synth_images_128 = []
            synth_images = []
    
            ws = w_latent.repeat(num_images, 1) + w_latent_offsets
            generated = self.generator.generate_original(ws, estimated_yaw, estimated_pitch, output_all=True, grad=True)
            
            synth_images_128 = generated['image_raw']
            synth_images = generated['image']
            loss = 0.0
            
            if use_segmentation:
                synth_foreground = target_segmentations_128*synth_images_128

                loss_l1 = self.loss_L1(synth_foreground, target_foreground_128) / num_foreground_px
                loss_lpips = self.loss_percept(synth_foreground, target_foreground_128).sum()
            else:
                loss_l1 = self.loss_L1(synth_images_128, target_images_128) / ( res ** 2 )
                loss_lpips = self.loss_percept(synth_images, target_images).sum()
            
            if self.use_ball_holder:
                ball_holder_loss_val = self.space_regularizer.space_regularizer_loss(self.generator, ws)
                loss += 0.0001 * ball_holder_loss_val
            
            loss += weight_l1 * loss_l1
            loss += weight_lpips * loss_lpips
            loss_text += f'PIX: {weight_l1*loss_l1.detach().cpu():.4f}  LPIPS: {weight_lpips*loss_lpips.sum().detach().cpu():.4f}  total: {loss.detach().cpu():.4f}'

            optimizer.zero_grad()

            if loss_lpips.sum() <= 0.006:
                break

            if count == 0:
                self.saveimg(target_images.clone().detach().cpu(), f'{out_folder_tune}/input.png')
            if count%20==0:
                self.saveimg(synth_images.clone().detach().cpu(), f'{out_folder_tune}/{str(count).zfill(4)}.png')

            loss.backward()
            optimizer.step()
            pbar.set_description(loss_text)
            
            self.use_ball_holder = i % self.locality_regularization_interval == 0
            
            if plot_progress:
                if i % image_log_step == 0:
                    log = []

                    raw_256 = F.interpolate(generated['image_raw'], size=(256, 256), mode='nearest')
                    for idx in range(len(target_images)):
                        log_image = generated['image'][idx].unsqueeze(0)
                        depth_image = self.normalize_depth(generated['image_depth'][idx], resize=256)
                        dr = torch.cat((depth_image, raw_256[idx].unsqueeze(0)), axis=-1)
                        log.append(torch.cat((target_images[idx].unsqueeze(0), log_image, dr), axis=-2))

                    #add default person
                    if output_progress_frames:
                        default_person = self.generator.generate_original(w_latent, yaws_circle[i%frames_per_rot], pitches_circle[i%frames_per_rot], output_all=True, grad=False)
                    else:
                        default_person = self.generator.generate_original(w_latent, 0.0, 0.0, output_all=True, grad=False)
                    log_image = default_person['image']
                    depth_image = self.normalize_depth(default_person['image_depth'], resize=512)

                    log.append(torch.cat((log_image, depth_image, torch.zeros_like(dr)), axis=-2))

                    Visualizer.show_tensor(torch.cat(log, axis=-1), ax=ax)

                    fig.canvas.draw()
                    hfig.update(fig)
                    if output_progress_frames:
                        progress_images.append(tensor_to_image(torch.cat(log, axis=-1)))
                plt.close(fig)
            count += 1
        if output_progress_frames:
            return progress_images
    
    def inversion_progressive( self, 
                  input_images, 
                  input_segmentations=None, 
                  w_opt=None,
                  w_offsets=None,
                  y_opt=None,
                  p_opt=None,
                  num_steps = 600,
                  initial_learning_rate = 5e-3,
                  initial_noise_factor = 0.02,
                  lr_rampdown_length = 0.25,
                  lr_rampup_length = 0.05,
                  noise_ramp_length = 0.75,
                  regularize_noise_weight = 1e5,
                  weight_vgg = 0.0,
                  weight_id = 0.0,
                  weight_pix = 0.01,
                  weight_face = 2.5,
                  weight_lpips = 0.8,
                  weight_wdist = 0.03,
                  weight_wdist_target = 0.01,
                  plot_progress = True,
                  output_progress_frames = False,
                  image_log_step = 50,
                  pbar_log_step = 5 ): 
    
        target_images = input_images.to(self.device)
        use_segmentation = input_segmentations is not None
        edge_size = 24
        
        num_images = len(target_images)

        if w_opt==None:
            w_opt = self.generator.get_average_w().clone().to(self.device) 
        else:
            w_opt = w_opt.to(self.device) 
        w_opt.requires_grad = True
        
        if y_opt==None:
            y_opt = torch.zeros((num_images), dtype=torch.float32, device=self.device, requires_grad=True)
        else:
            y_opt = y_opt.to(self.device) 
            y_opt.requires_grad = True
        if p_opt==None:
            p_opt = torch.zeros((num_images), dtype=torch.float32, device=self.device, requires_grad=True)
        else:
            p_opt = p_opt.to(self.device) 
            p_opt.requires_grad = True
            
        if w_offsets==None:
            w_offsets = torch.randn_like(w_opt.repeat(num_images, 1)) * initial_noise_factor
        else:
            w_offsets = w_offsets.to(self.device) 
        w_offsets.requires_grad = True
        
        w_noise_scale = self.generator.get_w_std() * initial_noise_factor
        w_noise = torch.randn_like(w_offsets) * w_noise_scale
        ws = w_opt.repeat(num_images, 1) + w_noise + w_offsets
        out = self.generator.generate_original(ws, y_opt, p_opt)
        target_images_128 = self.downsampler_128(target_images)
        segmentations = None 

        if use_segmentation:
            target_segmentations = input_segmentations.float().to(self.device)
            target_segmentations_128 = torch.round(self.downsampler_128(target_segmentations.repeat(1, 3, 1, 1))).to(torch.uint8).to(self.device)
    
            target_foreground = target_segmentations*target_images
            target_foreground_128 = target_segmentations_128*target_images_128
            num_foreground_px = target_segmentations.sum()
            num_foreground_128_px = target_segmentations_128.to(torch.uint8).sum() / 3
            print(f'{num_foreground_128_px} foreground px')
            
        if weight_vgg > 0:
            target_images_norm = (target_images + 1) * (255 / 2)
            target_features = self.vgg16(target_images_norm, resize_images=False, return_lpips=True)
        
        if weight_face > 0:
            target_faces_segmentations = self.segmenter.get_eyes_mouth_BiSeNet(target_images, dilate=4)
            target_faces_segmentations = target_faces_segmentations.any(dim=0).repeat(num_images, 1, 1, 1)
            num_face_px = target_faces_segmentations.sum()
            target_faces = target_images * target_faces_segmentations
        
        if weight_id > 0:
            target_ID_features = loss_ID.extract_feats(target_segmentations*target_images if use_segmentation else target_images)

        optimization_criteria = [w_opt] + [w_offsets] + [y_opt] + [p_opt] 
        optimizer = torch.optim.Adam(optimization_criteria, betas=(0.9, 0.999), lr=initial_learning_rate)
        
        if plot_progress:
            fig = plt.figure(figsize=(20, 20), dpi=100)
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            hfig = display(fig, display_id=True)
            ax = plt.gca()
            if output_progress_frames:
                progress_images = []
                # do a circle rotation for the default person
                frames_per_rot = 60
                steps = np.linspace(0, 1, frames_per_rot)
                pitches_circle = 0.3 * np.sin(steps * 2 * np.pi)
                yaws_circle = 0.3 * np.cos(steps * 2 * np.pi)
    
        pbar = tqdm(range(num_steps))
        
        for step in pbar:
            
            # Learning rate schedule.
            t = step / num_steps
            w_noise_scale = self.generator.get_w_std() * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * torch.cos(torch.tensor(lr_ramp * torch.pi))
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = initial_learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            loss = 0    

            log = []

            w_noise = torch.randn_like(w_offsets) * w_noise_scale
            ws = w_opt.repeat(num_images, 1) + w_noise + w_offsets
            out = self.generator.generate_original(ws, y_opt, p_opt, output_all=True, grad=True)
            synth_images_128 = out['image_raw']
            synth_images = out['image']
            
            if use_segmentation:# or segmentations==None:
                segmentations = self.segmenter.get_foreground_BiSeNet(synth_images, background_classes=[0])
                segmentations_128 = torch.round(self.downsampler_128(segmentations.repeat(1, 3, 1, 1))).to(torch.uint8).to(self.device)

                synth_images_foreground = segmentations * synth_images 
                synth_images_foreground_128 = segmentations_128*synth_images_128
 

            if weight_vgg > 0:
                synth_images_norm = (synth_images + 1) * (255 / 2) 
                features_vgg = self.vgg16(synth_images_norm, resize_images=False, return_lpips=True) 
                l_vgg = (target_features - features_vgg).square().sum()   
                loss += weight_vgg * l_vgg  

            if weight_pix > 0: 
                res = 128 
                l_pix = self.loss_L1(synth_images_foreground_128, target_foreground_128) / num_foreground_128_px if use_segmentation else self.loss_L1(synth_images_128, target_images_128) / ( res ** 2 )
                loss += weight_pix * l_pix

            if weight_lpips > 0:
                l_lpips = self.loss_percept(synth_images_foreground_128, target_foreground_128).sum() if use_segmentation else self.loss_percept(synth_images_128, target_images_128).sum()
                
                loss += weight_lpips * l_lpips

            if weight_face > 0:
                generated_faces = synth_images*target_faces_segmentations
                l_face = self.loss_percept(generated_faces, target_faces).sum() + 0.75*self.loss_L2(generated_faces, target_faces).sum() / num_face_px 
                loss += weight_face * l_face 

            if weight_wdist > 0:            
                l_wdist = torch.linalg.norm(w_offsets)
                loss += weight_wdist * l_wdist

                if step > 100 and weight_wdist * 0.99 > weight_wdist_target:
                    weight_wdist *= 0.99   
            
            # Optimizer step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            # if plotting, create log
            if plot_progress and ( step % image_log_step == 0 or step==(num_steps-1) ):
                for idx in range(num_images):  
                    synth_image = synth_images[idx].unsqueeze(0)
                    log_items = [target_images[idx].unsqueeze(0), synth_image]
                    if use_segmentation:
                        log_items.append(synth_images_foreground[idx].unsqueeze(0))
                        log_items.append(target_foreground[idx].unsqueeze(0))
                    if weight_face > 0:
                        log_items.append(generated_faces[idx].unsqueeze(0))
                    log_items.append(target_faces[idx].unsqueeze(0))
                    log.append(torch.cat(log_items, axis=-2))
                    
                if output_progress_frames:
                    generated = self.generator.generate_original(w_opt, yaws_circle[(step//image_log_step)%frames_per_rot], pitches_circle[(step//image_log_step)%frames_per_rot], output_all=True, grad=False)
                else:
                    generated = self.generator.generate_original(w_opt, 0, -0.1, output_all=True, grad=False)
                
                image = generated['image']
                depth_image = self.normalize_depth(generated['image_depth'], resize=512)
                            
                log_items = [image, depth_image]
                if use_segmentation:
                    fg = self.segmenter.get_foreground_BiSeNet(image)
                    log_items.append(fg * image)
                    log_items.append(torch.zeros_like(image))
                if weight_face > 0: 
                    face = self.segmenter.get_face_BiSeNet(image)
                    log_items.append(face * image)
                    
                log_items.append(torch.zeros_like(image))
                log.append(torch.cat(log_items, axis=-2))

                Visualizer.show_tensor(torch.cat(log, axis=-1), ax=ax)
                
                fig.canvas.draw()
                hfig.update(fig)
                
                if output_progress_frames:
                    progress_images.append(tensor_to_image(torch.cat(log, axis=-1)))
                
            if step % pbar_log_step == 0 or step==(num_steps-1):
                angles = '['
                for n, (y, p) in enumerate(zip(y_opt.cpu(), p_opt.cpu())):
                    if n != 0:
                        angles += '|'
                    angles += f'{y.clone().detach().numpy():<3.2f},{p.clone().detach().numpy():<3.2f}'
                desc = f'A={angles}] ' 

                desc += f'VGG={weight_vgg*l_vgg:<4.2f} ' if weight_vgg > 0 else ''
                desc += f'ID={weight_id*l_id:<4.2f} ' if weight_id > 0 else ''
                desc += f'PX={weight_pix*l_pix:<4.2f} ' if weight_pix > 0 else ''
                desc += f'LP={weight_lpips*l_lpips:<4.2f} ' if weight_lpips > 0 else ''
                desc += f'LF={weight_face*l_face:<4.2f} ' if weight_face > 0 else ''
                desc += f'WD={weight_wdist*l_wdist:<4.2f}(x{weight_wdist:.3f}) ' if weight_wdist > 0 else ''
                desc += f'L={float(loss):<5.2f}'
                pbar.set_description(f'{desc}')
                
        plt.close(fig)   
        
        if output_progress_frames:
            return w_opt, w_offsets, y_opt, p_opt, progress_images 
        else:
            return w_opt, w_offsets, y_opt, p_opt 
        
    def inversion( self, 
                  input_images, 
                  input_segmentations=None, 
                  num_steps = 600,
                  initial_learning_rate = 5e-3,
                  initial_noise_factor = 0.02,
                  lr_rampdown_length = 0.25,
                  lr_rampup_length = 0.05,
                  noise_ramp_length = 0.75,
                  regularize_noise_weight = 1e5,
                  weight_vgg = 0.0,
                  weight_id = 0.0,
                  weight_pix = 0.01,
                  weight_face = 2.5,
                  weight_lpips = 0.8,
                  weight_wdist = 0.03,
                  weight_wdist_target = 0.01,
                  plot_progress = True,
                  output_progress_frames = False,
                  image_log_step = 50,
                  pbar_log_step = 5,
                  name=None,
                  out_folder=None): 
    
        target_images = input_images.to(self.device)
        target_images = target_images.to(torch.float)
        use_segmentation = input_segmentations is not None
        edge_size = 24
        # os.makedirs(f'./inv/{name}', exist_ok=True)
        num_images = len(target_images)


        w_opt = self.generator.get_average_w().clone().to(self.device) 
        w_opt.requires_grad = True
        
        y_opt = torch.zeros((num_images), dtype=torch.float32, device=self.device, requires_grad=True)
        p_opt = torch.zeros((num_images), dtype=torch.float32, device=self.device, requires_grad=True)

        w_offsets = torch.randn_like(w_opt.repeat(num_images, 1)) * initial_noise_factor
        w_offsets.requires_grad = True
        
        w_noise_scale = self.generator.get_w_std() * initial_noise_factor
        w_noise = torch.randn_like(w_offsets) * w_noise_scale
        ws = w_opt.repeat(num_images, 1) + w_noise + w_offsets
        out = self.generator.generate_original(ws, y_opt, p_opt)
        target_images_128 = self.downsampler_128(target_images)
        segmentations = None 

        if use_segmentation:
            target_segmentations = input_segmentations.float().to(self.device)
            target_segmentations_128 = torch.round(self.downsampler_128(target_segmentations.repeat(1, 3, 1, 1))).to(torch.uint8).to(self.device)
    
            target_foreground = target_segmentations*target_images
            target_foreground_128 = target_segmentations_128*target_images_128
            num_foreground_px = target_segmentations.sum()
            num_foreground_128_px = target_segmentations_128.to(torch.uint8).sum() / 3
            print(f'{num_foreground_128_px} foreground px')
            
        if weight_vgg > 0:
            target_images_norm = (target_images + 1) * (255 / 2)
            target_features = self.vgg16(target_images_norm, resize_images=False, return_lpips=True)
        
        if weight_face > 0:
            target_faces_segmentations = self.segmenter.get_eyes_mouth_BiSeNet(target_images, dilate=1)
            target_faces_segmentations = target_faces_segmentations.any(dim=0).repeat(num_images, 1, 1, 1)
            num_face_px = target_faces_segmentations.sum()
            target_faces = target_images * target_faces_segmentations
        
        if weight_id > 0:
            target_ID_features = loss_ID.extract_feats(target_segmentations*target_images if use_segmentation else target_images)

        optimization_criteria = [w_opt] + [w_offsets] + [y_opt] + [p_opt] 
        optimizer = torch.optim.Adam(optimization_criteria, betas=(0.9, 0.999), lr=initial_learning_rate)
        
        if plot_progress:
            fig = plt.figure(figsize=(20, 20), dpi=100)
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            hfig = display(fig, display_id=True)
            ax = plt.gca()
            if output_progress_frames:
                progress_images = []
                # do a circle rotation for the default person
                frames_per_rot = 60
                steps = np.linspace(0, 1, frames_per_rot)
                pitches_circle = 0.3 * np.sin(steps * 2 * np.pi)
                yaws_circle = 0.3 * np.cos(steps * 2 * np.pi)
    
        pbar = tqdm(range(num_steps))
        count = 0
        out_folder_inv = os.path.join(out_folder, 'inv')
        os.makedirs(out_folder_inv, exist_ok=True)
        
        for step in pbar:
            
            # Learning rate schedule.
            t = step / num_steps
            w_noise_scale = self.generator.get_w_std() * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * torch.cos(torch.tensor(lr_ramp * torch.pi))
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = initial_learning_rate * lr_ramp
            if lr == 0:
                lr = 0.0001
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            loss = 0    

            log = []

            w_noise = torch.randn_like(w_offsets) * w_noise_scale
            ws = w_opt.repeat(num_images, 1) + w_noise + w_offsets
            out = self.generator.generate_original(ws, y_opt, p_opt, output_all=True, grad=True)
            synth_images_128 = out['image_raw']
            synth_images = out['image']
            
            if use_segmentation:# or segmentations==None:
                segmentations = self.segmenter.get_foreground_BiSeNet(synth_images, background_classes=[0])
                segmentations_128 = torch.round(self.downsampler_128(segmentations.repeat(1, 3, 1, 1))).to(torch.uint8).to(self.device)

                synth_images_foreground = segmentations * synth_images 
                synth_images_foreground_128 = segmentations_128*synth_images_128
 

            if weight_vgg > 0:
                synth_images_norm = (synth_images + 1) * (255 / 2) 
                features_vgg = self.vgg16(synth_images_norm, resize_images=False, return_lpips=True) 
                l_vgg = (target_features - features_vgg).square().sum()   
                loss += weight_vgg * l_vgg  

            if weight_pix > 0: 
                res = 128 
                l_pix = self.loss_L1(synth_images_foreground_128, target_foreground_128) / num_foreground_128_px if use_segmentation else self.loss_L1(synth_images_128, target_images_128) / ( res ** 2 )
                loss += weight_pix * l_pix

            if weight_lpips > 0:
                l_lpips = self.loss_percept(synth_images_foreground_128, target_foreground_128).sum() if use_segmentation else self.loss_percept(synth_images_128, target_images_128).sum()
                
                #l_lpips = loss_percept(synth_images_HD, target_images).sum() 

                loss += weight_lpips * l_lpips

            if weight_face > 0:
                generated_faces = synth_images*target_faces_segmentations
                l_face = self.loss_percept(generated_faces, target_faces).sum() + 0.75*self.loss_L2(generated_faces, target_faces).sum() / num_face_px 
                loss += weight_face * l_face 

            if weight_wdist > 0:            
                l_wdist = torch.linalg.norm(w_offsets)
                loss += weight_wdist * l_wdist

                if step > 100 and weight_wdist * 0.99 > weight_wdist_target:
                    weight_wdist *= 0.99   
            if count == 0:
                self.saveimg(target_images.clone().detach().cpu(), f'{out_folder_inv}/input.png')
            if count%20==0:
                self.saveimg(synth_images.clone().detach().cpu(), f'{out_folder_inv}/{str(count).zfill(4)}.png')

            # Optimizer step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            # if plotting, create log
            if plot_progress and ( step % image_log_step == 0 or step==(num_steps-1) ):
                for idx in range(num_images):  
                    synth_image = synth_images[idx].unsqueeze(0)
                    log_items = [target_images[idx].unsqueeze(0), synth_image]
                    if use_segmentation:
                        log_items.append(synth_images_foreground[idx].unsqueeze(0))
                        log_items.append(target_foreground[idx].unsqueeze(0))
                    if weight_face > 0:
                        log_items.append(generated_faces[idx].unsqueeze(0))
                    log_items.append(target_faces[idx].unsqueeze(0))
                    log.append(torch.cat(log_items, axis=-2))
                    
                if output_progress_frames:
                    generated = self.generator.generate_original(w_opt, yaws_circle[(step//image_log_step)%frames_per_rot], pitches_circle[(step//image_log_step)%frames_per_rot], output_all=True, grad=False)
                else:
                    generated = self.generator.generate_original(w_opt, 0, -0.1, output_all=True, grad=False)
                
                image = generated['image']
                depth_image = self.normalize_depth(generated['image_depth'], resize=512)
                            
                log_items = [image, depth_image]
                if use_segmentation:
                    fg = self.segmenter.get_foreground_BiSeNet(image)
                    log_items.append(fg * image)
                    log_items.append(torch.zeros_like(image))
                if weight_face > 0: 
                    face = self.segmenter.get_face_BiSeNet(image)
                    log_items.append(face * image)
                    
                log_items.append(torch.zeros_like(image))
                log.append(torch.cat(log_items, axis=-2))

                Visualizer.show_tensor(torch.cat(log, axis=-1), ax=ax)
                
                fig.canvas.draw()
                hfig.update(fig)
                
                if output_progress_frames:
                    progress_images.append(tensor_to_image(torch.cat(log, axis=-1)))
                
            if step % pbar_log_step == 0 or step==(num_steps-1):
                angles = '['
                for n, (y, p) in enumerate(zip(y_opt.cpu(), p_opt.cpu())):
                    if n != 0:
                        angles += '|'
                    angles += f'{y.clone().detach().numpy():<3.2f},{p.clone().detach().numpy():<3.2f}'
                desc = f'A={angles}] ' 

                desc += f'VGG={weight_vgg*l_vgg:<4.2f} ' if weight_vgg > 0 else ''
                desc += f'ID={weight_id*l_id:<4.2f} ' if weight_id > 0 else ''
                desc += f'PX={weight_pix*l_pix:<4.2f} ' if weight_pix > 0 else ''
                desc += f'LP={weight_lpips*l_lpips:<4.2f} ' if weight_lpips > 0 else ''
                desc += f'LF={weight_face*l_face:<4.2f} ' if weight_face > 0 else ''
                desc += f'WD={weight_wdist*l_wdist:<4.2f}(x{weight_wdist:.3f}) ' if weight_wdist > 0 else ''
                desc += f'L={float(loss):<5.2f}'
                pbar.set_description(f'{desc}')
                
            count += 1
        if plot_progress:
            plt.close(fig)   
        
        if output_progress_frames:
            return w_opt, w_offsets, y_opt, p_opt, progress_images 
        else:
            return w_opt, w_offsets, y_opt, p_opt 
        
    def saveimg(self, img, name):
        save_image((img+1)/2, name)

    def normalize_depth(self, depth_tensor, resize=None):
        depth_image = -depth_tensor
        depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 2 - 1

        if resize is not None:
            depth_image = F.interpolate(depth_image.repeat(1, 3, 1, 1), size=(resize, resize), mode='nearest')
        return depth_image

    def inversion_video(self,
                        w_person,
                        w_offsets,
                        input_images,
                        input_segmentations=None,  
                        face_segmentation=None, 
                        num_steps = 50,
                        learning_rate = 1e-2,
                        loss_threshold=0.25,
                        pbar_log_step = 5,
                        weight_vgg = 0.0,
                        weight_id = 0.0,
                        weight_pix = 0.25,
                        weight_face = 1.2,
                        weight_lpips = 1.0,
                        weight_wdist = 0.01,
                        weight_wprev = 0.02,
                        plot_progress = True,
                        image_log_step = 50,
                        out_folder = None): 
        if plot_progress:
            fig = plt.figure(figsize=(20, 20), dpi=100)
            hfig = display(fig, display_id=True)
            ax = plt.gca()
        num_frames = len(input_images)
        use_segmentation = input_segmentations is not None
        w_latent_offsets_video = []
        estimated_yaw_video = []
        estimated_pitch_video = [] 
        
        count = 0
        pbar = tqdm(range(num_frames))
        for idx in pbar: 
            target_image = input_images[idx].clone().to(self.device)
            target_image_128 = self.downsampler_128(target_image)
    
            if use_segmentation:
                target_segmentation = input_segmentations[idx].unsqueeze(0).clone().to(self.device)
                target_segmentation_128 = torch.round(self.downsampler_128(target_segmentation.repeat(1, 3, 1, 1))).to(torch.uint8).to(self.device)
    
                num_foreground_px = target_segmentation.sum()
                target_foreground = target_segmentation*target_image
                target_background = (1-target_segmentation)*target_image
                target_foreground_128 = target_segmentation_128*target_image_128
                
            if idx == 0:
                # first frame: start from average of known latents
                w_avg_known_offsets = torch.mean(w_offsets, axis=0, keepdims=True)
                start_offset = w_avg_known_offsets.clone().repeat([1, self.generator.get_num_ws(), 1])
                l_wprev = 0
                start_y = 0
                start_p = 0
                steps = 4*num_steps 
            else:
                # all other frames: start from previous frame
                start_offset = w_offset_opt.detach().clone()
                start_y = y_opt_frame.detach().clone()
                start_p = p_opt_frame.detach().clone()
                steps = num_steps
            
            if weight_face > 0:
                target_face_segmentation = self.segmenter.get_eyes_mouth_BiSeNet(target_image, dilate=8) if face_segmentation == None else face_segmentation
                num_face_px = target_face_segmentation.sum() 
                target_face = target_image * target_face_segmentation
            
            if weight_vgg > 0:
                target_images_norm = (target_images_128 + 1) * (255 / 2)
                target_features = self.vgg16(target_segmentations_128*target_images_norm if use_segmentation else target_images_norm, resize_images=False, return_lpips=True)

            w_offset_opt = torch.tensor(start_offset, dtype=torch.float32, device=self.device, requires_grad=True)  # pylint: disable=not-callable
            y_opt_frame = torch.tensor(start_y, dtype=torch.float32, device=self.device, requires_grad=True)
            p_opt_frame = torch.tensor(start_p, dtype=torch.float32, device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([w_offset_opt] + [y_opt_frame] + [p_opt_frame], betas=(0.9, 0.999), lr=learning_rate)
            
            for step in range(steps):
                ws = w_person.repeat([1, self.generator.get_num_ws(), 1]) + w_offset_opt
                out = self.generator.generate(ws, y_opt_frame, p_opt_frame, output_all=True, grad=True)
                synth_image_128 = out['image_raw']
                synth_image = out['image']        
                if use_segmentation:
                    synth_image_foreground = target_segmentation * synth_image 
                    synth_image_background = (1-target_segmentation) * synth_image 
                    synth_image_foreground_128 = target_segmentation_128 * synth_image_128
                    
                loss = 0

                if weight_vgg > 0:
                    synth_image_norm = (synth_image_128 + 1) * (255 / 2)
                    synth_features = self.vgg16(target_segmentation_128*synth_image_norm if use_segmentation else synth_image_norm, resize_images=False, return_lpips=True)
                    l_vgg = (target_features - synth_features).square().sum()   
                    loss += weight_vgg * l_vgg  

                if weight_id > 0:
                    l_id = self.loss_ID(synth_image_128, target_image_128)
                    loss += weight_id * l_id
                
                if weight_pix > 0: 
                    res = 512 
                    l_pix = self.loss_L1(synth_image_foreground, target_foreground) / num_foreground_px if use_segmentation else self.loss_L1(synth_image, target_image) / ( res ** 2 )
                    #l_pix = self.loss_L1(synth_image, target_image).sum() / ( res ** 2 )
                    loss += weight_pix * l_pix

                if weight_lpips > 0:
                    l_lpips = self.loss_percept(synth_image_foreground, target_foreground).sum() if use_segmentation else self.loss_percept(synth_image, target_image).sum()
                    #l_lpips = self.loss_percept(synth_image, target_image).sum()  + 0.2*self.loss_percept(synth_image_background, target_background).sum()
                    loss += weight_lpips * l_lpips 

                if weight_wdist > 0:
                    l_wdist = torch.linalg.norm(w_offset_opt)
                    loss += weight_wdist * l_wdist
                
                if idx > 0 and weight_wprev > 0:
                    l_wprev = torch.linalg.norm(w_offset_opt-start_offset)
                    loss += weight_wprev * l_wprev

                if weight_face > 0:
                    generated_face = synth_image*target_face_segmentation
                    l_face = self.loss_percept(generated_face, target_face).sum() + self.loss_L1(generated_face, target_face).sum() / num_face_px 
                    loss += weight_face * l_face

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                                                                             
                desc = f'[{idx}/{num_frames}] ({y_opt_frame.cpu():.2f},{p_opt_frame.cpu():.2f}) ' 
                if weight_vgg > 0: 
                    desc += f'VGG={weight_vgg*l_vgg:<4.2f} '
                if weight_id > 0:
                    desc += f'ID={weight_id*l_id:<4.2f} '
                if weight_pix > 0:
                    desc += f'PIX={weight_pix*l_pix:<4.2f} '
                if weight_lpips > 0:
                    desc += f'LPIPS={weight_lpips*l_lpips:<4.2f} '
                if weight_face > 0:
                    desc += f'FACE={weight_face*l_face:<4.2f} '
                if weight_wdist > 0:
                    desc += f'WD={weight_wdist*l_wdist:<4.2f} '
                if weight_wprev > 0:
                    desc += f'WP={weight_wprev*l_wprev:<4.4f} '
                    
                desc += f'loss {float(loss):<5.2f}'
                pbar.set_description(f'{desc}')
                
                if plot_progress and ( step % image_log_step == 0 or step == (num_steps-1) ): 
                    ref = target_segmentation.cpu()*target_image.cpu() if use_segmentation else target_image.cpu()
                    log = torch.cat((ref, generated_face.cpu(), torch.abs(synth_image.cpu()-ref), synth_image.cpu()), axis=-1)                                                   
                    Visualizer.show_tensor(log, ax=ax)
                    ax.text(30, 30, desc, color='white')
                    fig.canvas.draw()
                    hfig.update(fig)
                             
                if loss < loss_threshold: 
                    break
                if plot_progress:
                    plt.close(fig)
            
            count += 1
            w_latent_offsets_video.append(w_offset_opt.detach().cpu())
            estimated_yaw_video.append(y_opt_frame.detach().cpu())
            estimated_pitch_video.append(p_opt_frame.detach().cpu())
            
        w_latent_offsets_video = torch.cat(w_latent_offsets_video, axis=0) 
        estimated_yaw_video = torch.stack(estimated_yaw_video)
        estimated_pitch_video = torch.stack(estimated_pitch_video)
        
        return w_latent_offsets_video, estimated_yaw_video, estimated_pitch_video 
    
    def calculate_frame_flow(self, source_face, inset_face, source_yaw, new_yaw, source_pitch, new_pitch, angular_tolerance=45, output_frame_flow=False, previous_flow=None): 
        # flow is calculated in grayscale
        face_flow = cv.cvtColor(source_face, cv.COLOR_BGR2GRAY)
        inset_flow = cv.cvtColor(inset_face, cv.COLOR_BGR2GRAY)
        
        num_iter_flow = 7
        success = False
        num_tries = 3
        
        while not success and num_tries >= 0:
            if num_tries < 3:
                print(f'did not find a flow, trying again with {num_iter_flow} iterations')
            params = dict(pyr_scale=0.5,
                          levels=8,
                          winsize=25,
                          iterations=num_iter_flow, 
                          poly_n=5, 
                          poly_sigma=1.2,
                          flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
                         )
                    
            num_iter_flow += 1
            num_tries -= 1
            
            flow = cv.calcOpticalFlowFarneback(inset_flow, face_flow, previous_flow, **params)

            # get magnitudes and angles
            magnitudes, angles = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
            nonzero = magnitudes > 0
            eps = 0.5*np.mean(magnitudes[nonzero]) 

            #remove small vectors
            flow[magnitudes < eps] = 0
            magnitudes[magnitudes < eps] = 0
            nonzero = magnitudes > 0

            # save flow for plotting (before removing angles)
            flow_x = flow[:, :, 0].copy()
            flow_y = flow[:, :, 1].copy()

            flow_UV = np.zeros((512, 512, 2))
            flow_UV[..., 0][nonzero] = flow_x[nonzero]
            flow_UV[..., 1][nonzero] = flow_y[nonzero]

            dx = (source_yaw - new_yaw).cpu().numpy()
            dy = (source_pitch - new_pitch).cpu().numpy()
            # get direction
            d = (dx, dy)/np.linalg.norm((dx, dy))
            _, expected_angle = cv2.cartToPolar(np.array(d[0]), np.array(d[1]), angleInDegrees=True)

            # eval clockwise and counterclockwise angular difference
            angle_difference_1 = np.mod(angles-expected_angle, 360)
            angle_difference_2 = 360-np.mod(angles-expected_angle, 360)

            outlier_angles = np.minimum(angle_difference_1, angle_difference_2) > angular_tolerance
            flow[outlier_angles] = 0
            magnitudes[outlier_angles] = 0

            # update count of nonzero
            nonzero = magnitudes > 0

            # get normalized median direction vector    
            mean = np.mean(flow[nonzero], axis=0)
            mean_direction = mean/np.linalg.norm(mean, axis=-1)
            
            success = np.sum(nonzero) > 0 and not (np.isnan(np.max(magnitudes)) or np.isinf(np.max(magnitudes)))
        
        assert success, f'all flow got set to zero, that can\'t be right :( please retry'

        max_flow_length = np.max(magnitudes)
        median_direction_scale = max_flow_length*mean_direction
        return median_direction_scale, flow_UV, flow
        
    def inset_video(self,
                    input_video_images, # original image 450,450,3  list
                    input_face_tensors, # cropped image 124, 3, 512,512 tensor
                    landmarks_video, 
                    reference_face_landmarks, 
                    yaws_video, #8000
                    pitches_video, #8000
                    synth_video, # added
                    use_flow=False,
                    output_frame_flow=False,
                    exclude_face_in_flow=True,
                    original_yaws_video=None,
                    original_pitches_video=None,
                    scalar=1,
                    border_size = 42,
                    edge_size = 42,
                    edge_size_y_top = 8,
                    face_dilate = 0,
                    num_steps = 150,
                    learning_rate = 1e-2,
                    border_loss_threshold = 0.05,
                    weight_foreground = 1.0,
                    weight_border = 2.0,
                    weight_prev = 0.15,
                    weight_background = 0.0,
                    flow_multiplier = (0.75, 0.2),
                    use_w_dist = True,
                    plot_progress = True,
                    include_neck = False,
                    flow_directions = None,
                    return_flow_directions = False,
                    image_log_step = 50):
        
        if plot_progress:
            fig = plt.figure(figsize=(20, 20), dpi=100)
            hfig = display(fig, display_id=True)
            ax = plt.gca()
        
        N = len(input_video_images)
        H, W, C = input_video_images[0].shape
        border_sz_frame = 2*math.floor(((((border_size-face_dilate)/2)/512*min(W, H))-1)/2)-1 #transform border size to frame space and ensure border_sz is odd
        
        output_frames = np.zeros((N, H, W, C), dtype=np.uint8)
        output_faces = np.zeros((N, 512, 512, C), dtype=np.uint8)
        
        if use_flow:
            if output_frame_flow:
                output_flow = np.zeros((N, H, W, 2), dtype=np.float32)
            else:
                output_flow = np.zeros((N, 512, 512, 2), dtype=np.float32)
            output_vis = np.zeros((N, H, W, C), dtype=np.uint8)

        fg_prev = None
        if type(edge_size) is tuple:
            edge_size_h, edge_size_w = edge_size
        else:
            edge_size_h = edge_size
            edge_size_w = edge_size
        
        pbar = tqdm(enumerate(zip(input_video_images, input_face_tensors, landmarks_video, yaws_video, pitches_video, synth_video)), total=N)
        for i, (frame_image, face, landmarks, y, p, inset_image_edit) in pbar:
            frame = image_to_tensor(frame_image).to(self.device)
            inset_image_edit = torch.from_numpy(inset_image_edit/255*2-1).to(torch.float32).permute(2,0,1).unsqueeze(0).to(self.device)
            inset_image = inset_image_edit
            # ws_inset = w.detach().clone().unsqueeze(0).to(self.device)
            # inset_image_edit = self.generator.generate_original(ws_inset, y, p).clone()

            # if w_dist is not None:
            #     ws_inset += len_wdist * w_dist/torch.linalg.norm(w_dist)
                
            # if use_flow:
            #     face_sv = face.detach().clone()
            #     landmarks_sv = landmarks
                
            #     landmarks = landmarks + scalar*flow_directions[i]
            #     face = image_to_tensor(self.aligner.align_face_images(frame_image, landmarks, reference_face_landmarks, target_size=face.shape[-2:]))

            face = face.to(self.device).unsqueeze(0) if len(face.shape)<4 else face.to(self.device)
            
            background_classes = [0, 18, 16] if include_neck else [0, 18, 16, 14] 
            face_frame_foreground = self.segmenter.get_foreground_BiSeNet_original(face, dilate=face_dilate, neck_treatment=True, background_classes=background_classes)
            
            face_foreground = self.segmenter.get_foreground_BiSeNet_original(inset_image_edit, dilate=-face_dilate if face_dilate > 0 else 0)
            face_foreground_unite = torch.stack([face_foreground, face_frame_foreground], dim=0).any(dim=0).float()
            
            face_frame_background = (1-face_foreground)*face
            # pick only central area as relevant foreground
            face_foreground_edge = torch.zeros_like(face_foreground_unite)
            face_foreground_edge[:, :, edge_size_y_top:-edge_size_h, edge_size_w:-edge_size_w] = face_foreground_unite[:, :, edge_size_y_top:-edge_size_h, edge_size_w:-edge_size_w]
            num_foreground_px = face_foreground_edge.sum()
            inset_image_edit_foreground = inset_image_edit * face_foreground

            face_foreground_dilate = self.segmenter.get_foreground_BiSeNet_original(inset_image_edit, dilate=2)
            face_frame_foreground_dilate = self.segmenter.get_foreground_BiSeNet_original(face, dilate=2, background_classes=background_classes)
            # face_foreground_dilate = self.segmenter.get_foreground_BiSeNet_original(inset_image_edit, dilate=2*border_size)
            # face_frame_foreground_dilate = self.segmenter.get_foreground_BiSeNet_original(face, dilate=2*border_size, background_classes=background_classes)
            face_foreground_dilate = torch.stack([face_foreground_dilate, face_frame_foreground_dilate], dim=0).any(dim=0).float()

            face_foreground_border = face_foreground_dilate - face_foreground_edge

            
            face_foreground_dilate_half = self.segmenter.get_foreground_BiSeNet_original(inset_image_edit, dilate=1)
            face_frame_foreground_dilate_half = self.segmenter.get_foreground_BiSeNet_original(face, dilate=1, background_classes=background_classes)
            face_foreground_dilate_half = torch.stack([face_foreground_dilate_half, face_frame_foreground_dilate_half], dim=0).any(dim=0).float()
            
            # remove outer mask pixels to avoid edge artifacts
            mask_wo_edge = torch.zeros_like(inset_image).to(self.device).type(torch.float32) #transform binary image to get cut+paste mask
            mask_wo_edge[:, :, :-edge_size_h, edge_size_w:-edge_size_w] = face_foreground_dilate_half[:, :, :-edge_size_h, edge_size_w:-edge_size_w].repeat(1, 3, 1, 1)
            
        
            inset_image_np = tensor_to_image(inset_image)
            mask_wo_edge_np = tensor_to_image(mask_wo_edge, normalize=False).astype(np.float32)
            rect = np.zeros_like(mask_wo_edge_np)
            rect[3:-3, 3:-3, :] = np.ones_like(rect[3:-3, 3:-3, :])
            
            [transformed_inset_image, transformed_mask_image, transformed_rect_image] = self.aligner.align_face_images([inset_image_np, mask_wo_edge_np, rect], reference_face_landmarks, landmarks, target_size=(H, W), padding='zero')
            
            transformed_mask_image = transformed_rect_image * cv2.GaussianBlur(transformed_mask_image.astype(np.float), ksize=(border_sz_frame, border_sz_frame), sigmaX=0)# multiply by rect to avoid bleeding of gaussian blur outside of inpainting boundaries
            
            inpainted_frame_image = frame_image * (1-transformed_mask_image) + transformed_inset_image * transformed_mask_image
            
            inset_inpainted = self.aligner.align_face_images(inpainted_frame_image, landmarks, reference_face_landmarks, target_size=(512, 512), padding='replicate')
            
            output_frames[i, ...] = inpainted_frame_image
            output_faces[i, ...] = inset_inpainted
            
            if use_flow:
                # plot segmentation boundaries
                inset_foreground_debug = self.segmenter.get_foreground_BiSeNet_original(inset_image_edit)
                face_foreground_debug = self.segmenter.get_foreground_BiSeNet_original(face)
                
                # get segmentation boundary from segmentation using laplacian
                seg_im_inset = tensor_to_image(inset_foreground_debug)[..., np.newaxis] #np.repeat(tensor_to_image(inset_foreground_debug)[..., np.newaxis], 3, axis=-1)
                edge_foreground_inset = cv2.Laplacian(seg_im_inset, -1, ksize=9, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
                        
                # get segmentation boundary from segmentation using laplacian
                seg_im_face = tensor_to_image(face_foreground_debug)[..., np.newaxis] 
                edge_foreground_face = cv2.Laplacian(seg_im_face, -1, ksize=9, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
                
                transformed_mask_default = self.aligner.align_face_images(edge_foreground_inset, reference_face_landmarks, landmarks_sv, target_size=(H, W), padding='zero')
                transformed_mask_flow = self.aligner.align_face_images(edge_foreground_inset, reference_face_landmarks, landmarks, target_size=(H, W), padding='zero')
                transformed_mask_face = self.aligner.align_face_images(edge_foreground_face, reference_face_landmarks, landmarks, target_size=(H, W), padding='zero')
                
                transformed_mask_RGB = np.stack((transformed_mask_default, transformed_mask_flow, transformed_mask_face), axis=-1).astype(np.uint8)
                
                output_vis[i, ...] = transformed_mask_RGB
             
        if plot_progress:        
            plt.close(fig)  
        
        if use_flow:
            if return_flow_directions:
                return output_frames, output_faces, output_vis, output_flow, flow_directions 
            return output_frames, output_faces, output_vis, output_flow, None
        return output_frames, output_faces, None, None, None
    
