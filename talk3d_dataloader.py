import torch
import numpy as np
import PIL.Image as Image
from glob import glob

from Wav2Lip import audio
import os, random, cv2
from os.path import dirname, join, basename, isfile
from Wav2Lip.hparams import hparams
import pandas as pd

class Train_Dataset_nosync(torch.utils.data.Dataset):
    def __init__(self,
        data_root,
        opts,
        eye_smoothing=False,
        max_length=0):

        angle_dir = os.path.join(data_root, opts.cam_dir)
        aud_eo_dir = os.path.join(data_root, opts.audio_eo_dir)
        head_rot_dir = os.path.join(data_root, opts.angles_dir)
        landmark_dir = os.path.join(data_root, opts.landmark_dir)
        face_tensors_video_dir = os.path.join(data_root, opts.image_dir)
        face_segmentation_tensors_video_dir = os.path.join(data_root, opts.face_segmentation_dir)
        mouth_segmentation_tensors_video_dir = os.path.join(data_root, opts.mouth_segmentation_dir)
        torso_segmentation_tensors_video_dir = os.path.join(data_root, opts.torso_segmentation_dir)
        eye_dir = os.path.join(data_root, opts.eye_dir)
        
        self.angles = torch.load(angle_dir)
        self.head_rots = torch.load(head_rot_dir)
        landmarks = torch.load(landmark_dir)
        self.landmarks = torch.cat([landmarks[:,0:1,:],landmarks[:,2:4:,:]],dim = 1)
        aud_eo = torch.from_numpy(np.load(aud_eo_dir)).permute(0,2,1)
        self.images_root = face_tensors_video_dir + '/*.png'
        self.segmentations_root = face_segmentation_tensors_video_dir + '/*.png'
        self.mouth_segmentations_root = mouth_segmentation_tensors_video_dir + '/*.png'
        self.torso_segmentations_root = torso_segmentation_tensors_video_dir + '/*.png'
        self.length = min(self.angles.shape[0], len(glob(self.images_root)), len(glob(self.segmentations_root))) if max_length==0 else max_length
        
        self.aud_eo = aud_eo[:self.length]

        au_blink_info=pd.read_csv(eye_dir)
        eye_area = []
        try:
            au_blink = au_blink_info['AU45_r'].values
        except:
            au_blink = au_blink_info[' AU45_r'].values

        for au in au_blink :
            area = np.clip(au, 0, 2)
            eye_area.append(area)

        eye_area = np.array(eye_area)

        self.eye_area = torch.Tensor(eye_area).view(-1, 1)
        
        print(f'Total Dataset length: {self.length}')

    def get_audio_features(self, features, index, att_mode=2): 
        if att_mode == 0:
            return features[[index]]
        elif att_mode == 1:
            left = index - 8
            pad_left = 0
            if left < 0:
                pad_left = -left
                left = 0
            auds = features[left:index]
            if pad_left > 0:
                # pad may be longer than auds, so do not use zeros_like
                auds = torch.cat([torch.zeros(pad_left, *auds.shape[1:], device=auds.device, dtype=auds.dtype), auds], dim=0)
            return auds
        elif att_mode == 2:
            left = index - 4
            right = index + 4
            pad_left = 0
            pad_right = 0
            if left < 0:
                pad_left = -left
                left = 0
            if right > features.shape[0]:
                pad_right = right - features.shape[0]
                right = features.shape[0]
            auds = features[left:right]
            if pad_left > 0:
                auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
            if pad_right > 0:
                auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
            return auds

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        angle = self.angles[idx]
        yaw, pitch = angle[:1], angle[1:]
        image_dirs = sorted(glob(self.images_root))
        seg_dirs = sorted(glob(self.segmentations_root))
        mouth_seg_dirs = sorted(glob(self.mouth_segmentations_root))
        torso_seg_dirs = sorted(glob(self.torso_segmentations_root))
        image_dir = image_dirs[idx]
        seg_dir = seg_dirs[idx]
        mouth_seg_dir = mouth_seg_dirs[idx]
        torso_seg_dir = torso_seg_dirs[idx]
        image = np.array(Image.open(image_dir)).astype(np.float32)/127.5 -1
        image = torch.from_numpy(image).permute(2,0,1)

        seg = np.array(Image.open(seg_dir)).astype(np.float32)/127.5 -1
        seg = torch.from_numpy(seg).unsqueeze(0).round()
        if len(seg.shape) == 4:
            seg = seg[:,:,:,0]
        if seg.min()<-0.5:
            seg = (seg+1)/2

        mouthseg = np.array(Image.open(mouth_seg_dir)).astype(np.float32)/127.5 -1
        torsoseg = np.array(Image.open(torso_seg_dir)).astype(np.float32)/127.5 -1
        mouthseg = torch.from_numpy(mouthseg).unsqueeze(0).round()
        torsoseg = torch.from_numpy(torsoseg).unsqueeze(0).round()
        if len(mouthseg.shape) == 4:
            mouthseg = mouthseg[:,:,:,0]
        if mouthseg.min()<-0.5:
            mouthseg = (mouthseg+1)/2
        if len(torsoseg.shape) == 4:
            torsoseg = torsoseg[:,:,:,0]
        if torsoseg.min()<-0.5:
            torsoseg = (torsoseg+1)/2
        
        aud_eo = self.get_audio_features(self.aud_eo, idx)

        eye = self.eye_area[idx] 
        eye += (np.random.rand()-0.5) / 10

        head_rot = self.head_rots[idx]
        landmark = self.landmarks[idx]
        return image, seg, mouthseg, torsoseg, yaw, pitch, aud_eo, eye, head_rot, landmark


class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(self,
        data_root,
        opts,
        eye_smoothing,
        start_frame=0,
        set_max_length=0,
        is_OOD=False,
        ):

        angle_dir = os.path.join(data_root, opts.cam_dir)
        head_rot_dir = os.path.join(data_root, opts.angles_dir)
        landmark_dir = os.path.join(data_root, opts.landmark_dir)
        aud_eo_dir = os.path.join(data_root, opts.audio_eo_OOD_dir) if is_OOD else os.path.join(data_root, opts.audio_eo_dir)
        face_tensors_video_dir = os.path.join(data_root, opts.image_dir)
        face_segmentation_tensors_video_dir = os.path.join(data_root, opts.face_segmentation_dir)
        body_segmentation_tensors_video_dir = os.path.join(data_root, opts.body_segmentation_dir)
        mouth_segmentation_tensors_video_dir = os.path.join(data_root, opts.mouth_segmentation_dir)
        eye_dir = os.path.join(data_root, opts.eye_dir)
        
        self.angles = torch.load(angle_dir)
        self.head_rots = torch.load(head_rot_dir)
        self.landmarks = torch.load(landmark_dir)
        aud_eo = torch.from_numpy(np.load(aud_eo_dir)).permute(0,2,1)
        self.images_root = face_tensors_video_dir + '/*.png'
        self.face_segmentations_root = face_segmentation_tensors_video_dir + '/*.png'
        self.body_segmentations_root = body_segmentation_tensors_video_dir + '/*.png'
        self.mouth_segmentations_root = mouth_segmentation_tensors_video_dir + '/*.png'

        self.start_frame = start_frame
        self.max_length = self.angles.shape[0] - self.start_frame

        if set_max_length == -1:
            if is_OOD:
                self.max_length = int(aud_eo.shape[0])-1 # if OOD, align num of data with audio length
            else:
                self.max_length = int(self.angles.shape[0] * (1-opts.traintest_split_rate))-1 # if novel, align num of data with validation set
        elif set_max_length != 0:
            self.max_length = set_max_length
        
        print(f'Inference Dataset -- Start frame :{self.start_frame}, max_length : {self.max_length}')

        self.aud_eo = aud_eo[start_frame:start_frame+self.max_length]
        self.landmarks = torch.cat([self.landmarks[:,0:1,:],self.landmarks[:,2:4:,:]],dim = 1)

        au_blink_info=pd.read_csv(eye_dir)
        eye_area = []
        try:
            au_blink = au_blink_info['AU45_r'].values
        except:
            au_blink = au_blink_info[' AU45_r'].values
        
        for au in au_blink :
            area = np.clip(au, 0, 2)
            eye_area.append(area)

        eye_area = np.array(eye_area)
        if eye_smoothing:
            ori_eye = eye_area.copy()
            for i in range(ori_eye.shape[0]):
                start = max(0, i - 1)
                end = min(ori_eye.shape[0], i + 2)
                eye_area[i] = ori_eye[start:end].mean()

        self.eye_area = torch.Tensor(eye_area).view(-1, 1)
        

    def get_audio_features(self, features, index, att_mode=2): 
        if att_mode == 0:
            return features[[index]]
        elif att_mode == 1:
            left = index - 8
            pad_left = 0
            if left < 0:
                pad_left = -left
                left = 0
            auds = features[left:index]
            if pad_left > 0:
                # pad may be longer than auds, so do not use zeros_like
                auds = torch.cat([torch.zeros(pad_left, *auds.shape[1:], device=auds.device, dtype=auds.dtype), auds], dim=0)
            return auds
        elif att_mode == 2:
            left = index - 4
            right = index + 4
            pad_left = 0
            pad_right = 0
            if left < 0:
                pad_left = -left
                left = 0
            if right > features.shape[0]:
                pad_right = right - features.shape[0]
                right = features.shape[0]
            auds = features[left:right]
            if pad_left > 0:
                auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
            if pad_right > 0:
                auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
            return auds

    def __len__(self):
        return self.max_length

    def __getitem__(self, idx):
        angle = self.angles[idx+self.start_frame]
        yaw, pitch = angle[:1], angle[1:]
        image_dirs = sorted(glob(self.images_root))
        image_dir = image_dirs[idx+self.start_frame]
        image = np.array(Image.open(image_dir)).astype(np.float32)/127.5 -1
        image = torch.from_numpy(image).permute(2,0,1)

        face_seg_dirs = sorted(glob(self.face_segmentations_root))
        face_seg_dir = face_seg_dirs[idx+self.start_frame]
        face_seg = np.array(Image.open(face_seg_dir)).astype(np.float32)/127.5 -1
        face_seg = torch.from_numpy(face_seg).unsqueeze(0).round()
        if len(face_seg.shape) == 4:
            face_seg = face_seg[:,:,:,0]
        if face_seg.min()<-0.5:
            face_seg = (face_seg+1)/2
            
        body_seg_dirs = sorted(glob(self.body_segmentations_root))
        body_seg_dir = body_seg_dirs[idx+self.start_frame]
        body_seg = np.array(Image.open(body_seg_dir)).astype(np.float32)/127.5 -1
        body_seg = torch.from_numpy(body_seg).unsqueeze(0).round()
        if len(body_seg.shape) == 4:
            body_seg = body_seg[:,:,:,0]
        if body_seg.min()<-0.5:
            body_seg = (body_seg+1)/2

        mouth_seg_dirs = sorted(glob(self.mouth_segmentations_root))
        mouth_seg_dir = mouth_seg_dirs[idx+self.start_frame]
        mouthseg = np.array(Image.open(mouth_seg_dir)).astype(np.float32)/127.5 -1
        mouthseg = torch.from_numpy(mouthseg).unsqueeze(0).round()
        if len(mouthseg.shape) == 4:
            mouthseg = mouthseg[:,:,:,0]
        if mouthseg.min()<-0.5:
            mouthseg = (mouthseg+1)/2
        
        aud_eo = self.get_audio_features(self.aud_eo, idx)

        eye = self.eye_area[idx+self.start_frame]
        head_rot = self.head_rots[idx+self.start_frame]
        landmark = self.landmarks[idx+self.start_frame]
        
        return image, face_seg, body_seg, mouthseg, yaw, pitch, aud_eo, eye, head_rot, landmark

class Train_Dataset_sync(torch.utils.data.Dataset):
    def __init__(self,
        batchsize,
        data_root,
        opts,
        max_length=0,
        ):
        
        angle_dir = os.path.join(data_root, opts.cam_dir)
        head_rot_dir = os.path.join(data_root, opts.angles_dir)
        landmark_dir = os.path.join(data_root, opts.landmark_dir)
        aud_eo_dir = os.path.join(data_root, opts.audio_eo_dir)
        face_tensors_video_dir = os.path.join(data_root, opts.image_dir)
        face_segmentation_tensors_video_dir = os.path.join(data_root, opts.face_segmentation_dir)
        mouth_segmentation_tensors_video_dir = os.path.join(data_root, opts.mouth_segmentation_dir)
        torso_segmentation_tensors_video_dir = os.path.join(data_root, opts.torso_segmentation_dir)
        segmentation_tensors_video_dir = face_segmentation_tensors_video_dir
        
        
        eye_dir = os.path.join(data_root, opts.eye_dir)
        self.angles = torch.load(angle_dir)
        self.head_rots = torch.load(head_rot_dir)
        self.landmarks = torch.load(landmark_dir)
        aud_eo = torch.from_numpy(np.load(aud_eo_dir)).permute(0,2,1)
        self.images_root = face_tensors_video_dir + '/*.png'
        self.mouth_segmentations_root = mouth_segmentation_tensors_video_dir + '/*.png'
        self.torso_segmentations_root = torso_segmentation_tensors_video_dir + '/*.png'
        self.segmentation_tensors_video_dir = segmentation_tensors_video_dir + '/*.png'
        self.length = min(self.angles.shape[0], len(glob(self.images_root)), len(glob(self.mouth_segmentations_root))) if max_length==0 else max_length
        
        self.aud_eo = aud_eo[:self.length]
        
        self.landmarks = torch.cat([self.landmarks[:,0:1,:],self.landmarks[:,2:4:,:]],dim = 1)

        au_blink_info=pd.read_csv(eye_dir)
        eye_area = []
        try:
            au_blink = au_blink_info['AU45_r'].values
        except:
            au_blink = au_blink_info[' AU45_r'].values

        for au in au_blink :
            area = np.clip(au, 0, 2)
            eye_area.append(area)

        eye_area = np.array(eye_area)

        self.eye_area = torch.Tensor(eye_area).view(-1, 1)

        self.batchsize = batchsize
        self.syncnet_T = 5
        self.syncnet_mel_step_size = 16
        
        # sorting!  self.bbox_names: bbox coord (x1,y1,x2,y2)
        self.img_names = sorted(glob(self.images_root))
        self.bbox_names = sorted(glob(os.path.join(data_root, f'{opts.wav2lip_bbox_dir}/*.npy')))
        
        wavpath = os.path.join(data_root, "aud.wav")
        wav = audio.load_wav(wavpath, hparams.sample_rate)
        self.orig_mel = audio.melspectrogram(wav).T # (25597, 80)
        self.valid_idx = torch.arange(2, self.length-7)
        self.valid_idx = self.valid_idx[torch.randperm(self.valid_idx.shape[0])]
        print(f'Total Training Dataset length: {self.length}')
        
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, idx):
        start_id = idx#,self.get_frame_id(start_frame)
        #vidname = dirname(start_frame)

        window_fnames_img  = []
        window_fnames_bbox = []
        for frame_id in range(start_id, start_id + self.syncnet_T):
            frame_img  = self.img_names[frame_id]
            frame_bbox = self.bbox_names[frame_id]

            if not isfile(frame_img) or not isfile(frame_bbox):
                return None
            window_fnames_img.append(frame_img)
            window_fnames_bbox.append(frame_bbox)
            
        return window_fnames_img, window_fnames_bbox

    def read_window(self, window_fnames_img, window_fnames_bbox):
        if window_fnames_img is None or window_fnames_bbox is None: return None
        
        window_img = []
        window_bbox = []
        window_size = len(window_fnames_img)
        for i in range(window_size):
            img   = cv2.imread(window_fnames_img[i])
            coord = np.load(window_fnames_bbox[i])
            if img is None:
                return None
            if coord is None:
                return None
            
            window_img.append(img)
            window_bbox.append(coord)
        return window_img, window_bbox

    def crop_audio_window(self, spec, start_frame):
        start_frame_num = start_frame
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + self.syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert self.syncnet_T == 5
        #start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        start_frame_num = start_frame + 1
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + self.syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != self.syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window_img, window_bbox):
        # w/o transpose:(T,H,W,3) with transpose: 3 x T x H x W
        """ BGR """
        img = np.asarray(window_img) / 255.
        #img = np.transpose(x, (3, 0, 1, 2))
        
        bbox = np.array(window_bbox)
        return img, bbox
    
    def crop_with_resize(self, img, bbox, size=96):
        # img:  (T,H,W,3)
        # bbox: (T,4)
        results = []
        for i in range(bbox.shape[0]):
            cur_img = img[i,:,:,:] # (H,W,3)
            x1, y1, x2, y2 = bbox[i]//4      # 512->128 scaling.
            cur_img = cv2.resize(cur_img, (128,128))
            cur_img = cur_img[y1:y2, x1:x2, :]
            cur_img = cv2.resize(cur_img, (size,size)) 
            results.append(cur_img)
        
        results = np.array(results) # (T,96,96,3)
        results = np.transpose(results, (3, 0, 1, 2)) # (3, T, 96, 96)
        return results
        
    def get_wav2lip_data(self, idx):
        while 1:            
            window_fnames_img, window_fnames_bbox = self.get_window(idx)
            window_img, window_bbox = self.read_window(window_fnames_img, window_fnames_bbox)
            if window_img is None or window_bbox is None:
                continue

            mel = self.crop_audio_window(self.orig_mel.copy(), idx)
            
            if (mel.shape[0] != self.syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(self.orig_mel.copy(), idx)
            if indiv_mels is None: continue

            img, bbox = self.prepare_window(window_img, window_bbox)
            img = self.crop_with_resize(img, bbox, size=96) # crop with bbox and resize to 96=wav2lip syncnet trained size.
            img = torch.from_numpy(np.array(img))
            
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)

            return img, bbox, indiv_mels, mel


    def get_audio_features(self, features, index, att_mode=2): 
        if att_mode == 0:
            return features[[index]]
        elif att_mode == 1:
            left = index - 8
            pad_left = 0
            if left < 0:
                pad_left = -left
                left = 0
            auds = features[left:index]
            if pad_left > 0:
                # pad may be longer than auds, so do not use zeros_like
                auds = torch.cat([torch.zeros(pad_left, *auds.shape[1:], device=auds.device, dtype=auds.dtype), auds], dim=0)
            return auds
        elif att_mode == 2:
            left = index - 4
            right = index + 4
            pad_left = 0
            pad_right = 0
            if left < 0:
                pad_left = -left
                left = 0
            if right > features.shape[0]:
                pad_right = right - features.shape[0]
                right = features.shape[0]
            auds = features[left:right]
            if pad_left > 0:
                auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
            if pad_right > 0:
                auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
            return auds
        
    def __len__(self):
        return len(self.valid_idx)
        # return self.length
    
    def __getitem__(self, idx_):
        
        image_dirs = sorted(glob(self.images_root))
        
        seg_dirs = sorted(glob(self.segmentation_tensors_video_dir))
        mouth_seg_dirs = sorted(glob(self.mouth_segmentations_root))
        torso_seg_dirs = sorted(glob(self.torso_segmentations_root))
        
        imgs=[]
        segs=[]
        mouth_segs=[]
        torso_segs=[]
        yaws=[]
        pitchs=[]
        audio_eos=[]
        eyes=[]
        head_rots=[]
        landmarks=[]

        idx = int(self.valid_idx[idx_].item()) # wav2lip 5 frame window -> cut starting and ending frame
        # idx = random.randint( 2, self.length-7 )
        if self.batchsize==3:
            idxrange = range(1,4)
        elif self.batchsize == 2:
            idxrange = [1,3]
        elif self.batchsize==5:
            idxrange = range(0,5)
        elif self.batchsize==1:
            idxrange = range(2,3)
        elif self.batchsize==4:
            idxrange = range(0,4)

        for i in idxrange: #range(idx+lidx,idx+ridx):
            angle = self.angles[idx+i]
            yaw, pitch = angle[:1], angle[1:]
            

            image_dir = image_dirs[idx+i]
            seg_dir = seg_dirs[idx+i]
            mouth_seg_dir = mouth_seg_dirs[idx+i]
            torso_seg_dir = torso_seg_dirs[idx+i]
            image = np.array(Image.open(image_dir)).astype(np.float32)/127.5 -1
            image = torch.from_numpy(image).permute(2,0,1)

            seg = np.array(Image.open(seg_dir)).astype(np.float32)/127.5 -1
            seg = torch.from_numpy(seg).unsqueeze(0).round()
            if len(seg.shape) == 4:
                seg = seg[:,:,:,0]
            if seg.min()<-0.5:
                seg = (seg+1)/2
                
            mouthseg = np.array(Image.open(mouth_seg_dir)).astype(np.float32)/127.5 -1
            mouthseg = torch.from_numpy(mouthseg).unsqueeze(0).round()
            if len(mouthseg.shape) == 4:
                mouthseg = mouthseg[:,:,:,0]
            if mouthseg.min()<-0.5:
                mouthseg = (mouthseg+1)/2

            torsoseg = np.array(Image.open(torso_seg_dir)).astype(np.float32)/127.5 -1
            torsoseg = torch.from_numpy(torsoseg).unsqueeze(0).round()
            if len(torsoseg.shape) == 4:
                torsoseg = torsoseg[:,:,:,0]
            if torsoseg.min()<-0.5:
                torsoseg = (torsoseg+1)/2
            
            aud_eo = self.get_audio_features(self.aud_eo, idx+i)
            
            eye = self.eye_area[idx+i] 
            eye += (np.random.rand()-0.5) / 10

            imgs.append(image)
            segs.append(seg)
            mouth_segs.append(mouthseg)
            torso_segs.append(torsoseg)
            yaws.append(yaw)
            pitchs.append(pitch)
            audio_eos.append(aud_eo)
            eyes.append(eye)
            
            head_rots.append(self.head_rots[idx+i])
            landmarks.append(self.landmarks[idx+i])
        
        image = torch.stack(imgs)
        seg = torch.stack(segs)
        mouth_seg = torch.stack(mouth_segs)
        torso_seg = torch.stack(torso_segs)
        yaw = torch.stack(yaws)
        pitch = torch.stack(pitchs)
        aud_eo = torch.stack(audio_eos)
        eye = torch.stack(eyes)
        
        head_rots = torch.stack(head_rots)
        landmark = torch.stack(landmarks)
    
        wav2lip_frames, wav2lip_bbox, indiv_mels, mel = self.get_wav2lip_data(idx)
        
        return image, seg, mouth_seg, torso_seg, yaw, pitch, aud_eo, eye, wav2lip_frames, wav2lip_bbox, indiv_mels, mel, head_rots, landmark# xy_coordinate