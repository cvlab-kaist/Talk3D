import dnnlib
import legacy
from training.camera_utils import LookAtPoseSampler, FOV_to_intrinsics
import numpy as np
import torch
import math
import copy
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from eg3d.eg3d.torch_utils import misc

class EG3D_Generator: 
    def __init__(self, network_pkl, device='cuda', load_tuned=False, construct_G=False, versions=None, use_ddp=False):
        self.device=device
        self.use_ddp = use_ddp
        if construct_G:
            self.load_tuned_switch = True if load_tuned else False
            common_kwargs = {'c_dim': 25, 'img_resolution': 512, 'img_channels': 3} 
            G_kwargs = {'class_name': 'training.triplane.TriPlaneGenerator', 'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 2}, 'channel_base': 32768, 'channel_max': 512, 'fused_modconv_default': 'inference_only', 'rendering_kwargs': {'image_resolution': 512, 'disparity_space_sampling': False, 'clamp_mode': 'softplus', 'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC', 'c_gen_conditioning_zero': False, 'gpc_reg_prob': 0.5, 'c_scale': 1.0, 'superresolution_noise_mode': 'none', 'density_reg': 0.25, 'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0, 'sr_antialias': True, 'depth_resolution': 48, 'depth_resolution_importance': 48, 'ray_start': 2.25, 'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7, 'avg_camera_pivot': [0, 0, 0.2]}, 'num_fp16_res': 0, 'sr_num_fp16_res': 4, 'sr_kwargs': {'channel_base': 32768, 'channel_max': 512, 'fused_modconv_default': 'inference_only'}, 'conv_clamp': None}
            self.G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(self.device)
            self.G.register_buffer('dataset_label_std', torch.tensor([0.0664, 0.0295, 0.2720, 0.7343, 0.0279, 0.0178, 0.1280, 0.3457, 0.2721, 0.1274, 0.0679, 0.1832, 0.0000, 0.0000, 0.0000, 0.0000, 0.0079, 0.0000, 0.0000, 0.0000, 0.0079, 0.0000, 0.0000, 0.0000, 0.0000]).to(self.device))
            self.copy_tuned(network_pkl, use_ddp)
            self.G_tune = None

        elif load_tuned:
            self.G = None
            self.load_tuned(network_pkl)
            print("load tuned generator")

        else: #load pickle from original eg3d
            with dnnlib.util.open_url(network_pkl) as f:
                self.G = legacy.load_network_pkl(f)['G_ema'].eval().to(self.device)
            self.G_tune = None
            self.active_G = self.G
            
        self.set_camera_parameters(use_ddp=use_ddp)
        self.evaluate_average_w(use_ddp=use_ddp)
        self.BS = 4
        
    def tune(self, force=False):
        if self.G_tune == None or force:
            self.G_tune = copy.deepcopy(self.G).eval().to(self.device).float()
        self.G_tune.requires_grad_(True)
        self.G_tune.superresolution.requires_grad_(False)
        self.active_G = self.G_tune.to(self.device)
    
    def get_num_ws(self):
        return self.active_G.backbone.synthesis.num_ws
    
    def load_tuned(self, tuned_generator_path):
        import pickle
        with open(tuned_generator_path, 'rb') as f:
            self.G_tune = pickle.load(f).to(self.device)
        print(f'loaded tuned generator from {tuned_generator_path}')        
        self.active_G = self.G_tune
        
    def copy_tuned(self, tuned_generator_path, use_ddp): #this is for manipulating G network
        with dnnlib.util.open_url(tuned_generator_path) as f:
            pickle_ = legacy.load_network_pkl(f)
        
        for name, module in [('G_ema', self.G)]:
            if self.load_tuned_switch:
                misc.copy_params_and_buffers(pickle_, module, require_all=False)
            else:
                misc.copy_params_and_buffers(pickle_[name], module, require_all=False)
        self.active_G = self.G.eval()
        self.active_G.neural_rendering_resolution=128 # constructed G gets default by 64 -> manually change
        if use_ddp:
            for params in self.active_G.parameters():
                params.requires_grad = True
            self.active_G = DDP(self.active_G, device_ids=[self.device])
            

    def traineval_switch(self, select):
        if select == 'train':
            self.active_G.train()
        else:
            self.active_G.eval()

    def default(self):
        self.active_G = self.G
        
    def set_camera_parameters(self, focal_length=3.5, cam_pivot=[0, 0, 0.2], use_ddp = False):
        self.focal_length = focal_length
        self.intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=self.device)
        self.cam_pivot = torch.tensor(cam_pivot, device=self.device)
        if self.use_ddp:
            self.cam_radius = self.active_G.module.rendering_kwargs.get('avg_camera_radius', 2.7)
        else:
            self.cam_radius = self.active_G.rendering_kwargs.get('avg_camera_radius', 2.7)
        self.conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, self.cam_pivot, radius=self.cam_radius, device=self.device)
        self.conditioning_params = torch.cat([self.conditioning_cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1, 9)], 1)
        
    def evaluate_average_w(self, num_samples=10000, use_ddp=False):
        truncation_psi = 1
        truncation_cutoff = 14

        with torch.no_grad():
            if self.use_ddp:
                z_samples = np.random.RandomState(123).randn(num_samples, self.active_G.module.z_dim)
                w_samples = self.active_G.module.mapping(torch.from_numpy(z_samples).to(self.device), self.conditioning_params.repeat(num_samples, 1), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)  
            else:
                z_samples = np.random.RandomState(123).randn(num_samples, self.active_G.z_dim)
                w_samples = self.active_G.mapping(torch.from_numpy(z_samples).to(self.device), self.conditioning_params.repeat(num_samples, 1), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)   

            w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C] #
            w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
            self.w_avg_tensor = torch.from_numpy(w_avg[0]).to(self.device)
            self.w_std = (np.sum((w_samples - w_avg) ** 2) / num_samples) ** 0.5
    
    def get_ws(self, z_samples, truncation_psi=1, truncation_cutoff=14):
        return self.G.mapping(z_samples, self.conditioning_params.repeat(len(z_samples), 1), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff) 
    
    def get_random_ws(self, num_samples, truncation_psi=1, truncation_cutoff=14, seed=0):
        z_rand = torch.from_numpy(np.random.RandomState(seed).randn(num_samples, self.G.z_dim)).to(self.device)
        return self.get_ws(z_rand, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

    def get_average_w(self):
        return self.w_avg_tensor
    
    def get_w_std(self):
        return self.w_std
    
    # get camera parameters for generator from a single yaw and pitch or a list
    def get_camera_parameters(self, yaw=0.0, pitch=0.0, focal_length=None):
        focal_length = self.focal_length if focal_length == None else focal_length
        def is_list(variable):
            return type(variable) in (list, np.ndarray) or ( type(variable) is torch.Tensor and variable.dim() > 0 )
        
        # check if camera parameters are a list. for torch.tensors, check also if dimensionality is larger than 0, otherwise len() fails
        if is_list(yaw) or is_list(pitch) or is_list(focal_length):
            camera_params = []
            num_yaw = 1 if type(yaw)==float else len(yaw)
            num_pitch = 1 if type(pitch)==float else len(pitch)
            num_fl = 1 if type(focal_length)==float else len(focal_length)
            create_num_parameters = max(num_yaw, num_pitch, num_fl)
            for i in range(create_num_parameters):
                # check for each parameter if single float was passed, otherwise fill from list
                y = yaw if type(yaw)==float else yaw[i]
                p = pitch if type(pitch)==float else pitch[i]
                fl = focal_length if type(focal_length)==float else focal_length[i]
                intrinsics = torch.tensor([[fl, 0, 0.5], [0, fl, 0.5], [0, 0, 1]], device=self.device)        
                
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + y, np.pi/2 + p, self.cam_pivot, radius=self.cam_radius, device=self.device)
                camera_params.append(torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1))
            camera_params = torch.cat(camera_params, axis=0)
        else:
            intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=self.device)        
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + yaw, np.pi/2 + pitch, self.cam_pivot, radius=self.cam_radius, device=self.device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            
        return camera_params
   
    def generate(self, w, triplane_offset=None, yaw=0.0, pitch=0.0, focal_length=None, noise_mode='const', ID_triplane_dir=None, output_all=False, grad=False, ):                 
        # convert angles to camera parameter(s)
        
        camera_params = self.get_camera_parameters(yaw=yaw, pitch=pitch, focal_length=focal_length)
        # convert w to correct size
        # (,512) -> (1, 512)
        if len(w.shape) == 1:
            w = w.unsqueeze(0)
        
        # (x, 512) -> (x, 1, 512)
        if len(w.shape) == 2:
            w = w.unsqueeze(1)
            
        # repeat to (1, 14, 512)
        if w.shape[1] == 1:
            w = w.repeat(1, self.active_G.backbone.synthesis.num_ws, 1)
        
        num_images = max(len(w), len(camera_params))
        
        # replicate if single w was passed
        if w.shape[0] < num_images:
            w = w.repeat(num_images, 1, 1)
        
        # replicate if single camera parameter was passed
        if camera_params.shape[0] < num_images:
            camera_params = camera_params.repeat(num_images, 1)
        
        assert w.shape[0] == camera_params.shape[0], f'incompatible size of w ({w.shape}) and camera params ({camera_params.shape})'
        
        images = []
        images_raw = []
        depths = []
        self.active_G.to(self.device)
        
        for b in range(math.ceil(num_images/self.BS)):
            w_batch = w[b*self.BS:min((b+1)*self.BS, num_images), :, :]
            if triplane_offset != None:
                triplane_offset_batch = triplane_offset[b*self.BS:min((b+1)*self.BS, num_images), :, :]
            else:
                triplane_offset_batch = None

            camera_params_batch = camera_params[b*self.BS:min((b+1)*self.BS, num_images), :]  
            
            try:
                if grad:        
                    out = self.active_G.synthesis(w_batch, camera_params_batch, noise_mode=noise_mode, cache_backbone=True, delta_triplane=triplane_offset_batch, ID_triplane_dir=ID_triplane_dir) 
                else:
                    with torch.no_grad():  
                        out = self.active_G.synthesis(w_batch, camera_params_batch, noise_mode=noise_mode, cache_backbone=True, delta_triplane=triplane_offset_batch, ID_triplane_dir=ID_triplane_dir)
            except:
                if grad:        
                    out = self.active_G.module.synthesis(w_batch, camera_params_batch, noise_mode=noise_mode, cache_backbone=True, delta_triplane=triplane_offset_batch, ID_triplane_dir=ID_triplane_dir) 
                else:
                    with torch.no_grad():  
                        out = self.active_G.module.synthesis(w_batch, camera_params_batch, noise_mode=noise_mode, cache_backbone=True, delta_triplane=triplane_offset_batch, ID_triplane_dir=ID_triplane_dir)

            images.append(out['image'])
            if output_all:  
                images_raw.append(out['image_raw'])
                depths.append(out['image_depth'])
                
        images = torch.cat(images, axis=0)
        
        # return only output images
        if not output_all:
            return images
        
        images_raw = torch.cat(images_raw, axis=0)
        depths = torch.cat(depths, axis=0)
        return {'image': images, 'image_raw': images_raw, 'image_depth': depths}
    
    
    def generate_original(self, w, yaw=0.0, pitch=0.0, focal_length=None, noise_mode='const', output_all=False, grad=False):                 
        # convert angles to camera parameter(s)
        camera_params = self.get_camera_parameters(yaw=yaw, pitch=pitch, focal_length=focal_length)

        # convert w to correct size
        # (,512) -> (1, 512)
        if len(w.shape) == 1:
            w = w.unsqueeze(0)
        
        # (x, 512) -> (x, 1, 512)
        if len(w.shape) == 2:
            w = w.unsqueeze(1)
            
        # repeat to (1, 14, 512)
        if w.shape[1] == 1:
            w = w.repeat(1, self.active_G.backbone.synthesis.num_ws, 1)
        
        num_images = max(len(w), len(camera_params))
        
        # replicate if single w was passed
        if w.shape[0] < num_images:
            w = w.repeat(num_images, 1, 1)
        
        # replicate if single camera parameter was passed
        if camera_params.shape[0] < num_images:
            camera_params = camera_params.repeat(num_images, 1)
        
        assert w.shape[0] == camera_params.shape[0], f'incompatible size of w ({w.shape}) and camera params ({camera_params.shape})'
        
        images = []
        images_raw = []
        depths = []
        self.active_G.to(self.device)
        
        for b in range(math.ceil(num_images/self.BS)):
            w_batch = w[b*self.BS:min((b+1)*self.BS, num_images), :, :]           
            camera_params_batch = camera_params[b*self.BS:min((b+1)*self.BS, num_images), :]  
            
            if grad:        
                out = self.active_G.synthesis_original(w_batch, camera_params_batch, noise_mode=noise_mode) 
            else:
                with torch.no_grad():  
                    out = self.active_G.synthesis_original(w_batch, camera_params_batch, noise_mode=noise_mode)
            images.append(out['image'])
            if output_all:  
                images_raw.append(out['image_raw'])
                depths.append(out['image_depth'])
                
        images = torch.cat(images, axis=0)
        
        # return only output images
        if not output_all:
            return images
        
        images_raw = torch.cat(images_raw, axis=0)
        depths = torch.cat(depths, axis=0)
        return {'image': images, 'image_raw': images_raw, 'image_depth': depths}


    def normalize_depth(self, depth_tensor, resize=True):
        depth_image = -depth_tensor
        depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 2 - 1

        if resize:
            depth_image = F.interpolate(depth_image.repeat(1, 3, 1, 1), size=(512, 512), mode='nearest')
        return depth_image