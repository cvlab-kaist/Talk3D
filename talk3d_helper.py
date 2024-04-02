import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lpips
import json
from pytorch_msssim import ms_ssim
import torchvision
import math

import sys
sys.path.append('./triplanenet')
sys.path.append('./resnet1d')
from vive3D.model_irse import Backbone
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from training.PE import get_embedder

def save_image(img, name):
    img = (img+1)/2
    torchvision.utils.save_image(img, './' + name + '.png')

def color_coding(scoremap_list, epochs, iters, savedir):
    '''
    colorize scoremaps
    input : tensor (0-1) [num_blocks, batch, head, HxW, tokens]
    output : cmap transformed [H, Wxtokens] -> save to savedir
    '''
    cmap = cm.get_cmap('viridis')
    names = ['Down', 'Mid', 'Up']
    only_mid = False
    Agg_attention = False
    if not isinstance(scoremap_list, list):
        only_mid = True
        scoremap_list = [scoremap_list]
        
    for i, scoremaps in enumerate(scoremap_list):# scoremaps = [num_blocks, batch, head, HxW, tokens]
        name = names[i] if not only_mid else 'Mid'
        scoremapx8 = scoremaps[0][0]
        scoremapx8_avg = torch.mean(scoremapx8, 0)

        Agg_attention=True
        res = int((scoremapx8_avg.shape[0]//3)**0.5)
        scoremaps = scoremapx8_avg
        scoremap_transform = []
        for k in range(scoremaps.shape[-1]):
            norm_scoremap = scoremaps[:, k] # res^2 * 3, n_tokens
            norm_scoremap1 = ((norm_scoremap[:res**2] - norm_scoremap[:res**2].min()) / (norm_scoremap[:res**2].max() - norm_scoremap[:res**2].min()))
            norm_scoremap2 = ((norm_scoremap[res**2:res**2*2] - norm_scoremap[res**2:res**2*2].min()) / (norm_scoremap[res**2:res**2*2].max() - norm_scoremap[res**2:res**2*2].min()))
            norm_scoremap3 = ((norm_scoremap[res**2*2:] - norm_scoremap[res**2*2:].min()) / (norm_scoremap[res**2*2:].max() - norm_scoremap[res**2*2:].min()))
            
            norm_scoremap = torch.cat([norm_scoremap1.reshape(res,res,1).flip(0), norm_scoremap2.reshape(res,res,1), norm_scoremap3.reshape(res,res,1)], dim=-2)
            '''if Agg_attention:
                norm_scoremap[:, :res]      = ((norm_scoremap[:, :res] - norm_scoremap[:, :res].min()) / (norm_scoremap[:, :res].max() - norm_scoremap[:, :res].min()))
                norm_scoremap[:, res:res*2] = ((norm_scoremap[:, res:res*2] - norm_scoremap[:, res:res*2].min()) / (norm_scoremap[:, res:res*2].max() - norm_scoremap[:, res:res*2].min()))
                norm_scoremap[:, res*2:]    = ((norm_scoremap[:, res*2:] - norm_scoremap[:, res*2:].min()) / (norm_scoremap[:, res*2:].max() - norm_scoremap[:, res*2:].min()))
            else:
                norm_scoremap = ((norm_scoremap - norm_scoremap.min()) / (norm_scoremap.max() - norm_scoremap.min())) # 0 to 1'''
            scoremap_transform.append(torch.from_numpy(cmap(norm_scoremap)))
        scoremap_transform = torch.stack(scoremap_transform, dim=0)
        scoremap_transform = scoremap_transform.squeeze(-2).permute(0,3,1,2)
        # scoremap_transform = scoremap_transform.permute(0,3,1,2).flip(2)
        torchvision.utils.save_image(scoremap_transform, f'{savedir}/{epochs}_{iters}_{name}.png')


def recon_evaluation(real_frames, fake_frames, device): #return psnr, ssim, lpips, mse
    psnr_total = 0
    msssim_total = 0
    lpips_total = 0
    mse_total = 0
    lpips_calculator = lpips.LPIPS(net='alex').to(device)
    for real, fake in zip(real_frames, fake_frames):
        msssim = ms_ssim(real, fake, data_range=1, size_average=False ).item()
        lpips_ = lpips_calculator(real, fake).sum()
        real_int = ((real + 1)*127.5).astype(np.uint8)
        fake_int = ((fake + 1)*127.5).astype(np.uint8)#image should be 0-255 in psnr
        mse_int = np.mean((real_int - fake_int) ** 2) 
        mse = np.mean((real - fake) ** 2) 
        psnr = 20 * np.log10(255.0 / np.sqrt(mse_int))
        psnr_total += psnr
        msssim_total += msssim
        lpips_total += lpips_
        mse_total += mse

    return psnr_total/len(fake_frames), msssim_total/len(fake_frames), lpips_total/len(fake_frames), mse_total/len(fake_frames)


def calc_lmd(landmark_detector, real_frames, fake_frames):
    real_lms, fake_lms = [], []
    for real in real_frames:
        real_lm.append(landmark_detector.get_landmarks(real)) #input np image
    for fake in fake_frames:
        fake_lm.append(landmark_detector.get_landmarks(fake))
    return compare_landmarks(real_lms, fake_lms) #avg lm distance

#################### from https://github.com/lelechen63/3d_gan #########################
def compare_landmarks(real_lms, fake_lms):
    distances = []
    for real_lm, fake_lm in zip(real_lms, fake_lms):
        dis = (real_lm-fake_lm)**2
        dis = np.sum(dis,axis=1)
        dis = np.sqrt(dis)

        dis = np.sum(dis,axis=0)
        distances.append(dis)
        # dis_txt.write(rps[inx] + '\t' + str(dis) + '\n') 
    distances = np.array(distances)
    average_distance = distances.sum() / len(real_lms)
    return average_distance

def set_OSG_audio_encoders(hidden_dim=96):
    in_dim = 96
    out_dim = 32

    net = nn.Sequential(*[
            nn.Conv1d(in_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, out_dim, 3, 1, 1, bias=False),
        ])
    return net

        
class Blinkencoder(nn.Module):
    '''type == rand, constant, zeropad'''
    def __init__(self, in_dim, out_dim, n_layers, type_='constant'): # img_channels = 96
        super(Blinkencoder, self).__init__()
        self.type_ = type_
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.simplemlp = simpleMLP(33, out_dim, n_layers)
        # self.set_starting_vector(type_)
        
        self.eye_embedding = get_embedder(multires = 16, input_dim = 1, include_input = True)
        
        
    def set_starting_vector(self, type_):
        if type_ == 'rand':
            sv = torch.randn(1, self.in_dim)
        elif type_ == 'constant':
            sv = 1.0 / (10000 ** (torch.arange(0, self.in_dim, 1).float() / self.in_dim))
        elif type_ == 'zero_pad':
            sv = torch.zeros(1, self.in_dim)
            sv[0,0] = 1
        else:
            '''unexpected type given for blink encoder'''
            raise ValueError
            
        self.starting_vector = nn.Parameter(sv)
        #requires_grad = False
        pass
    
    def forward(self, scalar):
        # x = (scalar * self.starting_vector).unsqueeze(1) # B, F -> B, 1, F
        
        x = self.eye_embedding(scalar) 
        if self.type_ == 'zero_pad':
            return x
        else:
            x = self.simplemlp(x).unsqueeze(dim = 1)
            return x
        
class Cameraencoder(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers=3, type_='constant'): # img_channels = 96
        super(Cameraencoder, self).__init__()
        self.angle_embedding = get_embedder(multires = 10, input_dim = 3, include_input = True)
        self.landmark_embedding = get_embedder(multires = 6, input_dim = 6, include_input = True)
        
        self.encoder1 = simpleMLP(63, 64, n_layers)
        self.encoder2 = simpleMLP(78, 64, n_layers)
        
    def forward(self, angle, landmark):
        out1,out2 = None,None
        out1  = self.angle_embedding(angle)
        out2  = self.landmark_embedding(landmark)
        
        
        out1 = self.encoder1(out1).unsqueeze(dim = 1)
        out2 = self.encoder2(out2).unsqueeze(dim = 1)
        
        return out1, out2
        

class simpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers): # img_channels = 96
        super(simpleMLP, self).__init__()
        self.in_layer = nn.Linear(in_dim, out_dim)
        self.mlp_layers = nn.ModuleList(
            [nn.Linear(out_dim, out_dim) for _ in range(n_layers-1)]
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.in_layer(x)
        for mlp in self.mlp_layers:
            x = self.relu(x)
            x = mlp(x)
        return x


def visualize_ldm_and_img(img, ldm, iter_idx):
    plt.imshow(img.transpose(1,2,0), zorder=1)
    x, y = ldm[:, 0], ldm[:, 1]
    scat = plt.scatter(x, y, zorder=2, s=1.0)
    plt.savefig(f'./ldm_test/test{iter_idx}.png')
    scat.remove()

def makelossfunc(device):
    loss_L2 = torch.nn.MSELoss(reduction='sum').to(device) 
    loss_L1 = torch.nn.L1Loss().to(device) 
    loss_percept = lpips.LPIPS(net='alex').to(device)
    loss_ID = IDLoss().to(device)
    return loss_L2, loss_L1, loss_percept, loss_ID

def gen_interp_cam(length_, max_yaw, max_pitch, num_keyframes=120):
    # generate circular camera yaws and pitches
    # num_keyframes decide the number of steps within cameras
    yaws = []
    pitches = []
    yaw_range = max_yaw/180*math.pi
    pitch_range = max_pitch/180*math.pi
    for frame_idx in range(length_):
        yaw = yaw_range * np.sin(math.pi * frame_idx / (num_keyframes/2))
        pitch = pitch_range * np.cos(math.pi * frame_idx / (num_keyframes/2))        
        yaws.append(yaw)
        pitches.append(pitch)

    yaws = torch.Tensor(yaws).unsqueeze(-1).to(torch.float32)
    pitches = torch.Tensor(pitches).unsqueeze(-1).to(torch.float32)
    return yaws, pitches

def parser2yaml(opts, savedir, totalbs):
    optsdict = opts.__dict__
    optsdict['total_bs'] = totalbs
    with open(f'{savedir}/configs.json', 'w') as f:
        json.dump(optsdict, f, indent=2)
    pass

class IDLoss(torch.nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(f'./models/model_ir_se50.pth'))
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

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def _load_syncnet(checkpoint_path):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_syncnet_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load_syncnet(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model


def cosine_loss(a, v, y):
    logloss = nn.BCELoss()
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def get_sync_loss(syncnet, mel, g, device, use_ddp=False):
    syncnet_T = 5
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    if use_ddp:
        y = torch.ones(g.size(0), 1).float().to("cuda:{}".format(device))
    else:
        y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)


def img2mse(x, y, num=0):
    if num > 0:
        return torch.mean((x[num] - y[num]) ** 2)
    else:
        return torch.mean((x - y) ** 2)
    
def tensor2psnr(x,y,device, m = 2):
    x = torch.clamp(x, max=1, min=-1)
    y = torch.clamp(y, max=1, min=-1)
    mse =  torch.mean((x - y) ** 2)
    return -10. * torch.log(mse / (m**2)) / torch.log(torch.Tensor([10.]).to(device))

def tensor2ssim(x,y):
    x = tensor_to_image(x)
    y = tensor_to_image(y)
    simm_sum = 0
    for x_, y_ in zip(x, y):
        ssim_score = ssim(x_, y_, multichannel=True, channel_axis = -1)
        simm_sum+=ssim_score
    simm_avg = simm_sum/x.shape[0]
    return simm_avg

def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)

class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x): #torch.Size([8, 44, 16])
        half_w = int(self.win_size/2)
        x = x[:, :, 8-half_w:8+half_w] #torch.Size([8, 44, 16])
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x) 
        return x # [8, 64]

class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len]
        y = self.attentionConvNet(y) 
        y = self.attentionNet(y.view(1, self.seq_len)).view(1, self.seq_len, 1)
        return torch.sum(y * x, dim=1) # [1, dim_aud]

import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin



from argparse import ArgumentParser


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--master_port', type=str,default='12355', help='Port num')
        
        'project name'
        self.parser.add_argument('--short_configs', type=str, help='output folder name')

        'directores'
        self.parser.add_argument('--saveroot_path', type=str, help='Path to experiment output directory')
        self.parser.add_argument('--checkpoint_dir', type=str, default=None, help='Path to checkpoint')
        self.parser.add_argument('--data_root_dir', type=str, default='./data', help='Path to root dataset directory')
        self.parser.add_argument('--personal_id', type=str, default=None, help='name of the inference id')
        self.parser.add_argument('--generator_dir', type=str, default='G_tune.pkl', help='Personalized generator directory')
        self.parser.add_argument('--w_id_dir', type=str, default='inversion_w_person.pt', help='w_id directory')
        self.parser.add_argument('--cam_dir', type=str, default='inversion_0-0_angles.pt', help='camera angles directory')
        self.parser.add_argument('--ID_triplane_dir', type=str, default='ID_triplane.pt', help='raw triplane directory')
        self.parser.add_argument('--image_dir', type=str, default='image', help='train image dir')
        self.parser.add_argument('--face_segmentation_dir', type=str, default='faceseg', help='train seg dir')
        self.parser.add_argument('--body_segmentation_dir', type=str, default='bodyseg', help='train body dir')
        self.parser.add_argument('--mouth_segmentation_dir', type=str, default='mouthseg', help='train mouth seg dir')
        self.parser.add_argument('--torso_segmentation_dir', type=str, default='torsoseg', help='train torso seg dir')
        self.parser.add_argument('--wav2lip_bbox_dir', type=str, default='wav2lip_bbox', help='train syncnet bbox dir')
        self.parser.add_argument('--audio_eo_dir', type=str, default='aud_eo.npy', help='radnerf audio feature dir')
        self.parser.add_argument('--audio_eo_OOD_dir', type=str, default='syncobama_B.npy', help='radnerf audio feature dir')
        self.parser.add_argument('--eye_dir', type=str, default='au.csv', help='eye blink csv dir')
        self.parser.add_argument('--inference_wav_raw', type=str, default='aud.wav', help='raw wav file on mov file after inference')
        self.parser.add_argument('--inference_wav', type=str, default='aud_train.wav', help='add wav file on mov file after inference')
        self.parser.add_argument('--inference_wav_novel', type=str, default='aud_novel.wav', help='add wav file on mov file after inference')
        self.parser.add_argument('--inference_wav_OOD_raw', type=str, default='syncobama_B.wav', help='raw OOD wav file on mov file after inference')
        self.parser.add_argument('--inference_wav_OOD', type=str, default='syncobama_B_trim.wav', help='add wav file on mov file after inference')
        self.parser.add_argument('--facemesh_checkpoint_dir', type=str, default='./facemesh/facemesh.pth', help='facemesh model weight dir')
        self.parser.add_argument('--syncnet_weight_path', type=str, default='./Wav2Lip/checkpoints/lipsync_expert.pth', help='syncnet weights dir')
        self.parser.add_argument('--gfpgan_weights_dir', type=str, default='./GFPGAN/gfpgan/weights/GFPGANv1.3.pth', help='gfpgan weights dir')
        self.parser.add_argument('--angles_dir', type=str, default='angles.pt', help='camera pitch yaw roll')
        self.parser.add_argument('--landmark_dir', type=str, default='landmarks.pt', help='face landmark dir')
        
        
        'G settings'
        self.parser.add_argument('--focal_length', default=3.6, type=float)
        self.parser.add_argument('--camera_position', default=[0, 0.05, 0.2], type=list)
        self.parser.add_argument('--device', default='cuda', type=str)
        self.parser.add_argument('--loss_threshold', default=0.1, type=float)
        self.parser.add_argument('--use_tuned_G', default=True, type=bool, help='bool: using tuned generator')
        
        
        'model versions'
        self.parser.add_argument('--predict_deltaplane', action='store_false', help='bool: switch whether predict deltaplane or not')
        self.parser.add_argument('--num_null_vector', default=1, type=int)
        self.parser.add_argument('--use_GFPGAN', action='store_true', help='bool: using GFPGAN for SR module (finetuning)')


        'loss versions'
        self.parser.add_argument('--use_mouth_loss', action='store_false', help='bool: using optimizable w_id')
        self.parser.add_argument('--use_id_loss', action='store_false', help='bool: using tuned generator')
        self.parser.add_argument('--use_sync_loss', action='store_false', help='bool: using tuned generator')

        'hyperparams'
        self.parser.add_argument('--batchsize', default=2, type=int, help='batch size per gpu - To use syncnet loss, you must keep it maximum 5')
        self.parser.add_argument('--vis_opt_process', action='store_true', help='bool: using mlpconditioning')
        self.parser.add_argument('--max_G_tuning_iter', default=1000, type=int, help='maximum iterations for G_tuning')
        self.parser.add_argument('--learning_rate', default=0.0005, type=float, help='select training learning rate')
        self.parser.add_argument('--learning_rate_tune', default=0.0002, type=float, help='select training learning rate')
        self.parser.add_argument('--scheduler_step_size', default=2, type=int, help='select training hyperparams - decay in epochs')
        self.parser.add_argument('--gamma', default=0.98, type=float, help='select training hyperparams')
        self.parser.add_argument('--image_loss_weight', default=16, type=float, help='select training hyperparams')
        self.parser.add_argument('--mouth_loss_weight', default=15, type=float, help='select mouth loss weight')
        self.parser.add_argument('--id_loss_weight', default=1, type=float, help='select ID loss weight')
        self.parser.add_argument('--LPIPS_loss_weight', default=1, type=float, help='select LPIPS loss weight')
        self.parser.add_argument('--tuning_start_iter', default=80000, type=int, help='num of epochs for initializing SR tuning stage')
        self.parser.add_argument('--sync_loss_weight', default=0.01, type=float, help='select syncnet loss weight')
        self.parser.add_argument('--log_psnr', action='store_true', help='bool: logging psnr in tqdm')
        self.parser.add_argument('--num_gpus', default=1, type=float, help='number of GPU')

        'validation'
        self.parser.add_argument('--do_inference', action='store_true')
        self.parser.add_argument('--do_evaluation', action='store_true')
        self.parser.add_argument('--inf_camera_type', type=str, default='GT', help='facemesh model weight dir')
        self.parser.add_argument('--vis_score_map', action='store_true', help='bool: visualize attention score map')
        self.parser.add_argument('--do_inference_novel', action='store_true', help='bool: inference on novel audio')
        self.parser.add_argument('--do_inference_OOD', action='store_true', help='bool: inference on ood audio')
        self.parser.add_argument('--use_wandb', action='store_true', help='wandb toggle')
        self.parser.add_argument('--wandb_project_name', type=str, default='test', help='facemesh model weight dir')
        self.parser.add_argument('--val_max_length', default=200, type=int, help='select inference start frame')
        self.parser.add_argument('--inf_batchsize', default=4, type=int, help='batchsize while inferencing - this can increase fps')
        self.parser.add_argument('--traintest_split_rate', default=10/11, type=float, help='split rate of train and test set')
        self.parser.add_argument('--max_rotation_yaw', default=30, type=float, help='split rate of train and test set')
        self.parser.add_argument('--max_rotation_pitch', default=20, type=float, help='split rate of train and test set')
        self.parser.add_argument('--lip_smoothing', action='store_true', help='bool: smoothing audio feature in 5 window')
        self.parser.add_argument('--eye_smoothing', action='store_true', help='bool: smoothing eye feature in 5 window')
        self.parser.add_argument('--only_do_inference', action='store_true', help='bool: no training, only do inference')
        self.parser.add_argument('--recon_eval_type', default='face', help='recon_eval_type: select within [all, image, face, body] all means evaluate all')
        self.parser.add_argument('--eval_type', default='all', help='eval_type: select within [all, ID, FID, Sync, LMD, AUE, only_recon] all means evaluate all, only_recon means evaluate only for reconstruction metrics')
        self.parser.add_argument('--num_iter_inf', default=10000, type=int, help='number of iterations interval for inference step, saving model weights, and metric logging')


    def parse(self):
        opts = self.parser.parse_args()
        return opts
    
    

    
# def set_audio_encoders(device, select='residual', repeat_dims=14, requires_grad=False):
#     lastlayerinputdim = 64

#     if select == 'vanilla':
#         net = nn.Sequential(*[
#                 nn.Conv1d(lastlayerinputdim, 256, 3, 1, 1, bias=False),
#                 nn.BatchNorm1d(256),
#                 nn.GELU(),
#                 nn.Conv1d(256, 256, 3, 1, 1, bias=False),
#                 nn.BatchNorm1d(256),
#                 nn.GELU(),
#                 nn.Conv1d(256, 512, 3, 1, 1, bias=False),
#                 nn.BatchNorm1d(512),
#                 nn.GELU(),
#                 nn.Conv1d(512, 512, 3, 1, 1, bias=False),
#             ])
#     elif select == 'vanilla+':
#         net = nn.Sequential(*[
#                 nn.Conv1d(lastlayerinputdim, 256, 3, 1, 1, bias=False),
#                 nn.BatchNorm1d(256),
#                 nn.GELU(),
#                 nn.Conv1d(256, 256, 3, 1, 1, bias=False),
#                 nn.BatchNorm1d(256),
#                 nn.GELU(),
#                 nn.Conv1d(256, 512, 3, 1, 1, bias=False),
#                 nn.BatchNorm1d(512),
#                 nn.GELU(),
#                 nn.Conv1d(512, 512, 3, 1, 1, bias=False),
#                 nn.BatchNorm1d(512),
#                 nn.GELU(),
#                 nn.Conv1d(512, repeat_dims*512, 3, 1, 1, bias=False),
#             ])    
        
#     elif select == 'residual_vanilla+':
#         net = residual_vanilla(lastlayerinputdim, repeat_dims=repeat_dims) 
    
#     elif select == 'mlpcond':
#         net = multihead_vanilla(lastlayerinputdim, repeat_dims=repeat_dims)
    
#     else: #resnet1d 'residual'
#         # config = set_resnet_config()
#         # net = ResNet1D(**config)
#         kernel_size = 8
#         stride = 2
#         n_block = 10 # original 48
#         downsample_gap=2
#         increasefilter_gap=4
#         net = ResNet1D(in_channels=lastlayerinputdim, 
#             base_filters=128, # 64 for ResNet1D, 352 for ResNeXt1D
#             kernel_size=kernel_size, 
#             stride=stride, 
#             groups=32, 
#             n_block=n_block, 
#             n_classes=512,
#             downsample_gap=downsample_gap,
#             increasefilter_gap=increasefilter_gap)
        
#         for name, p_ in net.named_parameters():
#             if name == 'module.basicblock_list.0.bn1.bias' or name == 'module.basicblock_list.0.bn1.weight':
#                 p_.requires_grad = False

#     return net.to(device)

# def set_resnet_config():
#     pass

# def set_stage3_encoder(device, requires_grad=False):
#     lastlayerinputdim = 64
#     network = nn.Sequential(*[
#                 nn.Conv1d(lastlayerinputdim, 256, 3, 1, 1, bias=False),
#                 nn.BatchNorm1d(256),
#                 nn.GELU(),
#                 nn.Conv1d(256, 256, 3, 1, 1, bias=False),
#                 nn.BatchNorm1d(256),
#                 nn.GELU(),
#                 nn.Conv1d(256, 512, 3, 1, 1, bias=False),
#                 nn.BatchNorm1d(512),
#                 nn.GELU(),
#             ])
#     if not requires_grad:
#         for i in network.parameters():
#             i.requires_grad = False
        
#     return network.to(device)

# class residual_vanilla(torch.nn.Module):
#     def __init__(self, in_dim, repeat_dims):
#         super(residual_vanilla, self).__init__()
#         self.repeat_dims = repeat_dims
#         self.gelu = nn.GELU()
#         self.conv1 = nn.Conv1d(in_dim, 256, 3, 1, 1, bias=False)
#         self.bn1 = nn.BatchNorm1d(256)
#         self.conv2 = nn.Conv1d(256, 512, 3, 1, 1, bias=False)
#         self.bn2 = nn.BatchNorm1d(512)
#         self.conv3 = nn.Conv1d(512, 512, 3, 1, 1, bias=False)
#         self.bn3 = nn.BatchNorm1d(512)
#         self.conv4 = nn.Conv1d(512, 512, 3, 1, 1, bias=False)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.conv5 = nn.Conv1d(512, repeat_dims*512, 3, 1, 1, bias=False)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.gelu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.gelu(x)
#         x = self.conv3(x)
#         x_ = x.contiguous()
#         x = self.bn3(x)
#         x = self.gelu(x)
#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.gelu(x)
#         x = self.conv5(x)
#         x += x_.repeat(1,self.repeat_dims,1)
#         return x

# class multihead_vanilla(torch.nn.Module):
#     def __init__(self, in_dim, repeat_dims):
#         super(multihead_vanilla, self).__init__()
#         self.gelu = nn.GELU()
#         self.conv1 = nn.Conv1d(in_dim, 96, 3, 1, 1, bias=False)
#         self.bn1 = nn.BatchNorm1d(96)
#         self.conv2 = nn.Conv1d(96, 96, 3, 1, 1, bias=False)
#         self.bn2 = nn.BatchNorm1d(96)
#         self.conv3 = nn.Conv1d(96, 96, 3, 1, 1, bias=False)
#         self.bn3 = nn.BatchNorm1d(96)

#         self.conv4_1 = nn.Conv1d(96, 96, 3, 1, 1, bias=False)
#         self.conv4_2 = nn.Conv1d(96, 96, 3, 1, 1, bias=False)
#         self.conv4_3 = nn.Conv1d(96, 96, 3, 1, 1, bias=False)
#         self.bn4_1 = nn.BatchNorm1d(96)
#         self.bn4_2 = nn.BatchNorm1d(96)
#         self.bn4_3 = nn.BatchNorm1d(96)
#         self.conv5_1 = nn.Conv1d(96, 32, 3, 1, 1, bias=False)
#         self.conv5_2 = nn.Conv1d(96, 32, 3, 1, 1, bias=False)
#         self.conv5_3 = nn.Conv1d(96, 32, 3, 1, 1, bias=False)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.gelu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.gelu(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.gelu(x)
#         x1 = self.conv4_1(x)
#         x2 = self.conv4_2(x)
#         x3 = self.conv4_3(x)
#         x1 = self.bn4_1(x1)
#         x2 = self.bn4_2(x2)
#         x3 = self.bn4_3(x3)
#         x1 = self.conv5_1(x1)
#         x2 = self.conv5_2(x2)
#         x3 = self.conv5_3(x3)
#         out = torch.cat([x1,x2,x3], dim=-1).permute(0,2,1).unsqueeze(-2)
#         return out