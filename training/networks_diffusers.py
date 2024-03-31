from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_utils.ops import upfirdn2d
from positional_encodings.torch_encodings import PositionalEncoding3D
# from training.networks_stylegan2 import AffineMidBlock

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


class SynthesisBlock_noaffine(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        out_channels,                           # Number of output channels.
        # w_dim,                                  # Intermediate latent (W) dimensionality. #affine condition -> audio condition dimension
        # resolution,                             # Resolution of this block.
        # img_channels,                           # Number of output color channels.
        up                      = 0,
        down                    = 0,
        architecture            = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter         = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        roll_out                = False,
        conv_shortcut           = True,
        residual                = False
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.architecture = architecture
        self.conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
            
        if roll_out:
            self.conv0 = nn.Conv2d(in_channels*3, out_channels, 3, 1, 1)
            self.conv1 = nn.Conv2d(out_channels*3, out_channels, 3, 1, 1)
        else:
            self.conv0 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU()
        self.roll_out = roll_out
        if up == 0 and down == 0 and not residual:
            self.conv1x1 = nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        if conv_shortcut:
            self.shortcut_net = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.residual = residual

    def forward(self, input_tensor):
        x = input_tensor
        if self.roll_out:
            x = aware3d(x) #3d aware 2dconv
        
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.activation(x)
        if self.roll_out:
            x = aware3d(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        if self.residual:
            x += input_tensor
        elif self.conv_shortcut:
            x += self.shortcut_net(input_tensor) # residual 1x1 conv

        if self.up != 0:
            x = upfirdn2d.upsample2d(x, self.resample_filter)
        elif self.down != 0:
            x = upfirdn2d.downsample2d(x, self.resample_filter)
        else: #output layer
            if not self.residual:
                x = self.conv1x1(x)

        if self.architecture == 'skip':
            return x, input_tensor
        else:
            return x

class SynthesisBlock_keepdims(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        # out_channels,                           # Number of output channels.
        # w_dim,                                  # Intermediate latent (W) dimensionality. #affine condition -> audio condition dimension
        # resolution,                             # Resolution of this block.
        # img_channels,                           # Number of output color channels.
        up                      = 0,
        down                    = 0,
        architecture            = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter         = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        roll_out                = False,
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.architecture = architecture
        self.up = up
        self.down = down
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
            
        if roll_out:
            self.conv0 = nn.Conv2d(in_channels*3, in_channels, 3, 1, 1)
            self.conv1 = nn.Conv2d(in_channels*3, in_channels, 3, 1, 1)
        else:
            self.conv0 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
            self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.activation = nn.LeakyReLU()
        self.roll_out = roll_out
        if up == 0 and down == 0:
            self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, input_tensor):
        x = input_tensor
        if self.roll_out:
            x = aware3d(x) #3d aware 2dconv
        
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.activation(x)
        if self.roll_out:
            x = aware3d(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x += input_tensor #residual

        if self.up != 0:
            x = upfirdn2d.upsample2d(x, self.resample_filter)
        elif self.down != 0:
            x = upfirdn2d.downsample2d(x, self.resample_filter)

        if self.architecture == 'skip':
            return x, input_tensor
        else:
            return x


def aware3d(x, split = False):
    '''
    From Mimic3D -> 3D aware 2dconv
    [ C*3, H, W ] -> [ C*9, H, W ]
    '''
    if isinstance(x, list):
        x_xy, x_yz, x_zx = x
        B, _, H, W = x_xy.shape
        B *= 3
    else:
        x_ = x.view(-1, 3, x.shape[1]//3, x.shape[2], x.shape[3])
        x_xy, x_yz, x_zx = x_[:, 0], x_[:, 1], x_[:, 2]
        B, _, H, W = x.shape
    x_zy = x_yz.permute(0,1,3,2)
    x_xz = x_zx.permute(0,1,3,2)
    x_yx = x_xy.permute(0,1,3,2)

    x_zy_pz = x_zy.mean(dim=-1, keepdim=True).repeat(1,1,1,x_xy.shape[-1])
    x_xz_pz = x_xz.mean(dim=-2, keepdim=True).repeat(1,1,x_xy.shape[-2],1)
    x_xy_ = torch.cat([x_xy, x_zy_pz, x_xz_pz], 1)

    x_yx_px = x_yx.mean(dim=-2, keepdim=True).repeat(1,1,x_yz.shape[-2],1)
    x_xz_px = x_xz.mean(dim=-1, keepdim=True).repeat(1,1,1,x_yz.shape[-1])
    x_yz_ = torch.cat([x_yx_px, x_yz, x_xz_px], 1)

    x_yx_py = x_yx.mean(dim=-1, keepdim=True).repeat(1,1,1,x_zx.shape[-1])
    x_zy_py = x_zy.mean(dim=-2, keepdim=True).repeat(1,1,x_zx.shape[-2],1)
    x_zx_ = torch.cat([x_yx_py, x_zy_py, x_zx], 1)

    if split:
        return x_xy_, x_yz_, x_zx_
    else:
        x = torch.cat([x_xy_[:, None], x_yz_[:, None], x_zx_[:, None]], 1).view(B, -1, H, W)
        return x


class Synthesis_Distributed_Up(torch.nn.Module):
    def __init__(self, img_channels): # img_channels = 96
        super(Synthesis_Distributed_Up, self).__init__()
        self.Upnet1_xy = SynthesisBlock_noaffine(img_channels*8, img_channels//3*4, up=2, architecture = 'orig', conv_shortcut=False)
        self.Upnet2_xy = SynthesisBlock_noaffine(img_channels*4, img_channels//3*2, up=2, architecture = 'orig', conv_shortcut=False)
        self.Upnet3_xy = SynthesisBlock_noaffine(img_channels*2, img_channels//3,   up=2, architecture = 'orig', conv_shortcut=False)
        self.Upnet1_yz = SynthesisBlock_noaffine(img_channels*8, img_channels//3*4, up=2, architecture = 'orig', conv_shortcut=False)
        self.Upnet2_yz = SynthesisBlock_noaffine(img_channels*4, img_channels//3*2, up=2, architecture = 'orig', conv_shortcut=False)
        self.Upnet3_yz = SynthesisBlock_noaffine(img_channels*2, img_channels//3,   up=2, architecture = 'orig', conv_shortcut=False)
        self.Upnet1_zx = SynthesisBlock_noaffine(img_channels*8, img_channels//3*4, up=2, architecture = 'orig', conv_shortcut=False)
        self.Upnet2_zx = SynthesisBlock_noaffine(img_channels*4, img_channels//3*2, up=2, architecture = 'orig', conv_shortcut=False)
        self.Upnet3_zx = SynthesisBlock_noaffine(img_channels*2, img_channels//3,   up=2, architecture = 'orig', conv_shortcut=False)
        
    def forward(self, plane, residual_tuple):
        
        # plane += residual_tuple[-1]
        xy, yz, zx = aware3d(plane, split=True)
        xy = self.Upnet1_xy(xy)
        yz = self.Upnet1_yz(yz)
        zx = self.Upnet1_zx(zx)
        plane = torch.cat([xy, yz, zx], dim=1)

        plane += residual_tuple[-1]
        xy, yz, zx = aware3d(plane, split=True)
        xy = self.Upnet2_xy(xy)
        yz = self.Upnet2_yz(yz)
        zx = self.Upnet2_zx(zx)
        plane = torch.cat([xy, yz, zx], dim=1)
        
        plane += residual_tuple[-2]
        xy, yz, zx = aware3d(plane, split=True)
        xy = self.Upnet3_xy(xy)
        yz = self.Upnet3_yz(yz)
        zx = self.Upnet3_zx(zx)
        plane = torch.cat([xy, yz, zx], dim=1)
        
        return plane
    
class Synthesis_Distributed_Down(torch.nn.Module):
    def __init__(self, img_channels):
        super(Synthesis_Distributed_Down, self).__init__()
        self.Downnet1_xy = SynthesisBlock_noaffine(img_channels,   img_channels//3*2, down=2, architecture = 'orig', conv_shortcut=False)
        self.Downnet2_xy = SynthesisBlock_noaffine(img_channels*2, img_channels//3*4, down=2, architecture = 'orig', conv_shortcut=False)
        self.Downnet3_xy = SynthesisBlock_noaffine(img_channels*4, img_channels//3*8, down=2, architecture = 'orig', conv_shortcut=False)
        self.Downnet1_yz = SynthesisBlock_noaffine(img_channels,   img_channels//3*2, down=2, architecture = 'orig', conv_shortcut=False)
        self.Downnet2_yz = SynthesisBlock_noaffine(img_channels*2, img_channels//3*4, down=2, architecture = 'orig', conv_shortcut=False)
        self.Downnet3_yz = SynthesisBlock_noaffine(img_channels*4, img_channels//3*8, down=2, architecture = 'orig', conv_shortcut=False)
        self.Downnet1_zx = SynthesisBlock_noaffine(img_channels,   img_channels//3*2, down=2, architecture = 'orig', conv_shortcut=False)
        self.Downnet2_zx = SynthesisBlock_noaffine(img_channels*2, img_channels//3*4, down=2, architecture = 'orig', conv_shortcut=False)
        self.Downnet3_zx = SynthesisBlock_noaffine(img_channels*4, img_channels//3*8, down=2, architecture = 'orig', conv_shortcut=False)
        
    def forward(self, plane):
        # s0 = plane
        xy, yz, zx = aware3d(plane, split=True) #32*3, 32*3, 32*3
        xy = self.Downnet1_xy(xy)
        yz = self.Downnet1_yz(yz)
        zx = self.Downnet1_zx(zx)
        plane = torch.cat([xy, yz, zx], dim=1)
        
        s1 = plane
        xy, yz, zx = aware3d(plane, split=True) #64*3, 64*3, 64*3
        xy = self.Downnet2_xy(xy)
        yz = self.Downnet2_yz(yz)
        zx = self.Downnet2_zx(zx)
        plane = torch.cat([xy, yz, zx], dim=1)
        
        s2 = plane
        xy, yz, zx = aware3d(plane, split=True) #128*3, 128*3, 128*3
        xy = self.Downnet3_xy(xy)
        yz = self.Downnet3_yz(yz)
        zx = self.Downnet3_zx(zx)
        plane = torch.cat([xy, yz, zx], dim=1) # 256*3, 32, 32

        return plane, [s1, s2]
    
class Deltaplane_Predictor(torch.nn.Module):
    def __init__(self, img_channels=96, plane_res=256, vis_score_map=False):
        super(Deltaplane_Predictor, self).__init__()
        self.Downnet = Synthesis_Distributed_Down(img_channels)
        self.Upnet = Synthesis_Distributed_Up(img_channels)
        self.transformer = nn.ModuleList([Transformer2DModel(img_channels*8//3, img_channels*8//3, n_layers=2, lin_proj_out=False)])
        self.vis_score_map = vis_score_map # bool
        self.img_channels=img_channels
        self.imgch_small = img_channels//3 # for easy computation = 32

    def forward(self, input_plane, audio_feat):
        plane = input_plane
        scoremaps = []
        plane, res = self.Downnet(plane)
        
        for attn in self.transformer:
            outs = attn(
                plane,
                encoder_hidden_states=audio_feat,
                vis_score_map=self.vis_score_map
            )
            scoremaps.append(outs[1])
            plane = outs[0]
            
        #layernorm + activation
        #plane dim = [B, 3*C, H, W]
        out = self.Upnet(plane, res)
        if self.vis_score_map:
            scoremaps = torch.cat(scoremaps, dim=0)
        
        return out, scoremaps
    

class DistributedLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DistributedLinear, self).__init__()
        self.lin_xy = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_yz = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_zx = nn.Linear(in_channels, out_channels, bias=True)
        
    def forward(self, input_tensor):
        channel_dims = input_tensor.shape[-1]//3
        xy = self.lin_xy(input_tensor[:, :, :channel_dims])
        yz = self.lin_yz(input_tensor[:, :, channel_dims:channel_dims*2])
        zx = self.lin_zx(input_tensor[:, :, channel_dims*2:])
        
        return torch.cat([xy, yz, zx], dim=-1)
    
    
class DistributedGroupNorm(torch.nn.Module):
    def __init__(self, groupnorm_dims, in_channels):
        super(DistributedGroupNorm, self).__init__()
        self.GN_xy = nn.GroupNorm(groupnorm_dims, in_channels, eps=1e-05, affine=True)
        self.GN_yz = nn.GroupNorm(groupnorm_dims, in_channels, eps=1e-05, affine=True)
        self.GN_zx = nn.GroupNorm(groupnorm_dims, in_channels, eps=1e-05, affine=True)
        
    def forward(self, input_tensor):
        channel_dims = input_tensor.shape[1]//3
        xy = self.GN_xy(input_tensor[:, :channel_dims])
        yz = self.GN_yz(input_tensor[:, channel_dims:channel_dims*2])
        zx = self.GN_zx(input_tensor[:, channel_dims*2:])
        
        return torch.cat([xy, yz, zx], dim=1)

class DistributedResnetBlock2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, conv_shortcut=True, groupnorm_dims=32):
        super(DistributedResnetBlock2D, self).__init__()
        self.resblock_xy = ResnetBlock2D(in_channels//3, out_channels//3, conv_shortcut=conv_shortcut, groupnorm_dims=groupnorm_dims)
        self.resblock_yz = ResnetBlock2D(in_channels//3, out_channels//3, conv_shortcut=conv_shortcut, groupnorm_dims=groupnorm_dims)
        self.resblock_zx = ResnetBlock2D(in_channels//3, out_channels//3, conv_shortcut=conv_shortcut, groupnorm_dims=groupnorm_dims)

    def forward(self, input_tensor, audio_feat=None):
        channel_dims = input_tensor.shape[1]//3
        xy = self.resblock_xy(input_tensor[:, :channel_dims])
        yz = self.resblock_yz(input_tensor[:, channel_dims:channel_dims*2])
        zx = self.resblock_zx(input_tensor[:, channel_dims*2:])
        output_tensor = torch.cat([xy, yz, zx], dim=1)

        return output_tensor


class ResnetBlock2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, conv_shortcut=True, groupnorm_dims=32):
        super(ResnetBlock2D, self).__init__()
        self.norm1 = nn.GroupNorm(groupnorm_dims, in_channels, eps=1e-05, affine=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.time_emb_proj = nn.Linear(1280, out_channels, bias=True)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-05, affine=True)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = None
        if conv_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1
            )

    def forward(self, input_tensor, audio_feat=None):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if audio_feat != None:
            audio_feat = self.nonlinearity(audio_feat)
            audio_feat = self.time_emb_proj(audio_feat)[:, :, None, None]
            hidden_states = hidden_states + audio_feat
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor

class Transformer2DModel(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, resolution=32, lin_proj_out=True, positional_enc=True, distributed=True):
        super(Transformer2DModel, self).__init__()
        self.norm = DistributedGroupNorm(32, in_channels) if distributed else nn.GroupNorm(32, in_channels, eps=1e-06, affine=True)
        self.proj_in = DistributedLinear(in_channels, out_channels) if distributed else nn.Linear(in_channels, out_channels, bias=True)
        if distributed:
            self.transformer_blocks = nn.ModuleList(
                [DistributedTransformerBlock(out_channels) for _ in range(n_layers)]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [BasicTransformerBlock(out_channels) for _ in range(n_layers)]
            )
        self.lin_proj_out = lin_proj_out
        if lin_proj_out:
            self.proj_out = nn.Linear(out_channels, out_channels, bias=True)
            
        self.positional_enc = positional_enc
        if positional_enc:
            self.pos_emb = nn.Parameter(self.sinusoidal_embedding_for_triplane(resolution, out_channels),
                                               requires_grad=False)

    def forward(self, hidden_states, encoder_hidden_states=None, vis_score_map=False):
        scoremaps = []
        
        batch, _, height, width = hidden_states.shape
        res = hidden_states
        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * width, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            if self.positional_enc:
                hidden_states += self.pos_emb
            outs = block(hidden_states, encoder_hidden_states, vis_score_map=vis_score_map)
            scoremaps.append(outs[1])
            hidden_states = outs[0]
        
        # distributed proj out
        if self.lin_proj_out:
            hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, width, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        if vis_score_map:
            scoremaps = torch.stack(scoremaps, dim=0)

        return [hidden_states + res, scoremaps]
    
    
    def sinusoidal_embedding_for_triplane(self, resolution, dim):
        '''
        returns 3D PE of triplanes
        resolution : triplane resolution
        dim        : attention hidden dims
        '''
        pe = PositionalEncoding3D(dim)
        pe3d = pe(torch.zeros(1,resolution, resolution, resolution, dim))
        
        pe3d_x0 = torch.mean(pe3d[:, resolution//2-1:resolution//2+1, :, :], dim=1) 
        pe3d_y0 = torch.mean(pe3d[:, :, resolution//2-1:resolution//2+1, :], dim=2)
        pe3d_z0 = torch.mean(pe3d[:, :, :, resolution//2-1:resolution//2+1], dim=3)
        pe3d = torch.cat([pe3d_z0, pe3d_x0, pe3d_y0], dim=0).permute(0,3,1,2).reshape(-1, resolution*resolution).unsqueeze(0).permute(0,2,1) # xy, yz, zx

        return pe3d



class BasicTransformerBlock(nn.Module):
    def __init__(self, hidden_size):
        super(BasicTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn1 = Attention(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn2 = Attention(hidden_size, 64)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.ff = FeedForward(hidden_size, hidden_size)

    def forward(self, x, encoder_hidden_states=None, vis_score_map=False):
        residual = x

        x = self.norm1(x)
        x, _ = self.attn1(x)
        x = x + residual

        residual = x

        x = self.norm2(x)
        if encoder_hidden_states is not None:
            x, scoremap = self.attn2(x, encoder_hidden_states, vis_score_map=vis_score_map)
        else:
            x, scoremap = self.attn2(x)
        x = x + residual

        residual = x

        x = self.norm3(x)
        x = self.ff(x)
        x = x + residual
        return [x, scoremap]

class DistributedTransformerBlock(nn.Module):
    def __init__(self, hidden_size):
        super(DistributedTransformerBlock, self).__init__()
        self.norm1 = DistributedLayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn1 = DistributedAttention(hidden_size)
        self.norm2 = DistributedLayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn2 = Attention(hidden_size, 64)
        self.norm3 = DistributedLayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.ff = DistributedFeedForward(hidden_size)

    def split_planes(self, x, sd):
        '''
        sd: single split dimension
        split three planes into pixel-wise dimension
        spatial feature remains.
        B, H*W, C*3 -> B, H*W*3, C
        '''
        return torch.cat([x[..., :sd], x[..., sd:sd*2], x[..., sd*2:]], dim=-2)

    def unsplit_planes(self, x, sd):
        '''Undo split_planes function'''
        return torch.cat([x[:, :sd, :], x[:, sd:sd*2, :], x[:, sd*2:, :]], dim=-1)

    def forward(self, x, encoder_hidden_states=None, vis_score_map=False):
        residual = x
        sd=residual.shape[-1]//3 #single dim
        undo_sd = residual.shape[-2]
        
        x = self.norm1(x)
        x = self.attn1(x)
        x = x + residual

        residual = x

        x = self.norm2(x)
        x = self.split_planes(x, sd)
        residual = self.split_planes(residual, sd)
        
        if encoder_hidden_states is not None:
            x, scoremap = self.attn2(x, encoder_hidden_states, vis_score_map=vis_score_map)
        else:
            x, scoremap = self.attn2(x)
        x = x + residual

        x = self.unsplit_planes(x, undo_sd)
        residual = x

        x = self.norm3(x)
        x = self.ff(x)
        x = x + residual
        return [x, scoremap]

class DistributedLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-05, elementwise_affine=True):
        super(DistributedLayerNorm, self).__init__()
        self.norm_xy = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_yz = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_zx = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=elementwise_affine)
        
    def forward(self, x):
        hidden_dim = x.shape[-1]//3
        xy = self.norm_xy(x[..., :hidden_dim])
        yz = self.norm_yz(x[..., hidden_dim:hidden_dim*2])
        zx = self.norm_zx(x[..., hidden_dim*2:])
        return torch.cat([xy, yz, zx], dim=-1)

class DistributedFeedForward(nn.Module):
    def __init__(self, hidden_size):
        super(DistributedFeedForward, self).__init__()
        self.ff_xy = FeedForward(hidden_size, hidden_size)
        self.ff_yz = FeedForward(hidden_size, hidden_size)
        self.ff_zx = FeedForward(hidden_size, hidden_size)
        
    def forward(self, x):
        hidden_dim = x.shape[-1]//3
        xy = self.ff_xy(x[..., :hidden_dim])
        yz = self.ff_yz(x[..., hidden_dim:hidden_dim*2])
        zx = self.ff_zx(x[..., hidden_dim*2:])
        return torch.cat([xy, yz, zx], dim=-1)

class FeedForward(nn.Module):
    def __init__(self, in_features, out_features):
        super(FeedForward, self).__init__()

        self.net = nn.ModuleList(
            [
                GEGLU(in_features, out_features * 4),
                nn.Dropout(p=0.0, inplace=False),
                nn.Linear(out_features * 4, out_features, bias=True),
            ]
        )
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

class GEGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super(GEGLU, self).__init__()
        self.proj = nn.Linear(in_features, out_features * 2, bias=True)

    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return x1 * torch.nn.functional.gelu(x2)

class DistributedAttention(nn.Module):
    def __init__(self, inner_dim, cross_attention_dim=None, num_heads=None, dropout=0.0):
        super(DistributedAttention, self).__init__()
        self.att_xy = Attention(inner_dim, cross_attention_dim=cross_attention_dim, num_heads=num_heads, dropout=dropout)
        self.att_yz = Attention(inner_dim, cross_attention_dim=cross_attention_dim, num_heads=num_heads, dropout=dropout)
        self.att_zx = Attention(inner_dim, cross_attention_dim=cross_attention_dim, num_heads=num_heads, dropout=dropout)
        
    def forward(self, x):
        hidden_dim = x.shape[-1]//3
        xy, _ = self.att_xy(x[..., :hidden_dim])
        yz, _ = self.att_yz(x[..., hidden_dim:hidden_dim*2])
        zx, _ = self.att_zx(x[..., hidden_dim*2:])
        return torch.cat([xy, yz, zx], dim=-1)

class Attention(nn.Module):
    def __init__(
        self, inner_dim, cross_attention_dim=None, num_heads=None, dropout=0.0
    ):
        super(Attention, self).__init__()
        if num_heads is None:
            self.head_dim = 64
            self.num_heads = inner_dim // self.head_dim
        else:
            self.num_heads = num_heads
            self.head_dim = inner_dim // num_heads

        self.scale = self.head_dim**-0.5
        if cross_attention_dim is None:
            cross_attention_dim = inner_dim
        self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, inner_dim), nn.Dropout(dropout, inplace=False)]
        )
        
        self.xymask = torch.ones(1,1,32,32,1)*(-1e8)
        self.xymask[:, :, 8:16, 12:20, :] = 0

    def forward(self, hidden_states, encoder_hidden_states=None, vis_score_map=False):
        q = self.to_q(hidden_states)
        k = (
            self.to_k(encoder_hidden_states)
            if encoder_hidden_states is not None
            else self.to_k(hidden_states)
        )
        v = (
            self.to_v(encoder_hidden_states)
            if encoder_hidden_states is not None
            else self.to_v(hidden_states)
        )
        b, t, c = q.size()
        
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)

        for layer in self.to_out:
            attn_output = layer(attn_output)

        
        scores_vis = attn_weights.clone().detach().cpu() if vis_score_map else None

        return attn_output, scores_vis
