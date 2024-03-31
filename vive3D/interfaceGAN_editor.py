import numpy as np 
from vive3D import config
import os
import torch
import glob

class Editor:
    def __init__(self, boundary_dir=None, device='cuda'):
        self.device = device
        self.boundary_dir = boundary_dir if boundary_dir else config.BOUNDARY_PATH
        self.boundaries = {}
        paths = glob.glob(f'{self.boundary_dir}/*_boundary.npy')
        for path in paths:
            name = '-'.join(path.split('/')[-1].split('-')[1:]).split('_')[0] #if 'sg' in self.boundary_dir else path.split('/')[-1].split('_')[2]
            self.load_boundary(path, name)   
    
    def load_boundary(self, boundary_path, name):    
        boundary_tensor = torch.tensor(np.load(boundary_path)).to(self.device)
        self.boundaries[ name ] = boundary_tensor
    
    def get_boundaries(self):
        return self.boundaries.keys()
    
    def expand_tensor(self, tensor):
        while len(tensor.shape) < 3:
            tensor = tensor.unsqueeze(0)
        return tensor
                
    def multi_edit(self, w, edits, w_offset=None):
        if w_offset is not None:
            w_offset = self.expand_tensor(w_offset)
            
        w_edit = w
        for weight, boundary_type in edits:
            w_edit = self.edit(w_edit, boundary_type, weight)

        if w_offset is not None: 
            w_edit = self.expand_tensor(w_edit)
            w_edit = w_edit.repeat(1, w_offset.shape[1], 1).to(self.device) + w_offset.to(self.device)

        return w_edit
    
    def edit_list(self, w, edits, w_offset=None):
        if w_offset is not None:
            w_offset = self.expand_tensor(w_offset)
            
        w_edits = []
        for weight, boundary_type in edits:
            w_edit = self.edit(w, boundary_type, weight)
                
            if w_offset is not None: 
                w_edit = self.expand_tensor(w_edit)
                w_edit = w_edit.repeat(1, w_offset.shape[1], 1).to(self.device) + w_offset.to(self.device)
                
            w_edits.append(w_edit)
        return torch.cat(w_edits, axis=0)
    
    def edit(self, w, name='default', weight=0, w_offsets=None):
        if name == 'default' or weight == 0:
            if w_offsets == None:
                return w.to(self.device)
            else: 
                w = self.expand_tensor(w)
                w_offsets = self.expand_tensor(w_offsets)
                
                return w.to(self.device).repeat(1, w_offsets.shape[1], 1) + w_offsets.to(self.device)
        else:
            if name in self.boundaries:
                boundary_tensor = self.boundaries[ name ]
            else:
                boundary_dir = config.BOUNDARY_PATH
                success = False
                for p in os.listdir(boundary_dir):
                    success = name in p
                    if success:
                        boundary_path = f'{boundary_dir}/{p}'
                        break
                        
                #boundary_path = f'{config.BOUNDARY_PATH}/eg3d_ffhq_{name}_boundary.npy'
                assert success, print(f'editing boundary for "{name}" does not exist in {boundary_dir}.')
                
                boundary_tensor = torch.tensor(np.load(boundary_path)).to(self.device)
                self.boundaries[ name ] = boundary_tensor
        
        w_edit = (w + weight * boundary_tensor).to(self.device)
        if w_offsets == None:
            return w_edit
        else:
            w_offsets = self.expand_tensor(w_offsets)
            w_edit = self.expand_tensor(w_edit)
            return w_edit.repeat(w_offsets.shape[0], w_offsets.shape[1], 1).to(self.device) + w_offsets.to(self.device)