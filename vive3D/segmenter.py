from vive3D import config
import torch
from torchvision import transforms
from kornia.morphology import dilation, erosion
import numpy as np
import cv2

####################################################################################################################################
# BiSeNet SEGMENTER
# Labels: 
# { 'background': 0, 
#   'skin':       1, 
#   'left_brow':  2, 
#   'right_brow': 3, 
#   'left_eye':   4, 
#   'right_eye':  5, 
#   'eye_glass':  6, 
#   'left_ear':   7, 
#   'right_ear':  8, 
#   'earring':    9, 
#   'nose':      10, 
#   'mouth':     11, 
#   'up_lip':    12, 
#   'low_lip':   13, 
#   'neck':      14, 
#   'necklace':  15, 
#   'cloth':     16, 
#   'hair':      17, 
#   'hat':       18} 
####################################################################################################################################        
class Segmenter:
    def __init__(self, device='cuda', path=f'./models/79999_iter.pth'):
        from vive3D import BiSeNet
        self.device=device        
        self.segmentation_model = BiSeNet.BiSeNet(19).eval().to(device).requires_grad_(False)
        self.segmentation_model.load_state_dict(torch.load(path))
    
    def logical_or_reduce(self, *tensors):
        return torch.stack(tensors, dim=0).any(dim=0)

    def logical_and_reduce(self, *tensors):
        return torch.stack(tensors, dim=0).all(dim=0)

    def get_segmentation_BiSeNet(self, input_tensor):
        input_tensor = transforms.functional.normalize(input_tensor.clip(-1, 1).add(1).div(2), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        segmentation = self.segmentation_model(input_tensor)[0].argmax(dim=1, keepdim=True)

        return segmentation

    def get_foreground_BiSeNet_original(self, input_tensor, background_classes=[0, 18, 16, 14], dilate=0, neck_treatment=False):
        segmentation = self.get_segmentation_BiSeNet(input_tensor)
        
        if neck_treatment and 14 not in background_classes:
            is_foreground_neck = (segmentation == 14)
            is_foreground = self.logical_and_reduce(*[segmentation != cls for cls in background_classes+[14]])
            foreground_mask = is_foreground.float()
            if dilate > 0:
                foreground_mask = dilation(foreground_mask, torch.ones(dilate, dilate).to(self.device))
                
            foreground_mask = self.logical_or_reduce(*[foreground_mask, is_foreground_neck])
        else:   
            is_foreground = self.logical_and_reduce(*[segmentation != cls for cls in background_classes])
            foreground_mask = is_foreground.float()

            if dilate > 0:
                foreground_mask = dilation(foreground_mask, torch.ones(dilate, dilate).to(self.device))
        return foreground_mask

    def get_foreground_BiSeNet(self, input_tensor, background_classes=[0, 18, 16, 14], dilate_num=-1, erosion_num=-1, neck_treatment=False):
        segmentation = self.get_segmentation_BiSeNet(input_tensor)
        
        if neck_treatment and 14 not in background_classes:
            is_foreground_neck = (segmentation == 14)
            is_foreground = self.logical_and_reduce(*[segmentation != cls for cls in background_classes+[14]])
            foreground_mask = is_foreground.float()
            if dilate_num > 0:
                foreground_mask = dilation(foreground_mask, torch.ones(dilate_num, dilate_num).to(self.device))
            
            foreground_mask = self.logical_or_reduce(*[foreground_mask, is_foreground_neck])
        else:   
            is_foreground = self.logical_and_reduce(*[segmentation != cls for cls in background_classes])
            foreground_mask = is_foreground.float()

            if dilate_num > 0:
                foreground_mask = dilation(foreground_mask, torch.ones(dilate_num, dilate_num).to(self.device))
            elif erosion_num > 0:
                foreground_mask = erosion(foreground_mask, torch.ones(erosion_num, erosion_num).to(self.device))
        return foreground_mask

    def get_face_BiSeNet(self, input_tensor, dilate=0):
        
        segmentation = self.get_segmentation_BiSeNet(input_tensor)

        is_foreground = torch.logical_and(segmentation>=1, segmentation<=13)
        foreground_mask = is_foreground.float()

        if dilate > 0:
            foreground_mask = dilation(foreground_mask, torch.ones(dilate, dilate).to(self.device))
        return foreground_mask
    
    def get_face_and_hair_BiSeNet_naive_wide(self, input_tensor, dilate=20):
        segmentation = self.get_segmentation_BiSeNet(input_tensor)

        is_foreground = torch.logical_and(segmentation>=1, segmentation<=13)
        is_hair = torch.logical_and(segmentation==17, segmentation==17)
        foreground_mask = torch.logical_or(is_foreground, is_hair).float()

        if dilate > 0:
            foreground_mask = dilation(foreground_mask, torch.ones(dilate, dilate).to(self.device))
        return foreground_mask
    
    def get_body_BiSeNet(self, input_tensor, dilate=0, dilate_num=-1, erosion_num=-1):
        segmentation = self.get_segmentation_BiSeNet(input_tensor)

        is_foreground = torch.logical_and(segmentation>=1, segmentation<=17)
        is_hair = torch.logical_and(segmentation==17, segmentation==17)
        foreground_mask = torch.logical_or(is_foreground, is_hair).float()


        if dilate_num > 0:
            foreground_mask = dilation(foreground_mask, torch.ones(dilate_num, dilate_num).to(self.device))
        elif erosion_num > 0:
            foreground_mask = erosion(foreground_mask, torch.ones(erosion_num, erosion_num).to(self.device))

        return foreground_mask
    
    def get_torso_BiSeNet(self, input_tensor, dilate=0, dilate_num=-1, erosion_num=-1):
        segmentation = self.get_segmentation_BiSeNet(input_tensor)

        is_torso = torch.logical_and(segmentation>=14, segmentation<=16)
        # is_hair = torch.logical_and(segmentation==17, segmentation==17)
        foreground_mask = torch.logical_or(is_torso, is_torso).float()

        if dilate_num > 0:
            foreground_mask = dilation(foreground_mask, torch.ones(dilate_num, dilate_num).to(self.device))
        elif erosion_num > 0:
            foreground_mask = erosion(foreground_mask, torch.ones(erosion_num, erosion_num).to(self.device))
            
        return foreground_mask

    def get_face_and_hair_BiSeNet_naive_tight(self, input_tensor, dilate=20):
        segmentation = self.get_segmentation_BiSeNet(input_tensor)

        is_foreground = torch.logical_and(segmentation>=1, segmentation<=13)
        is_hair = torch.logical_and(segmentation==17, segmentation==17)
        foreground_mask = torch.logical_or(is_foreground, is_hair).float()

        if dilate > 0:
            foreground_mask = erosion(foreground_mask, torch.ones(dilate, dilate).to(self.device))
        return foreground_mask
    
    def get_face_and_hair_BiSeNet_naive(self, input_tensor, dilate_num=-1, erosion_num=-1):
        segmentation = self.get_segmentation_BiSeNet(input_tensor)

        is_foreground = torch.logical_and(segmentation>=1, segmentation<=13)
        is_hair = torch.logical_and(segmentation==17, segmentation==17)
        foreground_mask = torch.logical_or(is_foreground, is_hair).float()

        if dilate_num > 0:
            foreground_mask = dilation(foreground_mask, torch.ones(dilate_num, dilate_num).to(self.device))
        elif erosion_num > 0:
            foreground_mask = erosion(foreground_mask, torch.ones(erosion_num, erosion_num).to(self.device))
        return foreground_mask
    
    def get_face_and_hair_BiSeNet(self, input_tensor, dilate=0):
        segmentation = self.get_segmentation_BiSeNet(input_tensor)

        is_foreground = torch.logical_and(segmentation>=1, segmentation<=13)
        is_hair = torch.logical_and(segmentation==17, segmentation==17)
        foreground_mask = torch.logical_or(is_foreground, is_hair).float()

        if dilate > 0:
            foreground_mask = dilation(foreground_mask, torch.ones(dilate, dilate).to(self.device))
        return foreground_mask

    def get_eyes_mouth_BiSeNet(self, input_tensor, classes=[2, 3, 10, 11, 12, 13], dilate=0):
        segmentation = self.get_segmentation_BiSeNet(input_tensor)
        
        eyes = self.logical_or_reduce(*[segmentation == cls for cls in [4, 5, 6]])
        eyes = dilation(eyes.float(), torch.ones(24, 24).to(self.device)) #artificially dilate eye area because it is too small otherwise
        is_foreground = self.logical_or_reduce(*[segmentation == cls for cls in classes])
        is_foreground = self.logical_or_reduce(*[is_foreground, eyes])
        foreground_mask = is_foreground.float()

        if dilate > 0:
            foreground_mask = dilation(foreground_mask, torch.ones(dilate, dilate).to(self.device))
        return foreground_mask

    def get_mouth_BiSeNet(self, input_tensor, classes=[11, 12, 13], dilate=0):
        segmentation = self.get_segmentation_BiSeNet(input_tensor)
        
        is_foreground = self.logical_or_reduce(*[segmentation == cls for cls in classes])
        is_foreground = self.logical_or_reduce(*[is_foreground])
        foreground_mask = is_foreground.float()

        if dilate > 0:
            foreground_mask = dilation(foreground_mask, torch.ones(dilate, dilate).to(self.device))
        return foreground_mask
    