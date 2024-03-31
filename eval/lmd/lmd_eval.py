import numpy as np
import cv2, os, sys, subprocess, platform, torch
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat

sys.path.insert(0, 'third_part')
# sys.path.insert(0, 'third_part/GPEN')
# sys.path.insert(0, 'third_part/GFPGAN')

# 3dmm extraction
from third_part.face3d.util.preprocess import align_img
from third_part.face3d.util.load_mats import load_lm3d
from third_part.face3d.extract_kp_videos import KeypointExtractor

from utils import audio
from utils.ffhq_preprocess import Croper
from utils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from utils.inference_utils import options 

import warnings
warnings.filterwarnings("ignore")

args = options()

import torchvision.utils as vutils
from glob import glob


def load_video_to_frame(video_path):
    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)

    full_frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        y1, y2, x1, x2 = args.crop
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]
        frame = frame[y1:y2, x1:x2]
        full_frames.append(frame)

    
    # print ("[Step 0] Number of frames available for inference: "+str(len(full_frames)))
    # face detection & cropping, cropping the first frame as the style of FFHQ
    croper = Croper('checkpoints/shape_predictor_68_face_landmarks.dat')
    small_frames_RGB = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in full_frames]

    
    return small_frames_RGB
    # Loop through each frame and resize it
    for frame in small_frames_RGB:
        height, width, _ = frame.shape

        if height < target_size[0] or width < target_size[1]:
            resized_frame = cv2.resize(frame, target_size)
            full_frames_RGB.append(resized_frame)
        else:
            full_frames_RGB.append(frame)
    
    del small_frames_RGB

    full_frames_RGB, crop, quad = croper.crop(full_frames_RGB, xsize=512)

    clx, cly, crx, cry = crop
    lx, ly, rx, ry = quad
    lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
    oy1, oy2, ox1, ox2 = cly+ly, min(cly+ry, full_frames[0].shape[0]), clx+lx, min(clx+rx, full_frames[0].shape[1])
    # original_size = (ox2 - ox1, oy2 - oy1)
    frames_pil = [Image.fromarray(cv2.resize(frame,(256,256))) for frame in full_frames_RGB]
    

def load_video_to_frame_crop(video_path, cropped_size=(512, 512)):
    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)

    full_frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        y1, y2, x1, x2 = args.crop
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]
        frame = frame[y1:y2, x1:x2]
        full_frames.append(frame)

    
    # print ("[Step 0] Number of frames available for inference: "+str(len(full_frames)))
    # face detection & cropping, cropping the first frame as the style of FFHQ
    croper = Croper('checkpoints/shape_predictor_68_face_landmarks.dat')
    small_frames_RGB = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in full_frames]
    
    full_frames_RGB = []

    # Define the target size
    target_size = (512, 512)

    # Loop through each frame and resize it
    for frame in small_frames_RGB:
        height, width, _ = frame.shape

        if height < target_size[0] or width < target_size[1]:
            resized_frame = cv2.resize(frame, target_size)
            full_frames_RGB.append(resized_frame)
        else:
            full_frames_RGB.append(frame)
    
    del small_frames_RGB

    full_frames_RGB, crop, quad = croper.crop(full_frames_RGB, xsize=512)

    clx, cly, crx, cry = crop
    lx, ly, rx, ry = quad
    lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
    oy1, oy2, ox1, ox2 = cly+ly, min(cly+ry, full_frames[0].shape[0]), clx+lx, min(clx+rx, full_frames[0].shape[1])
    # original_size = (ox2 - ox1, oy2 - oy1)
    frames_pil = [Image.fromarray(cv2.resize(frame, cropped_size)) for frame in full_frames_RGB]
    
    return frames_pil

def main():    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(os.path.join('temp', args.tmp_dir), exist_ok=True)
    
    """ get frames from video """
    pred_frames = load_video_to_frame(args.pred)
    gt_frames = load_video_to_frame(args.gt)#[:len_pred_frames]
    
    length = min(len(pred_frames), len(gt_frames))
    
    pred_frames = pred_frames[:length]
    gt_frames = gt_frames[:length]
        
    """ extract landmark """
    kp_extractor = KeypointExtractor()
    
    lm_pred = kp_extractor.extract_keypoint(pred_frames, './temp/pred_landmarks.txt')
    lm_gt = kp_extractor.extract_keypoint(gt_frames, './temp/gt_landmarks.txt')
    
    """
    :17: Jaw
    48: Lip
    """
    
    result_dist = 0
    for i in range(length):
        # when extracing landmark, the imgsize is 256.
        
        cur_lm_pred = lm_pred[i] #/256
        cur_lm_gt = lm_gt[i] #/256
        
        """ calculating scaling factor """
        spx1, spy1 = cur_lm_pred[40] # left
        sgx1, sgy1 = cur_lm_gt[40]
        
        spx2, spy2 = cur_lm_pred[46] # right
        sgx2, sgy2 = cur_lm_gt[46]
        
        scale_factor = (spx2-spx1)/(sgx2-sgx1)
        
        # cur_lm_gt_scaled = cur_lm_gt
        # cur_lm_gt_scaled[:, 0] = cur_lm_gt[:, 0] * scale_factor
        
        """ calculating lip/jaw distance """
        # jaw_dist = np.sum((cur_lm_pred[:17] - cur_lm_gt_scaled[:17]) ** 2)
        # lip_dist = np.sum((cur_lm_pred[48:] - cur_lm_gt_scaled[48:]) ** 2)
        cur_lm_pred = cur_lm_pred[48:68]
        cur_lm_gt = cur_lm_gt[48:68]
        
        cur_lm_pred = cur_lm_pred -  cur_lm_pred.mean(0)
        cur_lm_gt=  cur_lm_gt -  cur_lm_gt.mean(0)

        mean_dist = np.sqrt(((cur_lm_pred - cur_lm_gt) ** 2).sum(1)).mean(0)
        
        result_dist += mean_dist


    with open('lmd_tmp.txt', 'w') as f:
        f.write('{:.3f}'.format(result_dist/length))
    
    
if __name__ == '__main__':
    main()
