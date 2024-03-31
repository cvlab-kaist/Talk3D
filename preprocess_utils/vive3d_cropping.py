import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import click
from datetime import datetime
import pickle
import torchvision
import numpy as np
from scipy import ndimage
    
import os
import sys
PATH = os.path.realpath(__file__)
folders = PATH.split(os.path.sep)
new_path = os.path.join('/', *folders[:-2])
sys.path.append(new_path)

from vive3D.visualizer import *
from vive3D.eg3d_generator_original import *
from vive3D.landmark_detector import *
from vive3D.video_tool import *
from vive3D.segmenter import *
from vive3D.inset_pipeline import *
from vive3D.aligner import *
from vive3D.interfaceGAN_editor import *
from vive3D.config import *
from preprocess_utils.pose_estimation import face_orientation



@click.command()
@click.option('-v', '--source_video', type=str, help='Path to source video', required=True)
@click.option('-g', '--generator_path', type=str, help='Path to pretrained_generator', required=True)
@click.option('-s', '--savepoint_path', type=str, help='Savepoint directory', default=None)
@click.option('-d', '--device', type=str, help='GPU device that should be used.', default='cuda')
@click.option('--focal_length', type=float, help='Generator Focal Length', default=3.6)
@click.option('--camera_position', type=(float, float, float), nargs=3, help='Generator Camera Position', default=(0, 0.05, 0.2))
@click.option('--start_sec', type=int, default=0)
@click.option('--end_sec', type=int, default=0)
@click.option('--frame_step', type=float, default=0.04)
@click.option('--train_test_split', type=float, default=10/11, help='')

def main(**config):
    _main(**config)


def _main(source_video, 
          generator_path, 
          savepoint_path,
          device,
          focal_length, 
          camera_position,
          start_sec, 
          end_sec, 
          frame_step,
          train_test_split):
    

    name = source_video.split('/')[-1].split('.')[0]
    device = torch.device(int(device))
    
    # create video tool instance for target video
    vid = VideoTool(source_video, set_fps=25)
                       
    # create new EG3D generator instance with appropriate camera parameters
    generator_path = './models/ffhq-fixed-triplane512-128.pkl'
    generator = EG3D_Generator(generator_path, device=device)
    generator.set_camera_parameters(focal_length=focal_length, cam_pivot=camera_position)
    
    # additionally required tools
    segmenter = Segmenter(device=device)
    landmark_detector = LandmarkDetector(device=device)
    align = Aligner(landmark_detector=landmark_detector, segmenter=segmenter, device=device)
    
    # evaluate average face as reference face
    average_face_tensors = generator.generate(generator.get_average_w(), yaw=[0.6, 0.0, -0.6], pitch=[-0.1, -0.1, -0.1])

    print(f'*******************************************************************************')
    print(f'Loading video {source_video.split("/")[-1]} from secs {start_sec}-{end_sec}')
    
    frames = vid.extract_frames_from_video(start_sec, end_sec, frame_step)
    
    face_tensors, segmentation_tensors, _, landmarks = align.get_face_tensors_from_frames(frames, reference_face=average_face_tensors[1], smooth_landmarks=True, smooth_sigma=2, return_foreground_images=True, name=name)
    original_res = frames[0].shape[-2]
    
    landmarks = torch.from_numpy(landmarks).to(torch.float32)/original_res #0~1
    landmarks = (landmarks - 0.5)*2 #-1~1

    os.makedirs(os.path.join(savepoint_path, f'{name}/'), exist_ok=True)
    torch.save(landmarks, os.path.join(savepoint_path, f'{name}/landmarks.pt'))

    angle_list = []
    for frame, landmark in zip(frames, landmarks):
        angle = face_orientation(frame, landmark)
        angle_list.append(angle)
    angle_np = np.array(angle_list)

    angle_np = gaussian_filter1d(angle_np, 1, axis=0, mode='nearest').astype(np.float32)
    os.makedirs(os.path.join(savepoint_path, f'{name}/'), exist_ok=True)
    torch.save(torch.from_numpy(angle_np), os.path.join(savepoint_path, f'{name}/angles.pt'))

    os.makedirs(os.path.join(savepoint_path, f'{name}/image'), exist_ok=True)
    os.makedirs(os.path.join(savepoint_path, f'{name}/faceseg'), exist_ok=True)
    os.makedirs(os.path.join(savepoint_path, f'{name}/mouthseg'), exist_ok=True)
    os.makedirs(os.path.join(savepoint_path, f'{name}/bodyseg'), exist_ok=True)
    os.makedirs(os.path.join(savepoint_path, f'{name}/torsoseg'), exist_ok=True)
    
    image_list = []
    for i, (face_tensor,segmentation_tensor) in tqdm(enumerate(zip(face_tensors, segmentation_tensors)), desc='Saving frames...'):
        mouthseg_tensor = segmenter.get_mouth_BiSeNet(face_tensor.clone().cuda()).cpu()
        seg1 = segmenter.get_face_and_hair_BiSeNet_naive(face_tensor.clone().cuda(), erosion_num=3).cpu()
        segmentation_tensor = remove_small_mask(seg1)
        bodyseg_tensor = segmenter.get_body_BiSeNet(face_tensor.clone().cuda(), erosion_num=3).cpu()
        torsoseg_tensor = segmenter.get_torso_BiSeNet(face_tensor.clone().cuda(), erosion_num=3).cpu()
        

        torchvision.utils.save_image((face_tensor+1)/2, os.path.join(savepoint_path, f'{name}/image/{str(i).zfill(5)}.png'))
        torchvision.utils.save_image((segmentation_tensor), os.path.join(savepoint_path, f'{name}/faceseg/{str(i).zfill(5)}.png'))
        torchvision.utils.save_image((mouthseg_tensor), os.path.join(savepoint_path, f'{name}/mouthseg/{str(i).zfill(5)}.png'))
        torchvision.utils.save_image((bodyseg_tensor), os.path.join(savepoint_path, f'{name}/bodyseg/{str(i).zfill(5)}.png'))
        torchvision.utils.save_image((torsoseg_tensor), os.path.join(savepoint_path, f'{name}/torsoseg/{str(i).zfill(5)}.png'))
        
        # validation split rule
        if i >= int(len(frames)*train_test_split):
            image_list.append(face_tensor)
        
    image_list = tensor_to_image(torch.cat(image_list, dim=0))
    vid.write_frames_to_video(image_list, os.path.join(savepoint_path, f'{name}/GT_video_recon_'))

    tmp_name = os.path.join(savepoint_path, f'{name}/GT_video_recon_.mp4')
    outvid_name = os.path.join(savepoint_path, f'{name}/GT_video_recon.mp4') # codec to libx264
    cmd = f'ffmpeg -i {tmp_name} -c:v libx264 -c:a aac {outvid_name}'
    os.system(cmd)
    cmd = f'rm {tmp_name}'
    os.system(cmd)
    
def remove_small_mask(mask):
    count = 0
    while len(mask.shape)>=3:
        mask = mask.squeeze(0)
        count += 1
    mask = mask.numpy()

    labeled_mask, num_features = ndimage.label(mask)

    if num_features == 1:
        result_mask = mask.copy()
    else:
        result_mask = mask.copy()
        sizes = ndimage.sum(mask, labeled_mask, range(1, num_features + 1))
        smallest_label = np.argmin(sizes) + 1
        result_mask[labeled_mask == smallest_label] = 0

    for _ in range(count):
        result_mask = result_mask[np.newaxis, ...]
    result_mask = torch.from_numpy(result_mask)
    return result_mask


if __name__ == '__main__':
    main()