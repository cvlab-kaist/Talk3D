import sys
import os
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
import cv2
    
sys.path.append('..')
from vive3D.visualizer import *
from vive3D.eg3d_generator_original import *
from vive3D.landmark_detector import *
from vive3D.video_tool import *
from vive3D.segmenter import *
from vive3D.inset_pipeline import *
from vive3D.aligner import *
from vive3D.interfaceGAN_editor import *
from vive3D.config import *
from vive3D.util import tensor_to_image
from preprocess_utils.pose_estimation import face_orientation

@click.command()
@click.option('--source_video', type=str, help='Path to source video: Place synthesized frames here')
@click.option('--target_video', type=str, help='Path to target video: GT frames here')
@click.option('-g', '--generator_path', type=str, default='./models/ffhq-fixed-triplane512-128.pkl', help='Path to pretrained_generator')
@click.option('-d', '--device', type=str, help='GPU device that should be used.', default='cuda')
@click.option('--focal_length', type=float, help='Generator Focal Length', default=3.6)
@click.option('--camera_position', type=(float, float, float), nargs=3, help='Generator Camera Position', default=(0, 0.05, 0.2))
@click.option('--frame_duration', type=int, default=100000)
@click.option('--sample_rate', type=int, default=16000)
@click.option('--traintest_split_rate', type=float, default=10/11)
def main(**config):
    _main(**config)

'''
This is post-processing code to align the cropping boundaries 
of different models(e.g. ad-nerf, rad-nerf, er-nerf) uniformly.
'''
def _main(source_video, 
          target_video, 
          generator_path,
          device,
          focal_length, 
          camera_position,
          frame_duration, 
          sample_rate,
          traintest_split_rate):
    
    device = torch.device(device)
    
    # aud_dir = source_video.replace('.mp4', '.wav')
    
    # cmd = f'ffmpeg -i {source_video} -f wav -ar {sample_rate} {aud_dir}'
    # os.system(cmd)
                       
    # create new EG3D generator instance with appropriate camera parameters
    generator = EG3D_Generator(generator_path, device)
    generator.set_camera_parameters(focal_length=focal_length, cam_pivot=camera_position)
    
    
    # additionally required tools
    segmenter = Segmenter(device=device)
    landmark_detector = LandmarkDetector(device=device)
    align = Aligner(landmark_detector=landmark_detector, segmenter=segmenter, device=device)
    
    # evaluate average face as reference face
    average_face_tensors = generator.generate(generator.get_average_w(), yaw=[0.6, 0.0, -0.6], pitch=[-0.1, -0.1, -0.1])
    # if output_intermediate:
    #     Visualizer.save_tensor_to_file(average_face_tensors[1], f'reference_face', out_folder=video_output_path)
    
    
    target_frames = extract_frames_from_video_custom(target_video) # gt (no crop) frames
    source_frames = extract_frames_from_video_custom(source_video) # synthesized frames
    print(f'*******************************************************************************')
    print(f'Loaded target video {target_video.split("/")[-1]} num of frames : {len(target_frames)}')
    print(f'Loaded gt video {source_video.split("/")[-1]} num of frames : {len(source_frames)}')
    
    print(f'*******************************************************************************')
    print(f'Preprocessing video frames...')
    
    face_tensors, landmarks = align.get_face_tensors_from_adnerf_frames(source_frames,
                                                                        target_frames, 
                                                                        reference_face=average_face_tensors[1], 
                                                                        get_all=False, 
                                                                        smooth_landmarks=True, 
                                                                        smooth_sigma=2, 
                                                                        traintest_split_rate=traintest_split_rate)
    
    
    face_tensors = tensor_to_image(face_tensors)
        
    output_path = f'./cropped'
    os.makedirs(output_path, exist_ok=True)
    name = source_frames.split('/')[-1].split('.')[0]
    
    video_writer = cv2.VideoWriter(f'{output_path}/{name}_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (512, 512))
    # video_writer = cv2.VideoWriter(f'{output_path}/{name}_output.mp4', cv2.VideoWriter_fourcc(*'H264'), 25, (512, 512))
    
    for i, (face_tensor) in tqdm(enumerate(face_tensors), desc='Saving frames...'):
        video_writer.write(face_tensor)
        
    video_writer.release()

    # cmd = f'ffmpeg -loglevel quiet -y -i {output_path}/{name}_output.mp4 -i {aud_dir} -c:v copy -c:a aac {output_path}/{ID}_{aud}_512res_OOD_recon.mp4'
    cmd = f'ffmpeg -loglevel quiet -y -i {output_path}/{name}_output.mp4 -c:v libx264 -c:a aac {output_path}/{ID}_512res_novel_recon.mp4'
    os.system(cmd)


def extract_frames_from_video_custom(video_dir, traintest_split_rate=-1):
    
    video_capture = cv2.VideoCapture(video_dir)
    
    if not video_capture.isOpened():
        raise FileNotFoundError(f"Video file not found: {video_dir}")

    video_length = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    
    saved_frames = []
    current_frame = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        if current_frame >= int(traintest_split_rate * video_length):
            saved_frames.append(frame)
                        
        current_frame += 1

    video_capture.release()

    return saved_frames


if __name__ == '__main__':
    main()
    
    
    
    
