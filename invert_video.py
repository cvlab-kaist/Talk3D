import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import click
from datetime import datetime
import pickle
    
from vive3D.visualizer import *
from vive3D.eg3d_generator_original import *
from vive3D.landmark_detector import *
from vive3D.video_tool import *
from vive3D.segmenter import *
from vive3D.inset_pipeline import *
from vive3D.aligner import *
from vive3D.interfaceGAN_editor import *
from vive3D.config import *

@click.command()
@click.option('-s', '--savepoint_path', type=str, help='Savepoint directory', required=True)
@click.option('-v', '--source_video', type=str, help='Path to source video', required=True)
@click.option('-f', '--frames_path', type=str, help='Path where to store video frames (optional)')
@click.option('--start_sec', type=int, default=0)
@click.option('--end_sec', type=int, default=0)
@click.option('--resize_video', type=int, default=1)
@click.option('--focal_length', type=float, help='Generator Focal Length', default=3.6)
@click.option('--camera_position', type=(float, float, float), nargs=3, help='Generator Camera Position', default=(0, 0.05, 0.2))
@click.option('--loss_threshold', type=float, default=0.2, help='Early stopping threshold for inversion. Empirically selected per video.')
@click.option('-d', '--device', type=str, help='GPU device that should be used.', default='cuda')


def main(**config):
    _main(**config)


def _main(savepoint_path,
          source_video, 
          frames_path,
          start_sec, 
          end_sec, 
          resize_video, 
          focal_length, 
          camera_position,
          loss_threshold,
          device):
    
    assert os.path.exists(savepoint_path), f'Savepoint path does not exist. Run personalize_generator.py first!'
    video_output_path = os.getcwd()+f'/video/{savepoint_path.split("/")[-1]}'
    os.makedirs(video_output_path, exist_ok=True)
    
    device = torch.device(device)
    
    # create video tool instance for target video
    vid = VideoTool(source_video, frames_path, set_fps=25)
    
    print(f'*******************************************************************************')
    print(f'Loading personalized generator from {savepoint_path}/G_tune.pkl')
    tuned_generator_path = f'{savepoint_path}/G_tune.pkl'
    assert os.path.exists(tuned_generator_path), f'Generator is not available at {tuned_generator_path}, please check savepoint_path'
    generator = EG3D_Generator(tuned_generator_path, device, load_tuned=True)
    generator.set_camera_parameters(focal_length=focal_length, cam_pivot=camera_position)
    
    print(f'*******************************************************************************')
    print(f'Loading video {source_video.split("/")[-1]} from secs {start_sec}-{end_sec} and cropping faces')
    
    # additionally required tools
    segmenter = Segmenter(device=device)
    landmark_detector = LandmarkDetector(device=device)
    align = Aligner(landmark_detector=landmark_detector, segmenter=segmenter, device=device)
    
    frames_video = vid.extract_frames_from_video(start_sec, end_sec, resize=resize_video)
    
    w_person = torch.load(f'{savepoint_path}/inversion_w_person.pt').to(device)
    w_offsets = torch.load(f'{savepoint_path}/inversion_w_offsets.pt').to(device)
    reference_neutral_face = generator.generate(w_person, 0.0, -0.1)

    face_tensors_video, segmentation_tensors_video, landmarks_video = align.get_face_tensors_from_frames(frames_video, reference_face=reference_neutral_face, smooth_landmarks=True)

    
    print(f'*******************************************************************************')
    print(f'Invert video sequence...')
    
    
    # create pipeline instance
    pipeline = Pipeline(generator, segmenter, align, device=device)
    
    selected_face_tensors = torch.load(f'{savepoint_path}/selected_face_tensors.pt').to(device)
    faces_accum_segmentation = segmenter.get_eyes_mouth_BiSeNet(selected_face_tensors.to(device), dilate=8).any(dim=0)

    w_offsets_video, yaws_video, pitches_video = pipeline.inversion_video(w_person, w_offsets, face_tensors_video, face_segmentation=faces_accum_segmentation, loss_threshold=loss_threshold, plot_progress=False)

    torch.save(w_offsets_video.cpu(), f'{savepoint_path}/inversion_{start_sec}-{end_sec}_w_offsets.pt')
    torch.save(torch.tensor(list(zip(yaws_video, pitches_video))).cpu(), f'{savepoint_path}/inversion_{start_sec}-{end_sec}_angles.pt')         
    

if __name__ == '__main__':
    main()