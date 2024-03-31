import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import click
from datetime import datetime
import pickle
import torchvision
    
from vive3D.visualizer import *
from vive3D.eg3d_generator import *
from vive3D.landmark_detector import *
from vive3D.video_tool import *
from vive3D.segmenter import *
from vive3D.inset_pipeline import *
from vive3D.aligner import *
from vive3D.interfaceGAN_editor import *
from vive3D.config import *


@click.command()
@click.option('-v', '--source_video', type=str, help='Path to source video', required=True)
@click.option('-g', '--generator_path', type=str, help='Path to pretrained_generator', required=True)
@click.option('-f', '--frames_path', type=str, help='Path where to store video frames (optional)')
@click.option('-s', '--savepoint_path', type=str, help='Savepoint directory', default=None)
@click.option('-d', '--device', type=str, help='GPU device that should be used.', default='cuda')
@click.option('--focal_length', type=float, help='Generator Focal Length', default=3.6)
@click.option('--camera_position', type=(float, float, float), nargs=3, help='Generator Camera Position', default=(0, 0.05, 0.2))
@click.option('--start_sec', type=int, default=0)
@click.option('--end_sec', type=int, default=0)
@click.option('--frame_step', type=float, default=0.04)
@click.option('--output_intermediate', type=bool, is_flag=True, default=False)
@click.option('--frame', type=int, multiple=True, help='Frame indices for face selection')
@click.option('--weight_vgg', type=float, default=0.0, help='Weight for VGG loss')
@click.option('--weight_id', type=float, default=0.0, help='Weight for ID loss')
@click.option('--weight_pix', type=float, default=0.05, help='Weight for pixel loss')
@click.option('--weight_face', type=float, default=2.0, help='Weight for loss on important face region')
@click.option('--weight_lpips', type=float, default=1.0, help='Weight for perceptual loss')
@click.option('--weight_wdist', type=float, default=0.05, help='Start weight for regularizer for offset from person latent')
@click.option('--weight_wdist_target', type=float, default=0.005, help='End weight for decreasing regularizer weight')
@click.option('--learning_rate', type=float, default=0.01, help='Initial learning rate')
@click.option('--directory_name', type=str, help='savepoint and inversion results folder name')
def main(**config):
    _main(**config)


def _main(source_video, 
          generator_path, 
          frames_path,
          savepoint_path,
          device,
          focal_length, 
          camera_position,
          start_sec, 
          end_sec, 
          frame_step,
          frame,
          output_intermediate,
          weight_vgg,
          weight_id,
          weight_pix,
          weight_face,
          weight_lpips,
          weight_wdist,
          weight_wdist_target,
          learning_rate,
          directory_name):
    
    
    print(f'*******************************************************************************')
    print(f'Setting up VIVE3D...')
    
    device = torch.device(device)
    # create video tool instance for target video
    vid = VideoTool(source_video, frames_path, set_fps=25)
    vid_name = source_video.split('/')[-1].split('.')[0]
    # create new EG3D generator instance with appropriate camera parameters
    generator = EG3D_Generator(generator_path, device, construct_G=True)
    generator.set_camera_parameters(focal_length=focal_length, cam_pivot=camera_position)
    
    # additionally required tools
    segmenter = Segmenter(device=device)
    landmark_detector = LandmarkDetector(device=device)
    align = Aligner(landmark_detector=landmark_detector, segmenter=segmenter, device=device)
    
    # create pipeline instance
    pipeline = Pipeline(generator, segmenter, align, device=device)
    
    # instantiate save paths if they don't exist
    load_from_savepoint = savepoint_path is not None
    if not load_from_savepoint:
        savepoint_folder = f'{vid.get_video_title()}_{directory_name}'
        savepoint_path = os.getcwd()+f'/savepoints/{savepoint_folder}'
        print(f'created new savepoint directory at {savepoint_path}!')
        
    os.makedirs(savepoint_path, exist_ok=True)

    video_output_path = os.getcwd()+f'/video/{savepoint_path.split("/")[-1]}'
    os.makedirs(video_output_path, exist_ok=True)
    
    # evaluate average face as reference face
    average_face_tensors = generator.generate_original(generator.get_average_w(), yaw=[0.6, 0.0, -0.6], pitch=[-0.1, -0.1, -0.1])
    
    if output_intermediate:
        Visualizer.save_tensor_to_file(average_face_tensors[1], f'reference_face', out_folder=video_output_path)
    
    print(f'*******************************************************************************')
    print(f'Loading video {source_video.split("/")[-1]} from secs {start_sec}-{end_sec}')
    
    frames = vid.extract_frames_from_video(start_sec, end_sec, frame_step)

    
    print(f'*******************************************************************************')
    print(f'Preprocessing video frames...')
    
    face_tensors, segmentation_tensors, _, _ = align.get_face_tensors_from_frames(frames, reference_face=average_face_tensors[1], smooth_landmarks=True, smooth_sigma=2, get_all=False, return_foreground_images=True, black_bg=True)
    # _, segmentation_tensors, face_tensors, landmarks = align.get_face_tensors_from_frames(frames, reference_face=average_face_tensors[1], smooth_landmarks=True, smooth_sigma=5, get_all=False, return_foreground_images=True, name=vid_name, black_bg=True)

    selected_face_tensors = [face_tensors[key] for key in frame]
    selected_segmentation_tensors = [segmentation_tensors[key] for key in frame]
    
    selected_face_tensors = torch.cat(selected_face_tensors, dim=0)
    selected_segmentation_tensors = torch.cat(selected_segmentation_tensors, dim=0)

    if output_intermediate:
        Visualizer.save_tensor_to_file(selected_face_tensors, f'selected_face', out_folder=video_output_path)
    
    torch.save(selected_face_tensors.cpu(), f'{savepoint_path}/selected_face_tensors.pt')
    torch.save(selected_segmentation_tensors.cpu(), f'{savepoint_path}/selected_face_tensors_segmentation.pt')


    print(f'*******************************************************************************')
    print(f'Joint inversion of {len(selected_face_tensors)} selected faces...')
    
    start_time = time.time()

    w_person, w_offsets, yaws, pitches = pipeline.inversion(selected_face_tensors, initial_learning_rate=learning_rate, weight_vgg=weight_vgg, weight_id=weight_id, weight_pix=weight_pix, weight_face=weight_face, weight_lpips=weight_lpips, weight_wdist=weight_wdist, weight_wdist_target=weight_wdist_target, plot_progress=False, name=vid_name, out_folder=video_output_path)
    
    elapsed_time = time.time() - start_time

    print(f'optimization duration: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}.')

    torch.save(w_person.cpu(), f'{savepoint_path}/inversion_w_person.pt')
    torch.save(w_offsets.cpu(), f'{savepoint_path}/inversion_w_offsets.pt')
    torch.save(torch.tensor(list(zip(yaws, pitches))).cpu(), f'{savepoint_path}/inversion_angles.pt')
    
    inversion_images = generator.generate_original(w_person + w_offsets, yaws, pitches)
    Visualizer.save_tensor_to_file(inversion_images, f'inverted_face', out_folder=video_output_path)
    
    print(f'*******************************************************************************')
    print(f'Personalized Generator Fine-Tuning...')
    
    pipeline.tune(w_person, w_offsets, yaws, pitches, selected_face_tensors, plot_progress=False, name=vid_name, out_folder=video_output_path) 

    generator_path = f'{savepoint_path}/G_tune.pkl'
    with open(generator_path, 'wb') as f:
        pickle.dump(generator.G_tune.cpu(), f)
        
    

if __name__ == '__main__':
    main()
