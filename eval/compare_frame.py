import cv2
import os, sys
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from segmenter import Segmenter
from calc_IDSIM import calculate_idsim

# from syncnet_python.SyncNetInstance_calc_scores import *

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
def img2mse(x, y): return torch.mean((x - y) ** 2)


def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))



def compare_videos(video1_path, video2_path):
    # Load videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) != int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)):
        print('Videos must have the same number of frames, but video 1 has {} frames and video 2 has {} frames'.format(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))))
    
    # Get video properties
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize metrics
    psnr_sum = 0
    ssim_sum = 0
    lpips_sum = 0

    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='alex')

    # Loop through frames
    for i in range(frame_count):
        # Read frames
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Check if frames were read successfully
        if not ret1 or not ret2:
            break

        # Resize frames to 512x512
        if frame1.shape != (512, 512, 3):
            frame1 = cv2.resize(frame1, (512, 512))
        if frame2.shape != (512, 512, 3):
            frame2 = cv2.resize(frame2, (512, 512))
        
        # Compute PSNR
        psnr = peak_signal_noise_ratio(frame1, frame2)
        psnr_sum += psnr

        # Compute SSIM
        ssim = structural_similarity(frame1, frame2, multichannel=True,channel_axis = -1)
        ssim_sum += ssim

        # Convert frames to tensors
        frame1_tensor = torch.from_numpy(frame1).unsqueeze(0).permute(0, 3, 1, 2).float()
        frame2_tensor = torch.from_numpy(frame2).unsqueeze(0).permute(0, 3, 1, 2).float()

        # Compute LPIPS
        lpips_val = lpips_model(frame1_tensor, frame2_tensor).item()
        lpips_sum += lpips_val

    # Compute averages
    # print(frame_count)
    psnr_avg = psnr_sum / frame_count
    ssim_avg = ssim_sum / frame_count
    lpips_avg = lpips_sum / frame_count

    # Release video captures
    cap1.release()
    cap2.release()

    return psnr_avg, ssim_avg, lpips_avg


def compare_img(img1_path, img2_path):
    # Load videos
    frame1 = cv2.imread(img1_path)
    frame2 = cv2.imread(img2_path)

    lpips_model = lpips.LPIPS(net='alex')

    if frame1.shape != (512, 512, 3):
        frame1 = cv2.resize(frame1, (512, 512))
    if frame2.shape != (512, 512, 3):
        frame2 = cv2.resize(frame2, (512, 512))

    psnr = peak_signal_noise_ratio(frame1, frame2)
    ssim = structural_similarity(frame1, frame2, multichannel=True,channel_axis = -1)

    frame1_tensor = torch.from_numpy(frame1).unsqueeze(0).permute(0, 3, 1, 2).float()
    frame2_tensor = torch.from_numpy(frame2).unsqueeze(0).permute(0, 3, 1, 2).float()

    lpips_val = lpips_model(frame1_tensor, frame2_tensor).item()

    return psnr, ssim, lpips_val



def compare_body_seg_videos(video1_path, video2_path, segmenter = None):
    # Load videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) != int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)):
        print('Videos must have the same number of frames, but video 1 has {} frames and video 2 has {} frames'.format(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))))
    
    # Get video properties
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize metrics
    psnr_sum = 0
    ssim_sum = 0
    lpips_sum = 0

    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='alex')

    # Loop through frames
    for i in range(frame_count):
        # Read frames
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Check if frames were read successfully
        if not ret1 or not ret2:
            break
        
        frame1 = torch.from_numpy(frame1)
        frame2 = torch.from_numpy(frame2)
        
        
        frame1 = segmenter.get_body_BiSeNet(frame1)
        frame2 = segmenter.get_body_BiSeNet(frame2)
        
        frame1 = frame1.numpy()
        frame2 = frame2.numpy()
        
        # Resize frames to 512x512
        if frame1.shape != (512, 512, 3):
            frame1 = cv2.resize(frame1, (512, 512))
        if frame2.shape != (512, 512, 3):
            frame2 = cv2.resize(frame2, (512, 512))
            
        
        
        # Compute PSNR
        psnr = peak_signal_noise_ratio(frame1, frame2)
        psnr_sum += psnr

        # Compute SSIM
        ssim = structural_similarity(frame1, frame2, multichannel=True)
        ssim_sum += ssim

        # Convert frames to tensors
        frame1_tensor = torch.from_numpy(frame1).unsqueeze(0).permute(0, 3, 1, 2).float()
        frame2_tensor = torch.from_numpy(frame2).unsqueeze(0).permute(0, 3, 1, 2).float()

        # Compute LPIPS
        lpips_val = lpips_model(frame1_tensor, frame2_tensor).item()
        lpips_sum += lpips_val

    # Compute averages
    psnr_avg = psnr_sum / frame_count
    ssim_avg = ssim_sum / frame_count
    lpips_avg = lpips_sum / frame_count

    # Release video captures
    cap1.release()
    cap2.release()

    return psnr_avg, ssim_avg, lpips_avg


import argparse
if __name__ == '__main__':
    
    a=1  
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ID', type=str, required=True)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='')
    parser.add_argument('--short_configs', type=str)
    
    
    
    # parser.add_argument('--video1', type=str, required=True)
    # parser.add_argument('--video2', type=str, required=True)
    args = parser.parse_args()
    
    
    args.directory_path = args.data_dir
    # blockPrint()
    # psnr_avg, ssim_avg, lpips_avg = compare_videos(args.video1, args.video2)
    # enablePrint()
    
    # print('PSNR: {:.3f}'.format(psnr_avg))
    # print('SSIM: {:.3f}'.format(ssim_avg))
    # print('LPIPS: {:.3f}'.format(lpips_avg))
    
    import os
    from tqdm import tqdm
    import cv2
    from moviepy.editor import VideoFileClip
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    
   

    directory_path = f'{args.directory_path}/full_frame/{args.ID}'
    args.short_configs = args.ID
    output_path = f'{args.directory_path}/result/{args.ID}'
    
    segmenter = Segmenter(device='cuda')
    
    
    
    if os.path.exists(output_path+f'/{args.short_configs}_full_frame_PSNR_SSIM_LPIPS_IDSIM.txt'):
        os.remove(output_path+f'/{args.short_configs}_full_frame_PSNR_SSIM_LPIPS_IDSIM.txt')
    
    GT_files = os.listdir(f'{directory_path}/{args.ID}_GT_novel_recon')
    novel_files = os.listdir(f'{directory_path}/{args.ID}_novel_recon')
    
    
    
    GT_files = sorted(GT_files, key=lambda x: int(x.split('.jpg')[0]))
    novel_files = sorted(novel_files, key=lambda x: int(x.split('.jpg')[0]))
    
    
    for GT_name,novel_name in tqdm(zip(GT_files,novel_files), desc="Processing Files"):
        with open(output_path+f'/{args.short_configs}_full_frame_PSNR_SSIM_LPIPS_IDSIM.txt', 'a') as output:

            psnr_avg, ssim_avg, lpips_avg = compare_img(directory_path+f'/{args.short_configs}_GT_novel_recon/'+novel_name ,directory_path+f'/{args.short_configs}_novel_recon/'+novel_name) 
            
            output.write(f"---------------------------------------------\n")
            output.write(f"File: {novel_name}\n")
            output.write(f"Processed Result:\n")
            output.write(f"PSNR :{psnr_avg:.3f}\n")
            output.write(f"SSIM :{ssim_avg:.3f}\n")
            output.write(f"LPIPS :{lpips_avg:.3f}\n")
            # output.write(f"ID-SIM Score :{fid_score:.3f}\n\n")
            
            
            
            
    GT_files = os.listdir(f'{directory_path}/{args.ID}_GT_novel_face_recon')
    novel_files = os.listdir(f'{directory_path}/{args.ID}_novel_face_recon')
    
    
    GT_files = sorted(GT_files, key=lambda x: int(x.split('.jpg')[0]))
    novel_files = sorted(novel_files, key=lambda x: int(x.split('.jpg')[0]))
    
    
    for GT_name,novel_name in tqdm(zip(GT_files,novel_files), desc="Processing Files"):
        with open(output_path+f'/{args.short_configs}_full_frame_PSNR_SSIM_LPIPS_IDSIM.txt', 'a') as output:
            
            psnr_avg, ssim_avg, lpips_avg = compare_img(directory_path+f'/{args.short_configs}_GT_novel_face_recon/'+novel_name ,directory_path+f'/{args.short_configs}_novel_face_recon/'+novel_name)

            output.write(f"---------------------------------------------\n")
            output.write(f"File: face_{novel_name}\n")
            output.write(f"Processed Result:\n")
            output.write(f"PSNR :{psnr_avg:.3f}\n")
            output.write(f"SSIM :{ssim_avg:.3f}\n")
            output.write(f"LPIPS :{lpips_avg:.3f}\n")
            # output.write(f"ID-SIM Score :{fid_score:.3f}\n\n")
    
    
    GT_files = os.listdir(f'{directory_path}/{args.ID}_GT_novel_body_recon')
    novel_files = os.listdir(f'{directory_path}/{args.ID}_novel_body_recon')
    
    
    GT_files = sorted(GT_files, key=lambda x: int(x.split('.jpg')[0]))
    novel_files = sorted(novel_files, key=lambda x: int(x.split('.jpg')[0]))
    
    
    for GT_name,novel_name in tqdm(zip(GT_files,novel_files), desc="Processing Files"):
        with open(output_path+f'/{args.short_configs}_full_frame_PSNR_SSIM_LPIPS_IDSIM.txt', 'a') as output:
 
            psnr_avg, ssim_avg, lpips_avg = compare_img(directory_path+f'/{args.short_configs}_GT_novel_body_recon/'+novel_name ,directory_path+f'/{args.short_configs}_novel_body_recon/'+novel_name)

            output.write(f"---------------------------------------------\n")
            output.write(f"File: body_{novel_name}\n")
            output.write(f"Processed Result:\n")
            output.write(f"PSNR :{psnr_avg:.3f}\n")
            output.write(f"SSIM :{ssim_avg:.3f}\n")
            output.write(f"LPIPS :{lpips_avg:.3f}\n")

    
    
            
    print(f'{args.short_configs}_PSNR_SSIM_LPIPS_IDSIM.txt is done')
    

