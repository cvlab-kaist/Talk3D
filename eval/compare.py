import cv2
import os, sys
import numpy as np
import torch
import lpips
import json
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from segmenter import Segmenter
from calc_IDSIM import calculate_idsim
from calc_FID import calculate_fid_score
from action_unit_error import compute_aue_from_csv

original_stdout = sys.stdout
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    # sys.stdout = sys.__stdout__
    sys.stdout=original_stdout
    
def img2mse(x, y): return torch.mean((x - y) ** 2)


def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))



def compare_videos(video1_path, video2_path, args):
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
    frame1 = cv2.imread(img1_path)
    frame2 = cv2.imread(img2_path)

    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='alex')

    if frame1.shape != (512, 512, 3):
        frame1 = cv2.resize(frame1, (512, 512))
    if frame2.shape != (512, 512, 3):
        frame2 = cv2.resize(frame2, (512, 512))
        
        # Compute PSNR
    psnr = peak_signal_noise_ratio(frame1, frame2)

        # Compute SSIM
    ssim = structural_similarity(frame1, frame2, multichannel=True,channel_axis = -1)
        # ssim_sum += ssim

        # Convert frames to tensors
    frame1_tensor = torch.from_numpy(frame1).unsqueeze(0).permute(0, 3, 1, 2).float()
    frame2_tensor = torch.from_numpy(frame2).unsqueeze(0).permute(0, 3, 1, 2).float()

        # Compute LPIPS
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--ID', type=str, required=True)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model")
    parser.add_argument('--short_configs', type=str)
    parser.add_argument('--input_video', type=str)
    parser.add_argument('--gt_video', type=str)
    parser.add_argument('--eval_type', default='all', type=str, help='select within : [all, ID, FID, Sync, LMD, AUE] all means evaluate all')
    parser.add_argument('--inf_type', default='novel', type=str)
    parser.add_argument('--eval_recon', action='store_true', help='Evaluation on reconstruction metrics: THIS ONLY CALCULATE WHOLE IMAGE AREA')
    
    args = parser.parse_args()
    
    
    
    import os
    from tqdm import tqdm
    import cv2
    from moviepy.editor import VideoFileClip
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

            
            
    # for file_name in tqdm(novel_face_recon_mp4_files, desc="Processing Files"):
    input_video = args.input_video
    gt_video_recon = os.path.join(args.data_dir, args.ID, 'GT_video_recon.mp4')
    gt_video_ood = os.path.join(args.data_dir, args.ID, 'GT_video_OOD.mp4') if args.inf_type != 'novel' else gt_video_recon
    print(f'Evaluating on {input_video}...')
    psnr_avg = ssim_avg = lpips_avg = idsim_score = fid_score = sync_score = lmd_score = aue_score = -1
    output_dict = {}
    with open(f'metric.txt', 'a') as output:
        
        #Reconstruction
        if args.eval_recon and args.eval_type == 'all':
            psnr_avg, ssim_avg, lpips_avg = compare_videos(gt_video_recon,input_video, args)
        
        #ID-SIM
        if args.eval_type == 'all' or args.eval_type == 'ID':
            idsim_score = calculate_idsim(gt_video_recon, input_video, args.ID)
        
        #FID
        if args.eval_type == 'all' or args.eval_type == 'FID':
            fid_score = calculate_fid_score(input_video, args.data_dir, args.ID)
        
        #Sync
        if args.eval_type == 'all' or args.eval_type == 'Sync':
            cmd = f'sh syncnet_python/calculate_scores_single_video.sh {input_video}'
            os.system(cmd)
            with open('syncnet_python/all_scores_tmp.txt', 'r') as f:
                sync_score = f.readline().split(' ')[0]
        
        #LMD
        if args.eval_type == 'all' or args.eval_type == 'LMD':
            cmd = f'sh lmd/lmd_eval.sh {input_video} {gt_video_ood}' 
            os.system(cmd)
            with open('lmd/lmd_tmp.txt', 'r') as f:
                lmd_score = f.readline()
            cmd = 'rm lmd/lmd_tmp.txt'
            os.system(cmd)
        
        #AUE
        if args.eval_type == 'all' or args.eval_type == 'AUE':
            #calc aue of 2 vids
            input_video_name = input_video.split('/')[-1]
            gt_video_name = gt_video_ood.split('/')[-1]
            cmd = f'sh au_detection.sh {input_video} {input_video_name} {os.getcwd()} au1.csv'
            os.system(cmd)
            cmd = f'sh au_detection.sh {gt_video_ood} {gt_video_name} {os.getcwd()} au2.csv'
            os.system(cmd)
            aue_score = compute_aue_from_csv('data_openface/au1.csv', 'data_openface/au2.csv')
            cmd = 'rm -rf data_openface'
            os.system(cmd)
        
        
        output_dict['PSNR'] = f'{psnr_avg:.3f}'
        output_dict['SSIM'] = f'{ssim_avg:.3f}'
        output_dict['LPIPS'] = f'{lpips_avg:.3f}'
        output_dict['ID-SIM'] = f'{idsim_score:.3f}'
        output_dict['FID'] = f'{fid_score:.3f}'
        output_dict['SYNC'] = f'{sync_score:.3f}'
        output_dict['LMD'] = f'{lmd_score:.3f}'
        output_dict['AUE'] = f'{aue_score:.3f}'
        
        # output.write(f"---------------------------------------------\n")
        # output.write(f"File: {input_video}\n")
        # output.write(f"Processed Result:\n")
        # output.write(f"PSNR :{psnr_avg:.3f}\n")
        # output.write(f"SSIM :{ssim_avg:.3f}\n")
        # output.write(f"LPIPS :{lpips_avg:.3f}\n")
        # output.write(f"ID-SIM :{idsim_score:.3f}\n")
        # output.write(f"FID :{fid_score:.3f}\n")
        # output.write(f"SYNC :{sync_score:.3f}\n")
        # output.write(f"LMD :{lmd_score:.3f}\n")
        # output.write(f"AUE :{aue_score:.3f}\n\n")
        
    with open('metric.json', 'w') as f : 
	    json.dump(output_dict, f, indent=4)
        
            
        
    

