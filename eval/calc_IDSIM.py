import argparse 
import time
import numpy as np
import os, shutil
import json
import sys
from PIL import Image
import multiprocessing as mp
import math
import torch
import torchvision.transforms as trans
import random

sys.path.append(".")
sys.path.append("..")

from metrics.mtcnn.mtcnn import MTCNN
from metrics.encoders.model_irse import IR_101
CIRCULAR_FACE_PATH = "metrics/models/CurricularFace_Backbone.pth" #model_paths['curricular_face']


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]

blockPrint()
facenet = IR_101(input_size=112)
facenet.load_state_dict(torch.load(CIRCULAR_FACE_PATH))
facenet.cuda()
facenet.eval()
mtcnn = MTCNN()
tensor_transform = trans.ToTensor()
id_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
enablePrint()

def id(path):
    im = Image.open(path)
    im, _ = mtcnn.align(im)
    if im is None:
        print("Image Not Found:", path)
        return None
    im_id = facenet(id_transform(im).unsqueeze(0).cuda())[0]
    im_id = im_id.cpu()
    return im_id

def calculate_idsim(ori_video_path, video_path, video_id):
    # Load video
    if os.path.exists(f"FRAMES/{video_id}"):
        shutil.rmtree(f"FRAMES/{video_id}")
    os.makedirs(f"FRAMES/{video_id}", exist_ok=True)
    ffmpeg_command = f"ffmpeg -loglevel quiet -i {ori_video_path} FRAMES/{video_id}/video1_%06d.png"
    os.system(ffmpeg_command)
    
    
    if os.path.exists(f"FRAMES/{video_id}_temp"):
        shutil.rmtree(f"FRAMES/{video_id}_temp")
    os.makedirs(f"FRAMES/{video_id}_temp", exist_ok=True)
    ffmpeg_command = f"ffmpeg -loglevel quiet -i {video_path} FRAMES/{video_id}_temp/video1_%06d.png"
    os.system(ffmpeg_command)
    
    gt_id, res_id = [], []
    
    for gt_path in sorted(os.listdir(os.path.join("FRAMES", video_id))):
        # print(gt_path)
        if gt_path.split(".")[-1] != "png":
            if gt_path.split(".")[-1] != "jpg":
                continue
        with torch.no_grad():
            gt_id.append(id(os.path.join("FRAMES", video_id, gt_path)).cpu())
            
    for res_path in sorted(os.listdir(os.path.join("FRAMES", f"{video_id}_temp"))):
        # print(res_path)
        if res_path.split(".")[-1] != "png":
            if res_path.split(".")[-1] != "jpg":
                continue
        with torch.no_grad():
            res_id.append(id(os.path.join("FRAMES",f"{video_id}_temp", res_path)).cpu())
            
    if len(gt_id) != len(res_id):
        shorter_length = min(len(gt_id), len(res_id))
        gt_id = gt_id[:shorter_length]
        res_id = res_id[:shorter_length]
    
    id_sim = []
    for gt_i, res_i in zip(gt_id, res_id):
        score = float(gt_i.dot(res_i))
        id_sim.append(score)
    id_score = np.mean(id_sim)
    
    shutil.rmtree(f"FRAMES/{video_id}_temp")
    
    return id_score

def calculate_idsim_frame(ori_video_path, video_path, video_id):
    # Load video
    if os.path.exists(f"FRAMES/{video_id}"):
        shutil.rmtree(f"FRAMES/{video_id}")
    os.makedirs(f"FRAMES/{video_id}", exist_ok=True)
    ffmpeg_command = f"ffmpeg -loglevel quiet -i {ori_video_path} FRAMES/{video_id}/video1_%06d.png"
    os.system(ffmpeg_command)
    
    
    if os.path.exists(f"FRAMES/{video_id}_temp"):
        shutil.rmtree(f"FRAMES/{video_id}_temp")
    os.makedirs(f"FRAMES/{video_id}_temp", exist_ok=True)
    ffmpeg_command = f"ffmpeg -loglevel quiet -i {video_path} FRAMES/{video_id}_temp/video1_%06d.png"
    os.system(ffmpeg_command)
    
    gt_id, res_id = [], []
    
    for gt_path in sorted(os.listdir(os.path.join("FRAMES", video_id))):
        # print(gt_path)
        if gt_path.split(".")[-1] != "png":
            if gt_path.split(".")[-1] != "jpg":
                continue
        with torch.no_grad():
            gt_id.append(id(os.path.join("FRAMES", video_id, gt_path)).cpu())
            
    for res_path in sorted(os.listdir(os.path.join("FRAMES", f"{video_id}_temp"))):
        # print(res_path)
        if res_path.split(".")[-1] != "png":
            if res_path.split(".")[-1] != "jpg":
                continue
        with torch.no_grad():
            res_id.append(id(os.path.join("FRAMES",f"{video_id}_temp", res_path)).cpu())
            
    if len(gt_id) != len(res_id):
        shorter_length = min(len(gt_id), len(res_id))
        gt_id = gt_id[:shorter_length]
        res_id = res_id[:shorter_length]
    
    id_sim = []
    for gt_i, res_i in zip(gt_id, res_id):
        score = float(gt_i.dot(res_i))
        id_sim.append(score)
    id_score = np.mean(id_sim)
    
    shutil.rmtree(f"FRAMES/{video_id}_temp")
    
    return id_score

        
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FRAMES score between two videos")
    parser.add_argument("video_path", help="Path to the first video")
    args = parser.parse_args()
    
    if True:
        video_id = os.path.basename(args.video_path).split(".")[0]
    else:
        raise NotImplementedError("Not implemented yet")
        args.video_id = args.force_video_id
    blockPrint()
    fid_score = calculate_idsim(args.video_path, video_id)
    enablePrint()
    print(f"ID-SIM Score: {fid_score}")
