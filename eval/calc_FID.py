import os, shutil
import sys
from cleanfid import fid
import argparse

# Function to calculate the FID score
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def calculate_fid_score(video_path, data_root, video_id):
    # Load video
    os.makedirs("FID/temp", exist_ok=True)
    cmd = f"ffmpeg -i {video_path} FID/temp/video1_%06d.png"
    os.system(cmd)

    score = fid.compute_fid("FID/temp", f"{data_root}/{video_id}/image")
    cmd = f"rm -r FID/temp"
    os.system(cmd)
    return score