import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import click
import torchvision
import sys
sys.path.append('./Wav2Lip')
import face_detection
import os
import numpy as np
import cv2

@click.command()
@click.option('-r', '--root_dir', type=str, help='Path to source video', required=True)
@click.option('--id', type=str, help='Path to source video', required=True)
@click.option('-i', '--source_images', type=str, help='Path to source video', required=True)
@click.option('-o', '--savepoint_path', type=str, help='Savepoint directory', default='wav2lip_bbox')

def main(**config):
    _main(**config)

def _main(root_dir,
          id,
          source_images,
          savepoint_path):
    
    images_dir = sorted(glob(os.path.join(root_dir, id, source_images, '*.png')))
    save_root = os.path.join(root_dir, id, savepoint_path)
    os.makedirs(save_root, exist_ok=True)
    fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False)

    for image_dir in tqdm(images_dir, desc='preprocessing wav2lip bounding box...'):
        name = image_dir.split('/')[-1].split('.')[0]
        save_dir = os.path.join(save_root, f'{name}.npy')
        image = cv2.imread(image_dir)
        results = fa.get_detections_for_image(image)
        np.save(save_dir, results)
        

if __name__ == '__main__':
    main()