import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

from os import listdir, path

if not path.isfile('./face_detection/detection/sfd/s3fd.pth'):
	raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
							before running this script!')

import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
import audio_process as audio
from hparams import hparams as hp
import face_detection

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument("--data_root", help="path of processed vive3d frames", required=True)
parser.add_argument("--video_path", help="path of video of which the processed vive3d frames", required=True)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)

args = parser.parse_args()

fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
									device='cuda:{}'.format(id)) for id in range(args.ngpu)]

def process_video_file(frames, save_path, args, gpu_id):
    #dirname = os.path.dirname(frames[0])
    fulldir = save_path #path.join(args.preprocessed_root)
    os.makedirs(fulldir, exist_ok=True)

    batches = []
    for i in range(0, len(frames), args.batch_size):
        mini_batch = []
        for k in range(i, i+args.batch_size, 1):
            mini_batch.append(cv2.imread(frames[i]))
        batches.append(mini_batch)

    i = -1
    for fb in tqdm(batches):
        preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))

        for j, f in enumerate(preds):
            i += 1
            if f is None:
                continue

            x1, y1, x2, y2 = f
            #cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), fb[j][y1:y2, x1:x2])
            np.save(path.join(fulldir, '{}.npy'.format(i)), np.array([x1, y1, x2, y2]))


def process_audio_file(vfile, save_path, args):
    template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'

    fulldir = save_path#path.join(args.preprocessed_root)
    #os.makedirs(fulldir, exist_ok=True)

    wavpath = path.join(fulldir, 'audio.wav')

    command = template.format(vfile, wavpath)
    subprocess.call(command, shell=True)
 
def main(args):
    print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))

    jpg_filelist = glob(path.join(args.data_root, '*.jpg'))
    png_filelist = glob(path.join(args.data_root, '*.png'))
    filelist = sorted(jpg_filelist + png_filelist)

    dirname = os.path.basename( os.path.dirname(filelist[0]) )
    save_path = os.path.join(args.preprocessed_root,dirname)
    os.makedirs( save_path, exist_ok=True)
    process_video_file(filelist, save_path, args, gpu_id=0)
    process_audio_file(args.video_path, save_path, args)

if __name__ == '__main__':
	main(args)