####################################################################################################################################
# VIDEO EXTRACTOR
####################################################################################################################################
import cv2
import os
from tqdm import tqdm
import numpy as np
import imageio

class VideoTool:
    def __init__(self, video_source_path, frames_path=None, crop_area=None, set_fps=None):
        self.video_source_path = video_source_path
        self.video_title = video_source_path.split('/')[-1].split('.')[0]
        self.crop_area = None
        
        if frames_path:
            self.frames_path = frames_path
            os.makedirs(frames_path, exist_ok=True)
        
        self.digits = 6
        self.frame_dtype = 'png'
        
        assert os.path.exists(video_source_path), print(f'{video_source_path} does not exist!')
        
        self.vidcap = cv2.VideoCapture(video_source_path, cv2.CAP_FFMPEG)
        self.fps = round(self.vidcap.get(cv2.CAP_PROP_FPS))
        if not set_fps == None:
            self.fps = set_fps
        self.frame_dims = self.get_frame_size()
        self.crop_area = crop_area
    
    def set_crop_area(self, area=[0, 0, 0, 0]):
        self.crop_area = area
    
    def crop(self, image):
        xmin, xmax, ymin, ymax = self.crop_area
        xmin = xmin if xmin >= 0 else 0
        ymin = ymin if ymin >= 0 else 0
        xmax = xmax if xmax >= 0 else self.frame_dims[1]
        ymax = ymax if ymax >= 0 else self.frame_dims[0]
        return image[ymin:ymax, xmin:xmax, :]
        
    def get_fps(self):
        return self.fps
    
    def get_video_title(self):
        return str(self.video_title)
    
    def get_frame_size(self):
        f, _ = self.extract_single_frame_from_video(0)
        return f.shape
    
    # TODO read from folder
    def get_range(self, start_sec, end_sec, input_image_transforms=None):
        frame_count = 0
        video_frames = []
        while frame_count < self.fps*(end_sec - start_sec):
            success, image = self.vidcap.read()
            assert success, f'could not read video {self.video_source_path} between seconds {start_sec} and {end_sec}'
            current_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if input_image_transforms:
                # convert image to tensor
                current_frame = input_image_transforms(current_frame).unsqueeze(0)

            video_frames.append(current_frame)
        return video_frames
    
    def convert_from_dtype(self, image, dtype='png'):
        if dtype == 'png':
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif dtype == '+pg':
            return cv2.cvtColor(image, cv2.COLOR_BRG2RGB)
        else:
            print(f'unkown dtype {dtype}')
    
    def set_position_to_sec(self, sec):
        self.vidcap.set(cv2.CAP_PROP_POS_MSEC, int(sec*1000))
        
    def extract_single_frame_from_video(self, sec, read_from_storage=False, store_frame=False, resize=0):
        if read_from_storage:
            frame_path = f'{self.frames_path}/{self.video_title}_{int(sec*1000):0^{self.digits}d}.{self.frame_dtype}'
            if os.path.exists(frame_path):
                image = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                if self.crop_area is not None:
                    image = self.crop(image)
                return self.convert_from_dtype(image, self.frame_dtype)
                
        success, image = self.vidcap.read()
        if not success:
            return None, False
        
        if self.crop_area is not None:
            image = self.crop(image)
        
        if resize > 0:
            image = cv2.resize(image, None, fx=resize, fy=resize, interpolation=cv2.INTER_CUBIC)
            
        if store_frame:
            if self.frame_dtype == 'png':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            cv2.imwrite(frame_path, image)
            
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), success

    def extract_frames_from_video(self, start_sec, end_sec=0, step_size=0, read_from_storage=False, store_frame=False, resize=0):
        frames = []
        frame_by_frame = step_size == 0
        
        step = 1/self.fps if frame_by_frame else step_size

        self.vidcap.set(cv2.CAP_PROP_POS_MSEC, int(start_sec*1000))
        
        end_sec = 1000000 if end_sec == 0 else end_sec
        
        frame_pos = start_sec
        while frame_pos < end_sec-step:
            frame, success = self.extract_single_frame_from_video(frame_pos, store_frame=store_frame, resize=resize)
            if not success:
                break
                
            frames.append(frame)
            
            frame_pos += step
            if not frame_by_frame: #update video location
                self.vidcap.set(cv2.CAP_PROP_POS_MSEC, int(frame_pos*1000))   
            
        print(f'extracted {len(frames)} frames from {self.video_source_path}')
        return frames
    
    def write_frames_to_video(self, frames, path, codec='mp4v', fps=25, use_imageio=False):
        if use_imageio:
            imageio.mimwrite(f'{path}.mp4', frames, fps=fps, quality=8, output_params=['-vf', f'fps={fps}'])
        else:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video = cv2.VideoWriter(f'{path}.mp4', fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))

            for frame in frames:
                video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            #cv2.destroyAllWindows()
            video.release()

    