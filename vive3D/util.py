import torch
import numpy as np
from torchvision import transforms
import cv2
from tqdm import tqdm
import os

input_image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
def image_to_tensor(image):    
    return input_image_transforms(image)
    
def tensor_to_image(tensor, normalize=True):
    if torch.is_tensor(tensor):
        image = tensor.detach().cpu().numpy().squeeze()
    else:
        image = tensor
        
    if normalize:
        image = 255 * ((image + 1) / 2)
        image = image.clip(0, 255).astype(np.uint8)

    if len(image.shape) == 3:
        image = image.transpose(1, 2, 0)
    elif len(image.shape) == 4:
        image = image.transpose(0, 2, 3, 1)
    return image

def join_videos(input_paths, output_path, cat_horizontal=True, crop_area=None, labels=None, time_range=None): #[xmin, ymin, xmax, ymax]
    captures = []
    heights = []
    widths = []
    lens = []
    for path in input_paths:
        assert os.path.exists(path), print(f'{path} does not exist!')
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        captures.append(cap)
        heights.append(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        widths.append(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        lens.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    
    unique_heights = list(set(heights))
    unique_widths = list(set(widths))
    unique_lens = list(set(lens))
    if cat_horizontal:
        assert len(unique_heights) == 1, print(f'video heights {heights} are incompatible for horizontal concatenation')
    else:
        assert len(unique_widths) == 1, print(f'video widths {widths} are incompatible for vertical concatenation')
    
    fps = round(captures[0].get(cv2.CAP_PROP_FPS)) 
    num_frames = min(unique_lens) if time_range == None else int((time_range[1] - time_range[0])*fps)
    if crop_area:
        crop_H = crop_area[3]-crop_area[1]
        crop_W = crop_area[2]-crop_area[0]
        target_H = int(crop_H) if cat_horizontal else int(len(input_paths)*crop_H)
        target_W = int(len(input_paths)*crop_W) if cat_horizontal else int(crop_W)
    else:
        target_H = int(unique_heights[0]) if cat_horizontal else int(np.sum(heights))
        target_W = int(np.sum(widths)) if cat_horizontal else int(unique_widths[0])
                                              
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(f'creating merged video of {len(captures)} files of size {target_W}x{target_H} at {output_path}')
    videowriter = cv2.VideoWriter(f'{output_path}.mp4', fourcc, fps, (target_W, target_H))
    
    if time_range is not None:
        for cap in captures:
            cap.set(cv2.CAP_PROP_POS_MSEC, int(time_range[0]*1000))
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    for f in tqdm(range(int(num_frames))):
        output_frame = np.zeros((target_H, target_W, 3), dtype=np.uint8)
        
        delta_H, delta_W = 0, 0
        for i, cap in enumerate(captures): 
            ret, frame = cap.read()
            H = heights[i] if not crop_area else crop_H
            W = widths[i] if not crop_area else crop_W
            
            if crop_area:
                frame = frame[crop_area[1]:crop_area[3], crop_area[0]:crop_area[2], :]
            
            if labels is not None:
                cv2.putText(frame, labels[i], (16, H-32), font, H/512, (255,255,255), 2, cv2.LINE_AA)
            
            output_frame[delta_H:delta_H+H, delta_W:delta_W+W, :] = frame
            delta_W += W if cat_horizontal else 0
            delta_H += H if not cat_horizontal else 0
            
        videowriter.write(output_frame)
            
    cv2.destroyAllWindows()
    videowriter.release()
    
    
def plot_angle_visualization(yaws_video, pitches_video, yaws_source=None, pitches_source=None):
    from scipy.ndimage import gaussian_filter1d
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    y = yaws_video
    p = pitches_video
    if torch.is_tensor(yaws_video):
        y = yaws_video.cpu().numpy()
    if torch.is_tensor(pitches_video):
        p = pitches_video.cpu().numpy()
    smooth_yaw = gaussian_filter1d(y, 1, axis=0, mode='nearest')
    smooth_pitch = gaussian_filter1d(p, 1, axis=0, mode='nearest')
    if yaws_source is not None:
        smooth_yaw_source = gaussian_filter1d(yaws_source, 1, axis=0, mode='nearest')
    if pitches_source is not None:
        smooth_pitch_source = gaussian_filter1d(pitches_source, 1, axis=0, mode='nearest')
    vis = []
    for i in tqdm(range(len(pitches_video))):
        fig = plt.figure(figsize=(15, 15))
        ax = plt.gca()
        ax.set_axisbelow(True)
        ax.xaxis.grid(color=(.8, .8, .8, .7), linestyle='dotted')
        ax.yaxis.grid(color=(.8, .8, .8, .7), linestyle='dotted')
        ax.xaxis.set_ticklabels([]) 
        ax.yaxis.set_ticklabels([]) 
        ax.set_xlim(-.3, .3) 
        ax.set_ylim(-.3, .3) 
        ax.axvline(x=0, color='white', linewidth=3)
        ax.axhline(y=0, color='white', linewidth=3)
        ax.set_facecolor((0.122, 0.122, 0.122))
    
        if yaws_source is not None and pitches_source is not None:
            ii = i
            di = 0
            while di < 8:
                if ii >= 0:
                    plt.scatter(smooth_yaw_source[ii], smooth_pitch_source[ii], c=[(0.87, 0.61, 0.00, (8-di)/8)], s=400)
                di+=1
                ii-=2
        ii = i
        di = 0
        while di < 8:
            if ii >= 0:
                plt.scatter(smooth_yaw[ii], smooth_pitch[ii], c=[(0.25, 0.78, 1.00, (8-di)/8)], s=400)
            di+=1
            ii-=2

        plt.tight_layout(pad=0)   
        ax.margins(0,0)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))
        vis.append(im)
        plt.close(fig)
    return vis
    