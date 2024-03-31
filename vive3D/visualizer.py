import cv2
import matplotlib.pyplot as plt
from vive3D.util import *
import math
import numpy as np
import torch
from PIL import Image

class Visualizer:    
    def plot(image, size='L', axes=False, title=None, dpi=100):
        if size=='L':
            W = 20
        elif size== 'M':
            W = 10
        else:
            W = 5
            
        h_im, w_im = image.shape[:2]
        H = int((h_im * W) / w_im)
        fig = plt.figure(figsize=(W, H), dpi=dpi)
        plt.imshow(image)
        if title:
            plt.title(title)
        if not axes:
            plt.axis('off')    
            
    def convert_to_grid(image_list, grid_width=5, labels=None, put_indices=False, color='black', scale_factor=1):
        image_list = tensor_to_image(image_list) if type(image_list) is torch.Tensor else image_list

        grid_height = math.ceil(len(image_list) / grid_width)
        print(f'creating grid with {grid_width} x {grid_height} cells for list of len {len(image_list)}')
        width = image_list[0].shape[1]//scale_factor
        height = image_list[0].shape[0]//scale_factor
        
        grid_view = 255*np.ones((grid_height*height, grid_width*width, 3), dtype=image_list[0].dtype)
        
        for idx, image in enumerate(image_list):
            x = idx % grid_width
            y = idx // grid_width
            grid_view[y*height:(y+1)*height, x*width:(x+1)*width, :] = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
            if labels is not None:
                col = (0, 0, 0) if color=='black' else (255, 255, 255)
                grid_view = cv2.putText(grid_view, f'{labels[idx]}', (20+x*width, 50+y*height), cv2.FONT_HERSHEY_SIMPLEX, 1.25, col, 2, cv2.LINE_AA)
            elif put_indices:
                col = (0, 0, 0) if color=='black' else (255, 255, 255)
                grid_view = cv2.putText(grid_view, f'{idx}', (20+x*width, 50+y*height), cv2.FONT_HERSHEY_SIMPLEX, 1.25, col, 2, cv2.LINE_AA)
        return grid_view

    def show_tensors(tensors, gh=None, gw=None, ax=None, normalize=True, title=None):
        if gh==None or gw==None:
            gw = len(tensors)
            gh = 1
        images = tensors.clone() if torch.is_tensor(tensors) else tensors
        if len(images.shape) == 3:
            images = images.unsqueeze(1)

        images = tensor_to_image(images, normalize)

        tiled_images = images

        _N, H, W, C = tiled_images.shape
        tiled_images = tiled_images.reshape(gh, gw, H, W, C)
        tiled_images = tiled_images.transpose(0, 2, 1, 3, 4) #-> gh, H, gw, W, C

        tiled_images = tiled_images.reshape(gh * H, gw * W, C)

        if ax is None:
            Visualizer.plot(tiled_images, 'L', title=title)
        else:
            ax.cla()
            plt.imshow(tiled_images)
            if title:
                plt.title(title)
            plt.axis('off')

    def show_tensor(tensor, ax=None, normalize=True, text=None, color='black', size=22):
        if len(tensor.shape)==2:
            tensor = tensor.unsqueeze(0) #.repeat(3, 1, 1)
        elif len(tensor.shape)==4:
            tensor = tensor[0]

        if tensor.shape[0] == 1: #single channel image
            tensor = tensor.repeat(3, 1, 1)
        
        image = tensor_to_image(tensor, normalize)
        
        if ax is None:
            Visualizer.plot(image, 'L', title=text)
        else:
            ax.cla()
            ax.imshow(image)
            ax.axis('off')
            if text:
                ax.text(10, 300, text, va='top', size=size, color=color, wrap=True)
    
    def save_tensor_to_file(tensor, filename='output', dtype='png', start=0, out_folder='', target_size=None):    
        image = tensor_to_image(tensor) if type(tensor) is torch.Tensor else tensor
        
        num_images = image.shape[0] if len(image.shape) == 4 else 1 
        for n in range(num_images):
            im = Image.fromarray(image[n].astype(np.uint8)) if num_images > 1 else Image.fromarray(image.astype(np.uint8))
            if target_size is not None:
                im = im.resize(target_size, Image.BICUBIC)
            path = f'{out_folder}/{filename}_{n+start:03d}' if num_images > 1 else f'{out_folder}/{filename}'
            
            im.save(f'{path}.{dtype}')
    
    def plot_flow_field(image, flow_field, stride=4, scale=1):
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        
        H, W, _ = image.shape
        #only plot a quiver every step pixels 
        x, y = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
        
        fig = Figure(figsize=(W/100, H/100))
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        ax.set_axis_off()
        fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        fig.tight_layout(pad=0)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.imshow(image)

        fx = flow_field[::stride, ::stride, 0]
        fy = flow_field[::stride, ::stride, 1]
        nz = np.logical_and(fx != 0, fy != 0) #only plot nonzero quivers
        ax.quiver(x[nz], y[nz], fx[nz], fy[nz], np.arctan2(fy[nz], fx[nz]), alpha=0.7, cmap='hsv', scale_units='xy', scale=scale, clim=(-np.pi, np.pi))

        canvas.draw()  # draw the canvas, cache the renderer

        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(H, W, 3)
        image = image[::-1, ...] #invert y, image is upside-down
        return image
   
    def overlay_images(image_canvas, image_overlay, alpha=0.7):
        mask_C = np.zeros(image_overlay.shape, dtype=np.uint8)
        for c in range(3):
            mask_C[:, :, c] = image_overlay[:, :, c] == 127
        
        mask = np.repeat(np.logical_not(np.all(mask_C, axis=-1))[..., np.newaxis], 3, axis=-1)
        
        image_out = image_canvas.copy()
        image_out[mask] = image_out[mask] * (1-alpha) + image_overlay[mask] * alpha
        
        return image_out
        
Visualizer.overlay_images = staticmethod(Visualizer.overlay_images)  
Visualizer.convert_to_grid = staticmethod(Visualizer.convert_to_grid)  
Visualizer.show_tensor = staticmethod(Visualizer.show_tensor)    
Visualizer.show_tensors = staticmethod(Visualizer.show_tensors)    
Visualizer.plot = staticmethod(Visualizer.plot)       
Visualizer.plot_flow_field = staticmethod(Visualizer.plot_flow_field)    
Visualizer.save_tensor_to_file = staticmethod(Visualizer.save_tensor_to_file)    