from vive3D.segmenter import Segmenter
from vive3D.landmark_detector import LandmarkDetector
from torchvision import transforms
import torch
import math
import numpy as np
from tqdm import tqdm 
from kornia.geometry import warp_affine
from vive3D.util import *
import cv2
from preprocess_utils.vive3d_cropping import remove_small_mask

class Aligner:
    def __init__(self, landmark_detector=None, segmenter=None, device='cuda'):
        self.device = device
        self.landmark_detector = LandmarkDetector(device) if landmark_detector is None else landmark_detector
        self.segmenter = Segmenter(device=device) if segmenter is None else segmenter
    
    def get_alignment_matrix(self, align_landmarks, target_landmarks):
        points1 = np.matrix(align_landmarks.astype(np.float64))
        points2 = np.matrix(target_landmarks.astype(np.float64))
        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2
        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T
        M = np.vstack([[np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)) ]])
        return M
    
    def get_rectangle_from_landmarks(self, landmarks, scale=1.0):
        lm_chin = landmarks[0: 17]  # left-right
        lm_eyebrow_left = landmarks[17: 22]  # left-right
        lm_eyebrow_right = landmarks[22: 27]  # left-right
        lm_nose = landmarks[27: 31]  # top-down
        lm_nostrils = landmarks[31: 36]  # top-down
        lm_eye_left = landmarks[36: 42]  # left-clockwise
        lm_eye_right = landmarks[42: 48]  # left-clockwise
        lm_mouth_outer = landmarks[48: 60]  # left-clockwise
        lm_mouth_inner = landmarks[60: 68]  # left-clockwise
        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg
        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)

        x *= scale
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        return c, x, y

    def align_face_images(self, image, align_landmarks, target_landmarks, target_size=256, padding='replicate', return_matrix=False):
        def warp_cv2(image, M, target_size, padding='reflect'):
            dtype = image.dtype
            if padding == 'reflect':
                borderMode=cv2.BORDER_REFLECT_101
            elif padding == 'replicate':
                borderMode =cv2.BORDER_REPLICATE 
            else:
                borderMode = cv2.BORDER_CONSTANT
            return np.clip(cv2.warpAffine(image.astype(np.float32), M, (target_size[1], target_size[0]), flags=cv2.INTER_LANCZOS4, borderMode=borderMode), 0, 255).astype(dtype)
        
        M = self.get_alignment_matrix(align_landmarks, target_landmarks)[0]

        if type(image) == list:
            output = [ warp_cv2(i, M, target_size, padding) for i in image ]
        else:
            output = warp_cv2(image, M, target_size, padding)

        return output
    
    # rigid alignment of landmarks
    def align_face_tensors(self, tensor, align_landmarks, target_landmarks, target_size=256, padding='zeros'):
        M = self.get_alignment_matrix(align_landmarks, target_landmarks)
        
        if type(tensor) == list: #do alignment for multiple tensors
            output = [ warp_affine(t.to(self.device).type(torch.float32), torch.tensor(M).to(self.device).type(torch.float32), dsize=target_size, padding_mode=padding) for t in tensor ]
        else:
            output: torch.tensor = warp_affine(tensor.to(self.device).type(torch.float32), torch.tensor(M).to(self.device).type(torch.float32), dsize=target_size, padding_mode=padding)
        return output
    
    def get_face_tensors_from_frames(self, frames, reference_face, get_all=False, smooth_landmarks=False, smooth_sigma=2, input_image_transforms=None, return_foreground_images=False, name=None, black_bg=False, return_matrix=False):
        if input_image_transforms==None:
            input_image_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        faces = []
        segmentations = []
        foreground_images = []
        matrices = []

        reference_face_landmarks = self.landmark_detector.get_landmarks(tensor_to_image(reference_face), get_all=get_all)

        transparency = 0.7*torch.tensor(self.checkerboard(reference_face.shape[-2:])[np.newaxis, ...].repeat(3, axis=0))
        
        landmarks = []
        #landmark_rects = []
        print(f'extracting landmarks...')
        for i, original_image in tqdm(enumerate(frames), total=len(frames)):
            # estimate landmarks for current image
            landmark = self.landmark_detector.get_landmarks(original_image, get_all=get_all)
            if i > 0 and np.array(landmark).shape != np.array(landmarks[-1]).shape:
                print(f'landmarks at frame {i} could not be successfully detected (shape {np.array(landmark).shape} vs {np.array(landmarks[-1]).shape}), using landmarks from previous frame...')
                landmark = landmarks[-1]
            
            landmarks.append(landmark)
        landmarks = np.stack(landmarks)
        
        if smooth_landmarks:
            print(f'smoothing landmarks...')
            from scipy.ndimage import gaussian_filter1d
            landmarks = gaussian_filter1d(landmarks, smooth_sigma, axis=0, mode='nearest').astype(np.int32)
        
        print(f'processing alignment and segmentation...')    
        for idx, original_image in tqdm(enumerate(frames), total=len(frames)):    
            # convert image to tensor
            warped_image = self.align_face_images(original_image, landmarks[idx], reference_face_landmarks, target_size=reference_face.shape[-2:])
            transformed_image = input_image_transforms(warped_image).unsqueeze(0).to(self.device)

            # face_foreground = self.segmenter.get_face_and_hair_BiSeNet_naive(transformed_image, background_classes=[0]).cpu() #original
            face_foreground = self.segmenter.get_face_and_hair_BiSeNet_naive(transformed_image, erosion_num=3).cpu() #face
            face_foreground = remove_small_mask(face_foreground)
            
            if black_bg:
                test = -torch.ones_like(face_foreground)
                foreground_image_black_bg = torch.where(face_foreground==0, test, (transformed_image.cpu()))
            face_background = ~(face_foreground.to(torch.bool))

            processed_image = (transformed_image.cpu())

            faces.append(processed_image)
            segmentations.append(face_foreground)
            if return_foreground_images:
                # foreground_images.append(face_foreground.cpu()*transformed_image.cpu()+face_background.cpu()*transparency) #체크무늬
                if black_bg:
                    foreground_images.append(foreground_image_black_bg)
                else:
                    foreground_images.append(face_foreground.cpu()*transformed_image.cpu())

            if return_matrix:
                matrices.append(M) # return cv2 warpaffine transform matrix

        if return_matrix:
            if return_foreground_images:
                foreground_images = torch.cat(foreground_images)
                return faces, segmentations, foreground_images, landmarks, matrices
            else:
                return faces, segmentations, landmarks, matrices
        else:
            if return_foreground_images:
                foreground_images = torch.cat(foreground_images)
                return faces, segmentations, foreground_images, landmarks
            else:
                return faces, segmentations, landmarks
        
    def checkerboard(self, size, blocksize=16):
        repeat_w = math.ceil(size[0]/(2*blocksize))
        repeat_h = math.ceil(size[1]/(2*blocksize))
        grid = np.kron([[1, 0] * repeat_w, [0, 1] * repeat_w] * repeat_h, np.ones((blocksize, blocksize)))
        return grid[:size[1], :size[0]]
    
    
    def visualize_face_tensors_from_frames(self, frames, reference_face, get_all=False, smooth_landmarks=False, smooth_sigma=2, input_image_transforms=None, return_foreground_images=False):
        if input_image_transforms==None:
            input_image_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        faces = []
        segmentations = []
        foreground_images = []
        
        reference_face_landmarks = self.landmark_detector.get_landmarks(tensor_to_image(reference_face), get_all=get_all)
        #reference_rect = self.get_rectangle_from_landmarks(reference_face_landmarks)

        transparency = 0.7*torch.tensor(self.checkerboard(reference_face.shape[-2:])[np.newaxis, ...].repeat(3, axis=0))
        
        landmarks = []
        #landmark_rects = []
        print(f'extracting landmarks...')
        for i, original_image in tqdm(enumerate(frames), total=len(frames)):
            # estimate landmarks for current image
            landmark = self.landmark_detector.get_landmarks(original_image, get_all=get_all)
            if i > 0 and np.array(landmark).shape != np.array(landmarks[-1]).shape:
                print(f'landmarks at frame {i} could not be successfully detected (shape {np.array(landmark).shape} vs {np.array(landmarks[-1]).shape}), using landmarks from previous frame...')
                landmark = landmarks[-1]

            landmarks.append(landmark)
        landmarks = np.stack(landmarks)
        
        if smooth_landmarks:
            print(f'smoothing landmarks...')
            from scipy.ndimage import gaussian_filter1d
            landmarks = gaussian_filter1d(landmarks, smooth_sigma, axis=0, mode='nearest').astype(np.int32)
        
        print(f'processing alignment and segmentation...')    
        for idx, original_image in tqdm(enumerate(frames), total=len(frames)):    
            warped_image = self.align_face_images(original_image, landmarks[idx], reference_face_landmarks, target_size=reference_face.shape[-2:])
            
            M = self.get_alignment_matrix(reference_face_landmarks, landmarks[idx])[0]
            #M = self.aligner.get_alignment_matrix(reference_face_landmarks, landmarks)
            M_rot = M[:, :2] #eliminate translation
            points = [(0, 0), (512, 0), (512, 512), (0, 512)]
            corner_points_frame = np.matmul(M_rot, points)

            transformed_image = input_image_transforms(warped_image).unsqueeze(0).to(self.device)

            face_foreground = self.segmenter.get_foreground_BiSeNet(transformed_image).cpu()
            face_background = ~(face_foreground.to(torch.bool))

            processed_image = transformed_image.cpu()

            faces.append(processed_image)
            segmentations.append(face_foreground)
            if return_foreground_images:
                foreground_images.append(face_foreground.cpu()*transformed_image.cpu()+face_background.cpu()*transparency)

        face_tensors = torch.cat(faces)
        segmentation_tensors = torch.cat(segmentations)
        
        if return_foreground_images:
            foreground_images = torch.cat(foreground_images)
            return face_tensors, segmentation_tensors, foreground_images, landmarks
        else:
            return face_tensors, segmentation_tensors, landmarks

    def get_face_tensors_from_adnerf_frames(self, 
                                            source_frames,
                                            target_frames,
                                            reference_face, 
                                            get_all=False, 
                                            smooth_landmarks=False,
                                            smooth_sigma=2,
                                            input_image_transforms=None, 
                                            traintest_split_rate=10/11):
        if input_image_transforms==None:
            input_image_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        faces = []

        reference_face_landmarks = self.landmark_detector.get_landmarks(tensor_to_image(reference_face), get_all=get_all)
        
        landmarks = []
        print(f'extracting landmarks...')
        for i, original_image in tqdm(enumerate(source_frames), total=len(source_frames)):
            # estimate landmarks for current image
            landmark = self.landmark_detector.get_landmarks(original_image, get_all=get_all)
            if i > 0 and np.array(landmark).shape != np.array(landmarks[-1]).shape:
                print(f'landmarks at frame {i} could not be successfully detected (shape {np.array(landmark).shape} vs {np.array(landmarks[-1]).shape}), using landmarks from previous frame...')
                landmark = landmarks[-1]
 
            landmarks.append(landmark)
        landmarks = np.stack(landmarks)
        
        if smooth_landmarks:
            print(f'smoothing landmarks...')
            from scipy.ndimage import gaussian_filter1d
            landmarks = gaussian_filter1d(landmarks, smooth_sigma, axis=0, mode='nearest').astype(np.int32)
        
        print(f'adnerf to vive3d : processing alignment and segmentation...')  
        print(f'starting from frame index {val_start_idx} : check for the frame alignment')   
        val_start_idx = int(traintest_split_rate*len(source_frames))
        for idx, adnerf_image in tqdm(enumerate(target_frames), total=len(source_frames)):    
            warped_image = self.align_face_images(adnerf_image, landmarks[idx+val_start_idx], reference_face_landmarks, target_size=reference_face.shape[-2:])
            transformed_image = input_image_transforms(warped_image).unsqueeze(0).to(self.device)

            processed_image = transformed_image.cpu()

            faces.append(processed_image)

        face_tensors = torch.cat(faces)
        

        return face_tensors, landmarks