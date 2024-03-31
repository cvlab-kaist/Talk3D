import numpy as np

class LandmarkDetector:
    def __init__(self, device='cuda'):
        import face_alignment
        try:
            self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=f'cuda:{device}') #_2D라고 되어있었음 TWO_D
        except:
            self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=str(device)) #_2D라고 되어있었음 TWO_D
    
    def get_landmarks(self, np_image, get_all=True):
        # estimate all 68 landmarks
        landmarks_68 = np.array(self.detector.get_landmarks_from_image(np_image))
        if not landmarks_68.all():
            return []

        if get_all:
             return landmarks_68[0, :, :]
        else:
            keypoint_indices=[30, 8, 45, 36, 64, 60] #nose, chin, l_eye, r_eye, l_mouth, r_mouth
            # pick a subset of desired landmarks
            return landmarks_68[0, keypoint_indices, :] 