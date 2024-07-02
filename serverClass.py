import time
import numpy as np
import cv2
import dlib
import matplotlib
import face_recognition
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x

def batch_face_encodings(image, rects):
    return face_recognition.face_encodings(image, rects)

class ServerClass ():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.auth_model = SimpleCNN().to(self.device)
        self.auth_model.load_state_dict(torch.load('auth.pth'))
        self.auth_model.eval()
        self.padding_percentage = .2

    def authenticate(self,imgL,imgR,cam,user_embeddings):

        self.h,self.w = imgL.shape[:2] 
        img1_rectified, img2_rectified = self.rectify_frames(imgL,imgR,cam)

        rects = face_recognition.batch_face_locations([img1_rectified,img2_rectified],number_of_times_to_upsample=0,batch_size=2)
        rectsL, rectsR = rects[0], rects[1]

        print(len(rectsL), len(rectsR))


        if rectsL and rectsR:

            encodingsL = face_recognition.face_encodings(img1_rectified, rectsL)
            encodingsR = face_recognition.face_encodings(img2_rectified, rectsR)

            # encodingsL and encodingR represent list of unorganized encodings
            # if there are multiple faces - we don't yet know how faces translate across camL/camR
            # We should be able to fetch a static pixel translation between cams based on matrix

            # We'll need to have [(person1L, person1R), (person2L, person2R)]

            # remove encodings from encodingsL that don't match user - we don't want to waste time w/ them
            #encodingsL = [enc for enc in encodingsL if face_recognition.compare_faces([user_embedding], enc)[0]]

            matches = []
            for idxL, encL in enumerate(encodingsL):
                distances = face_recognition.face_distance(encodingsR, encL)
                min_distance = min(distances)
                idxR = distances.tolist().index(min_distance)
                
                if min_distance < 0.6:
                    matches.append((idxL, idxR))


            # matched_pairs = [(personEnc1L, personEnc1R), (personEnc2L, personEnc2R)]
            matched_pairs = [(encodingsL[i], encodingsR[j], rectsL[i]) for i, j in matches]

            for embL,embR,rectL in matched_pairs:
                if face_recognition.compare_faces(user_embeddings, embL)[0] and face_recognition.compare_faces(user_embeddings, embR)[0]:
                    # embedding belongs to user - calculate disparity_map and authenticate embedding validity

                    # get_disparity takes about .6 seconds
                    # see if block size can be increased w/o accuracy falloff
                    disparity_map = self.get_disparity(
                        img1_rectified=img1_rectified, 
                        img2_rectified=img2_rectified, 
                        min_disp=-100, 
                        max_disp=100, 
                        block_size=1, 
                        uniquenessRatio=0, 
                        speckleWindowSize=0, 
                        speckleRange=1, 
                        disp12MaxDiff=0
                    )

                    #crop 224x224 of face 

                    height, width = disparity_map.shape[:2]

                    top,right,bottom,left = rectL

                    # Calculate dynamic padding based on the size of the face
                    pad_w = int((right - left) * self.padding_percentage)
                    pad_h = int((bottom - top) * self.padding_percentage)
                    
                    # Apply padding and ensure indices are within the image bounds
                    padded_top = max(top - pad_h, 0)
                    padded_bottom = min(bottom + pad_h, height)
                    padded_left = max(left - pad_w, 0)
                    padded_right = min(right + pad_w, width)
                    
                    # Extract the face region with padding
                    face_region = disparity_map[padded_top:padded_bottom, padded_left:padded_right]
                    face_normal = img1_rectified[padded_top:padded_bottom, padded_left:padded_right]

                    face_region_resized = cv2.resize(face_region, (224,224), interpolation=cv2.INTER_AREA)
                    face_normal_resized = cv2.resize(face_normal, (224,224), interpolation=cv2.INTER_AREA)
                    
                    #prep map for model input
                    disparity_tensor = torch.tensor(face_region_resized, dtype=torch.float32).unsqueeze(0)
                    disparity_tensor = disparity_tensor.unsqueeze(1)
                    disparity_tensor = disparity_tensor.to(self.device) 

                    with torch.no_grad():
                        output = self.auth_model(disparity_tensor)

                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        pred = torch.argmax(probabilities, dim=1)

                        if pred:
                            # embedding authenticity has been verified by auth_model
                            return True
                        else:
                            print('Embedding belongs to user - but is invalid!')

        print('no faces detected')
        return False 

    def rectify_frames(self,frameL,frameR,cam):
        self.h,self.w = frameL.shape[:2]

        mtx1 = cam['mtxL']
        dist1 = cam['distL']
        mtx2 = cam['mtxR']
        dist2 = cam['distR']
        R = cam['R']
        T = cam['T']

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, (self.w, self.h), R, T)

        # undistortion and rectification maps
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(mtx1, dist1, R1, P1, (self.w, self.h), cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(mtx2, dist2, R2, P2, (self.w, self.h), cv2.CV_32FC1)

        img1_rectified = cv2.remap(frameL, self.map1x, self.map1y, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(frameR, self.map2x, self.map2y, cv2.INTER_LINEAR)

        return img1_rectified,img2_rectified

    def get_disparity(self, img1_rectified, img2_rectified, min_disp=0, max_disp=128, block_size=11, 
                  uniquenessRatio=5, speckleWindowSize=200, speckleRange=2, disp12MaxDiff=0):
        # ------------------------------------------------------------
        # CALCULATE DISPARITY (DEPTH MAP)
        # Adapted from: https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
        # and: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html

        # StereoSGBM Parameter explanations:
        # https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html

        """
        # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        block_size = 11
        #min_disp = -128
        min_disp = 0 
        max_disp = 128
        # Maximum disparity minus minimum disparity. The value is always greater than zero.
        # In the current implementation, this parameter must be divisible by 16.
        # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
        # Normally, a value within the 5-15 range is good enough
        uniquenessRatio = 5
        # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
        # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckleWindowSize = 200
        # Maximum disparity variation within each connected component.
        # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
        # Normally, 1 or 2 is good enough.
        speckleRange = 2
        disp12MaxDiff = 0
        """

        num_disp = max_disp - min_disp

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            disp12MaxDiff=disp12MaxDiff,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
        )
        disparity_SGBM = stereo.compute(img1_rectified, img2_rectified)

        # Normalize the values to a range from 0..255 for a grayscale image
        disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                      beta=0, norm_type=cv2.NORM_MINMAX)
        disparity = np.uint8(disparity_SGBM)

        return disparity
