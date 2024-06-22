import time
import numpy as np
import cv2
import dlib
import matplotlib
import face_recognition
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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

class ServerClass ():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def authenticate(self,imgL,imgR,cam,user_embedding):
        self.h,self.w = imgL.shape[:2] 
        img1_rectified, img2_rectified = self.rectify_frames(imgL,imgR,cam)
        disparity_map = self.get_disparity(img1_rectified, img2_rectified)

        #returns imgs
        ret = self.get_faces(img1_rectified, img2_rectified) 

        if ret is True:

            print(self.rectsL)
            encodingsL = face_recognition.face_encodings(img1_rectified, self.rectsL)

            encodingsR = face_recognition.face_encodings(img2_rectified, self.rectsR)

            # encodingsL and encodingR represent list of unorganized encodingsL
            # if there are multiple faces - we don't yet know how faces translate across camL/camR
            # We should be able to fetch a static pixel translation between cams based on matrix

            # We'll need to have [(person1L, person1R), (person2L, person2R)]

            for encoding in encodingsL:
                ret = face_recognition.compare_faces([user_embedding], encoding)

            matches = []
            for idxL, encL in enumerate(encodingsL):
                distances = face_recognition.face_distance(encodingsR, encL[0])
                min_distance = min(distances)
                idxR = distances.tolist().index(min_distance)
                
                if min_distance < 0.6:
                    matches.append((idxL, idxR))

            # matched_pairs = [(personEnc1L, personEnc1R), (personEnc2L, personEnc2R)]
            matched_pairs = [(encodingsL[i], encodingsR[j]) for i, j in matches]

            print(len(matched_pairs))

            return True
    
        """
        plt.imshow(disparity_map, cmap='gray')
        plt.title('Disparity Map')
        plt.colorbar()
        plt.show()
        """

        return False 

    def rectify_frames(self,frameL,frameR,cam):
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

    def get_faces(self, frameL, frameR) -> bool:

        """

        TODO - verify face coords belong to same person  
        
        passes images to dlibs face recognition  
        returns true if faces are detected in both images

        """

        self.rectsL = face_recognition.face_locations(frameL)
        self.rectsR = face_recognition.face_locations(frameR)

        if self.rectsL and self.rectsR:
            return True
        
        """
        self.rectsL = self.detector(frameL)
        self.rectsR = self.detector(frameR)

        boxesL = [convert_and_trim_bb(frameL, r) for r in self.rectsL]
        boxesR = [convert_and_trim_bb(frameR, r) for r in self.rectsR]

        # Process detected faces from each cam
        #Grab face images if necessary
        facesL = []
        for (x, y, w, h) in boxesL:
            face = frameL[y:y+h, x:x+w]
            facesL.append(face)
            #frameL = cv2.rectangle(img1_rectified, (x,y), (x+w,y+h), (0,255,0))
        facesR = []
        for (x, y, w, h) in boxesR:
            face = frameR[y:y+h, x:x+w]
            facesR.append(face)
            #frameR = cv2.rectangle(img2_rectified, (x,y), (x+w,y+h), (0,255,0))
        """
        return False 

    def get_disparity(self,img1_rectified,img2_rectified):
        # ------------------------------------------------------------
        # CALCULATE DISPARITY (DEPTH MAP)
        # Adapted from: https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
        # and: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html

        # StereoSGBM Parameter explanations:
        # https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html

        # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        block_size = 11
        min_disp = -128
        max_disp = 128
        # Maximum disparity minus minimum disparity. The value is always greater than zero.
        # In the current implementation, this parameter must be divisible by 16.
        num_disp = max_disp - min_disp
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
