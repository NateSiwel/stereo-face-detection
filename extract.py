import time
from picamera2 import Picamera2
import numpy as np
import cv2
import dlib
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')   
import pickle
from clientClass import ClientClass

class Extract():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        self.camL = Picamera2(0)
        self.camR = Picamera2(1)

        configL = self.camL.create_video_configuration(main={"size":(1296, 972), 'format': 'RGB888'})
        configR = self.camR.create_video_configuration(main={"size":(1296, 972), 'format': 'RGB888'})

        self.camL.configure(configL)

        self.camR.configure(configR)

        self.camL.start()
        self.camR.start()

        self.frameL = None
        self.frameR = None

        self.sift = cv2.SIFT_create()

    def get_frames(self):
        self.frameL = self.camL.capture_array()
        self.frameR = self.camR.capture_array()

        return self.frameL, self.frameR

    def get_faces(self, frameL=None, frameR=None):
        if frameL is None:
            frameL = self.frameL
        if frameR is None:
            frameR = self.frameR

        self.rectsL = self.detector(frameL)
        self.rectsR = self.detector(frameR)

        boxesL = [convert_and_trim_bb(self.frameL, r) for r in self.rectsL]
        boxesR = [convert_and_trim_bb(self.frameR, r) for r in self.rectsR]

        return (boxesL, boxesR)

    def get_keypoints(self, imgL, imgR):
        self.kpL, self.desL = self.sift.detectAndCompute(imgL, None)
        self.kpR, self.desR = self.sift.detectAndCompute(imgR, None)

    def get_matches(self):
        # Match keypoints in both images
        # Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.matches = flann.knnMatch(self.desL, self.desR, k=2)

        # Keep good matches: calculate distinctive image features
        # Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
        # https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
        self.matchesMask = [[0, 0] for i in range(len(self.matches))]
        good = []
        pts1 = []
        pts2 = []

        for i, (m, n) in enumerate(self.matches):
            if m.distance < 0.7*n.distance:
                # Keep this keypoint pair
                self.matchesMask[i] = [1, 0]
                good.append(m)
                pts2.append(self.kpR[m.trainIdx].pt)
                pts1.append(self.kpL[m.queryIdx].pt)

        return pts1,pts2,good

    def draw_matches(self, matches):
        # Draw the keypoint matches between both pictures
        # Still based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=self.matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)

        keypoint_matches = cv2.drawMatchesKnn(
                self.frameL, self.kpL, self.frameR, self.kpR, self.matches, None, **draw_params)
        cv2.imshow("Keypoint matches", keypoint_matches)

    def get_fundamental_matrix(self, pts1, pts2):
        # ------------------------------------------------------------
        # STEREO RECTIFICATION

        # Calculate the fundamental matrix for the cameras
        # https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

        """
        # We select only inlier points
        pts1 = pts1[inliers.ravel() == 1]
        pts2 = pts2[inliers.ravel() == 1]
        """

        return pts1, pts2, fundamental_matrix

    def detect_landmarks(self,image):
        # Detect faces in the image
        faces = self.detector(image)

        print(faces)

        landmarks_list = []
        for face in faces:
            # Get the landmarks/parts for the face
            landmarks = self.landmark_predictor(image, face)

            # Convert landmarks to a list of (x, y) coordinates
            landmarks_coords = [(p.x, p.y) for p in landmarks.parts()]
            print(landmarks_coords)
            landmarks_list.append(landmarks_coords)

        return landmarks_list


    def destroy(self):
        self.camL.stop()
        self.camR.stop()

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

if __name__ == "__main__":
    with open('calibration/cams.pkl', 'rb') as file:
        cam = pickle.load(file)

    cams = Extract()
    client = ClientClass(cam)
 

    while True:

        #L and R represent left and right cams
        frameL, frameR = cams.get_frames()
        grayL, grayR  = cv2.cvtColor(frameL, cv2.COLOR_RGB2GRAY), cv2.cvtColor(frameR, cv2.COLOR_RGB2GRAY)
        # apply rectification 

        res = client.authenticate(frameL, frameR)
        print(res)

        """
        img1_rectified = cv2.remap(frameL, map1x, map1y, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(frameR, map2x, map2y, cv2.INTER_LINEAR)

        cv2.imshow('img1', img1_rectified)
        cv2.imshow('img2', img2_rectified)
        """

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cams.destroy()
    cv2.destroyAllWindows()
