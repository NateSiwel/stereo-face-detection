import cv2
import numpy as np
import pickle
from extract import Extract

class Calibrate:
    def __init__(self, chessboard_size, square_size, max_chessboards):
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.max_chessboards = max_chessboards
        self.objpoints = []
        self.imgpointsL = []
        self.imgpointsR = []
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size

    def add_corners(self, imgL, imgR):
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        retL, cornersL = cv2.findChessboardCorners(grayL, self.chessboard_size, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, self.chessboard_size, None)

        if retL and retR:
            self.objpoints.append(self.objp)
            self.imgpointsL.append(cornersL)
            self.imgpointsR.append(cornersR)
            return True
        else:
            return False

    def compute(self, img_shape):
        retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(self.objpoints, self.imgpointsL, img_shape, None, None)
        retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(self.objpoints, self.imgpointsR, img_shape, None, None)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        flags = cv2.CALIB_FIX_INTRINSIC
        ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpointsL, self.imgpointsR, mtxL, distL, mtxR, distR, img_shape, criteria=criteria, flags=flags)

        return ret, mtxL, distL, mtxR, distR, R, T

    def calibrate(self, cams):
        count = 0
        while count < self.max_chessboards:
            imgL, imgR = cams.get_frames()
            print('looped')
            cv2.imshow('right image', imgR)
            cv2.imshow('left image', imgL)

            if self.add_corners(imgL, imgR):
                count += 1
                print(f"Captured {count}/{self.max_chessboards} chessboards.")
                cv2.imshow('img', imgL)
            cv2.waitKey(0)
        img_shape = imgL.shape[1::-1]
        return self.compute(img_shape)

    def save_calibration(self, filename, calibration_data):
        with open(filename, 'wb') as f:
            pickle.dump(calibration_data, f)
        print(f"Calibration data saved to {filename}")

# Example usage
chessboard_size = (7, 7)  # Number of inside corners in the chessboard pattern (columns, rows)
square_size = 2.9# Size of a square in defined unit (meters or centimeters)
max_chessboards = 12  # chessboards to capture

calibration = Calibrate(chessboard_size, square_size, max_chessboards)

cams = Extract()
try:
    ret, mtxL, distL, mtxR, distR, R, T = calibration.calibrate(cams)
    calibration_data = {
        "ret": ret,
        "mtxL": mtxL,
        "distL": distL,
        "mtxR": mtxR,
        "distR": distR,
        "R": R,
        "T": T
    }
    calibration.save_calibration("calibration/cams.pkl", calibration_data)
finally:
    cams.destroy()
