import requests
import os
from dotenv import load_dotenv
import json
from types import SimpleNamespace
import numpy as np
import base64
import cv2
from flask import jsonify
import pickle
from picamera2 import Picamera2
import dlib
load_dotenv()

server_ip = os.getenv('server_ip')

server_url = f'https://{server_ip}:5000'
api_key = '123456'

def encode_img(image_array):
    _, buffer = cv2.imencode('.jpg', image_array)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

def numpy_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, dict):
        return {key: numpy_to_list(value) for key, value in data.items()}
    return data

#Add function to calibrate camera
class ClientClass():
    def __init__(self):

        self.detector = dlib.get_frontal_face_detector()
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

        with open('calibration/cams.pkl', 'rb') as file:
                self.cam = pickle.load(file)
                self.cam = numpy_to_list(self.cam)

        with open('ssl/key.pickle', 'rb') as handle:
            self.key = pickle.load(handle)

    def log_in(self):

        url = f"{server_url}/log_in"
        
        username = input("enter username: ")
        password = input("enter password: ")

        data = {
            "username": username,
            "password": password
        }

        headers = {"Content-Type": 'application/json'}
        
        response = requests.post(url, headers=headers, json=data)
        
        # Check the response status code
        if response.status_code == 200:
            # If credentials are valid, retrieve and return the access token
            access_token = response.json().get('access_token')
            print("Login successful. Access Token:", access_token)
            self.key = access_token
            with open('ssl/key.pickle', 'wb') as handle:
                pickle.dump(access_token, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return access_token
        else:
            # If credentials are invalid or any other error, print the error message
            error_message = response.json().get('error', 'An error occurred during login')
            print(f"Login failed: {error_message}")
            return None

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
        if not self.rectsL:
            return False
        self.rectsR = self.detector(frameR)

        if self.rectsL and self.rectsR:
            return True

        return False

    def authenticate(self, imgL, imgR):
        self.imgL, self.imgR = imgL, imgR

        headers = {"Content-Type": 'application/json',
                    "Authorization": f"Bearer {self.key}"
                   }
        #imgL = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
        imgL64 = encode_img(imgL)

        #imgR = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)
        imgR64 = encode_img(imgR)
        
        body = {"imgL":imgL64, "imgR":imgR64, "cam":self.cam}

        response = requests.post(server_url+'/authenticate', headers=headers, json=body, verify=False)
        status_code = response.status_code
        if status_code == 200:
            #Face authenticated
            json = response.json()
            message = json['message']
            #unlock method here
        else:
            #invalid face 
            json = response.json()
            message = json['error']
        return message

    def destroy(self):
        self.camL.stop()
        self.camR.stop()

if __name__ == "__main__":
    cams = ClientClass()

    while True:
        frameL, frameR = cams.get_frames()

        if cams.get_faces(frameL, frameR):
            res = cams.authenticate(frameL, frameR)
            print(res)
