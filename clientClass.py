import requests
import os
from dotenv import load_dotenv
import json
from types import SimpleNamespace
import numpy as np
import base64
import cv2
from flask import jsonify
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

class ClientClass():
    def __init__(self, cam):
        self.cam = cam
        self.cam = numpy_to_list(cam)

    def authenticate(self, imgL, imgR):
        self.imgL, self.imgR = imgL, imgR
        headers = {'API-Key': api_key, "Content-Type": 'application/json'}
        #imgL = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
        imgL64 = encode_img(imgL)

        #imgR = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)
        imgR64 = encode_img(imgR)
        
        body = {"imgL":imgL64, "imgR":imgR64, "cam":self.cam}

        response = requests.post(server_url+'/authenticate', headers=headers, json=body, verify=False)
        status_code = response.status_code
        if status_code == 200:
            json = response.json()
            message = json['message']
            return message 
        return "Invalid Request"
