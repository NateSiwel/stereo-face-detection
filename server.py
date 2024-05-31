from flask import Flask, make_response, request, jsonify
from serverClass import ServerClass
import numpy as np
import cv2
import json
import base64

app = Flask(__name__)
API_KEY = '123456'

def decode_img(base64_image):
    image_data = base64.b64decode(base64_image)
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    return image

def encode_img(image_array):
    _, buffer = cv2.imencode('.jpg', image_array)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

# Convert lists back to numpy arrays
def list_to_numpy(data):
    if isinstance(data, list):
        return np.array(data)
    if isinstance(data, dict):
        return {key: list_to_numpy(value) for key, value in data.items()}
    return data

server = ServerClass()
@app.route('/authenticate', methods=['POST'])
def authenticate():
    print('request received')
    if request.headers.get('API-Key') != API_KEY:
        return make_response(jsonify({'error': 'Unauthorized'}), 401)
    data = request.json

    imgL = data['imgL']
    imgR = data['imgR'] 
    cam = data['cam']

    imgL = decode_img(imgL)
    imgR = decode_img(imgR)

    cam = list_to_numpy(cam)

    if imgL is not None and imgR is not None and cam is not None:
        print('Rectifying Image')
        #server.authenticate()
        res = server.authenticate(imgL,imgR,cam=cam)
        print(res)

        return make_response(jsonify({'message':'Success!'}), 200)

    return make_response(jsonify({'error': 'Invalid Request'}), 400)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context=('ssl/cert.pem', 'ssl/key.pem'))
