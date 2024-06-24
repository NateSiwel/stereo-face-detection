from flask import Flask, make_response, request, jsonify
from serverClass import ServerClass
import numpy as np
import cv2
import json
import base64
import pickle

app = Flask(__name__)
API_KEY = '123456'

def decode_img(base64_image):
    image_data = base64.b64decode(base64_image)
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.COLOR_BGR2RGB)

    return image

def encode_img(image_array):
    """
    Was previosuly receiving images in GRAY to save data, 
    but face_recognition requires RGB
    """
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

counter = 0
data_array = []
#Backend logic that authenticates image of user 
@app.route('/authenticate', methods=['POST'])
def authenticate():
    print('request received')
    if request.headers.get('API-Key') != API_KEY:
        return make_response(jsonify({'error': 'Unauthorized'}), 401)

    #fetch user_emebdding from a db, to pass to server.authenticate
    user_embedding = np.array([0]) 
    if user_embedding is None:
        #Instance where user_embedding isn't found in database
        return make_response(jsonify({'error': 'Complete setup process'}))

    data = request.json

    """
    # remove this later
    global counter 
    global data_array
    counter += 1
    if counter > 40:
        with open('test_data.pickle', 'wb') as handle:
            pickle.dump(data_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return make_response(jsonify({'message':'successfully saved data!'}), 200)
    else:
        data_array.append(data)
        return make_response(jsonify({'message':f'looping - counter @ {counter}!'}), 200)
    """

    imgL = data['imgL']
    imgR = data['imgR'] 
    cam = data['cam']

    imgL = decode_img(imgL)
    imgR = decode_img(imgR)

    cam = list_to_numpy(cam)

    if imgL is not None and imgR is not None and cam is not None:
        print('Passing to server.authenticate')
        #server.authenticate()
        res = server.authenticate(imgL,imgR,cam=cam,user_embedding=user_embedding)
        if res:
            return make_response(jsonify({'message':'Auth successful'}), 200)
        else:
            return make_response(jsonify({'error':'Authentication Failed'}), 401)

    return make_response(jsonify({'error': 'Invalid Request'}), 400)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context=('ssl/cert.pem', 'ssl/key.pem'))
