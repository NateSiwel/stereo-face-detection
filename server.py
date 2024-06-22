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
    image = cv2.imdecode(image_array, cv2.COLOR_BGR2RGB)

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

#Backend logic that authenticates image of user 
@app.route('/authenticate', methods=['POST'])
def authenticate():
    print('request received')
    if request.headers.get('API-Key') != API_KEY:
        return make_response(jsonify({'error': 'Unauthorized'}), 401)

    #fetch user_emebdding from a db, to pass to server.authenticate
    user_embedding = np.array([-0.06581603, -0.00224692,  0.05033305, -0.02142783, -0.06770886,
       -0.10775353,  0.09814088, -0.12716433,  0.07386479, -0.02622478,
        0.26346737, -0.03806964, -0.28334403, -0.15818875,  0.00441827,
        0.11307094, -0.07292569, -0.13180599, -0.11482117, -0.10062823,
       -0.01925524, -0.05267199,  0.00670022,  0.00469728, -0.05628844,
       -0.26083213, -0.06063636, -0.06171902,  0.01865013, -0.08745471,
        0.03122865,  0.18768182, -0.15210827, -0.07999556,  0.00260305,
        0.10593551, -0.06907584, -0.08001573,  0.25822058, -0.03394725,
       -0.13499682,  0.01093885,  0.06424382,  0.27184838,  0.19476449,
       -0.03681045,  0.02888038,  0.00962743,  0.13205042, -0.28988746,
        0.01911257,  0.1463571 ,  0.18175685,  0.0747066 ,  0.06480634,
       -0.15684736,  0.04335513,  0.15626734, -0.25449699,  0.06240658,
       -0.02437693, -0.16583   , -0.0160768 , -0.082592  ,  0.18480711,
        0.09795678, -0.08436821, -0.05033356,  0.19224474, -0.14966905,
       -0.05298907,  0.05825811, -0.16567169, -0.18608718, -0.27286565,
        0.10731775,  0.34005117,  0.13032572, -0.14258303,  0.0504825 ,
       -0.03024056, -0.04730763,  0.02285877,  0.02755865, -0.05407654,
       -0.06901301, -0.10386412, -0.01080935,  0.18483657, -0.04005573,
       -0.06208696,  0.18424927, -0.0155708 , -0.00097101,  0.02803051,
       -0.02481522,  0.00306688,  0.01825804, -0.0990911 ,  0.03791631,
        0.02348158, -0.24515964,  0.02530631,  0.03793404, -0.17780685,
        0.09068563, -0.01795265,  0.01917171, -0.0286487 , -0.05659619,
       -0.04488274,  0.02835467,  0.13957471, -0.29675058,  0.29792106,
        0.21101409, -0.00865792,  0.13807109,  0.00430148,  0.05804594,
        0.01116218, -0.048469  , -0.08418012, -0.0963585 ,  0.06838287,
        0.01403427,  0.0443102 ,  0.0776477 ])
    if user_embedding is None:
        #Instance where user_embedding isn't found in database
        return make_response(jsonify({'error': 'Complete setup process'}))

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
        res = server.authenticate(imgL,imgR,cam=cam,user_embedding=user_embedding)
        print(res)

        return make_response(jsonify({'message':'Success!'}), 200)

    return make_response(jsonify({'error': 'Invalid Request'}), 400)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context=('ssl/cert.pem', 'ssl/key.pem'))
