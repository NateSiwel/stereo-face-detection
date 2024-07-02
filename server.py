from flask import Flask, make_response, request, jsonify
from serverClass import ServerClass
import numpy as np
import cv2
import base64
import pickle
from flask_sqlalchemy import SQLAlchemy
from models import User, Embedding
from config import app,db
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, verify_jwt_in_request
import face_recognition
import time

def decode_img(base64_image):
    image_data = base64.b64decode(base64_image)
    image_array = np.frombuffer(image_data, np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb

# Convert lists back to numpy arrays
def list_to_numpy(data):
    try:
        if isinstance(data, list):
            return np.array(data)
        if isinstance(data, dict):
            return {key: list_to_numpy(value) for key, value in data.items()}
        return data
    except Exception as e:
        print(e)
        return None

server = ServerClass()

API_KEY = '123456'
jwt = JWTManager(app)

@app.route('/log_in', methods=['POST'])
def log_in():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'msg': 'Missing username or password'}), 400

    user = User.query.filter_by(username=username).first()

    if user and user.verify_password(password):
        access_token = create_access_token(identity=user.id, expires_delta=False)
        return jsonify(access_token=access_token), 200
    else:
        return jsonify({'msg': 'Invalid credentials'}), 401

@app.route('/sign_up', methods=['POST'])
def sign_up():
    data = request.get_json()

    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'msg': 'Missing username or password'}), 400

    if User.query.filter_by(username=username).first() is not None:
        return jsonify({'msg': 'Username already exists'}), 409

    new_user = User(username=username)
    new_user.password = password

    db.session.add(new_user)
    db.session.commit()

    return jsonify({'msg': 'User created successfully'}), 201

@app.route('/add_embedding', methods=['POST'])
@jwt_required()
def add_embedding():
    current_user_id = get_jwt_identity()
    current_user = User.query.get(current_user_id)

    if not current_user:
        return make_response(jsonify({'msg': 'Unauthorized'}), 401)

    data = request.get_json(silent=True)

    if not data:
        return make_response(jsonify({'msg': 'Invalid Request'}), 400)

    imgL = data.get('imgL')
    name = data.get('name')

    imgL = decode_img(imgL)

    if imgL is not None:
        encoding = face_recognition.face_encodings(imgL)
        if encoding is not None:
            print(encoding)
            if len(encoding) == 1:
                encoding = encoding[0].tolist()
                new_embedding = Embedding(user_id=current_user_id,name=name,embedding=encoding) 
                db.session.add(new_embedding)
                db.session.commit()
                return make_response(jsonify({'msg':'Success! Embedding stored'}), 201)
            else:
                return make_response(jsonify({'msg':'Multiple faces detected, please retry'}), 400)
        else:
            return make_response(jsonify({'msg':'No face detected, please retry'}), 400)

    return make_response(jsonify({'msg': 'Invalid Request'}), 400)

counter = 0
data_array = []
#authenticates image of user 
@app.route('/authenticate', methods=['POST'])
@jwt_required()
def authenticate():
    current_user_id = get_jwt_identity()
    current_user = User.query.get(current_user_id)

    if not current_user:
        return make_response(jsonify({'msg': 'unauthorized'}), 401)

    embeddings = current_user.embeddings
    user_embeddings = [embedding.embedding for embedding in embeddings]

    #fetch user_emebdding from db, to pass to server.authenticate
    if not user_embeddings:
        #Instance where user_embedding isn't found in database
        return make_response(jsonify({'msg': 'Complete setup process'}))

    data = request.get_json(silent=True)

    """
    # remove this later
    global counter 
    global data_array
    counter += 1
    if counter > 300:
        with open('test_data.pickle', 'wb') as handle:
            pickle.dump(data_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return make_response(jsonify({'message':'successfully saved data!'}), 200)
    else:
        data_array.append(data)
        return make_response(jsonify({'message':f'looping - counter @ {counter}!'}), 200)
    """

    if not data:
        return make_response(jsonify({'msg': 'Invalid Request'}), 400)

    imgL = data.get('imgL')
    imgR = data.get('imgR') 
    cam = data.get('cam')

    imgL = decode_img(imgL)
    imgR = decode_img(imgR)

    cam = list_to_numpy(cam)

    # time to recieve data from raspi is about .25 seconds 

    start_time = time.time()
    #time for auth logic it about 1.75 seconds -- way too slow
    if imgL is not None and imgR is not None and cam is not None:
        print(f"passing {current_user}'s request to auth")
        res = server.authenticate(imgL,imgR,cam=cam,user_embeddings=user_embeddings)
        #print("--- %s seconds ---" % (time.time() - start_time))
        if res:
            return make_response(jsonify({'msg':'valid'}), 200)
        else:
            return make_response(jsonify({'msg':'invalid'}), 200)

    return make_response(jsonify({'msg': 'Invalid Request'}), 400)

if __name__ == '__main__':
    with app.app_context():
        #db.drop_all()
        db.create_all()
    app.run(host='0.0.0.0', port=5000, ssl_context=('ssl/cert.pem', 'ssl/key.pem'))
