from flask import Flask, request, jsonify
from serverClass import ServerClass

app = Flask(__name__)
API_KEY = '123456'

server = ServerClass()
@app.route('/upload', methods=['POST'])
def upload_image():
    print('request recieved')
    if request.headers.get('API-Key') != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401
    if request.headers.get('imgL') and request.headers.get('imgR'):
        imgL, imgR = request.headers.get('imgL'), request.headers.get('imgR')
        facesL,facesR=server.get_faces(imgL,imgR)
        print(facesL)

    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context=('ssl/cert.pem', 'ssl/key.pem'))
