from flask import Flask, request, jsonify

app = Flask(__name__)
API_KEY = '123456'

@app.route('/upload', methods=['POST'])
def upload_image():
    if request.headers.get('API-Key') != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context=('ssl/cert.pem', 'ssl/key.pem'))
