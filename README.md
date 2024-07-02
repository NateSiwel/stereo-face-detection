## cv2 face recognition - utilizes stereo vision to prevent presentation attacks

* Compute camera calibration paramaters via chessboard
* Store unlimited embeddings per account with { python3 clientClass.py add_embedding }
* Detect faces clientside w/ lightweight HOG model 
* Interact with web server once face is detected for the compute heavy tasks below
* Store/fetch user embeddings/credentials in psql database 
* Use dlib cnn model to classify face embeddings found in live image - compare to user embeddings in database 
* Rectify image for stereo vision task
* Compute depth map on authenticated faces
* Pass 224x224 depth map to CNN to verify authenticity of face (presentation attack)

### run from within ssl folder
* openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
