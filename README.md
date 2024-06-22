## cv2 face recognition - utilizes stereo vision to prevent presentation attacks

* Compute camera calibration paramaters 
* Detect faces clientside to know when to send images to server  
* Interact with web server once face is detected for the compute heavy tasks below
* Use dlib cnn to compare face embeddings found in image to user embedding
* Rectify image for stereo vision task
* Compute depth map on authenticated faces
* Use depth map to verify authenticity of face (not image, video, or presentation attack)

### run from within ssl folder
* openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes