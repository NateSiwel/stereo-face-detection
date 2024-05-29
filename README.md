## cv2 face recognition - utilizes stereo vision to prevent presentation attacks

* Compute camera calibration paramaters 
* Recognize faces
* Interact with web server once face is detected for the compute heavy tasks below
* Rectify image for stereo vision task
* Compute depth map 

### run from within ssl folder
* openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
