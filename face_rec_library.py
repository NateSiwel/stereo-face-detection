import face_recognition
import cv2

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))

ret, frame = cam.read()

face_locations = face_recognition.face_locations(frame, model='cnn')
for face in face_locations:
  y1,x2,y2,x1 = face
  my_face_encoding = face_recognition.face_encodings(frame, face_locations)[0]


# Have a script like this server side that runs comparisons on received images 
i = 0
while True:
    ret, frame = cam.read()

    face_locations = face_recognition.face_locations(frame, model='cnn')
    for face in face_locations:
      y1,x2,y2,x1 = face
      current_face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
      cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
      results = face_recognition.compare_faces([my_face_encoding], current_face_encoding)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
      break
    
cam.release()   
cv2.destroyAllWindows()