import imp
imp.find_module("cv2")
import cv2
import numpy as np
faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
cam = cv2.VideoCapture(0)
i=0
cv2.waitKey(2)
while True:
    ret,img=cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.5,5)
    for (x,y,w,h) in faces:
         i+=1
         cv2.imwrite("dataSet/User.1"+"."+str(i)+".jpg",img)
         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
         #cv2.waitKey(10)
    cv2.imshow("Capturing",img)
    cv2.waitKey(1)
    if i >= 250:
         break
cam.release()
cv2.destroyAllWindows()
