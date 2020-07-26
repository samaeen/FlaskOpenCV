import cv2
import numpy as np

face_cascade=cv2.CascadeClassifier('../../data/haarcascade_frontalface_default.xml')

eye_cascade=cv2.CascadeClassifier('../../data/haarcascade_eye.xml')

#cap=cv2.VideoCapture(0)

class faceDetection(object):
    """docstring for faceDetection"""
    def __init__(self, hello):
        self.hello=hello
    def update(self,img):
    def detect(self,img):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=3,minSize=(80, 80),flags=0)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray=gray[y:y+h,x:x+w]
            roi_color=img[y:y+h,x:x+w]
            eyes=eye_cascade.detectMultiScale(roi_gray)
            for(ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


        

'''
while True:
    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=3,minSize=(80, 80),flags=0)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        #print(faces)
        eyes=eye_cascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            #print('start point ',ex,'end point',ex+ew)
    cv2.imshow('img',img)
    if cv2.waitKey(30)& 0xff==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''