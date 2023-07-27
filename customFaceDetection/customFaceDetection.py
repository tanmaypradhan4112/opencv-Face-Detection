import numpy as np 
import cv2 as cv
import os

def rescaleFrame(img,scale=0.75):
    width = int(img.shape[1]*scale)
    height = int(img.shape[0]*scale)
    dimension = (width,height)
    return cv.resize(img,dimension,interpolation=cv.INTER_AREA)

people = ['rdj','selena','srk']

# for i in os.listdir(r'faces/train'):
#     people.append(i)


haar_cascade = cv.CascadeClassifier('haarFace.xml')

# features = np.load('features.npy')
# labels = np.load('labels.npy')

faces_recognizer = cv.face.LBPHFaceRecognizer_create()
faces_recognizer.read('face_trained.yml')

img = cv.imread(r'faces/val/srk/5.jpeg')
imgNew = rescaleFrame(img,scale=0.5)
gray = cv.cvtColor(imgNew,cv.COLOR_BGR2GRAY)

cv.imshow("open Face Detection",gray)

faces_rect = haar_cascade.detectMultiScale(gray,1.1,3)

for(x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+h]

    labels,confidence = faces_recognizer.predict(faces_roi)
    print(f'Label = {people[labels]} with a confidence of {confidence}')

    cv.putText(imgNew,str(people[labels]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    cv.rectangle(imgNew,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow("detected Face", imgNew)
cv.waitKey(0)