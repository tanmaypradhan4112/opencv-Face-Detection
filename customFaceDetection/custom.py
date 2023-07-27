import os
import cv2 as cv
import numpy as np 

people =['rdj','selena','srk']
DIR = r'faces/train'
# for i in os.listdir(r'faces/train'):
#     people.append(i)

haar_cascade = cv.CascadeClassifier('haarFace.xml')
features = []
labels= []

def create_train():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)
        print(path)

        for image in os.listdir(path):
            image_path = os.path.join(path,image)
            print(image_path)
            image_array = cv.imread(image_path)
            gray = cv.cvtColor(image_array,cv.COLOR_BGR2GRAY)
            face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)

            for(x,y,w,h) in face_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

# print(f'Length of the feature = {len(feature)}')
# print(f'Lenght of the label: {len(labels)}')
print("training is done")

features = np.array(features,dtype=object)
labels = np.array(labels)
faces_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer o the features
faces_recognizer.train(features,labels)
faces_recognizer.save('face_trained.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)

