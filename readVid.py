import cv2 as cv

vid = cv.VideoCapture(0)

def rescaleFrame(frame,scale=0.75):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimension = (width,height)
    return cv.resize(frame,dimension,interpolation=cv.INTER_AREA)

label = 'person'

while(True):
    isTrue, frame = vid.read()
    haar_cascade = cv.CascadeClassifier('haarFace.xml')
    faces_rect = haar_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=3)
    print(f'Number of faces found = {len(faces_rect)}')

    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,225,0),thickness=2)
        cv.putText(frame, label, (x, y), cv.FONT_HERSHEY_PLAIN, 1,(0,255,0), 1)

    cv.imshow('Video',frame)

    if cv.waitKey(20) & 0xFF==ord('x'):
        break

vid.release()
vid.destroyAllWindows() 