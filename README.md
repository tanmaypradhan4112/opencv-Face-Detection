# OpenCV Face Detection
## About the Repository
The mentioned repository focuses on a fundamental Face Detection project implemented using OpenCV, a powerful and popular open-source computer vision library. OpenCV (Open Source Computer Vision Library) is widely used for a wide range of computer vision tasks, such as image and video processing, object detection, and facial recognition.

## About  the Project
In this specific project, the Face Detection task is accomplished using a pre-trained model called Haar Cascade, which is natively provided by the OpenCV library. Haar Cascade is a machine learning-based object detection technique used to detect objects in images or videos.

## What is Haar Cascade?
The Haar Cascade model works by applying a series of Haar-like features (simple rectangular filters) to an image. These features are used to calculate the integral image and then apply a set of weak classifiers to the image at different scales and positions. The Haar Cascade model learns patterns from positive and negative training samples to differentiate between object regions (faces in this case) and non-object regions.

By leveraging the Haar Cascade model provided by OpenCV, this project enables the detection of faces in images or videos with high accuracy and efficiency. It serves as an essential building block for various applications, such as facial recognition systems, emotion detection, and even in security and surveillance systems.

## Algorithm for my Face Detection usinf Haar Cascade Project

1.  Import the necessary libraries, OpenCV (cv2) in this case.
    
2.  Create a VideoCapture object (vid) to access the video stream from the default camera (index 0).
    
3.  Define a function 'rescaleFrame' to resize the frame captured from the video stream to a desired scale.
    
4.  Set the label for the object to be detected, in this case, 'person'.
    
5.  Start an infinite loop to continuously read frames from the video stream.
    
6.  Inside the loop, use the Haar Cascade classifier (haar_cascade) to detect faces in the current frame.
    
7.  The 'detectMultiScale' method is used to detect faces. It takes the frame, scaleFactor (to specify how much the image size is reduced at each scale), and minNeighbors (to specify the minimum number of neighbor rectangles to retain as a face) as arguments.
    
8.  Use the 'len' function to get the number of faces detected (len(faces_rect)).
    
9.  Iterate through each detected face using a for loop.
    
10.  Draw a rectangle around each detected face using 'cv.rectangle' method and put the label 'person' near the top-left corner of the rectangle using 'cv.putText' method.
    
11.  Show the frame with detected faces using 'cv.imshow'.
    
12.  Check if the user presses the 'x' key (ASCII code 120) using 'cv.waitKey'. If 'x' is pressed, break the loop to stop the video stream.
    
13.  Release the video stream and close all windows using 'vid.release()' and 'cv.destroyAllWindows()' at the end.
    

> Note: Make sure you have a valid 'haarFace.xml' file for the Haar
> Cascade classifier for face detection, and the OpenCV library is
> properly installed before running the code.

## Result
![enter image description here](https://github.com/tanmaypradhan4112/opencv-Face-Detection/blob/main/resultImg.png?raw=true)
### Resources
https://github.com/opencv/opencv 



