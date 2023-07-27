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

## Custom Face Detection
In this project, I implemented a custom face detection system using the Haar Cascade method with the OpenCV library. By training the model on custom datasets, I achieved accurate and efficient face detection capabilities. The system can identify faces in real-time images, making it suitable for various applications, such as security systems, image analysis, and biometric authentication. The custom Haar Cascade model ensured reliable and precise face detection, enhancing the overall performance and usability of the application.

**Labels** : RDJ, Selena, SRK
**Libraries**: OpenCV, OS and Numpy

### Explanation of custom.py
- The code is for creating a custom face recognition system using Haar Cascade and LBPH (Local Binary Pattern Histogram) Face Recognizer in OpenCV.
- It starts by defining a list of people (actors/singers) for whom we want to recognize faces.
- It reads the images of each person from the "faces/train" directory and extracts facial features using Haar Cascade for face detection.
- The grayscale version of the face is extracted and stored in the "features" list, while the corresponding label (index of the person in the "people" list) is stored in the "labels" list.
- The function "create_train()" processes all images of each person, extracts facial features, and assigns corresponding labels.
- After processing all images, the LBPH Face Recognizer is created using "cv.face.LBPHFaceRecognizer_create()".
- The recognizer is then trained on the extracted features and labels using "faces_recognizer.train(features, labels)".
- The trained recognizer is saved to a file named "face_trained.yml".
- Additionally, the extracted features and labels are saved to "features.npy" and "labels.npy" files, respectively.
- The code completes the custom face recognition system by training the recognizer on custom data and saving the model for future use.

### Explanation of customFaceDetection.py
-   The code is written in Python and uses the OpenCV and NumPy libraries.
-   The "rescaleFrame" function is defined to resize the input image to a specific scale.
-   A list of people's names ("rdj", "selena", "srk") is created to represent different individuals in the image.
-   The Haar Cascade classifier is loaded from the 'haarFace.xml' file, which is used for face detection.
-   An LBPH (Local Binary Patterns Histogram) Face Recognizer is created and loaded from 'face_trained.yml'.
-   An image ('3.jpeg') is read from the 'faces/val/srk' directory and preprocessed by converting it to grayscale.
-   Face detection is performed on the preprocessed image using the Haar Cascade classifier, and the regions of detected faces are stored in "faces_rect."
-   For each detected face, the face region is used for face recognition using the LBPH Face Recognizer.
-   The label (person's name) and the confidence (degree of similarity) are extracted from the recognizer's predictions.
-   The label and confidence are printed on the image, and a green rectangle is drawn around each detected face.
-   The modified image with labelled faces is displayed using OpenCV's "imshow" function.
-   The program waits for a key press, and if the 'x' key is pressed, the image window is closed using "destroyAllWindows."

### Result

![enter image description here](https://github.com/tanmaypradhan4112/opencv-Face-Detection/blob/main/result/custom1.png?raw=true)![enter image description here](https://github.com/tanmaypradhan4112/opencv-Face-Detection/blob/main/result/custom3.png?raw=true)

### Resources
https://github.com/opencv/opencv 



