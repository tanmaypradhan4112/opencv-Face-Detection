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
-   The modified image with labeled faces is displayed using OpenCV's "imshow" function.
-   The program waits for a key press, and if the 'x' key is pressed, the image window is closed using "destroyAllWindows."
![enter image description here](https://github.com/tanmaypradhan4112/opencv-Face-Detection/blob/main/result/custom1.png?raw=true)![enter image description here](https://github.com/tanmaypradhan4112/opencv-Face-Detection/blob/main/result/custom3.png?raw=true)
