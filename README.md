# Face_recognition

# DATA
the train folder contains all the training images of the person we wnat to recognize
the test folder contains the testing images

# Content
Open-cv was used to detect faces using "haarcascade_frontalface"
the code consist of 4 functions
1. training data: to open the folder for providing the trainig images
2. Reshape : to detect the faces from the dataset reshape and resize it also to convert them into gray scale image for better recognition
3. train: To train the data once it is converted into suitable format and store the features of each img into an xml file using the LSTM algorithm
4. recognize: To recognize the person from the image using the classifier file.

The Tkinter library was used to provide the GUI interface
# Task 
To recognize the face of particular person and show the image heading as Matched or unkown face

