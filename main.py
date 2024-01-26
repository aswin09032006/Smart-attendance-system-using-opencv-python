import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Images'
images = []  # image files from the Images directory is stored in the images list
classNames = []  # extract the filenames without the extensions and store them in the classNames list
myList = os.listdir(
    path)  # The os.listdir(path) function in Python is used to list the contents of a directory. It takes a single argument, path, which is the path to the directory you want to list. The function returns a list of strings, where each string is the name of a file or directory in the specified directory.

for cl in myList:  # For each iteration of the loop, the current value of cl will be the name of one of the image files.
    curImg = cv2.imread(
        f'{path}/{cl}')  # The file path is constructed using the f-string syntax, which concatenates the path variable and the cl variable.
    images.append(curImg)  # images list is a list of all the images that were read from the Images directory
    classNames.append(os.path.splitext(cl)[
                          0])  # This line extracts the filename without the extension from the current value of cl and stores it in the classNames list. The os.path.splitext() function from the os library splits the filename into a filename part and a suffix part. The [0] index of the result (filename) is the filename without the extension, which is what is stored in the classNames list.


def findEncodings(images):
    encodeList = []  # encodeList is a list to store the extracted face encodings.
    for img in images:
        #  OpenCV's library loads images in BGR (Blue-Green-Red) color space by default.
        #  BGR color space is slightly more efficient to store and process than RGB color space, as it places the blue channel first, which is typically the least significant channel for human perception.
        img = cv2.cvtColor(img,
                           cv2.COLOR_BGR2RGB)  # cv2.cvtColor function converts the image from BGR color space to RGB, which is required for face recognition.
        encode = face_recognition.face_encodings(img)[
            0]  # face_recognition.face_encodings(img) extracts face encodings from the current image.
        # [0] selects the first encoding from the list, assuming there is only one face in the image
        encodeList.append(encode)  # Adds the extracted face encoding to the encodeList.
    return encodeList  # Returns the encodeList, containing all the extracted face encodings.


def markAttendance(name):
    with open('Attendance.csv',
              'r+') as f:  # This line of code opens a file named Attendance.csv for reading and writing using a context manager. The context manager ensures that the file is properly closed after the operations are complete, even if an exception occurs
        #  r+ allows you to both read and write to the file  # the opened file is assigned to the variable 'f', we can use 'f' to perform operations such as reading or writing
        myDataList = f.readlines()  # f.readlines() reads all lines from the file and stores them in a list.
        # myDataList is a list containing each line of the attendance data.
        nameList = []  # nameList is an empty list to store the extracted names.
        for line in myDataList:
            entry = line.split(',')
            nameList.append(
                entry[0])  # nameList.append(entry[0]) appends the first entry (name) from each line to the nameList
        if name not in nameList:  # Checks if the provided name is not already present in the nameList
            now = datetime.now()  # now = datetime.now() gets the current time and date.
            dtString = now.strftime('%H:%M:%S')  # formats the current time to a string in HH:MM:SS format
            f.writelines(f'\n{name},{dtString}')  # writes a new line to the file with the format 'name,timestamp'.


encodeListKnown = findEncodings(
    images)  # This line calls the findEncodings() function to extract face encodings from the list of images stored in the images variable
# The result of the findEncodings() function, which is a list of face encodings, is assigned to the variable encodeListKnown
print('Encoding Complete')  # prints a message to indicate that the encoding process is complete

cap = cv2.VideoCapture(0)  # 0 for using default webcam device

while True:
    success, img = cap.read()
    # success is a boolean value indicating whether the frame was successfully captured
    # img is the captured frame itself, represented as a NumPy array of image pixels.
    imgS = cv2.resize(img, (0, 0), None, 0.25,
                      0.25)  # This line of code resizes the captured frame (img) from the webcam to a smaller size
    # img represents the original image captured from the webcam
    # (0, 0) indicates that the aspect ratio of the resized image should be maintained
    # None specifies that the interpolation method should be chosen automatically by OpenCV based on the image type and the scaling factor.
    # interpolation methods are techniques used to estimate the values of pixels between known pixels
    # 0.25, 0.25 are the scaling factors for the width and height of the resized image, respectively. In this case, the image is resized to 25% of its original size in both dimensions
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # converts the resized image from BGR color space to RGB color space

    facesCurFrame = face_recognition.face_locations(
        imgS)  # face_recognition.face_locations() function is used to detect faces in an image
    encodesCurFrame = face_recognition.face_encodings(imgS,
                                                      facesCurFrame)  # This line extracts face encodings from each detected face in the current frame and stores them in encodesCurFrame as a list

    for encodeFace, faceLoc in zip(encodesCurFrame,
                                   facesCurFrame):  # The zip() function takes two or more iterables and creates a single iterator of tuples
        matches = face_recognition.compare_faces(encodeListKnown,
                                                 encodeFace)  # This line compares the current face encoding to all the known face encodings stored in encodeListKnown and stores the value in matches as a list of boolean value
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # encodeListKnown: A list of known face encodings, each represented as a 128-dimensional NumPy array. These encodings represent the facial features of the known individuals
        # encodeFace: A 128-dimensional NumPy array representing the facial features of the face to be compared. This encoding is typically extracted from a new image or frame
        # The function utilizes a technique called cosine similarity to measure the distance between facial encodings. Cosine similarity is a measure of similarity between two vectors, and it ranges from 0 to 1. A higher value indicates greater similarity, while a lower value indicates less similarity.
        # In the context of face recognition, the cosine similarity between two face encodings represents how closely the facial features of the two individuals match. A high cosine similarity suggests that the two faces are likely the same person, while a low cosine similarity suggests that they are different people.
        # faceDis is a list of facial distances, where each distance corresponds to a comparison with a known face.
        matchIndex = np.argmin(
            faceDis)  # The argmin() function takes a NumPy array as input and returns the index of the minimum element in the array.
        # Identifying the Closest Match: It determines the index of the minimum value in the faceDis list, which represents the facial distances between the current face encoding and all the known face encodings. This index corresponds to the known face encoding that has the smallest facial distance, indicating the closest match among the known faces.

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)