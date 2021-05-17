import cv2
import numpy as np
import dlib
from mlxtend.image import extract_face_landmarks
from utilities import *


def captureFrame(stamp):
    startTime = 180000
    videoCapture.set(cv2.CAP_PROP_POS_MSEC, startTime + stamp * 1000)
    frame, img = videoCapture.read()
    return frame, img


def export_csv():
    np.savetxt("features.csv", features, delimiter=",")
    np.savetxt("labels.csv", labels, delimiter=",")


data = []
labels = []

for j in [60]:
    for i in [10]:
        videoCapture = cv2.VideoCapture("/videos/Fold1_part2/" + str(j) + "/" + str(i) + ".mp4")
        timeStamp = 0
        frameRate = 1
        success, image = captureFrame(timeStamp)
        count = 0
        while success and count < 240:
            landmarks = extract_face_landmarks(image)
            if sum(sum(landmarks)) != 0:
                count += 1
                data.append(landmarks)
                labels.append([i])
                timeStamp += frameRate
                timeStamp = round(timeStamp, 2)
                success, image = captureFrame(timeStamp)
            else:
                timeStamp += frameRate
                timeStamp = round(timeStamp, 2)
                success, image = captureFrame(timeStamp)
                print("No frame detected!!!")

data = np.array(data)
labels = np.array(labels)
features = []

for dat in data:
    face = dat[36:68]
    eye_aspect_ratio = ear(face)
    mouth_aspect_ratio = mar(face)
    circ = circularity(face)
    mouth_eye_ratio = mouth_to_eye_ratio(face)
    features.append([eye_aspect_ratio, mouth_aspect_ratio, circ, mouth_eye_ratio])

features = np.array(features)
print(features.shape)
print(data.shape)
print(labels.shape)

export_csv()
