import numpy as np
from mlxtend.image import extract_face_landmarks
from utilities import *
import cv2
import sys

sys.stdin = open("input.txt", "r")

path = "/Volumes/Samsung T5/PyCharmProjects/Drowsiness-Detection-ML/videos/"


def capture_frame(start_time, stamp, cap):
    startTime = int(start_time)
    cap.set(cv2.CAP_PROP_POS_MSEC, startTime + stamp * 1000)
    frame, img = cap.read()
    return frame, img


def export_csv():
    with open("features.csv", "ab") as file:
        np.savetxt(file, features, delimiter=", ")
    with open("labels.csv", "ab") as file:
        np.savetxt(file, labels, delimiter=", ")


def extract(participant, extension, level, start_time):
    print("Participant: " + str(participant) + " Activity level: " + str(level))
    vCap = cv2.VideoCapture(path + str(participant) + "/" + str(level) + "." + str(extension))
    timeStamp = 0
    frameRate = 1
    success, image = capture_frame(start_time, timeStamp, vCap)
    count = 0
    while success and count < 240:
        landmarks = extract_face_landmarks(image)
        if landmarks is None:
            timeStamp += frameRate
            timeStamp = round(timeStamp, 2)
            success, image = capture_frame(start_time, timeStamp, vCap)
            continue
        if (sum(sum(landmarks))) != 0:
            count += 1
            data.append(landmarks)
            labels.append(int(level))
            timeStamp += frameRate
            timeStamp = round(timeStamp, 2)
            success, image = capture_frame(start_time, timeStamp, vCap)
            print(count)
        else:
            timeStamp += frameRate
            timeStamp = round(timeStamp, 2)
            success, image = capture_frame(start_time, timeStamp, vCap)
            print("No frame detected!!!")


data = []
labels = []

while True:
    print("New video?")
    option = int(input())
    if option == 1:
        print("Enter details:")
        participant_, extension_ = list(input().split())
        start_time_ = 180000
        for i in [0, 5, 10]:
            extract(participant_, extension_, i, start_time_)
    else:
        break

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
