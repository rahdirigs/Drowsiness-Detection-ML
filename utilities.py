from scipy.spatial import distance
import math


def ear(eye):
    param1 = distance.euclidean(eye[1], eye[5])
    param2 = distance.euclidean(eye[2], eye[4])
    param3 = distance.euclidean(eye[0], eye[3])
    ear_value = (param1 + param2) / (2.0 * param3)
    return ear_value


def mar(mouth):
    param1 = distance.euclidean(mouth[14], mouth[18])
    param2 = distance.euclidean(mouth[12], mouth[16])
    mar_value = param1 / param2
    return mar_value


def circularity(eye):
    param1 = distance.euclidean(eye[1], eye[4])
    radius = param1 / 2.0
    area = math.pi * (radius ** 2)
    circ_val = 0
    circ_val += distance.euclidean(eye[0], eye[1])
    circ_val += distance.euclidean(eye[1], eye[2])
    circ_val += distance.euclidean(eye[2], eye[3])
    circ_val += distance.euclidean(eye[3], eye[4])
    circ_val += distance.euclidean(eye[4], eye[5])
    circ_val += distance.euclidean(eye[5], eye[0])
    return 4 * math.pi * area / (circ_val ** 2)


def mouth_to_eye_ratio(face):
    ear_value = ear(face)
    mar_value = mar(face)
    mouth_to_eye_ratio_val = mar_value / ear_value
    return mouth_to_eye_ratio_val
