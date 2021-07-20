import joblib
from utilities import *
from imutils import face_utils
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

knn = joblib.load("model.pkl")
pred = "/Users/rect0r/mlxtend_data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pred)
ears = []
mars = []
circs = []
ratios = []
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 400)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2


def calibrate():
    data = []
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        if count == 200:
            break
        _, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(image, 0)
        count += 1
        print(count)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            data.append(shape)
            cv2.putText(image, "Calibrating...", bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.imshow("Output", image)
        cv2.waitKey(5)

    cv2.destroyAllWindows()
    cap.release()

    base_features = []
    for d in data:
        face = d[36:68]
        ear_value = ear(face)
        mar_value = mar(face)
        circ_value = circularity(face)
        mouth_eye_ratio = mouth_to_eye_ratio(face)
        ears.append(ear_value)
        mars.append(mar_value)
        circs.append(circ_value)
        ratios.append(mouth_eye_ratio)
        base_features.append([ear_value, mar_value, circ_value, mouth_eye_ratio])
    print(base_features)


def get_result(shape):
    face = shape[36:68]
    ear_value = (ear(face) - mean_ear) / std_ear
    mar_value = (mar(face) - mean_ear) / std_mar
    circ = (circularity(face) - mean_circ) / std_circ
    ratio = (mouth_to_eye_ratio(face) - mean_ratio) / std_ratio
    feature = [[ear_value, mar_value, circ, ratio]]
    _result = knn.predict(feature)
    print(_result)
    res_str = "Alert"
    if _result[0] == 10:
        res_str = "Drowsy"
    return res_str, [ear_value, mar_value, circ, ratio]


def live_demo():
    cap = cv2.VideoCapture(0)
    data = []
    result = []
    count = 0
    while True:
        if count == 200:
            break
        _, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(image, 0)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            result_val, features = get_result(shape)
            count += 1
            print(count)
            data.append(features)
            result.append(result_val)
            cv2.putText(image, result_val, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.imshow("Output", image)
        cv2.waitKey(300)

    cv2.destroyAllWindows()
    cap.release()
    return data, result


calibrate()
mean_ear = np.mean(ears)
mean_mar = np.mean(mars)
mean_circ = np.mean(circs)
mean_ratio = np.mean(ratios)
std_ear = np.std(ears)
std_mar = np.std(mars)
std_circ = np.std(circs)
std_ratio = np.std(ratios)

print(mean_ear, mean_mar, mean_circ, mean_ratio)
print(std_ear, std_mar, std_circ, std_ratio)

feats, res = live_demo()

plt.title("Results")
plt.xlabel("Frames")
plt.ylabel("State")
plt.plot(res, color="orange")
plt.show()
