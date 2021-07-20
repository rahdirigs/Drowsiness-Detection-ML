import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import sys
import joblib

sys.stdout = open("train_log.txt", "w+")

dataframe = pd.read_csv("normalised-data.csv", sep=",", header=None)


def calc_accuracy():
    cnt = 0
    for j in range(len(y_res)):
        if (y_res[j] == 5 and Y_test[j + train_idx] == 10) or (y_res[j] == 10 and Y_test[j + train_idx] == 5) or (y_res[j] == Y_test[j + train_idx]):
            cnt += 1
    return float(cnt / len(y_res))


train_percentage = 17 / 22
train_idx = int(len(dataframe) * train_percentage)
test_idx = len(dataframe) - train_idx

train_dataframe = dataframe[:train_idx]
test_dataframe = dataframe[-test_idx:]

X_train = train_dataframe.drop(4, axis=1)
Y_train = train_dataframe[4]

X_test = test_dataframe.drop(4, axis=1)
Y_test = test_dataframe[4]

accuracy = []
y_res = []
for i in range(1, 100):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, Y_train)
    y_res = model.predict(X_test)
    accuracy.append(calc_accuracy())

k_neighbors = accuracy.index(max(accuracy)) + 1
model = KNeighborsClassifier(n_neighbors=k_neighbors)
model.fit(X_train, Y_train)
y_res = model.predict(X_test)
print(y_res)
print('Max accuracy:', calc_accuracy())
print('k-value:', k_neighbors)
joblib.dump(model, "model.pkl")
