import numpy as np

data_file = open("data.csv", "r")
op_file = open("normalised-data.csv", "a")


def process():
    alert_user_ear = user_ear[:100]
    alert_user_mar = user_mar[:100]
    alert_user_circ = user_circ[:100]
    alert_user_ratio = user_ratio[:100]
    mean_ear = np.mean(alert_user_ear)
    mean_mar = np.mean(alert_user_mar)
    mean_circ = np.mean(alert_user_circ)
    mean_ratio = np.mean(alert_user_ratio)
    std_ear = np.std(user_ear)
    std_mar = np.std(user_mar)
    std_circ = np.std(user_circ)
    std_ratio = np.std(user_ratio)
    for k in range(len(user_ear)):
        user_ear[k] = (user_ear[k] - mean_ear) / std_ear
        user_mar[k] = (user_mar[k] - mean_mar) / std_mar
        user_circ[k] = (user_circ[k] - mean_circ) / std_circ
        user_ratio[k] = (user_ratio[k] - mean_ratio) / std_ratio
        normalised_ear.append(user_ear[k])
        normalised_mar.append(user_mar[k])
        normalised_circ.append(user_circ[k])
        normalised_ratio.append(user_ratio[k])
        y_val.append(user_yval[k])


normalised_ear = []
normalised_mar = []
normalised_circ = []
normalised_ratio = []
y_val = []

data = data_file.readlines()
for i in range(len(data)):
    data[i] = data[i].strip()
lines = []
for i in range(len(data)):
    line = data[i].split(", ")
    for j in range(len(line)):
        line[j] = float(line[j])
    lines.append(line)
print(len(lines))

user_ear = []
user_mar = []
user_circ = []
user_ratio = []
user_yval = []
zero = five = ten = False
for line in lines:
    if line[4] == float(0):
        if zero and five and ten:
            process()
            zero = five = ten = False
            user_ear.clear()
            user_mar.clear()
            user_circ.clear()
            user_yval.clear()
            user_ratio.clear()
        zero = True
    elif line[4] == float(5):
        five = True
    else:
        ten = True
    user_ear.append(line[0])
    user_mar.append(line[1])
    user_circ.append(line[2])
    user_ratio.append(line[3])
    user_yval.append(line[4])

process()
for i in range(len(lines)):
    op_string = '{:.18e}'.format(normalised_ear[i]) + ", " + '{:.18e}'.format(normalised_mar[i]) + ", " + '{:.18e}'.format(normalised_circ[i]) + ", " + '{:.18e}'.format(normalised_ratio[i]) + ", " + '{:.18e}'.format(y_val[i]) + "\n"
    op_file.write(op_string)
