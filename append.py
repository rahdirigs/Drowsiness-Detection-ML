import csv

labels = []
label_file = open("labels.csv", "r")
label_lines = label_file.readlines()
for line in label_lines:
    labels.append(float(line))

idx = 0
with open("features.csv", "r") as feature_file, open("data.csv", "w", newline='') as write_file:
    csv_reader = csv.reader(feature_file)
    csv_writer = csv.writer(write_file)
    for row in csv_reader:
        row.append(' {:.18e}'.format(labels[idx]))
        csv_writer.writerow(row)
        idx += 1
