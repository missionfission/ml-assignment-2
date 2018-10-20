import numpy as np
import math
import csv
import matplotlib.pyplot as plt

labels = {'HEALTHY': 1, 'MEDICATION': 2, 'SURGERY': 3}
lines = csv.reader(open("./Medical_data.csv"))
dataset = list(lines)
del dataset[0]
trainData = [dataset[index] for index in range(1, int(len(dataset)))]
x_train = [None] * len(dataset)
y_train = [None] * len(dataset)
for index in range(len(dataset)):
    x_train[index] = [
        1.0,
        float(dataset[index][1]),
        float(dataset[index][2]),
        float(dataset[index][3])
    ]
    y_train[index] = labels[dataset[index][0]]

lines = csv.reader(open("./test_medical.csv"))
dataset = list(lines)
del dataset[0]
testData = [dataset[index] for index in range(1, int(len(dataset)))]
x_test = [None] * len(dataset)
y_test = [None] * len(dataset)
for index in range(len(dataset)):
    x_test[index] = [
        1.0,
        float(dataset[index][1]),
        float(dataset[index][2]),
        float(dataset[index][3])
    ]
    y_test[index] = labels[dataset[index][0]]
