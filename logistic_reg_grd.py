import csv
import numpy as np
import math



labels = {'HEALTHY': 0, 'MEDICATION': 1, 'SURGERY': 2}
lines = csv.reader(open("./Medical_data.csv"))
dataset = list(lines)
del dataset[0]
trainData = [dataset[index] for index in range(1, int(len(dataset)))]
x_train = [None] * len(dataset)
y_train = np.zeros((len(dataset), 3), dtype=float)
for index in range(len(dataset)):
    x_train[index] = [
        1.0,
        float(dataset[index][1]),
        float(dataset[index][2]),
        float(dataset[index][3])
    ]
    y_train[index][labels[dataset[index][0]]] = 1.0

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

learning_rate = 1
num_iters = 1000
theta = np.zeros((len(x_train[0]), len(y_train[0])), dtype=float)
theta = gradient_descent(x_train, y_train, num_iters, learning_rate, theta)
predicted = np.dot(np.array(x_test), theta)
correct = 0
f_predicted = [None] * len(predicted)
for i in range(len(predicted)):
    f_predicted[i] = np.argmax(predicted[i])
    if (f_predicted[i] - y_test[i] < 0.5
            and f_predicted[i] - y_test[i] > -0.5):
        correct += 1

print(float(100 * correct / len(y_test)))
print(theta)
