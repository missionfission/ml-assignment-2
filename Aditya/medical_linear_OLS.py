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


def OLS(x_train, y_train, x_test, y_test, numclass):
    X = np.array(x_train)
    Y = np.array(y_train)
    beta = np.linalg.inv(np.dot(X.T, X))
    beta = np.dot(beta, np.dot(X.T, Y))
    print(beta)
    predicted = np.dot(np.array(x_test), beta)
    correct = 0
    f_predicted = [None] * len(predicted)
    # for i in range(len(predicted)):
    #     f_predicted[i] = np.argmax(predicted[i])
    #     if (f_predicted[i] - y_test[i] < 0.5
    #             and f_predicted[i] - y_test[i] > -0.5):
    #         correct += 1
    stats=np.zeros([numclass, numclass])
    for i in range(len(predicted)):
        f_predicted[i]=np.argmax(predicted[i])
        # print(str(f_predicted[i]))
        stats[int(f_predicted[i])][int(y_test[i])]+=1
    return stats
    # return float(100 * correct / len(y_test))

numclass=3
stats = OLS(x_train, y_train, x_test, y_test, numclass)
print(np.trace(stats)/np.sum(stats)*100)
print(stats)