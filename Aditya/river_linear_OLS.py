import csv
import numpy as np
import math
import matplotlib.pyplot as plt

def readfile(file):
    l1=csv.reader(open(file))
    data=list(l1)
    for i in range(1, len(data)):
        data[i]=[float(x) for x in data[i]]
    data=data[1:len(data)]
    # appending a constant starts
    data1=np.ones([len(data), len(data[0])+1])
    for i in range(len(data1)):
        for j in range(len(data[0])):
            data1[i][j+1]=data[i][j]
    data=data1
    # constant appende
    return data

import random
def setData(data, ratio):
    n1=len(data)
    train_size=int(ratio*n1)
    traindata=[]
    test=list(data)
    while len(traindata)<train_size:
        i=random.randrange(len(test))
        traindata.append(test.pop(i))
    return [traindata, test]


def OLS(x_train, y_train):
    X = np.array(x_train)
    Y = np.array(y_train)
    beta = np.linalg.pinv(np.dot(X.T, X))
    beta = np.dot(beta, np.dot(X.T, Y))
    plt.plot(x_train[0:100], y_train[0:100], "ro")
    print(beta)
    return beta

def getLoss(x_test, y_test, beta):
    plt.plot(x_test[0:100], y_test[0:100], "bo")
    predicted=np.dot(x_test, beta)
    # print(np.shape(predicted))
    plt.plot(x_test[0:100], predicted[0:100], "g")
    plt.show()
    loss=np.sum(np.square(predicted-y_test))/(len(x_test))
    return loss

dataset=readfile("Assignment2Data_4.csv")
ratio=0.7
[train_data, test_data]=setData(dataset, ratio)
x_train=np.zeros([len(train_data), len(train_data[0])-1])
x_test=np.zeros([len(test_data), len(test_data[0])-1])
y_train=np.zeros([len(train_data), 1])
y_test=np.zeros([len(test_data), 1])
for i in range(len(train_data)):
    y_train[i]=train_data[i][-1]
    x_train[i]=train_data[i][:-1]
for i in range(len(test_data)):
    y_test[i]=test_data[i][-1]
    x_test[i]=test_data[i][:-1]
beta = OLS(x_train, y_train)
loss=getLoss(x_test, y_test, beta)
print(loss)