# We normaize the data for 

import csv
import numpy as np
from performance_measures import *

def readfile(file):
    l1=csv.reader(open(file))
    data=list(l1)
    data=data[1:len(data)]
    for i in range(len(data)):
        if data[i][0]=="HEALTHY":
            data[i][0]=0
        elif data[i][0]=="MEDICATION":
            data[i][0]=1
        elif data[i][0]=="SURGERY":
            data[i][0]=2
        else:
            data[i][0]=3
        data[i]=[float(x) for x in data[i]]
        a=data[i][0]
        data[i][0]=data[i][-1]
        data[i][-1]=a
    # normalization starts
    max=np.amax(data, axis=0)
    max[-1]=1
    for i in range(len(data)):
        data[i]=np.divide(data[i], max)
    # normalization ends
    # appending constant starts
    data1=np.ones([len(data), len(data[0])+1])
    for i in range(len(data1)):
        for j in range(len(data[0])):
            data1[i][j+1]=data[i][j]
    # appending constant endsq
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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradient_descent(x_train, y_train, num_iters, learning_rate, theta, corr_class):
    X = np.array(x_train)
    y_train=(y_train==corr_class)
    Y = np.array(y_train)
    parameter = theta
    while (num_iters > 0):
        parameter = step_gradient(X, Y, learning_rate, parameter)
        num_iters -= 1
    return parameter


def step_gradient(X, Y, learning_rate, parameter):
    # cost = -np.dot(Y.T, np.log(sigmoid(np.dot(X, parameter)))) - np.dot(
    #     (1 - Y).T, np.log(1 - sigmoid(np.dot(X, parameter))))
    # print(np.shape(parameter))
    # print(np.shape(temp))
    grad = np.dot(X.T, Y)[:,0] - np.dot(X.T, sigmoid(np.dot(X, parameter)))
    parameter += learning_rate * (1 / len(X)) * grad
    return parameter

def checker(test, wt, numclass):
    stats=np.zeros([numclass, numclass])
    val=np.zeros(numclass)
    for i in range(len(test)):        
        actual=test[i][-1]
        for j in range(numclass):
            val[j]=sigmoid(np.dot(test[i][:-1],wt[j]))
        predicted=np.argmax(val)
        stats[int(actual), int(predicted)]+=1
    acc=np.trace(stats)/np.sum(stats)*100
    return stats


def readTest(file):
    labels = {'HEALTHY': 0, 'MEDICATION': 1, 'SURGERY': 2}
    lines = csv.reader(open(file))
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
    return [x_test, y_test]

dataset=readfile("../Medical_data.csv")
[x_test, y_test]=readTest("../test_medical.csv")
ratio=0.7
[train_data, test_data]=setData(dataset, ratio)
numclass=3
learning_rate=0.1
num_iters=1000
x_train=np.zeros([len(train_data), len(train_data[0])-1])
y_train=np.zeros([len(train_data),1])
for i in range(len(x_train)):
    for j in range(len(x_train[0])):
        x_train[i][j]=train_data[i][j]
    y_train[i]=train_data[i][-1]
parameter=np.ones([numclass, len(x_train[0])])
for i in range(numclass):
    parameter[i]=gradient_descent(x_train, y_train, num_iters, learning_rate, parameter[i], i)
stats=checker(test_data, parameter, numclass)
acc=np.trace(stats)/np.sum(stats)*100
print(acc)
print(stats)
print(np.mean(recall(stats)))
print(np.mean(precision(stats)))