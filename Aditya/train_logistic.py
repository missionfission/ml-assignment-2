# We normaize the data for 

import csv
import numpy as np

def readfile(file):
    l1=csv.reader(open(file))
    data=list(l1)
    for i in range(1, len(data)):
        if data[i][4]=="FIRST_AC":
            data[i][4]=1
        elif data[i][4]=="SECOND_AC":
            data[i][4]=2
        elif data[i][4]=="THIRD_AC":
            data[i][4]=3
        else:
            data[i][4]=4
        if data[i][5]=="male":
            data[i][5]=0
        elif data[i][5]=="female":
            data[i][5]=1
        else:
            data[i][5]=2
        data[i]=[float(x) for x in data[i]]
        data[i]=data[i][1:len(data[i])]
        a=data[i][0]
        data[i][0]=data[i][-1]
        data[i][-1]=a
    data=data[1:len(data)]
    # normalization starts
    max=np.amax(data, axis=0)
    # print(np.size(max))
    for i in range(len(data)):
        data[i]=np.divide(data[i], max)
    # normalization ends
    # appending a constant starts
    data1=np.ones([len(data), len(data[0])+1])
    for i in range(len(data1)):
        for j in range(len(data[0])):
            data1[i][j+1]=data[i][j]
    # constant appende
    return data1

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

def gradient_descent(x_train, y_train, num_iters, learning_rate, theta):
    X = np.array(x_train)
    Y = np.array(y_train)
    parameter = theta
    while (num_iters > 0):
        parameter = step_gradient(X, Y, learning_rate, parameter)
        num_iters -= 1
    return parameter


def step_gradient(X, Y, learning_rate, parameter):
    temp = (sigmoid(np.dot(X, parameter)) - Y)
    temp = np.dot(X.T, temp)
    # cost = -np.dot(Y.T, np.log(sigmoid(np.dot(X, parameter)))) - np.dot(
    #     (1 - Y).T, np.log(1 - sigmoid(np.dot(X, parameter))))
    parameter -= learning_rate * (1 / len(X)) * temp
    return parameter

def checker(test, wt, threshold, numclass):
    stats=np.zeros([numclass, numclass])
    for i in range(len(test)):
        actual=test[i][-1]
        predicted=sigmoid(np.dot(test[i][:-1], wt))
        if(predicted>=threshold and actual==1):
            stats[0][0]+=1
        elif(predicted>=threshold and actual==0):
            stats[0][1]+=1
        elif(predicted<=threshold and actual==1):
            stats[1][0]+=1
        else:
            stats[1][1]+=1
    return stats



dataset=readfile("railwayBookingList.csv")
ratio=0.7
[train_data, test_data]=setData(dataset, ratio)
numclass=2
learning_rate=0.1
num_iters=1000
threshold=0.5
x_train=np.zeros([len(train_data), len(train_data[0])-1])
y_train=np.zeros([len(train_data),1])
for i in range(len(x_train)):
    for j in range(len(x_train[0])):
        x_train[i][j]=train_data[i][j]
    y_train[i]=train_data[i][-1]
parameter=np.ones([len(x_train[0]), len(y_train[0])])
parameter=gradient_descent(x_train, y_train, num_iters, learning_rate, parameter)
stats=checker(test_data, parameter, threshold, numclass)
acc=np.trace(stats)/np.sum(stats)*100
print(acc)
print(stats)