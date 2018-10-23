import csv
import numpy as np
import math

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

dataset=readfile("railwayBookingList.csv")
ratio=0.7
[train_data, test_data]=setData(dataset, ratio)
numclass=2
x_train=np.zeros([len(train_data), len(train_data[0])-1])
x_test=np.zeros([len(test_data), len(test_data[0])-1])
y_train=np.zeros([len(train_data), numclass])
y_test=np.zeros([len(test_data), 1])
for i in range(len(train_data)):
    y_train[i][int(train_data[i][-1])]=1.0
    x_train[i]=train_data[i][:-1]
for i in range(len(test_data)):
    y_test[i]=test_data[i][-1]
    x_test[i]=test_data[i][:-1]
stats = OLS(x_train, y_train, x_test, y_test, numclass)
print(np.trace(stats)/np.sum(stats)*100)
print(stats)