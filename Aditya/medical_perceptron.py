# Multi-Class Problem
# One-vs.-rest (or one-vs.-all) strategy involves training a single classifier per class,
# with the samples of that class as positive samples and all other samples as negatives.
import csv
import numpy as np


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

def predict(data_row, wt, corr_class):
    # data_row is n*1 and wt is n-1*1
    # data_row contains actual value also
    actual=data_row[-1]
    # print(wt)
    # print(data_row[:-1])
    val=np.dot(np.transpose(data_row[:-1]),wt)
    # val=val1[0]
    deltaw=np.zeros(len(data_row)-1)
    if((val>0 and actual==corr_class) or (val<0 and actual!=corr_class)):
        deltaw=deltaw
    elif(val<=0 and actual==corr_class):
        deltaw=data_row[:-1]
    else:
        deltaw=-data_row[:-1]
    wt=wt+deltaw
    return wt

def percept(train_data, wt, corr_class):
    # print(len(train_data))
    for i in range(len(train_data)):
        # print(i)
        wt=predict(train_data[i], wt, corr_class)
    # acc_train=checker(train_data, wt, corr_class)
    # print("training accuracy=" + str(acc_train))
    return wt

def checker(test, wts, numclass):
    stats=np.zeros([numclass, numclass])
    val=np.zeros(numclass)
    for i in range(len(test)):        
        actual=test[i][-1]
        for j in range(numclass):
            val[j]=np.dot(np.transpose(test[i][:-1]),wts[j])
        predicted=np.argmax(val)
        stats[int(actual), int(predicted)]+=1
    acc=np.trace(stats)/np.sum(stats)*100
    return stats

data=readfile("Medical_data.csv")
ratio=0.65
numclass=3
numiter=10
[train, test]=setData(data, ratio)
wts=np.zeros([numclass, len(data[0])-1])
for i in range(numclass):
    acc_train=0
    for j in range(numiter):
        # print(wt)
        wts[i]=percept(train, wts[i], i)
stats=checker(test, wts, numclass)
accc=(np.trace(stats)/np.sum(stats)*100)
print(accc)
print(stats)


# data=readfile("Medical_data.csv")
# ratio=0.6
# acc1=np.zeros(100)
# numclass=3
# numiter=10
# acc1=np.zeros(100)
# while(ratio<0.8):
#     [train, test]=setData(data, ratio)
#     wts=np.zeros([numclass, len(data[0])-1])
#     for i in range(numclass):
#         acc_train=0
#         for j in range(numiter):
#             # print(wt)
#             wts[i]=percept(train, wts[i], i)
#     stats=checker(test, wts, numclass)
#     accc=(np.trace(stats)/np.sum(stats)*100)
#     # print(stats)
#     acc1[int(ratio*100)]=accc
#     ratio+=0.01
# print(np.max(acc1))