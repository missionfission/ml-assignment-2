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

def predict(data_row, wt):
    # data_row is n*1 and wt is n-1*1
    # data_row contains actual value also
    actual=data_row[-1]
    # print(wt)
    # print(data_row[:-1])
    val=np.dot(np.transpose(data_row[:-1]),wt)
    # val=val1[0]
    deltaw=np.zeros(len(data_row)-1)
    if((val>0 and actual==1) or (val<0 and actual==0)):
        deltaw=deltaw
    elif(val<=0 and actual==1):
        deltaw=data_row[:-1]
    else:
        deltaw=-data_row[:-1]
    wt=wt+deltaw
    return wt

def percept(train_data, wt):
    # print(len(train_data))
    for i in range(len(train_data)):
        # print(i)
        wt=predict(train_data[i], wt)
    return wt

def checker(test, wt):
    correct=0
    wrong=0
    for i in range(len(test)):        
        actual=test[i][-1]
        val=np.dot(np.transpose(test[i][:-1]),wt)
        if(val>=0 and actual==1):
            correct+=1
        elif(val<0 and actual==0):
            correct+=1
        else:
            wrong+=1
    acc=correct/(correct+wrong)*100
    return acc





data=readfile("railwayBookingList.csv")
ratio=0.6
acc1=np.zeros(100)
numclass=2
numiter=1
while ratio<0.8:
    # print(ratio*1000)
#     ratio=0.4
    [train, test]=setData(data, ratio)
    # print(m)
    # print(std)
    wt=np.ones(np.size(train[0])-1)
    acc_train=0
    for i in range(numiter):
        # print(wt)
        wt=percept(train, wt)

    acc=checker(test, wt)
    # print(acc)
    b1=int(ratio*100)
    acc1[b1]=acc
    ratio+=0.01
a1=np.argmax(acc1)
print(acc1[a1])
a1=a1/100
print(a1)