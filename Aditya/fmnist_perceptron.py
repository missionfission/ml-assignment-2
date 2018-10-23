import csv
import numpy as np
from mnist import MNIST
from numpy import linalg as la

def predict(data_row, row_label, wt, corr_class):
    # data_row is n*1 and wt is n-1*1
    # data_row contains actual value also
    actual=row_label
    # print(wt)
    # print(data_row[:-1])
    val=np.dot(np.transpose(data_row),wt)
    # val=val1[0]
    deltaw=np.zeros(len(data_row))
    if((val>0 and actual==corr_class) or (val<0 and actual!=corr_class)):
        deltaw=deltaw
    elif(val<=0 and actual==corr_class):
        deltaw=data_row
    else:
        deltaw=-data_row
    wt=wt+deltaw
    return wt

def percept(train_data, train_label, wt, corr_class):
    # print(len(train_data))
    for i in range(len(train_data)):
        # print(i)
        wt=predict(train_data[i], train_label[i], wt, corr_class)
    # acc_train=checker(train_data, wt, corr_class)
    # print("training accuracy=" + str(acc_train))
    return wt

def checker(test, test_label, wts, numclass):
    stats=np.zeros([numclass, numclass])
    val=np.zeros(numclass)
    for i in range(len(test)):        
        actual=test_label[i]
        for j in range(numclass):
            val[j]=np.dot(np.transpose(test[i]),wts[j])
        predicted=np.argmax(val)
        stats[int(actual), int(predicted)]+=1
    acc=np.trace(stats)/np.sum(stats)*100
    return stats

def shorten(egvec, train_img, test_img, num_features_new):
    egvec=egvec[:, -num_features_new:len(egvec)]
    train_img=np.dot(train_img, egvec)
    test_img=np.dot(test_img, egvec)
    return [train_img, test_img]


def readfile(filename):
    fsdata=MNIST(filename)
    [train_img, train_label]=fsdata.load_training()
    [test_img, test_label]=fsdata.load_testing()
    for i in range(len(train_img)):
        train_img[i]=list(train_img[i])
    train_label=list(train_label)
    for i in range(len(test_img)):
        test_img[i]=list(test_img[i])
    test_label=list(test_label)
    mu=np.mean(train_img, axis=0, dtype=np.float64)
    x=train_img-mu
    cov=np.dot(np.transpose(x), x)
    cov=np.divide(cov, len(x))
    [egval, egvec]=la.eig(cov)
    idx=np.argsort(egval)
    egval=egval[idx]
    egvec=egvec[:, idx]
    return [egvec, train_img, test_img, train_label, test_label]


[egvec, train_data_full, test_data_full, train_label, test_label]=readfile("fashion1")
num_features_new=50
numiter=5
numclass=10
while num_features_new<784:
    print(num_features_new)
    [train_data, test_data]=shorten(egvec, train_data_full, test_data_full, num_features_new)
    # normalization starts
    train_max=np.amax(train_data, axis=0)
    for i in range(len(train_data)):
        train_data[i]=np.divide(train_data[i], train_max)
    test_max=np.amax(test_data, axis=0)
    for i in range(len(test_data)):
        test_data[i]=np.divide(test_data[i], test_max)
    # normalization ends
    # appending constant starts
    train_data1=np.ones([len(train_data), len(train_data[0])+1])
    for i in range(len(train_data1)):
        for j in range(len(train_data[0])):
            train_data1[i][j+1]=train_data[i][j]
    train_data=train_data1
    test_data1=np.ones([len(test_data), len(test_data[0])+1])
    for i in range(len(test_data1)):
        for j in range(len(test_data[0])):
            test_data1[i][j+1]=test_data[i][j]
    test_data=test_data1
    # appending constant ends
    wts=np.zeros([numclass, len(train_data[0])])
    for k in range(numclass):
        for j in range(numiter):
            # print(wt)
            wts[k]=percept(train_data, train_label, wts[k], k)
    stats=checker(test_data, test_label, wts, numclass)
    acc=np.trace(stats)/np.sum(stats)*100
    print("acc value is: "+str(acc))
    print(stats)
    # print(acc)
    # f1.write(str(acc))
    # f1.write("\n")
    num_features_new+=10