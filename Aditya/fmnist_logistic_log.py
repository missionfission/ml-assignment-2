import csv
import numpy as np
# from mnist import MNIST
from numpy import linalg as la
from utils import mnist_reader
from performance_measures import *
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradient_descent(x_train, y_train, num_iters, learning_rate, theta, corr_class):
    X = np.array(x_train)
    # y_train=(y_train==corr_class)
    actual=np.zeros(len(y_train))
    for i in range(len(y_train)):
        actual[i]=int(train_label[i]==corr_class)
    # print(a1)
    # for i in range(25):
    #     print(str(actual[i])+" "+str(y_train[i]))
    # for i in range(len(y_train)):
    #     if(y_train[i]==corr_class):
    #         y_train[i]=1
    #     else:
    #         y_train[i]=0
    # Y = np.array(y_train)
    Y = np.array(actual)
    parameter = theta
    while (num_iters > 0):
        parameter = step_gradient(X, Y, learning_rate, parameter)
        num_iters -= 1
    return parameter


def step_gradient(X, Y, learning_rate, parameter):
    # print(np.shape(X))
    # print(np.shape(Y))
    # print(np.shape(sigmoid(np.dot(X, parameter))))
    # cost = -np.dot(Y.T, np.log(sigmoid(np.dot(X, parameter)))) - np.dot(
    #     (1 - Y).T, np.log(1 - sigmoid(np.dot(X, parameter))))

    grad = np.dot(X.T, Y) - np.dot(X.T, sigmoid(np.dot(X, parameter)))
    parameter += learning_rate * (1 / len(X)) * grad
    return parameter

def checker(test, label,  wt, numclass):
    stats=np.zeros([numclass, numclass])
    val=np.zeros(numclass)
    for i in range(len(test)):        
        actual=label[i]
        for j in range(numclass):
            val[j]=sigmoid(np.dot(test[i],wt[j]))
        predicted=np.argmax(val)
        stats[int(actual), int(predicted)]+=1
    acc=np.trace(stats)/np.sum(stats)*100
    return stats

def shorten(egvec, train_img, test_img, num_features_new):
    egvec=egvec[:, -num_features_new:len(egvec)]
    train_img=np.dot(train_img, egvec)
    test_img=np.dot(test_img, egvec)
    return [train_img, test_img]

def readfile(dir):
    train_img, train_label = mnist_reader.load_mnist(path=dir, kind='train')
    test_img, test_label = mnist_reader.load_mnist(path=dir, kind='t10k')
    mu=np.mean(train_img, axis=0, dtype=np.float64)
    x=train_img-mu
    cov=np.dot(np.transpose(x), x)
    cov=np.divide(cov, len(x))
    [egval, egvec]=la.eig(cov)
    idx=np.argsort(egval)
    egval=egval[idx]
    egvec=egvec[:, idx]
    return [egvec, train_img, test_img, train_label, test_label]

DATA_DIR='../data/fashion/'
[egvec, train_data_full, test_data_full, train_label, test_label]=readfile(DATA_DIR)
num_features_new=60
num_iters=1000
numclass=10
learning_rate=10
while num_features_new<62:
    print(num_features_new)
    [train_data, test_data]=shorten(egvec, train_data_full, test_data_full, num_features_new)
    # print(train_label)
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
    parameter=np.ones([numclass, len(train_data[0])])
    for i in range(numclass):
        parameter[i]=gradient_descent(train_data, train_label, num_iters, learning_rate, parameter[i], i)
    # print(parameter)
    stats=checker(test_data, test_label, parameter, numclass)
    acc=np.trace(stats)/np.sum(stats)*100
    print(acc)
    print(stats)
    num_features_new+=5
    print(np.mean(recall(stats)))
    print(np.mean(precision(stats)))