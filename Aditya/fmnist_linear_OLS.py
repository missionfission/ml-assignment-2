import csv
import numpy as np
from mnist import MNIST
from numpy import linalg as la


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
        stats[int(y_test[i])][int(f_predicted[i])]+=1
    return stats
    # return float(100 * correct / len(y_test))

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
num_features_new=35
numclass=10
train_label_new=np.zeros([len(train_label), numclass])
for i in range(len(train_label_new)):
    train_label_new[i][int(train_label[i])]=1.0

train_label=train_label_new
while num_features_new<784:
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
    stats = OLS(train_data, train_label, test_data, test_label, numclass)
    print(np.trace(stats)/np.sum(stats)*100)
    print(stats)
    num_features_new+=5