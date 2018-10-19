import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

Data =[]
target=[] #Y values
data=[] #X values
X=[]
Y=[[0 for x in range(0,40)] for y in range(0,400)]

def Reading_the_Data():
    DataSet=raw_input("Please enter the location of Atnt dataset to be classified")
    with open(DataSet,'r') as dataset:
        for line in dataset:
            line=line.strip()
            line=line.split(",")
            Data.append(line)

    #since we have target as first row and remaining column as data in the text file.
    #So we transform the dataset such that: data is a matrix of dimension 644*400
    Transposed_Data=np.transpose(Data)

    #Extracting X and Y from the transposed matrix
    for row in(Transposed_Data):
        data.append(row[1:]) #X values
        target.append(int(row[0])) #Y values
    #changing array into list
    data_list=list(data)

    #changing string values of X into Float plus addign 1 to each row
    for row in data_list:
        X.append([float(1)]+[float(i) for i in row])

    #changing Y matrix into matrix of dimension(400*40)
    for i in range (0,400):
        j=target[i]-1
        Y[i][j]=1
    #return X and Y
    return  X,Y


def K_fold(X):
    kf = KFold(n_splits=5)
    return kf.split(X)

def Linear_regression_classification(validation,X,Y):
    accuracy=[]
    Score=[]
    for train,test in validation:  # k values
    # we create instance of Neighbours Classifier and fit the data
        X_train=np.array([X[i] for i in train])
        Y_train=np.array([Y[i] for i in train])
        X_test = ([X[i]for i in test])
        Y_test = ([Y[i] for i in test] )

        Beta=np.dot(np.linalg.pinv(X_train),(Y_train))
        Beta = np.transpose(Beta)
        y_pred = [[0 for x in range(0, 40)] for y in range(0, 80)]

        for i in range(len(X_test)):
            for j in range(len(Beta)):
                for k in range(len(Beta[0])):
                    y_pred[i][j] = Beta[j][k] * X_test[i][k]

        index=[]
        index2=[]
        for i in range(len(y_pred)):
            maximum = max(y_pred[i])
            index.append(y_pred[i].index(maximum))

        for j in range(len(Y_test)):
            for i in range(len(Y_test[0])):
                if Y_test[j][i] == 1:
                    index2.append(j)
        Score.append(accuracy_score(index2 , index))

    for i in range (len(Score)):
        print "The accuracy of the Linear classification method for ATnT is",Score[i]*100,"for",i,"fold!"



if __name__ == '__main__':
    X_Y=Reading_the_Data()
    Train_Test=K_fold(X_Y[0])
    Linear_regression_classification(Train_Test,X_Y[0],X_Y[1])