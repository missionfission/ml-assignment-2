import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

Data =[]
target=[] #Y values
data=[] #X values
X=[]
p=100
#Y=[[0 for x in range(0,40)] for y in range(0,400)]
def Reading_the_Data():
    DataSet=raw_input("Please enter the location of Atnt dataset to be classified: ")
    with open(DataSet,'r') as dataset:
        for line in dataset:
            line=line.strip()
            line=line.split(",")
            Data.append(line)

    '''since we have target as first row and remaining column as data in the text file.
    So we transform the dataset such that: data is a matrix of dimension 644*400 and target is matrix of dimension 400*1'''
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



def Cross_Validation(data,target):
    j=0
    classes_X=[]
    classes_Y=[]
    Y_test=[]
    Y_train=[]
    X_test = []
    X_train = []
    #diving data into classes
    for i in range(1,41):
        classes_X.append(data[j:i*10])
        classes_Y.append(target[j:i*10])
        j=i*10
    # classes_X, classes_y = [list(10 vectors), list(10 vectors), ...], [list(10 values), list(10 values) ...]
    # len(classes_Y) = 40
    for j in range (0,5):  # j values for cross validation iteration
        Y_train_eachK = []
        Y_test_eachK = []
        X_train_eachK = []
        X_test_eachK = []
        for i in range(0,len(classes_Y)):
            Y_test_eachK+=(classes_Y[i][j:j+2])  # (0, 0)
            X_test_eachK+=(classes_X[i][j:j+2])  # (vector0, vector1)
            Y_train_eachK+=(classes_Y[i][j+2:])  # (0)*8 times
            X_train_eachK+=(classes_X[i][j+2:])  # (vector2, vector3 .. vector9]
            if j>0:
                Y_train_eachK += (classes_Y[i][0:j])
                X_train_eachK += (classes_X[i][0:j])
        Y_test.append(Y_test_eachK)
        Y_train.append(Y_train_eachK)
        X_test.append(X_test_eachK)
        X_train.append(X_train_eachK)
    return X_test,Y_test,X_train,Y_train

def Linear_regression_classification(X_Y):
    Score=[]
    count=0
    for i in range (0,5):  # k values
    # we create instance of Neighbours Classifier and fit the data
        X_train=np.array(X_Y[2][i])
        Y_train=np.array(X_Y[3][i])
        X_test = (X_Y[0][i])
        Y_test = (X_Y[1][i])
        correct_value_Y=0
        #print len(X_train),len(X_train[0]),len(X_test),len(X_test[0]),len(Y_train),len(Y_train[0]),len(Y_test),len(Y_test[0])
    # generating the value of all the coefficient and Y intercept in beta
        Beta=(np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X_train), X_train)), np.transpose(X_train)),
                    Y_train))
        Beta = np.transpose(Beta)
        y_pred = []
        for i in range(80):
            y_pred.append([])
        for i in range(80):
            for j in range(40):
                y_pred[i].append(0)
        for i in range(len(X_test)):
            for j in range(len(Beta)):
                for k in range(len(Beta[0])):
                    y_pred[i][j] = Beta[j][k]*X_test[i][k]

        index = []
        index2 = []
        for i in range(len(y_pred)):
            maximum = max(y_pred[i])
            index.append(y_pred[i].index(maximum))

        for j in range(len(Y_test)):
            for i in range(len(Y_test[0])):
                if Y_test[j][i] == 1:
                    index2.append(j)
        Score.append(accuracy_score(index2, index))

    for i in range (len(Score)):
        print "The accuracy of the Linear classification method for ATnT is",Score[i]*p,"for",i,"fold!"


if __name__ == '__main__':
    X_Y=Reading_the_Data()
    Train_Test=Cross_Validation(X_Y[0],X_Y[1])
    Linear_regression_classification(Train_Test)