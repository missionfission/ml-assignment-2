import numpy as np

class FLDA():
    def predict(self,test):
        y_flda = np.dot(test,self.w)
        y_flda = y_flda>0
        return y_flda
    def fit(self, train,y_train):
        self.n_classes = np.max(y)+1
        self.n_features=train.shape[1]
        mean_train0 = np.mean(train[y_train==0], axis = 0)
        mean_train1 = np.mean(train[y_train==1], axis = 0)
        var_train0 = np.cov(train[y_train==0].T.astype(float))
        var_train1 = np.cov(train[y_train==1].T.astype(float))
        mean_diff = (mean_train1-mean_train0).T
        self.w = np.dot(np.linalg.inv(var_train0 + var_train1), mean_diff)