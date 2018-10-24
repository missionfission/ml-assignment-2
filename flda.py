import numpy as np

class FLDA():
    def predict(self,test):
        y_flda = np.zeros((train.shape[0],num_classes))
        for i in range(num_classes):
            y_flda[:,i] = np.dot(test, w[i])
        y_flda = np.argmax(y_flda, axis = 1)
        return y_flda

    def fit(self, train,y_train):
        self.n_classes = np.max(y)+1
        self.n_features=train.shape[1]

        mean_train0 = []
        mean_train1 = []
        var_train0 =  []
        var_train1 =  []
        mean_diff = []
        self.w = []

        for i in range(self.n_classes):
            mean_train0.append(np.mean(train[y_train!=i], axis = 0))
            mean_train1.append(np.mean(train[y_train==i], axis = 0))
            var_train0.append(np.cov(train[y_train!=i].T.astype(float)))
            var_train1.append(np.cov(train[y_train==i].T.astype(float)))
            mean_diff.append((mean_train1[i] - mean_train0[i]).T)
        for i in range(n_classes):
            self.w.append(np.dot(np.linalg.inv(var_train0[i] + var_train1[i]), mean_diff[i]))