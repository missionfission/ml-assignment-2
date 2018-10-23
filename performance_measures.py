import numpy as np

def accuracy(y_pred, y_actual):
    return 100.0 * np.sum(y_pred == y_actual) / float(y_actual.shape[0])

def confusion_matrix(classes, y_pred, y_actual):
    k = len(classes)
    mat = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            mat[i][j] = np.sum(np.logical_and((y_actual == classes[i]), (y_pred == classes[j])))
            
    return mat

def precision(confusion_mat):
    k = confusion_mat.shape[0]
    prec = np.zeros(k)
    for i in range(k):
        prec[i] = float(confusion_mat[i][i]) / np.sum(confusion_mat[:,i])
    return prec

def recall(confusion_mat):
    k = confusion_mat.shape[0]
    rec = np.zeros(k)
    for i in range(k):
        rec[i] = float(confusion_mat[i][i]) / np.sum(confusion_mat[i])
    return rec

def fscore(rec, prec):
    return 2*rec*prec / (rec + prec)

def scores(y_true,y_pred):
        tp =(y_true == y_pred)
        tp_sum=np.bincount(y_true[tp].astype(int))
        pred_sum=np.bincount(y_pred.astype(int))
        true_sum=np.bincount(y_true.astype(int))
        precision = tp_sum/pred_sum
        recall = tp_sum/true_sum
        f_score = ((2) * precision * recall /
                   (precision + recall))
        precision = np.average(precision)
        recall = np.average(recall)
        f_score = np.average(f_score)

        return precision, recall, f_score