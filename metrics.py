import numpy as np

def mae(y_true,y_pred):
    return np.mean(np.abs(y_true-y_pred))

def rmse(y_true,y_pred):
    return np.square(np.mean((y_true-y_pred)**2))

# Normalize error in assigned power

def nep(y_true,y_pred):
    return np.sum(np.abs(y_true-y_pred))/np.sum(y_true)

def eac(y_true,y_pred):
    return 1- np.sum(np.abs(y_true-y_pred))/(np.sum(y_true)*2)