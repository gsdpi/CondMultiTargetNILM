import numpy as np

def mae(y_true,y_pred):
    return np.mean(np.abs(y_true-y_pred))

def rmse(y_true,y_pred):
    return np.square(np.mean((y_true-y_pred)**2))