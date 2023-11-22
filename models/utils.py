import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import random as python_random
import ipdb
import os

def get_windows(X,seqLen, stride):
    """
    It returns a numpy array with strided windows of length seqLen extracted from X.
    PARAMETERS
        X [numpy array]   -> 1D array
        seqLen [int]      -> Sequence lenght
        stride [int]      -> Stride of the windowing op.
    RETURN
        W [numpy array]   -> A matrix of windows vertically stacked 
    """

    N = len(X)
    excess = 0 if ((N-seqLen) %stride) == 0 else stride - ((N-seqLen) %stride) 
    X_pad  = np.concatenate([X,np.zeros(excess)])
    N = len(X_pad)
    n_windws = 1 + int((N-seqLen)/stride)
    W = []
    for i in range(n_windws):
        W.append(X_pad[ i*stride : seqLen+(i*stride) ])
    W = np.vstack(W)
    return W

def agg_windows(W,seqLen,stride):
    """
    Inverse of windowing op.
    PARAMETERS
        W [numpy array]   -> A matrix of windows vertically stacked 
        seqLen [int]      -> Sequence lenght
        stride [int]      -> Stride of the windowing op.
    RETURN
        X [numpy array]   -> 1D array
    """
    windws = W.shape[0]
    n = (windws-1)*stride + seqLen
    sum_arr = np.zeros((n))
    cont_arr = np.zeros((n))
    for w in range(windws):
        sum_arr[ w*stride : w*stride + seqLen] += W[w].ravel()
        cont_arr[ w*stride : w*stride + seqLen] += 1
    X = np.divide(sum_arr,cont_arr)
    return X



def oneHot(x,n_clss=None):
    x = np.int32(x)
    N = x.shape[0] 
    M = np.int32(np.max(x)+1) if n_clss==None else n_clss
    y = np.zeros((N,M)).astype(float)
    y[np.arange(N),x.squeeze()] = 1
    return y

# pinball loss
# https://www.kaggle.com/code/ulrich07/quantile-regression-with-keras

def qloss(qs):
    # Pinball loss for multiple quantiles
    
    def loss(y_true,y_pred):        
        q = tf.constant(np.array([qs]), dtype=tf.float32)        
        y_true = tf.stack([y_true]*y_pred.shape[2],axis=-1)
        e = y_true - y_pred
        v = tf.maximum(q*e, (q-1)*e)
        return K.mean(v)
    return loss

def reset_seeds(seed_value=39):
    # ref: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    # necessary for starting Numpy generated random numbers in a well-defined initial state.
    np.random.seed(seed_value)
    # necessary for starting core Python generated random numbers in a well-defined state.
    python_random.seed(seed_value)
    # set_seed() will make random number generation
    tf.random.set_seed(seed_value)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
    sys.path.insert(1, '../data/')
    from UKDALEData import UKDALEData
    plt.ion()
    dataGen = UKDALEData(path="../data/")
    trainMain,trainTargets, trainStates = dataGen.get_train_sequences(houses = 1, start = "2013-01-01",end="2016-01-01 03:00:00")

    W,SD,Y = get_windows(trainMain,1440,30)
