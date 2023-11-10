import numpy as np
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
