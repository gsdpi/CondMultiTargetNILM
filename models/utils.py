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

def agg_act(H,seqLen,stride):
    """
    Inverse of windowing op.
    PARAMETERS
        H [numpy array]   -> A matrix of activations vertically stacked 
        seqLen [int]      -> Sequence lenght
        stride [int]      -> Stride of the windowing op.
    RETURN
        X [numpy array]   -> 1D array
    """
    windws = H.shape[0]
    n = (windws-1)*stride + seqLen
    sum_arr = np.zeros((n,H.shape[-1]))
    for w in range(windws):
        sum_arr[ w*stride : w*stride + seqLen,:] += H[w,...]
        
    H = sum_arr/windws
    return H



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




def get_activations(chunk, min_off_duration=0, min_on_duration=0,
                    border=1, on_power_threshold=5):
    """Returns runs of an appliance.

    Most appliances spend a lot of their time off.  This function finds
    periods when the appliance is on.

    Parameters
    ----------
    chunk : pd.Series
    min_off_duration : int
        If min_off_duration > 0 then ignore 'off' periods less than
        min_off_duration seconds of sub-threshold power consumption
        (e.g. a washing machine might draw no power for a short
        period while the clothes soak.)  Defaults to 0.
    min_on_duration : int
        Any activation lasting less seconds than min_on_duration will be
        ignored.  Defaults to 0.
    border : int
        Number of rows to include before and after the detected activation
    on_power_threshold : int or float
        Watts

    Returns
    -------
    list of pd.Series.  Each series contains one activation.
    """
    chunk = pd.Series(chunk)
    when_on = chunk >= on_power_threshold

    # Find state changes
    state_changes = when_on.astype(np.int8).diff()
    del when_on
    switch_on_events = np.where(state_changes == 1)[0]
    switch_off_events = np.where(state_changes == -1)[0]
    del state_changes

    if len(switch_on_events) == 0 or len(switch_off_events) == 0:
        return []

    # Make sure events align
    if switch_off_events[0] < switch_on_events[0]:
        switch_off_events = switch_off_events[1:]
        if len(switch_off_events) == 0:
            return []
    if switch_on_events[-1] > switch_off_events[-1]:
        switch_on_events = switch_on_events[:-1]
        if len(switch_on_events) == 0:
            return []
    assert len(switch_on_events) == len(switch_off_events)

    # Smooth over off-durations less than min_off_duration
    if min_off_duration > 0:
        off_durations = (chunk.index[switch_on_events[1:]].values -
                         chunk.index[switch_off_events[:-1]].values)

        off_durations = timedelta64_to_secs(off_durations)

        above_threshold_off_durations = np.where(
            off_durations >= min_off_duration)[0]

        # Now remove off_events and on_events
        switch_off_events = switch_off_events[
            np.concatenate([above_threshold_off_durations,
                            [len(switch_off_events)-1]])]
        switch_on_events = switch_on_events[
            np.concatenate([[0], above_threshold_off_durations+1])]
    assert len(switch_on_events) == len(switch_off_events)

    activations = []
    for on, off in zip(switch_on_events, switch_off_events):
        duration = (chunk.index[off] - chunk.index[on]).total_seconds()
        if duration < min_on_duration:
            continue
        on -= 1 + border
        if on < 0:
            on = 0
        off += border
        activation = chunk.iloc[on:off]
        # throw away any activation with any NaN values
        if not activation.isnull().values.any():
            activations.append(activation)

    return activations


def timedelta64_to_secs(timedelta):
    """Convert `timedelta` to seconds.

    Parameters
    ----------
    timedelta : np.timedelta64

    Returns
    -------
    float : seconds
    """
    if len(timedelta) == 0:
        return np.array([])
    else:
        return timedelta / np.timedelta64(1, 's')
