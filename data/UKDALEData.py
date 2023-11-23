import pandas as pd
import numpy as np
import os
import ipdb



appliance_data = {
    "kettle": {
        "mean": 700,
        "std": 1000,
        'on_power_threshold': 2000,
        'max_on_power': 3998
    },
    "fridge": {
        "mean": 200,
        "std": 400,
        'on_power_threshold': 50,
        
      
    },
    "dish washer": {
        "mean": 700,
        "std": 700,
        'on_power_threshold': 10
    },
    
    "washing machine": {
        "mean": 400,
        "std": 700,
        'on_power_threshold': 20,
        'max_on_power': 3999
    },
    "microwave": {
        "mean": 500,
        "std": 800,
        "window":10,
        'on_power_threshold': 200,
       
    },
}
main_mean = 522
main_std = 814

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

def binarization(x,thrs):
    """
        Returns the ON states whe the app reaches a threshold. 
    """
    return np.where(x>= thrs,1,0).astype(int)



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


class UKDALEData(object):
    """
    Generates training and test sequence from  UK-DALE dataset.
    """

    def __init__(self,path = "/data/",dataset="UKDALE"):
        """
        PARAMETERS
            PATH:  PAth to the data.
        RETURNS
        """
        self.dataPath = os.path.join(path, dataset+".h5")
        self.rawData = pd.read_hdf(self.dataPath)
        self.apps = appliance_data.keys()

    def _get_sequences(self,houses = 1, start = "2013-01-01",end="2016-01-01",norm = True):
        """
        PARAMETERS
            Houses [list, integer]: ID of building from which the data is extracted
            start [sring]:          Start datetime of the returned sequence 
            end [sring]:            End datetime of the returned sequence
        RETURNS
            main [numpy array]:     Numpy array with the whole consumption for the selected sequence                   
            targets [list]:         List with all the numpy array for each of the appliances's consumptions and the selected sequence
            states [list]:         List with all the numpy array for each of the appliances states and the selected sequence
        """
        data =self.rawData.loc[start:end]
        data = data[data['House']==houses]
        main = data['main'].values
        if norm:
            main   =  (data['main'].values - main_mean)/ main_std
        targets = []
        states  = []
        for app in self.apps:
            target = data[app].values
            # Binarization
            states.append(binarization(target, appliance_data[app]["on_power_threshold"]))
            # Normalization
            if norm:
                target = (target-appliance_data[app]["mean"])/appliance_data[app]["std"]
            targets.append(target)
        return main, targets, states
        
    def get_train_sequences(self,houses = 1, start = "2013-01-01",end="2016-01-01",norm = True):
        """
        It returns training time series
        PARAMETERS
            Houses [list, integer]: ID of building from which the data is extracted
            start [sring]:          Start datetime of the returned sequence 
            end [sring]:            End datetime of the returned sequence
        RETURNS
            main [numpy array]:     Numpy array with the whole consumption for the selected sequence                   
            targets [list]:         List with all the numpy array for each of the appliances's consumptions and the selected sequence
            states [list]:         List with all the numpy array for each of the appliances states and the selected sequence
        """

        return self._get_sequences(houses, start,end,norm)
        

    def get_test_sequences(self,houses = 1, start = "2016-01-01",end="2016-07-01",norm=True):
        """
        It returns test time series
        PARAMETERS
            Houses [list, integer]: ID of building from which the data is extracted
            start [sring]:          Start datetime of the returned sequence 
            end [sring]:            End datetime of the returned sequence
        RETURNS
            main [numpy array]:     Numpy array with the whole consumption for the selected sequence                   
            targets [list]:         List with all the numpy array for each of the appliances's consumptions and the selected sequence
            states [list]:         List with all the numpy array for each of the appliances states and the selected sequence
        """

        return self._get_sequences(houses, start,end,norm)
        
    def get_app_mean_std(self):
        """
        RETURNS
            means [list]:    means of the stantdad normalization applied to appliances' data
            stds [list]:     stds of the stantdad normalization applied to appliances' data
            
        """
        means = [appliance_data[app]["mean"] for app in self.apps]
        stds = [appliance_data[app]["std"] for app in self.apps]
        return means, stds
    
    def get_main_mean_std(sef):
        """
        RETURNS
            mean:    mean of the stantdad normalization applied to main's data
            std :     stds of the stantdad normalization applied to main's data
            
        """
        return main_mean, main_std
    def get_app_data(self):
        return appliance_data



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.ion()
    dataGen = UKDALEData(path=".")
    trainMain,trainTargets, trainStates = dataGen.get_train_sequences(houses = 1, start = "2013-01-01",end="2016-01-01",norm=False)
    # Test overall pre-processing
    plt.figure("Main")
    plt.clf()
    plt.plot(trainMain[:10000])

    plt.figure("Targers")
    plt.clf()
    for ii,app in enumerate(appliance_data.keys()):
        plt.subplot(3,2,ii+1)
        plt.plot(trainTargets[ii][:10000])
        plt.plot(trainStates[ii][:10000])
        plt.title(app)




