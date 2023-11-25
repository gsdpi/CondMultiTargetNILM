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
main_mean = 389
main_std = 445

def binarization(x,thrs):
    """
        Returns the ON states whe the app reaches a threshold. 
    """
    return np.where(x>= thrs,1,0).astype(int)


######### Quantile filter ###############
# Functions for quantile filter from: https://github.com/sambaiga/UNETNiLM/blob/master/src/data/load_data.py:
# get_perceptile()
# generate sequences()
# quatile_filter()

def get_percentile(data,p=50):
    """[summary]
    
    Arguments:
        data {[type]} -- [description]
        quantile {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    return np.percentile(data, p, axis=1, interpolation="nearest")

def generate_sequences(sequence_length, data):
    sequence_length = sequence_length - 1 if sequence_length% 2==0 else sequence_length
    units_to_pad = sequence_length // 2
    new_mains = np.pad(data, (units_to_pad,units_to_pad),'constant',constant_values=(0,0))
    new_mains = np.array([new_mains[i:i + sequence_length] for i in range(len(new_mains) - sequence_length+1)])
    return new_mains

def quantile_filter(sequence_length, data, p=50):
    new_mains = generate_sequences(sequence_length, data)
    new_mains = get_percentile(new_mains, p)
    return new_mains



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




