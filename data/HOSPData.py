import pandas as pd
import numpy as np
import os
import ipdb



appliance_data = {
    "Lifts": {
        "mean": 35,
        "std": 21,
        "meter_name": "CGBT-2.Montante0"
    },

    "X-ray": {
        "mean": 31,
        "std": 18,
        "meter_name": "Radiologia1"
    },
    "Rehab": {
        "mean": 20,
        "std": 6,
        "meter_name": "RehabilitacionA"
    },
    "Data Center": {
        "mean": 57.120370,
        "std": 3.463719,
        "meter_name": "CPD"
    },

    "Floors": {
        "mean": 31,
        "std": 7,
        "meter_name": "Plantas_2-7"
    }

}

main_mean = 469
main_std  = 126




class HOSPData(object):
    """
    Generates training and test sequence from  HOSPITAL dataset.
    """

    def __init__(self,path = "/data/",dataset="HOSP"):
        """
        PARAMETERS
            PATH:  PAth to the data.
        
        """
        self.dataPath = os.path.join(path, dataset+".h5")
        self.rawData = pd.read_hdf(self.dataPath)
        self.rawData.dropna(axis=0, inplace=True)
        
        self.apps = appliance_data.keys()

    def _get_sequences(self, start = "2013-01-01",end="2016-01-01",norm= True):
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
        
        main = data['CGBT-2.Red-Grupo'].values
        if norm:
            main   =  (main - main_mean)/ main_std
        targets = []
        states  = []
        for app in self.apps:
            meterID = appliance_data[app]["meter_name"]
            target = data[meterID].values
            # Binarization
            states.append(None)
            # Normalization
            if norm:
                target = (target-appliance_data[app]["mean"])/appliance_data[app]["std"]
            targets.append(target)
        return main, targets, states
        
    def get_train_sequences(self, start = "2018-04-01",end="2019-02-28",norm = True):
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

        return self._get_sequences( start,end,norm)
        

    def get_test_sequences(self, start = "2018-03-01",end="2018-04-01",norm = True):
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

        return self._get_sequences( start,end,norm)
        
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
    dataGen = HOSPData(path=".")
    trainMain,trainTargets, trainStates = dataGen.get_train_sequences( start = "2018-04-01",end="2019-02-28",norm=True)
    # Test overall pre-processing
    plt.figure("Main")
    plt.clf()
    plt.plot(trainMain[:10000])

    plt.figure("Targets")
    plt.clf()
    for ii,app in enumerate(appliance_data.keys()):
        plt.subplot(3,2,ii+1)
        plt.plot(trainTargets[ii][:10000])
        plt.title(app)

