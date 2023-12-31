
from models import *
import pandas as pd
from address import *
import json


# Model factory pattern
def modelGen(modelID:str,data,params:dict={},verbose=True,debug = False):
    '''
    ARGUMENTS
        modelID (str)                       ID that indicates the model type
        data    (featExtraction object)     Data object needed to train
        params  (dict)                      the params that define the model 
    '''
    data = data
    modelID = modelID
    params  = params

    if verbose:
        print("Building model")

    if not params and not debug:
        if verbose:
            print("loading best hyperparameters")
        params_path  = get_param_path(modelID)
        
        with open(params_path) as f:
            params = json.load(f)
        
        #df_params    = pd.read_csv(params_path,index_col=0)
        #params       = ast.literal_eval(df_params.loc[data.dataID,'params'])[0]


    #TODO: Make it more generic: https://stackoverflow.com/questions/456672/class-factory-in-python 
    for cls in BaseModel.__subclasses__():
        print(cls.get_model_name())
        if cls.is_model_for(modelID):
            return cls(data,params)
    raise Exception("Model not implemented")
    
    


if __name__ == "__main__":
    import ipdb
    from data import HOSPData,UKDALEData
    import matplotlib.pyplot as plt

    # # Ejemplo lectura hospital
    # dataGen = HOSPData(path="./data/")
    # trainMain,trainTargets, trainStates = dataGen.get_train_sequences( start = "2018-04-01",end="2019-02-28")
    # testMain,testTargets, testStates = dataGen.get_train_sequences( start = "2018-03-01",end="2018-04-01")
    # app_data = dataGen.get_app_data()

    # data= {"X_train":trainMain,
    #        "Y_train":trainTargets,
    #        "Z_train":trainStates,
    #         "X_test":testMain,
    #         "Y_test":testTargets,
    #         "apps": app_data.keys()
    #         }  
    
    # TEST UKDALE AND U-NET
    plt.ion()
    dataGen = UKDALEData(path="./data/")
    trainMain,trainTargets, trainStates = dataGen.get_train_sequences(houses = 1, start = "2014-01-01",end="2016-02-01")
    testMain,testTargets, testStates = dataGen.get_train_sequences(houses = 1, start = "2016-01-01",end="2016-07-01")
    app_data = dataGen.get_app_data()

    data= {"X_train":trainMain,
           "Y_train":trainTargets,
           "Z_train":trainStates,
            "X_test":testMain,
            "Y_test":testTargets,
            "apps": app_data.keys()
            }  


    params = {'sequence_length':100,'stride':50,'epochs':2}
    model = modelGen("UNET",data,params)
    X,Y = model.preprocessing(data["X_train"],data["Y_train"],data["Z_train"])
    