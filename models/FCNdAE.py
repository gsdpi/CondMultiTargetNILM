# Implementation of tcn from https://arxiv.org/pdf/1803.01271.pdf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from .utils import get_windows,agg_windows,oneHot
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from .BaseModel import BaseModel
import ipdb
layers = tf.keras.layers
K = tf.keras.backend
initialiazers = tf.keras.initializers





class FCNdAE(BaseModel):
    def __init__(self,data, params: dict, **kwargs) -> None:      
        
        # Data
        self.X_train = data["X_train"]
        self.Y_train = data["Y_train"]

        self.training_hist =None
        self.model =None

        # Net params
        self.apps            = data["apps"]
        self.n_apps          = len(self.apps)
        self.sequence_length = params.get('sequence_length',100)
        self.stride = params.get('stride',10)
        
        self.convBlocks  = params.get('convBlocks',2)
        self.ksize       =  params.get('ksize',4)
        self.filters     = params.get('filters',8)
        self.filters_lt  = params.get('filters_lt',26)

        # Training params
        self.epochs = params.get('epochs', 10)
        self.patience = params.get('patience', 15)
        self.batch_size = params.get('batch_size',128)
        self.data_balance = params.get('data_balance',False)
        
        # Metrics
        self.metrics     = [mean_squared_error, mean_absolute_error,r2_score]        
        

        # Model
        print(f"Building model: {FCNdAE.get_model_name} [{FCNdAE.get_model_type}]")
        self.model = []
        for app in range(self.n_apps):
            self.model.append(self.create_model())

        #############################################################################


    def _createEnc(self, x):
        self.layersEnc = []
        filters = [i*2*self.filters for i in range(self.convBlocks)]; filters[0] = self.filters
        
        for ll in range(self.convBlocks):
            self.layersEnc.append(layers.Conv2D(kernel_size=(self.ksize,1),
                                                filters=filters[ll],
                                                strides=(1,1),
                                                padding = "same",
                                                activation="relu",
                                                kernel_initializer= self.initializer_relu,
                                                bias_initializer= self.initializer_relu,
                                                name=f"Enc_conv{ll}_1")
                                 )
            
            
            self.layersEnc.append(layers.Conv2D(kernel_size=(self.ksize,1),
                                                filters=filters[ll],
                                                strides=(1,1),
                                                padding = "same",
                                                activation="relu",
                                                kernel_initializer= self.initializer_relu,
                                                bias_initializer= self.initializer_relu,
                                                name=f"Enc_conv{ll}_2")
                                 )
            self.layersEnc.append(layers.Conv2D(kernel_size=(self.ksize,1),
                                                filters=filters[ll]*2,
                                                strides=(2,1),
                                                padding = "same",
                                                activation="relu",
                                                kernel_initializer= self.initializer_relu,
                                                bias_initializer= self.initializer_relu,
                                                name=f"Enc_conv{ll}_3")
                                 )
        # building enc
        y = x
        for ll,_ in enumerate(self.layersEnc):
            y = self.layersEnc[ll](y)
        return y

    def _createDec(self,z):
        self.layersDec = []
        filters = [i*2*self.filters for i in reversed(range(self.convBlocks))]; filters[-1] = self.filters
        for ll in reversed(range(self.convBlocks)):
            self.layersDec.append(layers.Conv2DTranspose(kernel_size=(self.ksize,1),
                                                         filters=filters[ll]*2,strides=(2,1),
                                                         padding = "same",
                                                         activation="relu",
                                                         kernel_initializer= self.initializer_relu,
                                                         bias_initializer= self.initializer_relu,
                                                         name=f"Dec_conv{ll}_1")
                                    )
            self.layersDec.append(layers.Conv2D(kernel_size=(self.ksize,1),
                                                         filters=filters[ll],strides=(1,1),
                                                         padding = "same",
                                                         activation="relu",
                                                         kernel_initializer= self.initializer_relu,
                                                         bias_initializer= self.initializer_relu,
                                                         name=f"Dec_conv{ll}_2")
                                    )
            self.layersDec.append(layers.Conv2D(kernel_size=(self.ksize,1),
                                                         filters=filters[ll],strides=(1,1),
                                                         padding = "same",
                                                         activation="relu",
                                                         kernel_initializer= self.initializer_relu,
                                                         bias_initializer= self.initializer_relu,
                                                         name=f"Dec_conv{ll}_3")
                                    )
        y = z
        for layer in self.layersDec:
            y = layer(y)
        return y

    def create_model(self,verbose=True):
        
        # Initializers
        self.initializer_relu = initialiazers.VarianceScaling()
        self.initializer_linear = initialiazers.RandomNormal(0.,0.02)


        # Input layers
        self.main_input_layer = layers.Input(dtype = tf.float32,shape=[self.sequence_length,1,1],name='main_input')
        
        # FCN dAE 
        # Encoder 
        self.z = self._createEnc(self.main_input_layer)
        # Bottleneck
        self.z = layers.Conv2D(kernel_size=(int(self.sequence_length/(2*self.convBlocks)),1),
                               filters=self.filters_lt,
                               strides=(1,1),
                               padding = "valid",
                               activation=None,
                               kernel_initializer= self.initializer_linear ,
                               bias_initializer= self.initializer_linear,
                               name="z")(self.z)
        
        self.z_dec_input = layers.Conv2DTranspose(kernel_size=(int(self.sequence_length/(2*self.convBlocks)),1),
                                                  filters=self.filters_lt,
                                                  strides=(1,1),
                                                  padding = "valid",
                                                  activation=None,
                                                  kernel_initializer= self.initializer_linear,
                                                  bias_initializer= self.initializer_linear,
                                                  name="D_input")(self.z)                    
        # Decoder
        self.y = self._createDec(self.z_dec_input) 

        #Output
        self.y = layers.Conv2D(kernel_size=(1,1),
                               filters=1,
                               strides=(1,1),
                               padding = "same",
                               activation=None,
                                kernel_initializer= self.initializer_linear,bias_initializer= self.initializer_linear,name="out_conv")(self.y)
        self.y = layers.Flatten()(self.y)


        model = tf.keras.Model(inputs=[self.main_input_layer],outputs = [self.y])



        model.compile( optimizer='adam',
                      loss=tf.keras.losses.MeanSquaredError(),
                     metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
        
        if verbose:
            model.summary()

        return model
    
    def train(self):
        print(f"Training model: {FCNdAE.get_model_name} [{FCNdAE.get_model_type}]")
        print("Prepocesing data")
        X, Y = self.preprocessing(self.X_train,self.Y_train,method='train')
        
        # Training M single target models
        self.training_hist = []
        for ii,app in enumerate(self.apps):
            print(f"Training model for {app} individual consumption")
            model_app = self.model[ii]
            train_X, v_X, train_Y,v_Y =  train_test_split(X,Y[ii], test_size=.15,random_state=10)  
            ES_cb = tf.keras.callbacks.EarlyStopping( monitor="val_loss",
                                            min_delta=0.001,
                                            patience=self.patience,                                            
                                            baseline=None,
                                            restore_best_weights=True)
            
            self.training_hist.append(model_app.fit([train_X], train_Y ,
                                                batch_size=self.batch_size,
                                                epochs=self.epochs,
                                                validation_data=([v_X], v_Y),
                                                callbacks=[ES_cb]))
        

        return self.training_hist        

    def predict_sample(self,X):
        """
            It predicts a unique sample
            PARAMETERS
                X [numpy array]  -> Input sample with the main consumption to be disaggregated.                 
            RETURN
                y [list] -> list of arrays with the disaggregations
        """
        y = []
        X = X.reshape(-1,self.sequence_length,1,1)
                
        for app in range(self.n_apps):
            
            y.append(self.model[app].predict(X))

        return y
    
    def predict(self,X):
        """
            It predicts the time serie. It first gets the windowns and then the disaggregations will be computed
            PARAMETERS
                X [numpy array]  -> Input sample with the main consumption to be disaggregated.                 
                SD [numpy array] -> Modulation input. If None all available apps will be disaggregated. 
                                    If SD is a one hot, the indicated app will be disaggregated
            RETURN
                y [list/numpy array] -> array or list of arrays (if SD == None) with the disaggregations
        """
        N = len(X)
        X = self.preprocessing(X,None,method='test')
        Y = []
        for app in range(self.n_apps):
            y_w = self.model[app].predict([X],batch_size=300)
            y  = agg_windows(y_w,self.sequence_length,self.stride)
            Y.append(y[:N])
        return Y
        
    def store(self,path):
        if self.training_hist==None:
            raise Exception("The model has not been trained yet")
        for ii,app in enumerate(self.apps):
            savePath = os.path.join(path, f"{FCNdAE.get_model_name()}_{app}")
            print(f"saving weights of model {FCNdAE.get_model_name()} and individual consump. {app} in path: {savePath}")
            self.model[ii].save_weights(savePath)
        return None
    
    def load(self,path):
        
        if self.model==None:
            raise Exception("The model has not been defined yet")
        for ii,app in enumerate(self.apps):
            loadPath = os.path.join(path, f"{FCNdAE.get_model_name()}_{app}")
            print(f"restoring weights of model {FCNdAE.get_model_name()} from path: {loadPath}")
            self.model[ii].load_weights(loadPath)
        return None


    def preprocessing(self,main,targets,method='train'):
        ###### Inputs ######
        # Windowing op.
        W_main = get_windows(main,self.sequence_length,self.stride)

        if method == "test":
            return W_main.reshape(-1,self.sequence_length,1,1)
        else:
            ##### Targets ######
            W_apps = []
            S_D    = []
            for app in range(self.n_apps):                
                W_apps.append(get_windows(targets[app],self.sequence_length,self.stride))
            
            # Reshaping data for 2D conv layers
            W_main = W_main.reshape(-1,self.sequence_length,1,1)
            return W_main, W_apps

    
    @classmethod
    def get_model_type(cls):
        return "keras" # Aquí se puede indicar qué tipo de modelo es: RRNN, keras, scikit-lear, etc.
    
    @classmethod
    def get_model_name(cls):
        return "FCNdAE" # Aquí se puede indicar un ID que identifique el modelo
    
    @classmethod
    def target(cls):
        return "single-target" 
    @classmethod
    def is_model_for(cls,name):
        return cls.get_model_name() == name 



##########################################
# Unit testing
##########################################


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
    sys.path.insert(1, '../data/')
    from UKDALEData import UKDALEData
    plt.ion()
    dataGen = UKDALEData(path="../data/")
    trainMain,trainTargets, trainStates = dataGen.get_train_sequences(houses = 1, start = "2014-01-01",end="2014-02-01")
    testMain,testTargets, testStates = dataGen.get_train_sequences(houses = 1, start = "2016-01-01",end="2016-07-01")
    app_data = dataGen.get_app_data()

    data= {"X_train":trainMain,
           "Y_train":trainTargets,
           "Y_states":trainStates,
            "X_test":testMain,
            "Y_test":testTargets,
            "apps": app_data.keys()
            }  


    model = FCNdAE(data,{'sequence_length':400,'stride':200,'epochs':2})
    # Preprocessing test
    X,Y = model.preprocessing(data["X_train"],data["Y_train"])
    X = X.squeeze()
    
    idx =33

    plt.figure("Data")
    plt.clf()
    plt.plot(X[idx],label="main")
    for ii,app in enumerate(app_data.keys()):        
        plt.plot(Y[ii][idx],label=app)
    plt.legend()
    
    # Training test
    hist = model.train()
    model.store('.')
    model.load('.')

    # # Inference test
    y_est = model.predict_sample(X[idx])
    plt.figure("Estimation sample")
    plt.clf()
    plt.plot(X[idx],label='main')
    for ii,app in enumerate(app_data.keys()):
        plt.plot(y_est[ii].squeeze(),label=app)
    plt.legend()

    
    # Inferring a entire sequence
    X_ts = trainMain[:10000]
    Y_ts=model.predict(X_ts)
    plt.figure("Estimation time series")
    plt.clf()
    plt.plot(X_ts,label='main')
    for ii,app in enumerate(app_data.keys()):
        plt.plot(Y_ts[ii],label=app)
    plt.legend()