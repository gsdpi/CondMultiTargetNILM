# Implementation of tcn from https://arxiv.org/pdf/1803.01271.pdf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from utils import get_windows,oneHot
from sklearn.model_selection import train_test_split
import tensorflow as tf

import ipdb
layers = tf.keras.layers
K = tf.keras.backend
initialiazers = tf.keras.initializers


class filmLayer(tf.keras.layers.Layer):
    """Linear modulation on the input. Apply an afin transformation on the inputs"""
    def __init__(self, activation=tf.nn.relu,name='filmLayer',**kwargs):
        super(filmLayer, self).__init__(name=name, **kwargs)
        self.activation=activation

        #[-1,n_params]   [-1,k,1,c] 

    def build(self,input_shape):
        assert isinstance(input_shape, list)
        feature_map_shape, FiLM_params = input_shape
        self.n_feature_maps = feature_map_shape[-1]
        assert(int(2 * self.n_feature_maps)==FiLM_params[1])
        super(filmLayer, self).build(input_shape)

    def call(self, inputs):
        assert isinstance(inputs, list)
        conv_output, FiLM_params = inputs
        dim = conv_output.get_shape().as_list()[1]
        FiLM_params = tf.expand_dims(FiLM_params,[1])
        FiLM_params = tf.expand_dims(FiLM_params,[1])
        FiLM_params = tf.tile(FiLM_params,[1,dim,1,1])


        gammas = FiLM_params[:, :, :, :self.n_feature_maps]
        betas  = FiLM_params[:, :, :, self.n_feature_maps:]

        m  = tf.keras.backend.mean(conv_output,axis=1,keepdims=True)
        std = tf.keras.backend.std(conv_output,axis=1,keepdims=True)
        conv_out_ = (conv_output -m)/std
        z = (1-gammas)*conv_out_ + betas

        return self.activation(z)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]




class multiFCNdAE(object):
    def __init__(self,data, params: dict, **kwargs) -> None:      
        
        # Data
        self.X_train = data["X_train"]
        self.Y_train = data["Y_train"]

        self.X_test = data["X_test"]
        self.Y_test = data["Y_test"]

        self.N_capas = params.get("N_capas", 3)
        self.testMetrics = []

        # Net params
        self.n_apps          = len(self.Y_train)
        self.sequence_length = params.get('sequence_length',99)
        self.sequence_stride = params.get('stride',10)
        
        self.convBlocks  = params.get('convBlocks',2)
        self.filters        = params.get('filters',32)
        self.filters_lt     = params.get('filters_lt',26)
        self.n_layersFilmGen  = params.get('n_layersFilmGen',4)

        # Training params
        self.epochs = params.get('epochs', 10)
        self.patience = params.get('patience', 15)
        self.batch_size = params.get('batch_size',128)
        self.data_balance = params.get('data_balance',False)

        # Storage params
        self.save_model_path = params.get('save-model-path', None)
        self.load_model_path = params.get('pretrained-model-path',None)
        
        # Metrics
        self.metrics     = [mean_squared_error, mean_absolute_error,r2_score]        
        

        # Model
        print(f"Building model: {multiFCNdAE.get_model_name} [{multiFCNdAE.get_model_type}]")
        self.model = self.create_model() 

        #############################################################################

    def _create_filmGen(self,S_D):
        params = []
        self.layersFiLMGEN = []
        self.outFiLMGenLayers = []
        # FiLM Generator
        for ll in range(1,self.n_layersFilmGen+1):
            self.layersFiLMGEN.append(layers.Dense(units = 32, activation=None,name=f"FiLM_GEN_{ll}"))
            self.layersFiLMGEN.append(layers.LeakyReLU(alpha=0.2,name=f"FiLM_GEN_act_{ll}"))

        # Output FiLM Gen
        for ll in range(1,self.convBlocks+1):
            self.outFiLMGenLayers.append(layers.Dense(units = self.filters*2*ll, activation="linear",name=f"FiLM_GEN_params_conv{ll}_1"))
            self.outFiLMGenLayers.append(layers.Dense(units = self.filters*2*ll, activation="linear",name=f"FiLM_GEN_params_conv{ll}_2"))
            self.outFiLMGenLayers.append(layers.Dense(units = self.filters*4*ll, activation="linear",name=f"FiLM_GEN_params_conv{ll}_3"))

        # Building FiLM Gen
        for layer in self.layersFiLMGEN:
            S_D = layer(S_D)
        
        for layer in self.outFiLMGenLayers:
            params.append(layer(S_D))

        return params

    def _createEnc(self, x,filmParams):
        self.layersEnc = []
        self.layersFiLMEnc = []
        filters = [i*2*self.filters for i in range(self.convBlocks)]; filters[0] = self.filters
        
        for ll in range(self.convBlocks):
            self.layersEnc.append(layers.Conv2D(kernel_size=(4,1),
                                                filters=filters[ll],
                                                strides=(1,1),
                                                padding = "same",
                                                activation="linear",
                                                kernel_initializer= self.initializer_relu,
                                                bias_initializer= self.initializer_relu,
                                                name=f"Enc_conv{ll}_1")
                                 )
            
            self.layersFiLMEnc.append(filmLayer(name=f"FiLM_conv{ll}_1"))
            self.layersEnc.append(layers.Conv2D(kernel_size=(4,1),
                                                filters=filters[ll],
                                                strides=(1,1),
                                                padding = "same",
                                                activation="linear",
                                                kernel_initializer= self.initializer_relu,
                                                bias_initializer= self.initializer_relu,
                                                name=f"Enc_conv{ll}_2")
                                 )
            self.layersFiLMEnc.append(filmLayer(name=f"FiLM_conv{ll}_2"))
            self.layersEnc.append(layers.Conv2D(kernel_size=(4,1),
                                                filters=filters[ll]*2,
                                                strides=(2,1),
                                                padding = "same",
                                                activation="linear",
                                                kernel_initializer= self.initializer_relu,
                                                bias_initializer= self.initializer_relu,
                                                name=f"Enc_conv{ll}_3")
                                 )
            self.layersFiLMEnc.append(filmLayer(name=f"FiLM_conv{ll}_3"))
        # building enc
        y = x
        for ll,_ in enumerate(self.layersEnc):
            y = self.layersEnc[ll](y)
            y = self.layersFiLMEnc[ll]([y,filmParams[ll]])

        return y

    def _createDec(self,z):
        self.layersDec = []
        filters = [i*2*self.filters for i in reversed(range(self.convBlocks))]; filters[-1] = self.filters
        for ll in reversed(range(self.convBlocks)):
            self.layersDec.append(layers.Conv2DTranspose(kernel_size=(4,1),
                                                         filters=filters[ll]*2,strides=(2,1),
                                                         padding = "same",
                                                         activation="relu",
                                                         kernel_initializer= self.initializer_relu,
                                                         bias_initializer= self.initializer_relu,
                                                         name=f"Dec_conv{ll}_1")
                                    )
            self.layersDec.append(layers.Conv2D(kernel_size=(4,1),
                                                         filters=filters[ll],strides=(1,1),
                                                         padding = "same",
                                                         activation="relu",
                                                         kernel_initializer= self.initializer_relu,
                                                         bias_initializer= self.initializer_relu,
                                                         name=f"Dec_conv{ll}_2")
                                    )
            self.layersDec.append(layers.Conv2D(kernel_size=(4,1),
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

        self.FiLMParams = []
        
        
        # Initializers
        self.initializer_relu = initialiazers.VarianceScaling()
        self.initializer_linear = initialiazers.RandomNormal(0.,0.02)


        # Input layers
        self.main_input_layer = layers.Input(dtype = tf.float32,shape=[self.sequence_length,1,1],name='main_input')
        self.mod_input_layer  = layers.Input(dtype = tf.float32,shape=[self.n_apps],name='mod_input')

        # FiLM Gen
        self.FiLMParams = self._create_filmGen(self.mod_input_layer)

        # FCN dAE (main)
        # Encoder 
        self.z = self._createEnc(self.main_input_layer,self.FiLMParams)
        # Bottleneck
        self.z = layers.Conv2D(kernel_size=(int(self.sequence_length/4.),1),
                               filters=self.filters_lt,
                               strides=(1,1),
                               padding = "valid",
                               activation=None,
                               kernel_initializer= self.initializer_linear ,
                               bias_initializer= self.initializer_linear,
                               name="z")(self.z)
        
        self.z_dec_input = layers.Conv2DTranspose(kernel_size=(int(self.sequence_length/4.),1),
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


        model = tf.keras.Model(inputs=[self.main_input_layer,self.mod_input_layer],outputs = [self.y])



        model.compile( optimizer='adam',
                      loss=tf.keras.losses.MeanSquaredError(),
                     metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
        
        if verbose:
            model.summary()

        return model
    
    def train(self):
        print(f"Training model: {multiFCNdAE.get_model_name} [{multiFCNdAE.get_model_type}]")
        print("Prepocesing data")
        X,S_D, Y = self.preprocessing(self.X_train,self.Y_train,method='train')
        
        train_X, v_X, train_SD,v_SD,train_Y,v_Y =  train_test_split(X,S_D,Y, test_size=.15,random_state=10)  
        ES_cb = tf.keras.callbacks.EarlyStopping( monitor="val_loss",
                                        min_delta=0.001,
                                        patience=self.patience,                                            
                                        baseline=None,
                                        restore_best_weights=True)
        
        self.training_hist = self.model.fit([train_X,train_SD], train_Y ,
                                            batch_size=self.batch_size,
                                            epochs=self.epochs,
                                            validation_data=([v_X,v_SD], v_Y),
                                            callbacks=[ES_cb])
        


        return self.training_hist        

    def predict(self,X):
        return None
        
    def store(self,path):
        # Método para guardar los pesos en path 
        return None

    def preprocessing(self,main,targets,method='train'):
        ###### Inputs ######
        # Windowing op.
        W_main = get_windows(main,self.sequence_length,self.sequence_stride)
        # Repeting windows for each app
        W_main = np.tile(W_main,(self.n_apps,1))

        if method == "test":
            return W_main.reshape(-1,self.sequence_length,1,1)
        else:
            ##### Targets ######
            W_apps = []
            S_D    = []
            for app in range(self.n_apps):
                
                W_apps.append(get_windows(targets[app],self.sequence_length,self.sequence_stride))
                sd = oneHot([app], n_clss = self.n_apps)
                sd = np.tile(sd,(W_apps[-1].shape[0],1))
                S_D.append(sd)

            W_apps = np.vstack(W_apps)
            S_D = np.vstack(S_D)
            
            # Reshaping data for 2D conv layers
            W_main = W_main.reshape(-1,self.sequence_length,1,1)
            return W_main, S_D,W_apps

    
    @classmethod
    def get_model_type(cls):
        return "keras" # Aquí se puede indicar qué tipo de modelo es: RRNN, keras, scikit-lear, etc.
    
    @classmethod
    def get_model_name(cls):
        return "multiFCNdAE" # Aquí se puede indicar un ID que identifique el modelo
    

    
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


    model = multiFCNdAE(data,{'sequence_length':400,'stride':400,'epochs':10})
    # Preprocessing test
    X,SD,Y = model.preprocessing(data["X_train"],data["Y_train"])
    X = X.squeeze()
    SD = SD.squeeze()
    Y  = Y.squeeze()
    plt.figure("X SD")
    plt.clf()
    plt.subplot(2,1,1)
    plt.imshow(X,aspect='auto',interpolation=None)
    plt.subplot(2,1,2)
    plt.imshow(SD,aspect='auto',interpolation=None)

    plt.figure("Y")
    plt.clf()
    plt.imshow(Y,aspect='auto',interpolation=None)
    
    idx =2100

    plt.figure("X")
    plt.clf()
    for i in range(30):
        plt.subplot(2,1,1)
        plt.plot(X[idx+i,:])
        plt.subplot(2,1,2)
        plt.plot(Y[idx+i,:])
    
    hist = model.train()
