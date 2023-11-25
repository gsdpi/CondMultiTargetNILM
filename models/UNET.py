# Implementation of UNET from https://github.com/sambaiga/UNETNiLM/tree/master
#                             https://github.com/jonasbuchberger/energy_disaggregation/tree/master/src/models 

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from .utils import get_windows,agg_windows,qloss,oneHot
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from .BaseModel import BaseModel
import ipdb
layers = tf.keras.layers
K = tf.keras.backend
initialiazers = tf.keras.initializers


#  F.nll_loss(F.log_softmax(logits, 1), z)
def multiLabelLoss(y_true,y_pred):
    #ipdb.set_trace()
    nll = tf.keras.losses.CategoricalCrossentropy(axis=1,from_logits=False,reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    
    return nll(y_true,tf.nn.softmax(y_pred,axis=1))

## Conv block
class Conv1D(layers.Layer):
    """
        Conv block
    """
    def __init__(self,
                 kernels,
                 kernel_size=3,
                 stride=2,
                 padding = "same",
                 activation=tf.nn.leaky_relu,
                 name='con1D',
                 **kwargs):
        super(Conv1D, self).__init__(name=name, **kwargs)
        

        self.conv = layers.Conv1D(filters=kernels,
                                  kernel_size=kernel_size,
                                  strides = stride,
                                  padding=padding,
                                  kernel_initializer = "glorot_uniform",
                                  activation = None,
                                  name = f"{name}_conv")
        self.Batch = layers.BatchNormalization(name = f"{name}_batch")
        self.activation =activation 
        

    def call(self, inputs):
        
        feat = self.conv(inputs)
        feat = self.Batch(feat)
        
        return self.activation(feat)

## Transposed Conv block
class DeConv1D(layers.Layer):
    """
        DeConv block
    """
    def __init__(self,
                 kernels,
                 kernel_size=3,
                 stride=2,
                 padding = "same",
                 activation=tf.nn.leaky_relu,
                 name='DeCon1D',
                 **kwargs):
        super(DeConv1D, self).__init__(name=name, **kwargs)
        

        self.conv = layers.Conv1DTranspose(filters=kernels,
                                  kernel_size=kernel_size,
                                  strides = stride,
                                  padding=padding,
                                  activation = None,
                                  kernel_initializer = "glorot_uniform",
                                  name = f"{name}_conv")
        self.Batch = layers.BatchNormalization(name = f"{name}_batch")
        self.activation =activation 
        

    def call(self, inputs):
        
        feat = self.conv(inputs)
        feat = self.Batch(feat)
        
        return self.activation(feat)


# Upsampling block

class Up(layers.Layer):
    """
        Up block
    """
    def __init__(self,
                 kernels,
                 name='up',
                 **kwargs):
        super(Up, self).__init__(name=name, **kwargs)
        

        self.conv = Conv1D(kernels,stride=1)
        self.upsample= DeConv1D(kernels)

    def call(self, inputs):
        x1 = inputs[0]
        x2 = inputs[1]
        
        x1 = self.upsample(x1)
        diff = x2.shape[1]-x1.shape[1]
        paddings = tf.constant([[0, 0,], [abs(diff//2), abs(diff - diff//2)],[0,0]])
        if diff>0:
            x1 = tf.pad(x1,paddings,"CONSTANT")
        else:
            x2 = tf.pad(x2,paddings,"CONSTANT")
        
        x2 = tf.concat([x1,x2],axis=-1)
        return self.conv(x2)


class Encoder(layers.Layer):
    """
        Output CNN Encoder
    """
    def __init__(self,
                 kernels,
                 N_layers,
                 name='Encoder',
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        

        self.layers = [Conv1D(kernels = kernels//2**(N_layers-1),stride=1)]

        for l in range(1,N_layers-1):
            self.layers.append(Conv1D(kernels=kernels//2**(N_layers-l-1)))
        
        self.layers.append(Conv1D(kernels =kernels, stride=1))


    def call(self, inputs):
        y = inputs
        for layer in self.layers:
            y = layer(y)
        return y



class UNET(BaseModel):
    def __init__(self,data, params: dict, **kwargs) -> None:      
        
        # Data
        self.X_train = data["X_train"]
        self.Y_train = data["Y_train"]
        

        self.training_hist =None
        self.model =None

        # Net params
        self.main_mean       = data["main_data"][0]
        self.main_std        = data["main_data"][1]
        self.app_data        = data["app_data"]
        self.apps            = self.app_data.keys()
        self.n_apps          = len(self.apps)
        self.sequence_length = params.get('sequence_length',100)
        self.stride = params.get('stride',1)
        
        self.nLayers         = params.get('nLayers',5)
        self.ksize           =  params.get('ksize',3)
        self.filters         = params.get('filters',16)
        self.filterCNNoutput = params.get('filterCNNoutput',128)
        self.pool_size       = params.get("pool_size",16)
        self.Nh              = params.get("Nh",1024)
        self.quantiles       = params.get("quatiles",[0.025,0.1,0.5,0.9,0.975])
        self.n_quantiles       = len(self.quantiles)

        # Training params
        self.epochs = params.get('epochs', 10)
        self.patience = params.get('patience', 30)
        self.batch_size = params.get('batch_size',128)
        self.data_balance = params.get('data_balance',False)
        
        # Metrics
        self.metrics     = [mean_squared_error, mean_absolute_error,r2_score]        
        

        # Model
        print(f"Building model: {UNET.get_model_name()} [{UNET.get_model_type()}]")
        self.model = self.create_model()

        #############################################################################

    

    def create_model(self,verbose=True):
        

        # Input layers
        self.input_layer = layers.Input(dtype = tf.float32,shape=[self.sequence_length,1],name='main_input')
        
    
        # Encoder 
        self.layers  = [Conv1D(self.filters,self.ksize,stride=1)]
        feats = self.filters

        for ii,_ in enumerate(range(self.nLayers-1)):
            self.layers.append(Conv1D(feats*2,self.ksize,name=f"conv_{ii}"))
            feats *= 2

        # Decoder
        for ii,_ in enumerate(range(self.nLayers-1)):
            self.layers.append(Up(feats//2,name=f"up_{ii}"))
            feats //=2

        self.layers.append(Conv1D(self.n_apps,kernel_size=1,stride=1,name="out_conv"))
             
        # Building model
        y = [self.layers[0](self.input_layer)]

        # building encoder
        for layer in self.layers[1:self.nLayers]:
            y.append(layer(y[-1]))
        
        # building decoder
        for ii,layer in enumerate(self.layers[self.nLayers:-1]):
            y[-1] = layer([ y[-1], y[-2 -ii]])

        self.zm = self.layers[-1](y[-1])
        
        # Output CNN Encoder
        self.zc = Encoder(self.filterCNNoutput,N_layers=self.nLayers//2)(self.zm)
        
        # Adaptative avg pooling
        poolStride = (self.zc.shape[1]//self.pool_size)  
        poolKernel = self.zc.shape[1] - (self.pool_size-1)*poolStride  
        self.zc = layers.AveragePooling1D(pool_size=poolKernel,strides=poolStride)(self.zc)

        # Flatten
        self.zc = layers.Flatten()(self.zc)

        # MLP h TODO: encapsulate this in a subclass MLP
        self.h = layers.Dense(self.Nh,activation=None)(self.zc)
        self.h = layers.BatchNormalization(name = f"batch_MLP")(self.h)
        self.h = tf.nn.leaky_relu(self.h)
        self.h = layers.Dropout(0.1)(self.h)


        self.power_quantiles = layers.Dense(self.n_apps*self.n_quantiles)(self.h)
        self.power_quantiles = layers.Reshape((self.n_apps,self.n_quantiles),name="out_power")(self.power_quantiles)
        
        
        
        # Keras model
        model = tf.keras.Model(inputs=[self.input_layer],outputs = [self.power_quantiles])


        model.compile( optimizer='adam',loss=qloss(self.quantiles))
        
        if verbose:
            model.summary()

        return model
    
    def train(self):
        print(f"Training model: {UNET.get_model_name()} [{UNET.get_model_type()}]")
        print("Prepocesing data")
        X, Y = self.preprocessing(self.X_train,self.Y_train,method='train')
        
        # Training M single target models
        self.training_hist = []
        
        
        train_X, v_X, train_Y,v_Y, =  train_test_split(X,Y, test_size=.15,random_state=10)  
        ES_cb = tf.keras.callbacks.EarlyStopping( monitor="val_loss",
                                        min_delta=0.0001,
                                        patience=self.patience,                                            
                                        baseline=None,
                                        restore_best_weights=True)
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
        
        self.training_hist = self.model.fit([train_X],[train_Y] ,
                                            batch_size=self.batch_size,
                                            epochs=self.epochs,
                                            validation_data=([v_X], [v_Y]),
                                            callbacks=[ES_cb])
        

        return self.training_hist        

    def predict_sample(self,X):
        """
            It predicts a unique sample
            PARAMETERS
                X [numpy array]  -> Input sample with the main consumption to be disaggregated.                 
            RETURN
                y [list] -> list of arrays with the disaggregations
        """
        
        X = X.reshape(-1,self.sequence_length,1)
        Y_est = self.model.predict(X)                   
        Y_est = [Y_est[:,app,self.n_quantiles//2] for app in range(self.n_apps)]
        return Y_est
    
    def predict(self,X_):
        """
            It predicts the time serie. It first gets the windowns and then the disaggregations will be computed
            PARAMETERS
                X [numpy array]  -> Input sample with the main consumption to be disaggregated. Without normalization                
            RETURN
                Y [list/numpy array] -> array or list of arrays  with the disaggregations. Without normalization
                Z [list/numpy array] -> array or list of arrays with the states.
        """
        X = np.copy(X_)
        N = len(X)
        # Normalization
        X = (X-self.main_mean)/self.main_std
        # Windowing
        X = np.pad(X,(self.sequence_length-1,0),'constant', constant_values=(0,0))
        X = get_windows(X,self.sequence_length,1)
        X = X.reshape(-1,self.sequence_length,1)
        # Prediction
        Y_est = self.model.predict(X,batch_size=500)
        Y = []
        for ii,app in enumerate(self.app_data.values()):
            y  = Y_est[:,ii,self.n_quantiles//2+1]            
            # Denormalization
            y  = (y * app["std"]) + app["mean"]
            Y.append(y[:N])
            

        return Y



    def evaluate(self, X_, Y_,Z,metrics):
        """
            It computes the metrics between Y_ and Z_ and the estimation obtained from X_. 
            PARAMETERS
                X_ [numpy array]  -> Input sample with the main consumption to be disaggregated. Without normalization                
                Y_ [numpy array]  -> Ground Truth output consumptions
                Z_ [numpy array]  -> Ground Truth output states
                metrics [list]    -> List with the metrcis to be applied
            RETURN
                SCORES [list] -> list with the scores for all the appliances and metrics
        """
        scores = []
        Y = np.copy(Y_)
        X = np.copy(X_)

        Y_est = self.predict(X)
        for metric in metrics:
            for ii,app in enumerate(self.apps):                
                scores.append([app,metric(Y[ii],Y_est[ii]),metric.__name__,self.get_model_name()] )
            
        return scores


    def store(self,path):
        if self.training_hist==None:
            raise Exception("The model has not been trained yet")
        
        savePath = os.path.join(path, f"{UNET.get_model_name()}")
        print(f"saving weights of model {UNET.get_model_name()} in path: {savePath}")
        self.model.save_weights(savePath)
        return None
    
    def load(self,path):
        
        if self.model==None:
            raise Exception("The model has not been defined yet")
        
        loadPath = os.path.join(path, f"{UNET.get_model_name()}")
        print(f"restoring weights of model {UNET.get_model_name()} from path: {loadPath}")
        self.model.load_weights(loadPath)
        return None


    def preprocessing(self,main,targets,method='train'):
        ###### Inputs ######
        # Windowing op.
        W_main = get_windows(main,self.sequence_length,self.stride)

        if method == "test":
            return W_main.reshape(-1,self.sequence_length,1)
        else:
            ##### Targets ######
            W_apps = []
            W_states = []
            for app in range(self.n_apps):                
                W_apps.append(get_windows(targets[app],self.sequence_length,self.stride)[:,[-1]])
            
            # Reshaping data for 2D conv layers
            W_main = W_main.reshape(-1,self.sequence_length,1)
            W_apps = np.hstack(W_apps)

            return W_main, W_apps

    
    @classmethod
    def get_model_type(cls):
        return "keras" # Aquí se puede indicar qué tipo de modelo es: RRNN, keras, scikit-lear, etc.
    
    @classmethod
    def get_model_name(cls):
        return "UNET" # Aquí se puede indicar un ID que identifique el modelo
    
    @classmethod
    def target(cls):
        return "multi-target" 
    
    @classmethod
    def tast(cls):
        return "single-task" 
    @classmethod
    def is_model_for(cls,name):
        return cls.get_model_name() == name 








##########################################
# Unit testing
##########################################


if __name__ == "__main__":
    pass