
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





class dAE(BaseModel):
    def __init__(self,data, params: dict, **kwargs) -> None:      
        
        # Data
        self.X_train = data["X_train"]
        self.Y_train = data["Y_train"]

        self.training_hist =None
        self.model =None

        # Net params
        self.main_mean       = data["main_data"][0]
        self.main_std       = data["main_data"][1]
        self.app_data        = data["app_data"]
        self.apps            = list(self.app_data.keys())
        self.n_apps          = len(self.apps)
        
        # Net params
        self.sequence_length = params.get('sequence_length',100)
        self.stride = params.get('stride',10)
        

        # Training params
        self.epochs = params.get('epochs', 10)
        self.patience = params.get('patience', 15)
        self.batch_size = params.get('batch_size',128)

                

        # Model
        print(f"Building model: {dAE.get_model_name()} [{dAE.get_model_type()}]")
        self.model = []
        for app in range(self.n_apps):
            self.model.append(self.create_model())

        #############################################################################


    # Implementation of dAE from: https://github.com/nilmtk/nilmtk-contrib/blob/master/nilmtk_contrib/disaggregate/dae.py 
    def create_model(self,verbose=True):
        
        model = tf.keras.models.Sequential()
        model.add(layers.Conv1D(8, 4, activation="linear", input_shape=(self.sequence_length, 1), padding="same", strides=1))
        model.add(layers.Flatten())
        model.add(layers.Dense((self.sequence_length)*8, activation='relu'))
        model.add(layers.Dropout(.2))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense((self.sequence_length)*8, activation='relu'))
        model.add(layers.Dropout(.2))
        model.add(layers.Reshape(((self.sequence_length), 8)))
        model.add(layers.Conv1D(1, 4, activation="linear", padding="same", strides=1))
        model.compile(loss='mse', optimizer='adam')
        if verbose:
            model.summary()

        return model
                

    
    def train(self):
        print(f"Training model: {dAE.get_model_name()} [{dAE.get_model_type()}]")
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
            
            self.training_hist.append(model_app.fit([train_X], [train_Y] ,
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
        X = X.reshape(-1,self.sequence_length,1)
                
        for app in range(self.n_apps):
            
            y.append(self.model[app].predict(X))

        return y
    
    def predict(self,X_):
        """
            It predicts the time serie. It first gets the windowns and then the disaggregations will be computed
            PARAMETERS
                X [numpy array]  -> Input sample with the main consumption to be disaggregated. Without normalization                
            RETURN
                Y [list/numpy array] -> array or list of arrays (if SD == None) with the disaggregations. Without normalization
        """
        N = len(X_)
        X = np.copy(X_)
        self.stride=self.sequence_length//3
        X = (X-self.main_mean)/self.main_std
        X = self.preprocessing(X,None,method='test')
        Y = []
        for ii,app in enumerate(self.app_data.values()):
            y_w = self.model[ii].predict([X],batch_size=300)
            y  = agg_windows(y_w,self.sequence_length,self.stride)
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
            for ii,app in enumerate(self.app_data.values()):                
                scores.append([self.apps[ii],metric(Y[ii],Y_est[ii]),metric.__name__,self.get_model_name()] )
            
        return scores

    def store(self,path):
        if self.training_hist==None:
            raise Exception("The model has not been trained yet")
        for ii,app in enumerate(self.apps):
            savePath = os.path.join(path, f"{dAE.get_model_name()}_{app}")
            print(f"saving weights of model {dAE.get_model_name()} and individual consump. {app} in path: {savePath}")
            self.model[ii].save_weights(savePath)
        return None
    
    def load(self,path):
        
        if self.model==None:
            raise Exception("The model has not been defined yet")
        for ii,app in enumerate(self.apps):
            loadPath = os.path.join(path, f"{dAE.get_model_name()}_{app}")
            print(f"restoring weights of model {dAE.get_model_name()} from path: {loadPath}")
            self.model[ii].load_weights(loadPath)
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
            S_D    = []
            for app in range(self.n_apps):                
                W_apps.append(get_windows(targets[app],self.sequence_length,self.stride))
            
            # Reshaping data for 2D conv layers
            W_main = W_main.reshape(-1,self.sequence_length,1)
            return W_main, W_apps

    
    @classmethod
    def get_model_type(cls):
        return "keras" # Aquí se puede indicar qué tipo de modelo es: RRNN, keras, scikit-lear, etc.
    
    @classmethod
    def get_model_name(cls):
        return "dAE" # Aquí se puede indicar un ID que identifique el modelo
    
    @classmethod
    def target(cls):
        return "single-target" 
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