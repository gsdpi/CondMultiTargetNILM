# Implementation of tcn from https://arxiv.org/pdf/1803.01271.pdf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

class BaseModel(object):
    def __init__(self,data, params: dict, **kwargs) -> None:      
        #############################################################################
        # Aquí se tratan los parámetros del modelo. Esto es necesario porque estos modelos contienen muchos hiperparámetros
        
        self.N_capas = params.get("N_capas", 3)
        self.testMetrics = []
        self.metrics     = [mean_squared_error, mean_absolute_error,r2_score]
        #############################################################################
        # Los datos de entrenamiento vienen en el parametro data:
        #     - Vienen pre-procesados.
        #     - data suele ser un objeto o diccionario con: 
        #         data.X_Train
        #         data.Y_Train
        #         data.X_Test
        #         data.Y_Test
        # El formato del objeto Data puede variar de aplicación en aplicación    
        self.X_train = data.X_train
        self.X_test  = data.X_test
        self.y_train = data.y_train
        self.y_test  = data.y_test


        #############################################################################

        # También se crea el modelo. Si es una red aquí se define el grafo. 
        # La creación del modelo se encapsula en la función "create_model"
        # Ejemplo de lectura de parámetros:
        #    param1 = params.get("N_capas", 3)

        self.model = self.create_model() 

        #############################################################################

    def create_model(self):
        # Aquí se define la red, SVC, árbol, ..., lo que sea.

        model = None

        return model
    
    def train(self):
        
        # Se lanza el entrenamiento de los modelos. El código para lanzar el entrenamiento depende mucho del modelo. 

        # Cuanda acaba el entrenamiento y obtenemos los pesos óptimos, las métricas de error para los datos de test son calculadas.
        self.y_test_est = self.predict(self.X_test)
        self.testMetrics = [metric(self.y_test,self.y_test_est) for metric in self.metrics] 
        

    def predict(self,X):
        # Método para predicir una o varias muestras.        
        # El código puede variar dependiendo del modelo
        return self.model.predict(X)
        
    def store(self,path):
        # Método para guardar los pesos en path 
        return None
    
    def load(self, path):
        # Método para cargar los pesos desde el path indicado 
        return None

    ##########   MÉTODOS DE LAS CLASES    ##########
    # Estos métodos se pueden llamar sin instar un objeto de la clase
    # Ej.: import model; model.get_model_type()
    
    @classmethod
    def get_model_type(cls):
        return "Generico" # Aquí se puede indicar qué tipo de modelo es: RRNN, keras, scikit-lear, etc.
    
    @classmethod
    def get_model_name(cls):
        return "esn" # Aquí se puede indicar un ID que identifique el modelo
    

    
##########################################
# Unit testing
##########################################


if __name__ == "__main__":
    # Este código solo se ejecuta si el script de ejecución principal es BaseModel.py:
    #   run BaseModel.py
    
    # Aquí se puede escribir un código de prueba para probar por separado 
    
    pass