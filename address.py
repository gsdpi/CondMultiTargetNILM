import os

path_here = os.path.abspath('')
dataset_dir = str(path_here)+'/data/'
model_params_path= str(path_here)+'/results/params/'
path_results_metrics = str(path_here)+'/results/'

def get_param_path(modelID):
    return os.path.join(model_params_path,modelID+'.json')
def get_model_path(dataset,modelID):
    return os.path.join(path_results_metrics,dataset,modelID)