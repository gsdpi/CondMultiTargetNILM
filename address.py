import os

path_here = os.path.abspath('')
dataset_dir = str(path_here)+'/data/'
results_grid_search = str(path_here)+'/results/params/'
path_results_metrics = str(path_here)+'/results/'

def get_param_path(modelID):
    return os.path.join(results_grid_search,modelID+'.csv')
