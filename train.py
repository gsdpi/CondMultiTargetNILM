#SCRIPT FOR TRAINING THE MODELS

import ipdb
import numpy as np
from data import HOSPData,UKDALEData
import matplotlib.pyplot as plt
from modelGen import modelGen
import argparse
import warnings
warnings.filterwarnings("ignore")
from models.utils import reset_seeds
from address import get_model_path



reset_seeds(seed_value=39)

# Defining program params
parser = argparse.ArgumentParser(description="Script to train the UKDALE models.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-dataset",    action="store", help="selected dataset: UKDALE or HOSP ",default="UKDALE")
parser.add_argument("-trainHouse", action="store", help="train house for UKDALE", type = int,  default=1)
parser.add_argument("-testHouse",  action="store", help="test house for UKDALE" , type = int,  default=1)
parser.add_argument("-trainDates", action="store", help="Dates for training",     nargs='+', default=["2015-01-01","2016-01-01"])
parser.add_argument("-testDates",  action="store", help="Dates for test",        nargs='+', default=["2016-01-01","2016-07-01"])
parser.add_argument("-models",     action="store", help="Models to be trained",   nargs='+', default=["FCNdAE","multiFCNdAE","multiUNET"])
parser.add_argument("-seqLen",    action="store", help="Sequence length ",       type = int, default= 500)
parser.add_argument("-epochs",    action="store", help="Training epochs ",       type = int, default= 100)

args = parser.parse_args()

DATASET        = args.dataset
TRAINING_HOUSE = args.trainHouse
TEST_HOUSE     = args.testHouse
TRAINING_DATES = args.trainDates
TEST_DATES     = args.testDates
MODELS         = args.models
SEQ_LEN        = args.seqLen     
EPOCHS         = args.epochs   

# Reading data
if DATASET == "UKDALE":
    dataGen = UKDALEData(path="./data/")
    trainMain,trainTargets, trainStates = dataGen.get_train_sequences(houses = TRAINING_HOUSE, start = TRAINING_DATES[0],end=TRAINING_DATES[1])
    testMain,testTargets, testStates = dataGen.get_train_sequences(houses = TEST_HOUSE, start=TEST_DATES[0],end=TEST_DATES[1],norm = False)
    app_data = dataGen.get_app_data()
else:
    dataGen = HOSPData(path="./data/")
    trainMain,trainTargets, trainStates = dataGen.get_train_sequences(start = TRAINING_DATES[0],end=TRAINING_DATES[1])
    testMain,testTargets, testStates = dataGen.get_test_sequences( start=TEST_DATES[0],end=TEST_DATES[1],norm=False)
    app_data = dataGen.get_app_data()
# Data dict
data= {"X_train":trainMain,
        "Y_train":trainTargets,
        "Y_states":trainStates,
        "Z_train":trainStates,
        "X_test":testMain,
        "Y_test":testTargets,
        "Z_test":testStates,
        "app_data":dataGen.get_app_data(),
        "main_data": dataGen.get_main_mean_std()
        }  
 
print(f"EL modelo es {MODELS}")
for modelID in MODELS:
   params = {"epochs":EPOCHS,"sequence_length":SEQ_LEN}
   model = modelGen(modelID,data,params)
   model.train()
   model.store(get_model_path(DATASET,modelID))
   del model





