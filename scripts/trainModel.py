
print("Importing libraries...")
import json
import skopt
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
import botsFactoryLib as botsFactoryLib

from histDataHandler import loadSuchData
from sklearn.metrics import mean_squared_error

modelName = "tradingViewEyeBacktestModel"

timePeriods = [
    2,
    6,
    12,
    24,
    4*24,
    8*24,
    14*24,
    20*24,
    30*24,
    45*24,
    3*30*24,
    6*30*24,
    ]

rollingMeanWindow = 24
predictionHorizon = 30
frequency="1h"

modelParamsDict = {
    "timePeriods":timePeriods,
    "rollingMeanWindow":rollingMeanWindow,
    "predictionHorizon":predictionHorizon,
    "frequency":frequency,
}

initialDate = "2018-01-01"

BTCUSDT_1m = loadSuchData(startDate=initialDate,ticker="BTCUSDT",frequency=frequency)

print("generating target")
BTCUSDT_1m = botsFactoryLib.genTarget(data=BTCUSDT_1m, rollingMeanWindow=rollingMeanWindow, predictionHorizon=predictionHorizon)

print("generating date features")
BTCUSDT_1m = botsFactoryLib.genDate(BTCUSDT_1m)

print("generating technical indicators")
BTCUSDT_1m = botsFactoryLib.genTechnicalIndicators(BTCUSDT_1m,timePeriods)

print("rescaling")
BTCUSDT_1m = botsFactoryLib.rescale_gen(BTCUSDT_1m,modelName,save=True)

print("generating cyclical features")
BTCUSDT_1m = botsFactoryLib.cyclicalTime(BTCUSDT_1m)



def splitDataRandom(
    TrainSize,
    ValSize,
    TestSize,
    data):
    """
    Split data into Train, Val and Test sets.
    Each sample in Train, Val and Test are randomly picked from data
    1- randomly pick a sample from data
    2- roll a random number between 0 and totSize
    3- if the number is smaller than TrainSize, add the sample to Train, else if it is smaller than TrainSize+ValSize, add the sample to Val, else add the sample to Test
    4- remove the sample from data so it doesn't get picked again
    """
    totSize = TrainSize+ValSize+TestSize
    trainCutOff = TrainSize/totSize
    valCutOff = (TrainSize+ValSize)/totSize

    # initialize Train, Val and Test
    TrainSel = []
    ValSel = []
    TestSel = []

    dataLen = len(data)

    #initialize loop
    for i in range(dataLen):
        # roll random float between 0 and 1
        randomRoll = np.random.rand()

        # add sample to Train, Val or Test
        if randomRoll < trainCutOff:
            TrainSel.append(i)
        elif randomRoll < valCutOff:
            ValSel.append(i)
        else:
            TestSel.append(i)
    
    Train = data.iloc[TrainSel]
    Val = data.iloc[ValSel]
    Test = data.iloc[TestSel]

    return Train, Val, Test

x_all = BTCUSDT_1m.drop(["Target"],axis=1)
y_all = BTCUSDT_1m["Target"]

Train, Val, Test = splitDataRandom(TrainSize=0.84,ValSize=0.08,TestSize=0.08,data=BTCUSDT_1m)

x_train = Train.drop(["Target"],axis=1)
y_train = Train["Target"]

x_val = Val.drop(["Target"],axis=1)
y_val = Val["Target"]

x_test = Test.drop(["Target"],axis=1)
y_test = Test["Target"]

def rmse_error(y, yhat):
    return np.sqrt( mean_squared_error(y, yhat))

def train_model(params):
    learning_rate = params[0]
    subsample = params[1]
    colsample_bytree = params[2]
    max_depth = params[3]
    print(params)
    model_xgb = xgb.XGBRegressor(
        eta=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_depth =max_depth,
        tree_method='gpu_hist',
        predictor="gpu_predictor",
    )
    model_xgb.fit(x_train, y_train)
    yhat = model_xgb.predict(x_val)
    error = rmse_error(y_val, yhat)
    print(f"ERROR: {error}")
    return error

space = [
        (.001, .6, 'log-uniform'), #learning rate
        (0.2, 1.0),    # subsample         
        (0.1, 1.0),     # colsample bytree  
        (5, 12)         # max_depth         
        ]

results_gp = skopt.gp_minimize(train_model, space, random_state=42, verbose=1, n_calls=50, n_random_starts=10)

params = results_gp.x

learning_rate = params[0]
subsample = params[1]
colsample_bytree = params[2]
max_depth = params[3]

model_xgb_bayes = xgb.XGBRegressor(
        eta=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_depth =max_depth,
        tree_method='gpu_hist',
        predictor="gpu_predictor",
    )

def saveModel(model, modelName, modelParamsDict):
    with open( f"models/{modelName}/{modelName}.pkl", 'wb' ) as file:
        pickle.dump( model, file )
    # save model params dict to same folder
    with open( f"models/{modelName}/{modelName}_params.json", 'w' ) as file:
        json.dump( modelParamsDict, file )

selectedModel = model_xgb_bayes.fit(x_all, y_all)
saveModel(selectedModel, modelName, modelParamsDict)

