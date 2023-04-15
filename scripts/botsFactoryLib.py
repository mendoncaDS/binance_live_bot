
import os
import pickle
import talib

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def genTarget(data, rollingMeanWindow, predictionHorizon):
    """
    Generate target column for prediction.
    The target column is the rolling mean of the Close price.
    The rolling mean is shifted by predictionHorizon.
    """
    currentMean = talib.MA(data["Close"],timeperiod=rollingMeanWindow)
    futureMean = talib.MA(data["Close"],timeperiod=rollingMeanWindow).shift(-predictionHorizon)
    # compute percentual difference between current and future mean
    target = 100+((futureMean-currentMean)*100/currentMean)
    return pd.concat([data,target.rename("Target")],axis=1)

def genDate(data):
    """
    Generate date columns from index.
    """

    data["month"] = data.index.month
    data["day"] = data.index.day
    data["dayofweek"] = data.index.dayofweek
    data["hour"] = data.index.hour

    return data

def genBollingerBand(data, timeperiods, colname):
    """
    Generate Bollinger Bands.
    """
    newCols = pd.DataFrame()
    for timeperiod in timeperiods:
        newCols["{}_upper_band_{}m".format(colname, timeperiod)], newCols["{}_middle_band_{}m".format(colname, timeperiod)], newCols["{}_lower_band_{}m".format(colname, timeperiod)] = talib.BBANDS(data[colname], timeperiod=timeperiod)
    return newCols

def genRSI(data, timeperiods, colname):
    """
    Generate RSI.
    """
    newCols = pd.DataFrame()
    for timeperiod in timeperiods:
        newCols["{}_rsi_{}m".format(colname, timeperiod)] = talib.RSI(data[colname], timeperiod=timeperiod)
    return newCols

def genPercentChange(data, timeperiods, colname):
    """
    Generate percent change.
    """
    newCols = pd.DataFrame()
    for timeperiod in timeperiods:
        newCols["{}_percent_change_{}m".format(colname, timeperiod)] = data[colname].pct_change(timeperiod)
    return newCols

def genTechnicalIndicators(data, timeperiods):
       """
       Generate technical indicators.
       """

       # Close
       data = pd.concat([data, genBollingerBand(data, timeperiods=timeperiods, colname="Close")], axis=1)
       data = pd.concat([data, genRSI(data, timeperiods=timeperiods, colname="Close")], axis=1)
       data = pd.concat([data, genPercentChange(data, timeperiods=timeperiods, colname="Close")], axis=1)

       #Volume
       data = pd.concat([data, genBollingerBand(data, timeperiods=timeperiods, colname="Volume")], axis=1)
       data = pd.concat([data, genRSI(data, timeperiods=timeperiods, colname="Volume")], axis=1)
       data = pd.concat([data, genPercentChange(data, timeperiods=timeperiods, colname="Volume")], axis=1)

       # drop all na and infinite values
       data = data.replace([np.inf, -np.inf], np.nan)
       data = data.dropna()

       return data

def rescale_gen(data,modelName,save=False):
    """
    Rescale data using MinMaxScaler and save scaler to model folder
    """
    mms = MinMaxScaler()
    dir = f"models/{modelName}/scalers/"
    # if dir doesnt exist, create it
    if not os.path.exists(dir):
        os.makedirs(dir)

    for i in data.columns:
        # do not scale cyclical features
        # month	day	dayofweek	hour	minute
        if (i == "month") | (i == "day") | (i == "dayofweek") | (i == "hour") | (i == "minute"):
            continue
        data[i] = mms.fit_transform(data[[i]])
        if save: pickle.dump(mms, open(f"{dir}/{i}_scaler.pkl", 'wb'))

    return data

# rescale function to load rescalers from folder and apply them to corresponding columns
def rescale_load(data,modelName):
    """
    Rescale data using existing scalers from model folder
    """
    scalers = {}
    for i in data.columns:
        if (i == "month") | (i == "day") | (i == "dayofweek") | (i == "hour") | (i == "minute"):
            continue
        
        mms = pickle.load(open(f'models/{modelName}/scalers/{i}_scaler.pkl', 'rb'))
        scalers[f"{i}_scaler"] = mms
        data[i] = mms.transform(data[[i]])

    return scalers

def rescale_supply(data,scalers):
    """
    Rescale data using existing scalers from model folder
    """
    for i in data.columns:
        if (i == "month") | (i == "day") | (i == "dayofweek") | (i == "hour") | (i == "minute"):
            continue

        data[i] = scalers[f"{i}_scaler"].transform(data[[i]])

    return data

def cyclicalTime(data):

    data['month_sin'] = data['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
    data['month_cos'] = data['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

    data['day_of_month_sin'] = data['day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
    data['day_of_month_cos'] = data['day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )

    data['dayofweek_cos'] = data['dayofweek'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )
    data['dayofweek_sin'] = data['dayofweek'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )

    data['hour_sin'] = data['hour'].apply( lambda x: np.sin( x * ( 2. * np.pi/24 ) ) )
    data['hour_cos'] = data['hour'].apply( lambda x: np.cos( x * ( 2. * np.pi/24 ) ) )


    # drop month, day, dayofweek, hour and minute columns from data
    data = data.drop([
        'month',
        'day',
        'dayofweek',
        'hour',
        ], axis=1)

    return data

def processData(data, timePeriods, modelName=None, scalers=None):
    data = genDate(data)
    data = genTechnicalIndicators(data,timePeriods)

    if scalers == None:
        scalers = rescale_load(data,modelName)
        return scalers

    data = rescale_supply(data,scalers)
    data = cyclicalTime(data)
    return data


