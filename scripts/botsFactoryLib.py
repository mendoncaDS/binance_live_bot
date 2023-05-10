
import os
import json
import talib
import pickle
import numpy as np
import pandas as pd
import vectorbt as vbt

from matplotlib import pyplot as plt
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

def processOHL(data):
    """
    Generate OHLV.
    """
    newCols = pd.DataFrame()
    data["Open"] = data[["Open","Close"]].apply(lambda x: (x["Open"]-x["Close"])/x["Close"], axis=1)
    data["High"] = data[["High","Close"]].apply(lambda x: (x["High"]-x["Close"])/x["Close"], axis=1)
    data["Low"] = data[["Low","Close"]].apply(lambda x: (x["Low"]-x["Close"])/x["Close"], axis=1)
    
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

def genATR(data, timeperiods):
    """
    Generate ATR.
    """
    newCols = pd.DataFrame()
    for timeperiod in timeperiods:
        newCols["atr_{}m".format(timeperiod)] = talib.ATR(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)
    return newCols

def genOBV(data):
    """
    Generate OBV.
    """
    newCols = pd.DataFrame()
    newCols["OBV"] = talib.OBV(data["Close"],data["Volume"])
    return newCols

def genTechnicalIndicators(data, timeperiods):
       """
       Generate technical indicators.
       """

       # Close
       data = pd.concat([data, genBollingerBand(data, timeperiods=timeperiods, colname="Close")], axis=1)
       data = pd.concat([data, genRSI(data, timeperiods=timeperiods, colname="Close")], axis=1)
       data = pd.concat([data, genPercentChange(data, timeperiods=timeperiods, colname="Close")], axis=1)
       data = pd.concat([data, genATR(data, timeperiods=timeperiods)], axis=1)

       #Volume
       data = pd.concat([data, genBollingerBand(data, timeperiods=timeperiods, colname="Volume")], axis=1)
       data = pd.concat([data, genRSI(data, timeperiods=timeperiods, colname="Volume")], axis=1)
       data = pd.concat([data, genPercentChange(data, timeperiods=timeperiods, colname="Volume")], axis=1)
       data = pd.concat([data, genOBV(data)], axis=1)
       
       #OBV
       #data["OBV"] = talib.OBV(data["Close"],data["Volume"])

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
        if (i == "month") | (i == "day") | (i == "dayofweek") | (i == "hour") | (i == "minute") | (i == "Open") | (i == "High") | (i == "Low"):
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
        if (i == "month") | (i == "day") | (i == "dayofweek") | (i == "hour") | (i == "minute") | (i == "Open") | (i == "High") | (i == "Low"):
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
    dataCopy = data.copy()
    dataCopy = genDate(dataCopy)
    dataCopy = processOHL(dataCopy)
    dataCopy = genTechnicalIndicators(dataCopy,timePeriods)

    if scalers == None:
        scalers = rescale_load(dataCopy,modelName)
        return scalers

    dataCopy = rescale_supply(dataCopy,scalers)
    dataCopy = cyclicalTime(dataCopy)
    return dataCopy


# ----------


def genPreds(data,model,modelParamsDict,targetScaler,scalers):

    processedData = processData(data,modelParamsDict["timePeriods"],scalers=scalers)
    predictedMA = talib.SMA(data["Close"], timeperiod=modelParamsDict["rollingMeanWindow"]).rename("Current_SMA")
    preds = pd.DataFrame(targetScaler.inverse_transform(model.predict(processedData).reshape(1, -1))).T

    df = pd.concat([data["Close"][-len(preds):],predictedMA[-len(preds):]],axis=1)
    df["Prediction"] = preds.values
    df["Prediction"] = df["Prediction"]*df["Close"]/100

    return pd.DataFrame(df["Prediction"]).shift(1).dropna()

def genExposure(data,exposureConstant,exposureMultiplier,colname):
    return data.apply(lambda x: min(max(((x[colname]-x["Current_SMA"])/x["Current_SMA"])*exposureMultiplier+exposureConstant,0),1) if ((x["Shifted_Close"] < x[colname])) else 0, axis=1)

def genExposureDown(preds,exposureConstant,exposureMultiplier,colname):
    return preds.apply(lambda x: -max(min(((x[colname]-x["Current_SMA"])/x["Current_SMA"])*exposureMultiplier+exposureConstant,0),-1) if ((x["Shifted_Close"] > x[colname])) else 0, axis=1)

def backtestFromExposureSettings(data,exposureConstant,exposureMultiplier,colname,freq):

    exposure = genExposure(data,exposureConstant,exposureMultiplier,colname)
    pf = vbt.Portfolio.from_orders(
            data["Open"],
            exposure,
            size_type='targetpercent',
            freq=freq,
            )
    return pf

def genShiftedSMA(data,rollingMeanWindow, predictionHorizon):
    return talib.SMA(data["Close"],rollingMeanWindow).shift(-predictionHorizon+1)

def searchExposureBacktest(data,exposureConstants,exposureMultipliers,maPeriod,model,modelParamsDict,targetScaler,scalers):
    
    currentSMA = talib.SMA(data["Close"], timeperiod=maPeriod).rename("Current_SMA")

    shiftedSMA = pd.DataFrame({f"Shifted_SMA":talib.SMA(data["Close"],maPeriod).shift(-(maPeriod+1))})
    currentShiftedSMA = pd.concat([currentSMA,shiftedSMA["Shifted_SMA"]],axis=1).dropna()

    preds = genPreds(data,model,modelParamsDict,targetScaler,scalers)
    currentPredictedSMA = pd.concat([currentSMA,preds],axis=1).dropna()

    pfShiftedSMA_DF = pd.DataFrame()
    pfPred_DF = pd.DataFrame()

    for constant in exposureConstants:
        for multiplier in exposureMultipliers:

            print(f"_{constant}_{multiplier}_", end="\r", flush=True)

            pfShiftedSMA = backtestFromExposureSettings(pd.concat([data["Close"].shift(1).rename("Shifted_Close"),data["Open"],currentShiftedSMA],axis=1).dropna(),constant,multiplier,"Shifted_SMA",modelParamsDict["frequency"])
            pfShiftedSMA_DF.loc[f"{constant}_{multiplier}","Shifted_SMA"] = pfShiftedSMA.sharpe_ratio()

            pfPred = backtestFromExposureSettings(pd.concat([data["Close"].shift(1).rename("Shifted_Close"),data["Open"],currentPredictedSMA],axis=1).dropna(),constant,multiplier,"Prediction",modelParamsDict["frequency"])
            pfPred_DF.loc[f"{constant}_{multiplier}","Pred"] = pfPred.sharpe_ratio()
    
    backtests_DF = pd.concat([pfShiftedSMA_DF,pfPred_DF],axis=1)
    backtests_DF["diff"] = backtests_DF["Pred"]-backtests_DF["Shifted_SMA"]

    return backtests_DF

def loadModel(modelName, data):
    with open(f"models/{modelName}/{modelName}.pkl", 'rb') as pickle_file:
        model = pickle.load(pickle_file)

    with open( f"models/{modelName}/{modelName}_params.json", 'r' ) as file:
        modelParamsDict = json.load(file)

    with open(f"models/{modelName}/scalers/Target_scaler.pkl", 'rb') as pickle_file:
        targetScaler = pickle.load(pickle_file)

    scalers = processData(data,modelParamsDict["timePeriods"],modelName)

    return model, modelParamsDict, targetScaler, scalers



def backtestModels(maPeriod, exposureConstants, exposureMultipliers, pair, interval):
    
    if not os.path.exists("backtesting_ohlcv_data.csv"):
        start_date = "2017-01-01"
        data = vbt.BinanceData.download(
            pair,
            start=start_date,
            interval=interval).get(["Open", "High", "Low", "Close", "Volume"])
        data.to_csv("backtesting_ohlcv_data.csv")
    else:
        data = pd.read_csv("backtesting_ohlcv_data.csv", index_col=0)
        data.index = pd.to_datetime(data.index)
    
    modelName = f"articleModelSMA{maPeriod}"
    model, modelParamsDict, targetScaler, scalers = loadModel(modelName, data)
    
    backtests_DF = searchExposureBacktest(data, exposureConstants, exposureMultipliers, maPeriod, model, modelParamsDict, targetScaler, scalers)
    
    folder_name = f"modelBacktests/backtest_maPeriod_{maPeriod}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Sort and save the DataFrame ordered by each column
    backtests_DF_sorted_shifted_sma = backtests_DF.sort_values(by="Shifted_SMA",ascending=False)
    backtests_DF_sorted_shifted_sma.to_csv(f"{folder_name}/backtest_sorted_shifted_sma.csv")

    backtests_DF_sorted_pred = backtests_DF.sort_values(by="Pred",ascending=False)
    backtests_DF_sorted_pred.to_csv(f"{folder_name}/backtest_sorted_pred.csv")

    backtests_DF_sorted_diff = backtests_DF.sort_values(by="diff")
    backtests_DF_sorted_diff.to_csv(f"{folder_name}/backtest_sorted_diff.csv")
    
    fig, axs = plt.subplots(2, 1, figsize=(20,6), sharex=True)
    backtests_DF[["Shifted_SMA", "Pred"]].replace([np.inf, -np.inf], np.nan).hist(bins=100, ax=axs)
    plt.savefig(f"{folder_name}/histogram.png")
    plt.close(fig)


def backtest_model_bayes(params, model):

    exposureConstant = params[0]
    exposureMultiplier = params[1]

    if not os.path.exists("backtesting_ohlcv_data.csv"):
        start_date = "2017-01-01"
        data = vbt.BinanceData.download(
            "BTCUSDT",
            start=start_date,
            interval="1h").get(["Open", "High", "Low", "Close", "Volume"])
        data.to_csv("backtesting_ohlcv_data.csv")
    else:
        data = pd.read_csv("backtesting_ohlcv_data.csv", index_col=0)
        data.index = pd.to_datetime(data.index) 
    
    modelName = f"articleModelSMA{model}"
    loadedModel, modelParamsDict, targetScaler, scalers = loadModel(modelName, data)

    currentSMA = talib.SMA(data["Close"], timeperiod=model).rename("Current_SMA")

    preds = genPreds(data,loadedModel,modelParamsDict,targetScaler,scalers)
    currentPredictedSMA = pd.concat([currentSMA,preds],axis=1).dropna()

    error = -backtestFromExposureSettings(pd.concat([data["Close"].shift(1).rename("Shifted_Close"),data["Open"],currentPredictedSMA],axis=1).dropna(),exposureConstant,exposureMultiplier,"Prediction","1h").sharpe_ratio()
    
    if np.isinf(error) or np.isnan(error):
        return 1000000
    
    return error
