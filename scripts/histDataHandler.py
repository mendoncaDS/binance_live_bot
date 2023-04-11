
import dateutil.relativedelta

import pandas as pd
import datetime as dt

from binance_historical_data import BinanceDataDumper


def intToStringWithLeadingZeros(integer,numberOfDigits):
    string = str(integer)
    while len(string) < numberOfDigits:
        string = "0" + string
    return string

def dumpSuchData(pathToDump,ticker,frequency,startDate,endDate):
    data_dumper = BinanceDataDumper(
        path_dir_where_to_dump=pathToDump,
        data_type="klines",
        data_frequency=frequency,
    )
    data_dumper.dump_data(
        tickers=ticker,
        date_start=startDate,
        date_end=endDate,
        is_to_update_existing=False
    )

def loadOneMonthOf1mData(year,month,frequency):
    data = pd.read_csv(f"data/spot/monthly/klines/BTCUSDT/{frequency}/BTCUSDT-{frequency}-{year}-{intToStringWithLeadingZeros(month,2)}.csv",header=None)
    data[0] = pd.to_datetime(data[0], unit="ms")
    data.set_index(0,inplace=True)
    data = data.iloc[:,0:5]
    data.rename(columns={1:"Open",2:"High",3:"Low",4:"Close",5:"Volume"},inplace=True)
    data.index.name = "Timestamp"
    return data

def loadSuchData(ticker,frequency,startDate,endDate=None,pathToDump="data"):
    
    if type(startDate) == str:
        startDate = dt.datetime.strptime(startDate,"%Y-%m-%d").date()
    if endDate == None:
        endDate = dt.datetime.now().date()
    elif type(endDate) == str:
        endDate = dt.datetime.strptime(endDate,"%Y-%m-%d").date()

    dumpSuchData(pathToDump,ticker,frequency,startDate,endDate)
    
    if endDate.replace(day=1) == dt.datetime.now().date().replace(day=1):
        endDate = endDate - dateutil.relativedelta.relativedelta(months=1)
    startYear = startDate.year
    startMonth = startDate.month
    endYear = endDate.year
    endMonth = endDate.month

    dataList = []

    if startYear == endYear:
        for month in range(startMonth,endMonth+1):
            dataList.append(loadOneMonthOf1mData(startYear,month,frequency))
    else:
        for month in range(startMonth,13):
            dataList.append(loadOneMonthOf1mData(startYear,month,frequency))
        for year in range(startYear+1,endYear):
            for month in range(1,13):
                dataList.append(loadOneMonthOf1mData(year,month,frequency))
        for month in range(1,endMonth+1):
            dataList.append(loadOneMonthOf1mData(endYear,month,frequency))
    if len(dataList)>0:
        return pd.concat(dataList)
    else:
        return None
