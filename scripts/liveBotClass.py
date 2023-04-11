
import os
import json
import math
import pytz
import talib
import socket
import pickle
import asyncio

import pandas as pd
import datetime as dt

from dotenv import dotenv_values
from vectorbt import BinanceData
from binance.client import Client

from botsFactoryLib import processData

class tradingBot:
    
    # Initialize the class
    def __init__(
        self,
        modelName,
        symbol,
        minPctChange,
        exposureMultiplier,
        API,
        ):

        self.name = f"{modelName}_bot"
        print(f"\nHello, I'm {self.name}.\n")
        self.symbol = symbol
        self.minPctChange = minPctChange
        self.exposureMultiplier = exposureMultiplier
        self.API = API
        self.modelName = modelName

        self.initializeBot()

        print("Starting trading.\n")
    #-----


    
    ##### INITIALIZATION METHODS #####
    def initializeBot(self):
        # INITIALIZE FIRST VARIABLES
        self.initializeFolder()

        # INITIALIZE LOG
        self.initializeLog()

        # INITIALIZE MARKET DATA
        self.initializeData()
        
        # INITIALIZE PREDICTIONS
        self.initializePredictions()

        # LOAD MODEL AND SCALERS
        self.initializedModelAndScalers()

    def initializeFolder(self):
    
        self.botFolderPath = f"botsStates/{self.name}"

        if not os.path.exists(self.botFolderPath):
            os.makedirs(self.botFolderPath)

    def initializeLog(self):
        if not os.path.exists(f"{self.botFolderPath}/{self.name}_log.csv"):
            pd.DataFrame([
                'symbol',
                'pfValue',
                'orderId',
                'executedQty',
                'cummulativeQuoteQty',
                'effectivePrice',
                'side',
                'status',
                'type',
                'transactTime']).T.to_csv(f"{self.botFolderPath}/{self.name}_log.csv",header=0,index=False)

    def initializeData(self):
        with open( f"models/{self.modelName}/{self.modelName}_params.json", 'r' ) as file:
            self.modelParamsDict = json.load( file )
        self.dataLength = int(max(self.modelParamsDict["timePeriods"])*2)
        self.Portfolio = pd.DataFrame()
        if not os.path.exists(f"{self.botFolderPath}/{self.name}_data.csv"):
            print("Data not found, downloading.\n")
            self.marketData = BinanceData.download(
                self.symbol,
                start = (dt.datetime.now(pytz.timezone("UTC"))-dt.timedelta(minutes=self.dataLength)).strftime("%Y-%m-%d %H:%M"),
                interval="1m").get(["Open", "High", "Low", "Close", "Volume"])
            self.marketData = pd.concat([self.marketData[:"2023-03-24 11:27:00+00:00"],self.marketData["2023-03-24 14:00:00+00:00":]])
            self.marketData.to_csv(f"{self.botFolderPath}/{self.name}_data.csv")

        # else, read the csv into a dataframe, and load it into self.marketData
        else:
            print("Data found, loading.\n")
            self.marketData = pd.read_csv(f"{self.botFolderPath}/{self.name}_data.csv",index_col=0,parse_dates=True)

        self.lastMarketDataTS = self.marketData.index[-1]
        self.refreshPortfolio()
        print("Data initialized.\n")

    def initializePredictions(self):
        
        if not os.path.exists(f"{self.botFolderPath}/{self.name}_preds.csv"):
            pd.DataFrame([
                'Timestamp',
                'pfVal',
                'cryptoRatio',
                'prediction'
            ]).T.to_csv(f"{self.botFolderPath}/{self.name}_preds.csv",mode="a",header=0,index=False)
    
    def initializedModelAndScalers(self):
        with open(f"models/{self.modelName}/{self.modelName}.pkl", 'rb') as pickle_file:
            self.model = pickle.load(pickle_file)
        self.targetScaler = pickle.load(open(f"models/{self.modelName}/scalers/Target_scaler.pkl", 'rb'))
        self.scalers = processData(self.marketData,self.modelParamsDict["timePeriods"],self.modelName)
    #-----



    ##### REFRESH BOT #####
    def refreshAll(self):
        # update balances and average price
        self.refreshPortfolio()
        # update market data
        self.refreshSaveData()
        # update prediction to enable decision making
        self.refreshPred()

    def refreshPortfolio(self):
        
        while True:
            try:
                client = Client(self.API[0], self.API[1], testnet=True)
                balances = pd.DataFrame.from_records(client.get_account()["balances"])
                self.price = float(client.get_avg_price(symbol=self.symbol)["price"])
                client.close_connection()
            
            except Exception as e:
                if isinstance(e, socket.error):
                    print(f"Connection error:\n{e}")
                else:
                    print(f"Ooops there was a problem refreshing the portfolio:\n{e}")
            else:
            
                newNominal = balances[(balances["asset"]==self.symbol[:-4]) | (balances["asset"]=="USDT")]["free"].values


                self.Portfolio["nominal"] = newNominal
                self.Portfolio["nominal"] = self.Portfolio["nominal"].astype("float")

                self.Portfolio["inUSD"] = self.Portfolio["nominal"]*[self.price,1]
                
                self.pfValUSD = self.Portfolio["inUSD"].sum()
                self.pfValNonUSD = (self.Portfolio["inUSD"]/[self.price,self.price]).sum()

                self.cryptoRatio = self.Portfolio["inUSD"][0]/self.Portfolio["inUSD"].sum()
                break
    
    def refreshSaveData(self):
        
        if dt.datetime.now(pytz.timezone("UTC")) - dt.datetime.fromtimestamp(self.lastMarketDataTS.value/1000000000,tz=pytz.timezone("UTC")) > dt.timedelta(seconds=121):
            newRow = BinanceData.download(
                self.symbol,
                start = self.lastMarketDataTS + dt.timedelta(minutes=1),
                end = dt.datetime.now(pytz.timezone("UTC")) - dt.timedelta(minutes=1),
                interval="1m").get(["Open", "High", "Low", "Close", "Volume"])
            
            self.marketData = pd.concat([self.marketData,newRow]).iloc[-self.dataLength:,:]
            self.lastMarketDataTS = self.marketData.index[-1]

            # SAVE
            newRow.to_csv(f"{self.botFolderPath}/{self.name}_data.csv",mode="a",header=False)

    def refreshPred(self):

        processedData = processData(self.marketData,self.modelParamsDict["timePeriods"],scalers=self.scalers)
        modelPred = self.model.predict(processedData.iloc[-1:])
        descaledModelPred = self.targetScaler.inverse_transform(modelPred.reshape(-1, 1))[0][0]
        self.currentPrediction = (descaledModelPred/100)*self.price
    
    def saveFinish(self):
        self.refreshPortfolio()
        predDict = {"Timestamp":self.lastMarketDataTS,"pfVal":self.pfValUSD,"cryptoRatio":self.cryptoRatio,"prediction":self.currentPrediction}
        predDF = pd.DataFrame(predDict,index=[0])
        predDF.to_csv(f"{self.botFolderPath}/{self.name}_preds.csv",mode="a",index=False,header=False)
    #-----



    ##### MAKE ORDERS #####
    def smartSignals(self):
        
        # in this example the model predicts a moving average. predictedMA is the current value of the moving average the model predicts
        predictedMA = talib.MA(self.marketData["Close"],timeperiod=self.modelParamsDict["rollingMeanWindow"]).iloc[-1]

        # if price is above current and predicted MA, this means the price is going down. thus, sell all
        if (self.price > predictedMA) & (self.price > self.currentPrediction):
            self.targetRatio = 0

        # else, compute predicted percentual change of the MA and use it to generate the target ratio of exposure
        else:
            percentualPred = (self.currentPrediction-predictedMA)/predictedMA

            # the operation below is mostly arbitrary and can be optimized through backtests
            # it converts the predicted percentual change of the MA into a target ratio of exposure
            self.targetRatio = min(max(percentualPred*self.exposureMultiplier,0),1)
        
        # if the target ratio is below minPctChange, set it to 0
        if self.targetRatio < self.minPctChange:
            self.targetRatio = 0
        
        # with targetRatio set, we now place orders to achieve such ratio
        self.achieveIdealPortfolio()

    def achieveIdealPortfolio(self,saveOrder=True):
        
        # utility functions (basically to avoid minimum notional)
        def upThenDown():
            self.placeOrder(sell=False,amount=(self.pfValNonUSD*(1-self.cryptoRatio)),saveOrder=saveOrder, refreshPf=True)
            self.placeOrder(sell=True,amount=(self.pfValNonUSD*(1-self.targetRatio)),saveOrder=saveOrder)
        def downThenUp():
            self.placeOrder(sell=True,amount=(self.pfValNonUSD*self.cryptoRatio),saveOrder=saveOrder, refreshPf=True)
            self.placeOrder(sell=False,amount=(self.pfValNonUSD*self.targetRatio),saveOrder=saveOrder)

        # if the difference between the target ratio and the current ratio is less than minPctChange, do nothing
        percentChange = self.targetRatio-self.cryptoRatio
        if abs(percentChange)>self.minPctChange:
            
            minNotionalThreshold = 12
            minNotionalRatio = minNotionalThreshold/self.pfValUSD
            
            # avoid minimum notional
            if abs(percentChange)<minNotionalRatio:
                if (self.cryptoRatio>1.2*minNotionalRatio):
                    downThenUp()
                else:
                    upThenDown()
                    
            else:
                self.placeOrder(sell=(percentChange<0),amount=abs(self.pfValNonUSD*percentChange),saveOrder=saveOrder)
           
    def placeOrder(self, sell, amount, saveOrder=True, refreshPf=False):
        
        def roundAndSendOrder(self, client, sell, amount):
            
            amountToOrder = math.floor(amount*10000)/10000
            print(f"\n-----> {'SELL' if sell else 'BUY'} {amountToOrder} {self.symbol[:-4]} | {round(amountToOrder*self.price,2)} USD | {round((amountToOrder*self.price*100)/self.pfValUSD,2)}% <-----\n")

            if sell:
                order = client.order_market_sell(
                    symbol= self.symbol,
                    quantity = amountToOrder)
            else:
                order = client.order_market_buy(
                    symbol= self.symbol,
                    quantity = amountToOrder)
            
            return order

        while True:
            try:
                client = Client(self.API[0], self.API[1], testnet=True)
                
                # SELL
                if sell:
                    amountCrypto = self.pfValNonUSD*self.cryptoRatio
                    if amount > amountCrypto:
                        print("Not enough crypto to sell!")
                        amount = amountCrypto*0.9999
                    
                    order = roundAndSendOrder(self, client, sell, amount)
                
                # BUY
                else:
                    amountUSD = self.pfValNonUSD*(1-self.cryptoRatio)
                    if amount > amountUSD:
                        print("Not enough USD to buy!")
                        amount = amountUSD*0.9999

                    order = roundAndSendOrder(self, client, sell, amount)

                client.close_connection()
            
            except Exception as e:
                if isinstance(e, socket.error):
                    print(f"Connection error:\n{e}")
                else:
                    print(f"Ooops there was a problem placing an order:\n{e}")
            else:
                if refreshPf: self.refreshPortfolio()
                if saveOrder: self.saveOrder(order)
                break
        
    def saveOrder(self,order):
        order['effectivePrice'] = [round(float(order['cummulativeQuoteQty'])/float(order['executedQty']),2)]
        order['pfValue'] = self.pfValUSD
        order.pop('fills')
        orderDF = pd.DataFrame.from_dict(order)
        orderDF = orderDF[['symbol','pfValue','orderId','executedQty','cummulativeQuoteQty','effectivePrice','side','status','type','transactTime']].copy()

        orderDF["transactTime"] = pd.to_datetime(orderDF["transactTime"],unit="ms")
        orderDF.to_csv(f"{self.botFolderPath}/{self.name}_log.csv",mode="a",index=False,header=False)
    #-----
    


    ##### ASYNC LOOP #####
    async def mainLiveTradeLoop(self):

        while True:
            print("\n---|---|---|---|---|---|---|---|---|---\n")
            while True:

                self.refreshAll()

                now = dt.datetime.now(pytz.timezone("UTC"))
                timeGap = now.minute-self.lastMarketDataTS.minute

                if timeGap <= 1:

                    self.smartSignals()
                    self.saveFinish()

                    break

                print(f"DATA IS LATE: now-{now.minute} vs last-{self.lastMarketDataTS.minute+1} -> timeGap = {timeGap}\n")
                print("Sleeping 1 second")

                await asyncio.sleep(1)
        
            now = dt.datetime.now(pytz.timezone("UTC"))
            timeToWait = round(61-(now.second+(now.microsecond)/1000000),4)

            print(f"Seconds now: {now.second}")
            print(f"Waiting {timeToWait} seconds")

            await asyncio.sleep(timeToWait) 
    #-----



config = dotenv_values(".env")

api_key_testnet = config["API_Key_Testnet"]
api_secret_testnet = config["Secret_Key_Testnet"]

modelName = config["modelName"]


async def main():

    smartBot1 = tradingBot(
        modelName=modelName,
        symbol="BTCUSDT",
        minPctChange=2/100,
        exposureMultiplier=100,
        API = [api_key_testnet, api_secret_testnet],
        )

    await asyncio.gather(
        smartBot1.mainLiveTradeLoop()
    )

if __name__ == "__main__":
    asyncio.run(main())
