
import math
import json
import talib
import os.path

import numpy as np
import pandas as pd
import datetime as dt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from dotenv import dotenv_values
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")

def mainLayout(log, histData, preds):

    # PAGE HEADER
    st.header(f"{botName} Dashboard")

    # INITIALIZE TABS
    tradingTab, metricsTab, logsTab, aboutTab, settingsTab = st.tabs([
        "📈 Trading",
        "📊 Metrics",
        "📝 Logs",
        "👀 About",
        "⚙ Settings"
        ])

    # LOAD MODEL PARAMETERS
    with open( f"models/{modelName}/{modelName}_params.json", 'r' ) as file:
            modelParamsDict = json.load( file )
    
    # INITIALIZE INITIAL BOT STATE VARIABLES
    pfValueInitial = preds["pfVal"].iloc[0]
    timeStarted = preds.index[0]
    assetBeginningPrice = histData[histData.index==timeStarted.strftime("%Y-%m-%d %H:%M")]["Close"][0]

    # TOTAL SECONDS RUNNING
    nowTS = dt.datetime.now(dt.timezone.utc).timestamp()
    startedTS = (timeStarted).timestamp()
    deltaTS = nowTS-startedTS
    totalSecondsRunning = deltaTS+10

    # GUARANTEE SLIDER PERSISTANCE
    sliderValkey = "sliderVal"

    def lookBackRangeSlider(sliderPos=max(math.ceil(totalSecondsRunning/60/60),2)):
        return st.slider(
        "Graph range:",
        2,
        max(math.ceil(totalSecondsRunning/60/60),2)+10,
        sliderPos,
        key=sliderValkey)


    # ------------------------------------------------------------TABS------------------------------------------------------------


    # ----------------------------------------TRADING TAB----------------------------------------

    with tradingTab:


        # --------------------PREPARE DATA--------------------

        # ---------- SLIDER ----------
        if sliderValkey not in st.session_state:
            nHoursBack =  totalSecondsRunning/60/60-lookBackRangeSlider()-1
        else:
            nHoursBack = totalSecondsRunning/60/60-lookBackRangeSlider(st.session_state[sliderValkey]+(1 if st.session_state[sliderValkey] == math.ceil(totalSecondsRunning/60/60)-1 else 0))-1
            
        timeStampLookBack = (timeStarted+dt.timedelta(hours=nHoursBack+1)).strftime("%Y-%m-%d %H:%M")

        # trim log df
        logTrim = log[log["transactTime"]>=timeStampLookBack]

        # trim preds df
        predsTrim = preds[timeStampLookBack:]


        # ---------- CREATE PLOTTING DATA ----------
        
        # ohlcv
        histDataPlot = histData[timeStampLookBack:]
        
        # asset performance
        assetPerformancePlot = ((histDataPlot["Close"]*100/assetBeginningPrice)-100)[timeStampLookBack:]
        
        # portfolio value
        pfValPlot = predsTrim["pfVal"]
        
        # target MA
        predictionMAPlot = talib.MA(histData["Close"],timeperiod=modelParamsDict["rollingMeanWindow"]).rename("target")[timeStampLookBack:]
        
        # error
        errorPlot = pd.DataFrame()
        errorPlot["error"] = ((predsTrim["prediction"] - predictionMAPlot.shift(-modelParamsDict["predictionHorizon"]))/predictionMAPlot)*100
        errorPlot.dropna(inplace=True)
        errorPlot = errorPlot.sort_index()

        # assetRatio
        # extend asset ratio df back to start of graph
        dateRange = pd.date_range(start=pd.Timestamp(timeStampLookBack).round('H'),freq='1H',periods=len(histDataPlot))
        dateRangeSeries = pd.Series(dtype='float64').reindex(dateRange)
        dateRangeSeries.name = "dateRange"
        
        assetRatioPlot = predsTrim["assetRatio"]
        assetRatioPlot.name = "assetRatio"
        assetRatioPlot = pd.merge(assetRatioPlot.tz_localize(None),dateRangeSeries,left_index=True,right_index=True,how="outer")["assetRatio"]
        
    
        # -------------------- PLOT --------------------


        # ---------- FIG 1 ----------
        # PORTFOLIO VS UNDERLYING ASSET
        if len(predsTrim)>1:
            fig1 = px.line(x=assetPerformancePlot.index,y=assetPerformancePlot).update_traces(line=dict(color="gray"))
            
            fig1.update_layout(yaxis_tickformat = '%')

            fig1.add_traces(list(px.line(x=predsTrim.index, y=(pfValPlot/pfValueInitial)*100-100).update_traces(line=dict(color="blue")).select_traces()))
            
        else: fig1 = px.line()
        # ----------


        # ---------- FIG 2 ----------
        # RATIO OF EXPOSURE
        fig2 = go.Scatter(x=assetRatioPlot.index,y=assetRatioPlot*100,mode="lines",marker_color='Orange')
        # ----------


        # -------------------- ASSEMBLE LEFT FIGURE --------------------
        Fig1 = make_subplots(rows=2, cols=1,shared_xaxes=True,subplot_titles=["Portfolio vs. Underlying Asset","Asset Exposure"])

        # fig1
        for d in fig1.data:
            Fig1.add_trace(d, row=1, col=1)
        # fig2
        Fig1.add_trace(fig2, row=2, col=1)

        # update figure layout
        Fig1.update_layout(showlegend=False,height=600)
        # --------------------------------------------------------------------------------



        # ---------- FIG 3 ----------
        # PRICE
        fig3 = go.Candlestick(
            x = histDataPlot.index,
            open = histDataPlot["Open"],
            high = histDataPlot["High"],
            low = histDataPlot["Low"],
            close = histDataPlot["Close"],
            increasing_line_color = "rgba(220,220,220,0.8)",
            decreasing_line_color = "rgba(128,128,128,0.8)",
        )
        # ----------


        # ---------- FIG 4 ----------
        # PREDICTION ERROR
        fig4 = go.Scatter(x=errorPlot.index,y=errorPlot["error"],mode="markers",marker_color='red')
        # ----------


        # -------------------- ASSEMBLE RIGHT FIGURE --------------------
        Fig2 = make_subplots(rows=2, cols=1,shared_xaxes=True,subplot_titles=["Price, Current Target, Shifted Target, Model prediction", "Prediction Error"])

        # fig3
        Fig2.add_trace(fig3, row=1, col=1)
        # preds, target, shifted target
        Fig2.add_trace(go.Scatter(x=predsTrim.index,y=predsTrim["prediction"],mode="lines",marker_color='aqua',opacity=.5), row=1, col=1)
        Fig2.add_trace(go.Scatter(x=predictionMAPlot.index,y=predictionMAPlot,mode="lines",marker_color='red ',opacity=.5), row=1, col=1)
        Fig2.add_trace(go.Scatter(x=predictionMAPlot.shift(-modelParamsDict["predictionHorizon"]).index,y=predictionMAPlot.shift(-modelParamsDict["predictionHorizon"]),mode="lines",marker_color='green ',opacity=.5), row=1, col=1)
        
        # fig4
        Fig2.add_trace(fig4, row=2, col=1)

        # hide range slider
        Fig2.update_xaxes(rangeslider_visible=False)

        # update figure layout
        Fig2.update_layout(showlegend=False,height=600)
        # --------------------------------------------------------------------------------


        # render figures in page
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(Fig1,True)
        with col2:
            st.plotly_chart(Fig2,True)

    # ---------------------------------------- METRICS TAB ----------------------------------------

    with metricsTab:

        # -------------------- PEPARE DATA --------------------

        pfValue = preds["pfVal"].iloc[-1]
        prediction = preds['prediction'].iloc[-1]
        delta = pfValue-pfValueInitial

        assetPerformance = histData.iloc[-1]["Close"]
        hoursRunning = math.floor((totalSecondsRunning)/60/60)

        if len(log)>0:
            totalSecondsSinceTraded = ((dt.datetime.now()+dt.timedelta(hours=3)) - log["transactTime"].max()).seconds

            hoursSinceTraded = math.floor((totalSecondsSinceTraded)/60/60)
            minutesSinceTraded = math.floor((totalSecondsSinceTraded/60)-hoursSinceTraded*60)
            secondsSinceTraded = math.floor(totalSecondsSinceTraded-hoursSinceTraded*60*60-minutesSinceTraded*60)


        # -------------------- RENDER PAGE --------------------

        st.subheader("Metrics")
        
        col1, col2 = st.columns(2)

        with col1:
            col11, col12, col13 = st.columns(3)

            with col11:
                st.markdown(f"**Time running:**<br/>{hoursRunning}h",unsafe_allow_html=True)
                if len(log)>0:
                    st.markdown(f"**Time since last trade:**<br/>{hoursSinceTraded}h {minutesSinceTraded}m {secondsSinceTraded}s",unsafe_allow_html=True)
                    st.markdown(f"**Number of trades:**<br/>{math.floor((len(logTrim))/2)}",unsafe_allow_html=True)
                
                st.markdown(f"**Current prediction ABS:**<br/>{round(prediction,2)} USD",unsafe_allow_html=True)
                st.markdown(f"**Current prediction %:**<br/>{round(((prediction-histData['Close'][-1])/histData['Close'][-1])*100,2)}%",unsafe_allow_html=True)

            with col12:
                st.markdown(f"**Portfolio Initial value:**<br/>{round(pfValueInitial,2)} USD",unsafe_allow_html=True)
                st.markdown(f"**Portfolio Current value:**<br/>{round(pfValue,2)} USD",unsafe_allow_html=True)
                st.markdown(f"**Portfolio Delta:**<br/>{round(delta,2)} USD",unsafe_allow_html=True)
                st.markdown(f"**Portfolio % Delta:**<br/>{round((delta/pfValueInitial)*100,2)}%",unsafe_allow_html=True)

            with col13:
                st.markdown(f"**BTC initial price:**<br/>{assetBeginningPrice} USD",unsafe_allow_html=True)
                st.markdown(f"**BTC current price:**<br/>{assetPerformance} USD",unsafe_allow_html=True)
                st.markdown(f"**BTC price Delta:**<br/>{round(assetPerformance - assetBeginningPrice,2)} USD",unsafe_allow_html=True)
                st.markdown(f"**BTC % price Delta:**<br/>{round((assetPerformance - assetBeginningPrice)/assetBeginningPrice*100,2)}%",unsafe_allow_html=True)
                st.markdown(f"**Portfolio vs. BTC % Delta:**<br/>{round((delta/pfValueInitial)*100-(assetPerformance - assetBeginningPrice)/assetBeginningPrice*100,2)}%",unsafe_allow_html=True)
        
        
        # -------------------- PLOT PORTFOLIO --------------------
        with col2:

            st.markdown(f"**Current portfolio:**",unsafe_allow_html=True)
            pfVal = preds["pfVal"].iloc[-1]
            assetRatio = preds["assetRatio"].iloc[-1]
            inUSD = {"USDT":pfVal*(1-assetRatio),"ASSET":pfVal*assetRatio}
            inUSD = pd.DataFrame(inUSD,index=[0]).T
            fig = px.pie(inUSD,values=0,color=inUSD.index,names=inUSD.index,color_discrete_map={"USDT":"Green","ASSET":"Orange"}, height=500)
            st.plotly_chart(fig,True)

    # ---------------------------------------- LOGS TAB ----------------------------------------

    with logsTab:

        st.subheader("Logs")
        
        colsSelec=["pfValue","effectivePrice","cummulativeQuoteQty","side","transactTime"]
        st.dataframe(
            log.sort_values(
                by="orderId",
                ascending=False)[colsSelec].rename(columns={
                    "pfValue":"Portfolio",
                    "effectivePrice":"Price",
                    "cummulativeQuoteQty":"amount USD"}),
            height=(len(log) + 1) * 35 + 3,
            use_container_width=True)

    # ---------------------------------------- ABOUT TAB ----------------------------------------

    with aboutTab:

        st.subheader("About")

        aboutMsg1 = f"Hello! This is the bot UI. Here you can see graphs, logs and metrics about the bot's performance."
        aboutMsg2 = f"This bot's trading revolves aroung a moving average prediction made by an algorithm.  \n The period of the MA being predicted is {modelParamsDict['rollingMeanWindow']}, and the prediction horizon is {modelParamsDict['predictionHorizon']} \n"
        aboutMsg3 = f"You can see the prediction in the 'Trading' tab. Here is an overview of the graphs:"

        aboutMsg4 = f"  - Portfolio vs. Underlying Asset:  \n-> Portfolio is blue  \n -> Asset is gray  \n-> Y axis is in %"
        aboutMsg5 = f"  - Price, Current Target, Shifted Target, Model prediction:  \n-> Price is in candlestick format  \n-> Current Target is red (current value of MA)  \n-> Shifted Target is green (predicted MA at the time it was predicted)  \n-> Model prediction is light blue"
        aboutMsg6 = f"  - Asset Exposure:  \n-> % of portfolio's exposure to asset"
        aboutMsg7 = f"  - Prediction Error:  \n-> % of error between model prediction and target  \n-> Notice it is late: the model predicts a future value, not current."
        
        st.markdown(
            aboutMsg1+"  \n"+aboutMsg2+"  \n"+aboutMsg3+"  \n"+aboutMsg4+"  \n  \n"+aboutMsg5+"  \n  \n"+aboutMsg6+"  \n  \n"+aboutMsg7
        )


    # ---------------------------------------- SETTINGS TAB ----------------------------------------

    with settingsTab:
        
        if 'checkbox_value' not in st.session_state:
            st.session_state.checkbox_value = True

        autoRefresh = st.checkbox("Auto Refresh", value=st.session_state.checkbox_value)
        st.session_state.checkbox_value = autoRefresh
        
        minVal = 2
        maxVal = 59
        sliderVal = minVal
        sliderVal = st.slider(f"Select auto refresh interval:", min_value=minVal, max_value=maxVal, value=sliderVal)
    
        if autoRefresh:

            st_autorefresh(interval= sliderVal*60*1000, key="dataframerefresh")


def loadFromCSV():
    log = pd.read_csv(f"{path}/{botName}_log.csv",index_col=0)
    log["transactTime"] = pd.to_datetime(log["transactTime"])
    log["pfValue"] = log["pfValue"].astype('float')

    histData = pd.read_csv(f"{path}/{botName}_data.csv",index_col=0)
    histData.index = pd.to_datetime(histData.index)
    
    preds = pd.read_csv(f"{path}/{botName}_preds.csv",index_col=0)
    preds.index= pd.to_datetime(preds.index)

    return log, histData, preds


modelName = dotenv_values(".env")["modelName"]

botName = modelName+"_bot"

path = f"botsStates/{botName}"


if os.path.isfile(f"{path}/{botName}_log.csv") and os.path.isfile(f"{path}/{botName}_data.csv") and os.path.isfile(f"{path}/{botName}_preds.csv"):
    log, histData, preds = loadFromCSV()
    if len(preds)>0:
        mainLayout(log, histData, preds)
else:
    st.header("Data not found")
    st_autorefresh(interval= 2 * 1000)

