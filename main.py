# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 04:13:53 2018

@author: Eric Guo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import TechnicalIndicators as ti


# appdata_root = "/storage/emulated/0/AppData/"
appdata_root = "C:/AI/AIDATA/"
# code_root = "/storage/emulated/0/qpython/projects/"
code_root = "C:/AI/AIPROJECTS/"

data_path = appdata_root + "temp/coinmetrics/"
dataset_path = appdata_root + "dataset/coinmetrics/"
code_path = code_root + "TradingBot/"

btc_filename = data_path + "btc.csv"
price_filename = dataset_path + "price.csv"

col_names = ["Date", "txVolume", "txCount", "MarketCapUSD", "PriceUSD", "ExchangeVolumeUSD", "GeneratedCoins", "Fees", "ActiveAddresses"]
headers = ["date", "txVolume(USD)", "txCount", "marketcap(USD)", "price(USD)", "exchangeVolume(USD)", "generatedCoins", "fees", "activeAddresses"]

# building dataframe of closing prices
def build_dataframe_price():

    # build an index of symbols, btc first
    # with data files (names) downloaded from coinmetrics
    # exclude sp500, gold, liborusd, dxy
    symbols = ["btc"]
    filenames = [btc_filename]
    for filename in sorted(os.listdir(data_path)):
        symbol = os.path.splitext(filename)[0]
        if (symbol != "sp500") and (symbol != "gold") and (symbol != "liborusd") and (symbol != "dxy") and (symbol != "btc"):
            symbols.append(symbol)
            filenames.append(data_path + filename)

    # Price dataframe with/without nan
    df_all = pd.DataFrame({"Date" : []})
    df_nonan = pd.DataFrame({"Date" : []})
    
    for index, symbol in enumerate(symbols):
        # read symbol data (date & price) to dataframe
        #dftemp = pd.read_csv(filenames[index], names=col_names, index_col="Date", parse_dates=True, usecols=["Date", "PriceUSD"], na_values=["nan"])
        dftemp = pd.read_csv(filenames[index], index_col=False, usecols=['date', 'price(USD)'])
        dftemp.set_index('date', inplace=True)
        
        # save symbol dataframe to file (date & price), one symbol per file
        dftemp.to_csv(dataset_path + symbol + "_price.csv", sep = ",")
        
        if index == 0:
            df_all = dftemp.copy()
            df_nonan = dftemp.copy()
        else:
            df_all = df_all.join(dftemp)
            df_nonan = df_nonan.join(dftemp, how='inner')
            
        df_all = df_all.rename(columns = {"price(USD)": symbol})
        df_nonan = df_nonan.rename(columns = {"price(USD)": symbol})
        
    # back fill data
    df_all.fillna(method="ffill", inplace=True)
    df_all.fillna(method="bfill", inplace=True)
    
    df_all.index = pd.to_datetime(df_all.index)
    df_nonan.index = pd.to_datetime(df_nonan.index)
    
    return df_all, df_nonan


def test_ti():

    #data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #df = pd.DataFrame(data=data)
    #print("dataframe before: ")
    #print(df)
    #df_out = ti.calculate_cci(df, 3)
    #print("dataframe after: ")
    #print(df_out)
    
    plt.show()
    plt.savefig(code_path + "plot.png")

    
if __name__ == "__main__":
    
    #test_ti()

    # build dataframe
    df_price, df_price_nonan = build_dataframe_price()
    df_dr = ti.calculate_daily_returns(df_price)
    df_sr = ti.calculate_sharpe_ratio(df_dr)
    print(df_sr)
    
    #print(df_price["btc"])
    #df_price["btc"].plot()
    ##plt.show()
    #plt.savefig(code_path + "plot.png")