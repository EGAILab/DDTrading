# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 11:06:53 2018

@author: Eric Guo
"""

import numpy as np
import pandas as pd
import math

# True Range (TR) & Average True Range (ATR)
# True Range (TR) is defined as the greatest of the following
#   Method 1: Current High less the current Low
#   Method 2: Current High less the previous Close (absolute value)
#   Method 3: Current Low less the previous Close (absolute value)
# Current ATR = [(Prior ATR x 13) + Current TR] / 14
#   Or: moving average of the 14 TRs
# untested, may be wrong
def calculate_atr(df_close, df_high, df_low, look_back=14):
    atr = lambda close, high, low: max((high[-1] - low[-1]), (high[-1] - close[-2]), (low[-1] - close[-2])).mean()
    # may be wrong
    df_atr = (df_close, df_high, df_low).rolling(window=look_back).apply(atr)
    return df_atr

# Standard Deviation (STD)
def calculate_std(df_price, look_back):
    df_std = df_price.rolling(window=look_back).std()
    df_std = df_std[(look_back-1):]
    return df_std

# Daily Returns (%)
def calculate_daily_returns(df_price):
    df_dr = (df_price / df_price.shift(1)) - 1
    df_dr.ix[0, :] = 0
    return df_dr

# Sharpe Ratio
# SR = (Mean portfolio return − Risk-free rate)/Standard deviation of portfolio return
# SR_annual = sqrt(252) * SR_daily
# SR_annual = sqrt(52) * SR_weekly
# SR_annual = sqrt(12) * SR_monthly
def calculate_sharpe_ratio(df_daily_returns, risk_free_rate=0):
    years = df_daily_returns.index.year.unique()
    df_sr = pd.DataFrame(columns=df_daily_returns.columns.values)
    
    for year in years:
        str_year = str(year)
        series_sr = (df_daily_returns.loc[str_year].mean() - risk_free_rate) / df_daily_returns.loc[str_year].std()
        df_sr = df_sr.append(series_sr, ignore_index=True)
    df_sr.insert(0, column='year', value=years)
    df_sr.set_index('year', inplace=True)
    # convert daily to annually 
    df_sr = df_sr * math.sqrt(252)
    return df_sr
    
# Mean Absolute Deviation (MAD)
# x[] = xi - x.mean()
# MAD = x[].mean()
def calculate_mad(df_price, look_back):
    mad = lambda x: np.fabs(x - x.mean()).mean()
    df_mad = df_price.rolling(window=look_back).apply(mad)
    df_mad = df_mad[(look_back-1):]
    return df_mad

# Simple Moving Average (SMA)
def calculate_sma(df_price, look_back):
    df_sma = df_price.rolling(window=look_back, min_periods=1).mean()
    df_sma = df_sma[(look_back-1):]
    return df_sma

# Exponential Moving Average (EMA)
# Initial SMA/EMA: 10-period sum / 10
# Multiplier: (2 / (Time periods + 1) )
# EMA: {Close - EMA(previous day)} x multiplier + EMA(previous day)
def calculate_ema(df_price, look_back=10):
    # df_sma = calculate_sma(df_closing_price, look_back)
    # multiplier = 2 / (look_back + 1)
    return df_price.ewm(span=look_back).mean()


    
# Leading Indicators
#   Designed to lead price movements
#   Most represent a form of price momentum over a fixed lookback period
#   which is the number of periods used to calculate the indicator

# Relative Strength Index (RSI)
# Momentum oscillator that measures the speed and change of price movements.
# Range: 0 - 100
# RSI = 100 - 100 / (1 + RS)
# RS = Average Gain / Average Loss
# First Average Gain = Sum of Gains over the past 14 periods / 14
# First Average Loss = Sum of Losses over the past 14 periods / 14
# The second, and subsequent, calculations are based on the prior averages and the current gain loss
# Average Gain = [(previous Average Gain) x 13 + current Gain] / 14
# Average Loss = [(previous Average Loss) x 13 + current Loss] / 14
# The default look-back period for RSI is 14
# RSI is considered overbought when above 70 and oversold when below 30
def calculate_rsi(df, look_back=14):
    df_gain = df.copy()
    df_gain.where(df_gain > 0, 0, inplace=True)
    df_loss = df.copy()
    df_loss.where(df_loss < 0, 0, inplace=True)
    df_loss = df_loss.abs()
    
    # df_avg_gain[0, :] = df_gain.ix[:14, :].mean()
    df_avg_gain = df_gain.rolling(window=look_back, min_periods=1).mean()
    df_avg_gain = df_avg_gain[(look_back-1):]
    df_avg_loss = df_loss.rolling(window=look_back, min_periods=1).mean()
    df_avg_loss = df_avg_loss[(look_back-1):]
    
    df_rs = df_avg_gain / df_avg_loss
    # print "rs is: ", df_rs
    df_rsi = 100 - 100 / (1 + df_rs)
    # print "rsi: ", df_rsi
    return df_rsi
    
# Commodity Channel Index (CCI)
# CCI = (Typical Price - 20-period SMA of TP) / (.015 x Mean Deviation)
# Typical Price (TP) = (High + Low + Close) / 3
# Constant = .015
# As a coincident indicator
#   surges above +100 reflect strong price action that can signal the start of an uptrend
#   Plunges below -100 reflect weak price action that can signal the start of a downtrend.
# As a leading indicator
#   chartists can look for overbought or oversold conditions that may foreshadow a mean reversion. 
def calculate_cci(df_typical_price, look_back, constant=0.015):
    df_sma = calculate_sma(df_typical_price, look_back)
    df_mad = calculate_mad(df_typical_price, look_back)
    df_cci = (df_typical_price[(look_back-1):] - df_sma) / (constant * df_mad)
    return df_cci

# Stochastic Oscillator

# Williams %R

# Lagging Indicators
#   Follow the price action and are commonly referred to as trend-following indicators. 
#   Trend-following indicators work best when markets or securities develop strong trends.

# Moving Averages (exponential, simple, weighted, variable)

# Moving Average Convergence / Divergence (MACD)
# MACD line: (12-day EMA - 26-day EMA)
# Signal line: 9-day EMA of MACD line
# MACD Histogram: MACD line - Signal line

# Oscillator
#   An oscillator is an indicator that fluctuates above and below a centerline
#   or between set levels as its value changes over time.

# Centered Oscillators - MACD, ROC 
# Banded Oscillators - RSI, Stochastic Oscillator, CCI

# Rate of Change (ROC)
# ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100
def calculate_roc(df_closing_price, look_back):
    roc = lambda x: (x[-1] - x[0]) / x[0] * 100
    df_roc = df_closing_price.rolling(window=look_back).apply(roc)
    df_roc = df_roc[(look_back-1):]
    return df_roc



# *** Technical Overlays ***

# http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators
    
# Bollinger Bands
# Middle Band = 20-day simple moving average (SMA)
# Upper Band = 20-day SMA + (20-day standard deviation of price x 2)
# Lower Band = 20-day SMA - (20-day standard deviation of price x 2)
# By default use 20-day moving average and multiplier of 2
# Multiplier is set to 2.1 for 50-day SMA, 1.9 for 10-day SMA
# According to Bollinger, the bands should contain 88-89% of price action, which makes a move outside the bands significant
def calculate_bollinger_bands(df_closing_price, look_back=20, std_multiplier=2):
    df_m = calculate_sma(df_closing_price, look_back)
    df_std = calculate_std(df_closing_price, look_back)
    df_u = df_m + (df_std * std_multiplier)
    df_l = df_m - (df_std * std_multiplier)
    return df_m, df_u, df_l

# Chandelier Exit
#   Chandelier Exit (long) = 22-day High - ATR(22) x 3 
#   Chandelier Exit (short) = 22-day Low + ATR(22) x 3
# By default uses 22-periods (22 trading days per month) and a multiplier of 3
# not tested
def calculate_ce(df_close, df_high, df_low, look_back=22, multiplier=3):
    df_ce_long = df_high.rolling(window=look_back).max() - (calculate_atr(df_close, df_high, df_low, look_back) * multiplier)
    df_ce_short = df_low.rolling(window=look_back).min() + (calculate_atr(df_close, df_high, df_low, look_back) * multiplier)
    return df_ce_long, df_ce_short

# Ichimoku Clouds
#   Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
#   Kijun-sen (Base Line): (26-period high + 26-period low)/2))
#   Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
#   Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
#   Chikou Span (Lagging Span): Close plotted 26 days in the past
# The Cloud (Kumo) is the most prominent feature of the Ichimoku Cloud plots
# The Leading Span A (green) and Leading Span B (red) form the Cloud
# the trend is up when prices are above the Cloud
# the trend is down when prices are below the Cloud
# and flat when prices are in the Cloud
# the uptrend is strengthened when the Leading Span A (green cloud line) is rising and above the Leading Span B (red cloud line). This situation produces a green Cloud
# a downtrend is reinforced when the Leading Span A (green cloud line) is falling and below the Leading Span B (red cloud line). This situation produces a red Cloud
# untested
def calculate_ichimoku(df_close, df_high, df_low, look_back_tenkan=9, look_back_kijun=26, look_back_senkou=52, look_back_chikou=26):
    df_tenkan = (df_high.rolling(window=look_back_tenkan).max() + df_low.rolling(window=look_back_tenkan).min()) / 2
    df_kijun = (df_high.rolling(window=look_back_kijun).max() + df_low.rolling(window=look_back_kijun).min()) / 2
    df_senkou_a = (df_tenkan + df_kijun) / 2
    df_senkou_b = (df_high.rolling(window=look_back_senkou).max() + df_low.rolling(window=look_back_senkou).min()) / 2
    df_chikou = df_close.shift(look_back_chikou)
    return df_tenkan, df_kijun, df_senkou_a, df_senkou_b, df_chikou

# Kaufman's Adaptive Moving Average (KAMA)
# Efficiency Ratio (ER) = Change / Volatility
#   Change = ABS(Close - Close (10 periods ago))
#   Volatility = Sum10(ABS(Close - Prior Close))
#   Volatility is the sum of the absolute value of the last ten price changes (Close - Prior Close)
# Smoothing Constant (SC)           
#   SC = [ER x (fastest SC - slowest SC) + slowest SC]2
#   SC = [ER x (2/(2+1) - 2/(30+1)) + 2/(30+1)]2
# Current KAMA = Prior KAMA + SC x (Price - Prior KAMA)
# untested, messy
def calculate_kama(df_close, look_back_change=10, look_back_volatility=10, fast_sc=2, slow_sc=30):
    # Efficiency Ratio(ER)
    change = lambda x: np.fabs(x[-1] - x[0])
    df_change = df_close.rolling(window=look_back_change).apply(change)
    df_daily_volatility = np.fabs(df_close - df_close.shift(1))
    df_volatility = df_daily_volatility.rolling(window=look_back_volatility).sum()
    df_er = df_change / df_volatility
    # Smoothing Constant(SC)
    df_sc = (df_er * (2 / (fast_sc + 1) - 2 / (slow_sc + 1)) + 2 / (slow_sc + 1)) ** 2
    ## KAMA
    df_kama = pd.DataFrame(columns=df_sc.columns.values)
    series_sma = df_close[0:10].mean()
    df_kama.append(series_sma, ignore_index=True)
    for index, row in df_sc[10:].iterrows():
        df_temp = df_kama.tail(1) + df_sc.iloc[[index]] * (df_close.iloc[[index]] - df_kama.tail(1))
        df_kama.append(df_temp, ignore_index=True)
    df_sc_index = df_sc[9:].index
    df_kama = pd.DataFrame(data=df_kama, index=df_sc_index)
    return df_kama

# Keltner Channels
# Middle Line: 20-day exponential moving average 
# Upper Channel Line: 20-day EMA + (2 x ATR(10))
# Lower Channel Line: 20-day EMA - (2 x ATR(10))
# untested
def calculate_kc(df_close, df_high, df_low, look_back_ema=20, look_back_atr=10, multiplier=2):
    df_m = calculate_ema(df_close, look_back=look_back_ema)
    df_atr = calculate_atr(df_close, df_high, df_low, look_back_atr)
    df_u = df_m + (multiplier * df_atr)
    df_l = df_m - (multiplier * df_atr)
    return df_m, df_u, df_l

# Moving Average Envelopes
# Upper Envelope: 20-day SMA + (20-day SMA x .025)
# Lower Envelope: 20-day SMA - (20-day SMA x .025)
# untested
def calculate_mae(df_price, look_back=20, multiplier=0.025):
    df_m = calculate_sma(df_price, look_back)
    df_u = df_m * (1 + multiplier)
    df_l = df_m * (1 - multiplier)
    return df_m, df_u, df_l

    