# %%
# import all needed libraries
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path
from time import time

# turn off FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# data_path
data_path = Path(__file__).parent.parent / "dataset"
# read csv data
data = pl.read_csv(data_path / "time_series_covid19_confirmed_global.csv").with_columns(pl.col("Country/Region").alias("Country"))

# get the countries' name
countries = data["Country"]
# extract the starting date up to the last date
x = data.columns[4:-1]
# print(x)
print(len(x))

# %%
## last-date confirmed cases analysis ##
# get the rank of each country based on the number of last date confirmed cases,
# save it in a new data column 'Rank_Global'
# select top ten rank and save it under new data column 'Selected_Global'
data = data.with_columns([
    pl.col(x[-1]).rank("dense", descending=True).alias("Rank_Global"),
    # (pl.col("Rank_Global") <= 10).alias("Selected_Global")
]).with_columns(
    (pl.col("Rank_Global") <= 10).alias("Selected_Global")
).filter(pl.col("Selected_Global") == True)

# plot top ten countries based on the number of last date confirmed cases
fig = plt.figure(figsize=(20,10)) 
ax = fig.add_subplot(111)
top_countries = []
for i in range(data.shape[0]):
    if data["Selected_Global"][i]:
        y = data.filter(
            data["Country"] == data["Country"][i]
        )
        # print(y)
        y = y.select(
            pl.col(x)
        ).unpivot().to_series()
        # .select(pl.col("value")).to_series()
        lbl = str(data["Country"][i])
        # if data["Country"][i].isna():
            
        # else:
        #     lbl = str(data["Country"][i]) + " - " + str(data["Country"][i])
        # print((len(x), len(y)))
        ax.plot(x, y, label=lbl)
        top_countries.append(lbl)


ind = [i for i in range(0, len(x), 7)]
date = [x[i] for i in ind]
plt.xticks(ind, date, rotation=60)

# title, label, and legend
ax.set_title("COVID-19 Confirmed - Global", fontsize=18, fontweight='bold')
ax.set_xlabel('Time', fontsize=15)
ax.set_ylabel('Number of confirmed COVID-19 cases', fontsize=15)
ax.legend()

fig.savefig('Top Ten Countries.jpg', bbox_inches = 'tight')

# %%
# Simple Moving Average
def SMA(src, period):
    return src.rolling_mean(window_size=period)

# %%
# Weighted Moving Average
def WMA(src:pl.Series, period):
    weights = pl.Series([i+1 for i in range(period)])
    return src.rolling_mean(window_size=period, weights=weights)

# %%
# Exponential Moving Average
def EMA(src:pl.Series, alpha):
    return src.ewm_mean(alpha=alpha)

# %%
# Hull Moving Average
def HMA(src:pl.Series, period):
    # print(period)
    wma_half = WMA(src, period // 2)
    wma_full = WMA(src, period)
    hma_input = 2 * wma_half - wma_full
    hma = pl.select(pl.repeat(None, period-1, dtype=pl.Float64)).to_series()
    hma.append(WMA(hma_input[period-1:], int(math.sqrt(period))))
    return hma

# %%
# Weighted Exponential Moving Average
def WEMA(src:pl.Series, period, alpha):
    return EMA(WMA(src, period), alpha=alpha)

# %%
# Hull Weighted Exponential Moving Average
def HullWEMA(src, period, alpha):
    return EMA(HMA(src, period), alpha=alpha)

# %%
# Forecast Error Criteria
def errCalc(src:pl.Series, pred:pl.Series, startInd:int):
    diff = src[startInd:] - pred[startInd:]
    mse = (diff).pow(2).mean()
    rmse = mse**0.5
    mae = diff.abs().mean()
    mape = (diff.abs() / src[startInd:]).mean() * 100
    mase = (diff.abs() / (src.diff().abs().mean())).mean()
    return mse, rmse, mae, mape, mase

# %%
# Train-Test Phase
# <<INPUT>> - read the 'considered' country data
country = 'US'
country = 'Argentina'

def opt_hama(country="US"):
    data = pl.read_csv(data_path / f"{country}_all.csv")
    res = {}
    x = data.columns[4:]
    # print(country)
    procData = data.select(pl.col(x)).unpivot().select(pl.col("value")).to_series()
    # print(procData)

    split = 0.8
    trainData = procData[:int(split * len(procData))]
    testData = procData[int(split * len(procData)):]
    res["Train"] = len(trainData)
    res["Test"] = len(testData)
    res["Total"] = len(procData)

    period = 7
    iter = 100

    finPredHMA = HMA(procData, period)
    finMse, finRmse, finMae, finMape, finMase = errCalc(procData, finPredHMA, period-1)
    res["HMA"] = {"MSE": finMse, "RMSE": finRmse, "MAE": finMae, "MAPE": finMape, "MASE": finMase}

    initMape = 100
    finPredWEMA = []
    alpha = 0
    for i in range(1,iter):
        wema = WEMA(trainData, period, i/iter)
        mse, rmse, mae, mape, mase = errCalc(trainData, wema, period-1)
        if mape < initMape:
            initMape = mape
            finPredWEMA = wema
            alpha = i/iter
            finMse, finRmse, finMae, finMape, finMase = errCalc(trainData, finPredWEMA, period-1)
    wema_alpha = alpha
    res["WEMA"] = {"alpha": alpha, "MSE": finMse, "RMSE": finRmse, "MAE": finMae, "MAPE": finMape, "MASE": finMase}

    initMape = 100
    finPredHullWEMA = []
    alpha = 0
    for i in range(1,iter):
        hullWema = HullWEMA(trainData, period, i/iter)
        mse, rmse, mae, mape, mase = errCalc(trainData, hullWema, period)
        if mape < initMape:
            initMape = mape
            finPredHullWEMA = hullWema
            alpha = i/iter
            finMse, finRmse, finMae, finMape, finMase = errCalc(trainData, finPredHullWEMA, period)
    hma_alpha = alpha
    # print(finPredHullWEMA.head(10))
    res["Hull-WEMA"] = {"alpha": alpha, "MSE": finMse, "RMSE": finRmse, "MAE": finMae, "MAPE": finMape, "MASE": finMase}

    alpha = wema_alpha
    wemaTest = WEMA(testData, period, alpha)
    finMse, finRmse, finMae, finMape, finMase = errCalc(testData, wemaTest, period-1)
    res["WEMA-Test"] = {"alpha": alpha, "MSE": finMse, "RMSE": finRmse, "MAE": finMae, "MAPE": finMape, "MASE": finMase}

    alpha = hma_alpha
    hullWemaTest = HullWEMA(testData, period, alpha)
    finMse, finRmse, finMae, finMape, finMase = errCalc(testData, hullWemaTest, period)
    res["Hull-WEMA-Test"] = {"alpha": alpha, "MSE": finMse, "RMSE": finRmse, "MAE": finMae, "MAPE": finMape, "MASE": finMase}

    allWEMA = finPredWEMA.append(wemaTest)
    allHullWEMA = finPredHullWEMA.append(hullWemaTest)

    # fig = plt.figure(figsize=(20,10)) 
    # ax = fig.add_subplot(111)
    # ax.plot(x, procData, label="Actual")
    # ax.plot(x, finPredHMA, label="HMA Prediction")
    # ax.plot(x, allWEMA, label="WEMA Prediction")
    # ax.plot(x, allHullWEMA, label="Hull WEMA Prediction")

    # ind = [i for i in range(0, len(x), 7)]
    # date = [x[i] for i in ind]
    # plt.xticks(ind, date, rotation=60)
    # plt.legend()

    # ax.set_title("Prediction Plot - " + country, fontsize=18, fontweight='bold')
    # ax.set_xlabel('Time', fontsize=15)
    # ax.set_ylabel('Number of confirmed COVID-19 cases', fontsize=15)

    # xTrain = int(len(trainData)/2)
    # yTrain = int(max(trainData))
    # xTest = len(trainData) + int(len(testData)/2)
    # yTest = int(min(testData))
    # ax.text(xTrain, yTrain, 'Train', fontsize=12, color='blue', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
    # ax.text(xTest, yTest, 'Test', fontsize=12, color='red', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
    # plt.axvline(x=len(trainData), color='k', linestyle='--')

    # fig.savefig('Prediction Plot - ' + country + '.jpg', bbox_inches = 'tight')
    # plt.close(fig)
    return res


top_countries = [
    'Argentina', 
    'Brazil', 
    'Colombia', 
    'India',
    'Mexico', 
    'Peru', 
    'Russia', 
    'South Africa', 
    'Spain', 
    'US'
]
res = {}
start_tick = time()
for country in top_countries:
    res[country] = opt_hama(country)
    print((
        country, 
        res[country]["WEMA"]["alpha"], 
        res[country]["WEMA"]["MAPE"], 
        res[country]["Hull-WEMA"]["alpha"],
        res[country]["Hull-WEMA"]["MAPE"],
    ))

print("Time taken: ", time() - start_tick)

# Scatter plot for methods accuracy comparison

# Reading an excel file using Python
# import xlrd
 
# # Give the location of the file
# loc = ("mape_mase.xlsx")

# # plot the graph
# fig = plt.figure(figsize=(15,8))
# # make two subplots
# ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=2)
# ax2 = plt.subplot2grid((1, 4), (0, 2), colspan=2)

# # To open Workbook and sheet 1 (MAPE)
# wb = xlrd.open_workbook(loc)
# sheet = wb.sheet_by_index(0)

# # Get column names
# xs = []
# for i in range(sheet.ncols):
#     xs.append(sheet.cell_value(0, i))

# # Get row values
# wema_mape = sheet.row_values(1)
# hma_mape = sheet.row_values(2)
# hullWema_mape = sheet.row_values(3)

# # Get maximum y values
# max_y = max(max(wema_mape[1:], hma_mape[1:], hullWema_mape[1:]))

# # plot the MAPE values
# ax1.scatter(xs[1:], wema_mape[1:], c='r', label='WEMA')
# ax1.scatter(xs[1:], hma_mape[1:], c='g', label='HMA')
# ax1.scatter(xs[1:], hullWema_mape[1:], c='b', label='Hull-WEMA')

# # title, label, limit, ticks and legend
# ax1.set_title("MAPE Comparison", fontsize=18, fontweight='bold')
# ax1.set_xlabel('Countries', fontsize=15)
# ax1.set_ylabel('Error', fontsize=15)
# ax1.set_ylim(0, 2 * max_y)
# ax1.set_xticklabels(xs[1:], rotation=60)
# ax1.legend()

# # To open Workbook and sheet 2 (MASE)
# wb = xlrd.open_workbook(loc)
# sheet = wb.sheet_by_index(1)
 
# # Get column names
# xs = []
# for i in range(sheet.ncols):
#     xs.append(sheet.cell_value(0, i))

# # Get row values
# wema_mase = sheet.row_values(1)
# hma_mase = sheet.row_values(2)
# hullWema_mase = sheet.row_values(3)

# # Get maximum y values
# max_y = max(max(wema_mase[1:], hma_mase[1:], hullWema_mase[1:]))

# # plot the MAPE values
# ax2.scatter(xs[1:], wema_mase[1:], c='r', label='WEMA')
# ax2.scatter(xs[1:], hma_mase[1:], c='g', label='HMA')
# ax2.scatter(xs[1:], hullWema_mase[1:], c='b', label='Hull-WEMA')

# # title, label, limit, ticks and legend
# ax2.set_title("MASE Comparison", fontsize=18, fontweight='bold')
# ax2.set_xlabel('Countries', fontsize=15)
# ax2.set_ylabel('Error', fontsize=15)
# ax2.set_ylim(0, 2 * max_y)
# ax2.set_xticklabels(xs[1:], rotation=60)
# ax2.yaxis.tick_right()
# ax2.legend()

# %%



