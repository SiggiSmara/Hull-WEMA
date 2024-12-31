# %%
# import all needed libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path

# turn off FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# data_path
data_path = Path(__file__).parent.parent / "dataset"
# read csv data
data = pd.read_csv(data_path / "time_series_covid19_confirmed_global.csv", index_col=0)

# get the countries' name
countries = data["Country/Region"]
# extract the starting date up to the last date
x = list(data.columns)
# print(x)
x = x[3:]
print(len(x))

# %%
## last-date confirmed cases analysis ##
# get the rank of each country based on the number of last date confirmed cases,
# save it in a new data column 'Rank_Global'
# select top ten rank and save it under new data column 'Selected_Global'
data["Rank_Global"] = data[x[-1]].rank(ascending=False)
data["Selected_Global"] = data["Rank_Global"] <= 10

# plot top ten countries based on the number of last date confirmed cases
fig = plt.figure(figsize=(20,10)) 
ax = fig.add_subplot(111)
top_countries = []
for i in range(len(countries)):
    if data["Selected_Global"][i]:
        y = data.iloc[i,3:-2]
        if np.isnan(data.index[i]):
            lbl = str(data.iloc[i,0])
        else:
            lbl = str(data.iloc[i,0]) + " - " + str(data.index[i])
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
    sma = [None] * len(src)
    for i in range(len(src)-period+1):
        sma[i+period-1] = np.average(src[i:(i+period)])
    return sma

# %%
# Weighted Moving Average
def WMA(src, period):
    wma = [None] * len(src)
    wts = np.array([i+1 for i in reversed(range(period))])
    for i in range(len(src)-period+1):
        wma[i+period-1] = np.average(list(reversed(src[i:(i+period)])), weights = wts)
    return wma

# %%
# Exponential Moving Average
def EMA(src, alpha):
    ema = []
    for i in range(len(src)):
        if i == 0:
            ema.append(src[i])
        else:
            ema.append((alpha * src[i]) + ((1 - alpha) * ema[i-1]))
    return ema

# %%
# Hull Moving Average
def HMA(src, period):
    hma = [None] * len(src)
    inp = [None] * len(src)
    wmaHalf = WMA(src, math.floor(period/2))
    wmaFull = WMA(src, period)
    inp[period-2] = src[period-2]   
    for i in range(len(src)-period+1):
        inp[i+period-1] = (2 * wmaHalf[i+period-1]) - wmaFull[i+period-1]
    hma[(period-2):] = WMA(inp[(period-2):], math.floor(math.sqrt(period)))
    return hma

# %%
# Weighted Exponential Moving Average
def WEMA(src, period, alpha):   
    wema = [None] * len(src)
    wmaFull = WMA(src, period)
    wema[period-2] = src[period-2]   
    for i in range(len(src)-period+1):
        wema[i+period-1] = ((alpha * wmaFull[i+period-1]) + ((1 - alpha) * wema[i+period-2]))
    return wema

# %%
# Hull Weighted Exponential Moving Average
def HullWEMA(src, period, alpha):   
    hullWema = [None] * len(src)
    hma = HMA(src, period)
    hullWema[period-2] = src[period-2]   
    for i in range(len(src)-period+1):
        hullWema[i+period-1] = ((alpha * hma[i+period-1]) + ((1 - alpha) * hullWema[i+period-2]))
    return hullWema

# %%
# Forecast Error Criteria
def errCalc(src, pred, startInd):
    # Mean Square Error (MSE) & Root Mean Square Error (RMSE)
    diff2 = [(src[i] - pred[i])**2 for i in range(startInd, len(src))]
    sumDiff2 = sum(diff2)
    mse = sumDiff2 / len(diff2)
    rmse = math.sqrt(mse)
    # Mean Absolute Error (MAE)
    diffAbs = [abs(src[i] - pred[i]) for i in range(startInd, len(src))]
    sumDiffAbs = sum(diffAbs)
    mae = sumDiffAbs / len(diffAbs)
    # Mean Absolute Percentage Error (MAPE)
    diffMape = [abs((src[i] - pred[i]) / src[i]) for i in range(startInd, len(src))]
    sumDiffMape = sum(diffMape)
    mape = (sumDiffMape / len(diffMape)) * 100
    # Mean Absolute Scale Error (MASE)
    diffAct = [abs(src[i] - src[i-1]) for i in range(startInd, len(src))]
    sumDiffAct = sum(diffAct)
    diffQt = [abs((src[i] - pred[i]) / ((1 / (len(diffAct) - 1)) * sumDiffAct)) for i in range(startInd, len(src))]
    sumDiffQt = sum(diffQt)
    mase = sumDiffQt / len(diffQt)
    return mse, rmse, mae, mape, mase

# %%
# Train-Test Phase
# <<INPUT>> - read the 'considered' country data
country = 'US'
country = 'Argentina'
def opt_hama(country="US"):
    data = pd.read_csv(data_path /  f"{country}_all.csv", index_col=0)
    res = {}
    # extract the starting date up to the last date
    x = list(data.columns)
    x = x[3:]

    procData = data.iloc[0,3:]

    # <<INPUT>> - train test split ratio
    split = 0.8

    trainData = procData[0:int(split * len(procData))]
    testData = procData[int(split * len(procData)):]
    # print("Country: " + country)
    res["Train"] = len(trainData)
    res["Test"] = len(testData)
    res["Total"] = len(procData)
    # print("Train: {}, Test: {}, Total: {}".format(len(trainData), len(testData), len(procData)))
    # print("Train First Date: {}, Train Last Date: {}, Test First Date: {}, Test Last Date: {}"
        # .format(x[0], x[int(split * len(procData))-1], x[int(split * len(procData))], x[-1]))

    # <<INPUT>> - period and iteration number
    period = 7
    iter = 100

    # HMA Prediction
    finPredHMA = HMA(procData, period)
    # print(finPredHMA[:10])
    finMse, finRmse, finMae, finMape, finMase = errCalc(procData, finPredHMA, period-1)
    # print()
    # print("-----------HMA----------")
    res["HMA"] = {
        "MSE": float(finMse),
        "RMSE": finRmse,
        "MAE": finMae,
        "MAPE": float(finMape),
        "MASE": float(finMase)
    }
    # print('MSE: {}, RMSE: {}, MAE: {}, MAPE: {}, MASE: {}, Prediction: {}'.format(finMse, finRmse, finMae, finMape, finMase, [])) #finPredHMA

    # WEMA iteration - Train Phase
    initMape = 100
    finPredWEMA = []
    alpha = 0
    for i in range(iter):
        wema = WEMA(trainData, period, i/iter)
        mse, rmse, mae, mape, mase = errCalc(trainData, wema, period-1)
        if mape < initMape:
            initMape = mape
            finPredWEMA = wema
            alpha = i/iter
            finMse, finRmse, finMae, finMape, finMase = errCalc(trainData, finPredWEMA, period-1)
    wema_alpha = alpha
    # print()
    # print("-----------WEMA - Train----------")
    res["WEMA"] = {
        "alpha": alpha,
        "MSE": finMse,
        "RMSE": finRmse,
        "MAE": finMae,
        "MAPE": float(finMape),
        "MASE": float(finMase)
    }
    # print('alpha: {}, MSE: {}, RMSE: {}, MAE: {}, MAPE: {}, MASE: {}, Prediction: {}'.format(alpha, finMse, finRmse, finMae, finMape, finMase, [])) #finPredWEMA

    # Hull-WEMA iteration - Train Phase
    initMape = 100
    finPredHullWEMA = []
    alpha = 0
    for i in range(iter):
        hullWema = HullWEMA(trainData, period, i/iter)
        mse, rmse, mae, mape, mase = errCalc(trainData, hullWema, period)
        if mape < initMape:
            # print((i/iter, mape, initMape))
            initMape = mape
            finPredHullWEMA = hullWema
            alpha = i/iter
            finMse, finRmse, finMae, finMape, finMase = errCalc(trainData, finPredHullWEMA, period)
    hma_alpha = alpha
    # print()
    # print("-----------Hull-WEMA - Train----------")
    res["Hull-WEMA"] = {
        "alpha": alpha,
        "MSE": finMse,
        "RMSE": finRmse,
        "MAE": finMae,
        "MAPE": float(finMape),
        "MASE": float(finMase)
    }
    # print('alpha: {}, MSE: {}, RMSE: {}, MAE: {}, MAPE: {}, MASE: {}, Prediction: {}'.format(alpha, finMse, finRmse, finMae, finMape, finMase, [])) #finPredHullWEMA

    # WEMA - Test Phase
    # <<INPUT>> - best alpha from Train Phase
    alpha = 0.99
    alpha = wema_alpha

    wemaTest = WEMA(testData, period, alpha)
    finMse, finRmse, finMae, finMape, finMase = errCalc(testData, wemaTest, period-1)
    # print("-----------WEMA - Test----------")
    res["WEMA-Test"] = {
        "alpha": alpha,
        "MSE": finMse,
        "RMSE": finRmse,
        "MAE": finMae,
        "MAPE": finMape,
        "MASE": finMase
    }
    # print('alpha: {}, MSE: {}, RMSE: {}, MAE: {}, MAPE: {}, MASE: {}, Prediction: {}'.format(alpha, finMse, finRmse, finMae, finMape, finMase, [])) #wemaTest

    # Hull-WEMA - Test Phase
    # <<INPUT>> - best alpha from Train Phase
    alpha = 0.99
    alpha = hma_alpha

    hullWemaTest = HullWEMA(testData, period, alpha)
    finMse, finRmse, finMae, finMape, finMase = errCalc(testData, hullWemaTest, period-1)
    # print()
    # print("-----------Hull-WEMA - Test----------")
    res["Hull-WEMA-Test"] = {
        "alpha": alpha,
        "MSE": finMse,
        "RMSE": finRmse,
        "MAE": finMae,
        "MAPE": finMape,
        "MASE": finMase
    }
    # print('alpha: {}, MSE: {}, RMSE: {}, MAE: {}, MAPE: {}, MASE: {}, Prediction: {}'.format(alpha, finMse, finRmse, finMae, finMape, finMase, [])) #hullWemaTest

    # combine the Train-Test results for WEMA and Hull-WEMA
    allWEMA = finPredWEMA + wemaTest
    allHullWEMA = finPredHullWEMA + hullWemaTest

    # plot the actual and prediction results
    fig = plt.figure(figsize=(20,10)) 
    ax = fig.add_subplot(111)
    ax.plot(x, procData, label="Actual")
    ax.plot(x, finPredHMA, label="HMA Prediction")
    ax.plot(x, allWEMA, label="WEMA Prediction")
    ax.plot(x, allHullWEMA, label="Hull WEMA Prediction")

    ind = [i for i in range(0, len(x), 7)]
    date = [x[i] for i in ind]
    plt.xticks(ind, date, rotation=60)
    plt.legend()

    # title
    ax.set_title("Prediction Plot - " + country, fontsize=18, fontweight='bold')

    # axis title
    ax.set_xlabel('Time', fontsize=15)
    ax.set_ylabel('Number of confirmed COVID-19 cases', fontsize=15)

    # text
    xTrain = int(len(trainData)/2)
    yTrain = int(max(trainData))
    xTest = len(trainData) + int(len(testData)/2)
    yTest = int(min(testData))
    ax.text(xTrain, yTrain, 'Train', fontsize=12, color='blue', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
    ax.text(xTest, yTest, 'Test', fontsize=12, color='red', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
    plt.axvline(x=len(trainData), color='k', linestyle='--')

    fig.savefig('Prediction Plot - ' + country + '.jpg', bbox_inches = 'tight')
    plt.close(fig)
    # alternative plot
    # divide HMA prediction results for Train and Test phases
    trainHMA = finPredHMA[0:int(split * len(finPredHMA))]
    testHMA = finPredHMA[int(split * len(finPredHMA)):]
    # divide the x axis values for Train and Test phases
    x1 = x[0:int(split * len(procData))]
    x2 = x[int(split * len(procData)):]

    # plot the graph
    fig = plt.figure(figsize=(15,8))
    # make two subplots
    ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=3)
    ax2 = plt.subplot2grid((1, 5), (0, 3), colspan=2)

    # First subplot for TRAIN phase
    ax1.plot(x1, procData[0:int(split * len(procData))], label="Actual")
    ax1.plot(x1, trainHMA, label="HMA Prediction")
    ax1.plot(x1, finPredWEMA, label="WEMA Prediction")
    ax1.plot(x1, finPredHullWEMA, label="Hull WEMA Prediction")
    # title and axis titles
    ax1.set_title("Train Phase", fontsize=18, fontweight='bold')
    ax1.set_xlabel('Time', fontsize=15)
    ax1.set_ylabel('Number of confirmed COVID-19 cases', fontsize=15)
    # xticks and legends
    ind = [i for i in range(0, len(x1), 14)]
    date = [x1[i] for i in ind]
    ax1.set_xticks(ind)
    ax1.set_xticklabels(date, rotation=40)
    ax1.legend()

    # Second subplot for TEST phase
    ax2.plot(x2, procData[int(split * len(procData)):], label="Actual")
    ax2.plot(x2, testHMA, label="HMA Prediction")
    ax2.plot(x2, wemaTest, label="WEMA Prediction")
    ax2.plot(x2, hullWemaTest, label="Hull WEMA Prediction")
    # title and axis titles
    ax2.set_title("Test Phase", fontsize=18, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=15)
    ax2.yaxis.tick_right()
    # xticks and legends
    ind = [i for i in range(0, len(x2), 14)]
    date = [x2[i] for i in ind]
    ax2.set_xticks(ind)
    ax2.set_xticklabels(date, rotation=40)
    ax2.legend()

    fig.suptitle(country, fontsize=24, fontweight='bold')

    fig.savefig('Train-Test Prediction Plot - ' + country + '.jpg', bbox_inches = 'tight')
    plt.close(fig)
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
for country in top_countries:
    # print(country)
    res[country] = opt_hama(country)
    print((
        country, 
        res[country]["WEMA"]["alpha"], 
        res[country]["WEMA"]["MAPE"], 
        res[country]["Hull-WEMA"]["alpha"],
        res[country]["Hull-WEMA"]["MAPE"],
    ))


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



