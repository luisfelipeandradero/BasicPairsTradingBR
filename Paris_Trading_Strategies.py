###################################################
###  Evaluating pairs-trading strategies in the ###
###  Brazilian market                           ###
# Authors: Felipe Andrade and Carlos Trucíos      #
###################################################

import os
#os.chdir("/home/ctrucios/Dropbox/Students/IC/IC-Pairs Trading/Codes")
os.chdir("/Users/Felipe/Iniciação Científica Dropbox/Felipe Andrade/IC-Pairs Trading/Codes")
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import utils

# Read Data
file_name = "../DadosBR/precos_diarios_ibrx.csv"
daily_prices = pd.read_csv(file_name).iloc[: , 2:]
type = "cointegration" #distances1, distances2, cointegration, correlation are also possible

# Setup
n_training = 252   # 12 months
n_trading = 21   # testar com 21 (um mes), 63 (3 meses) e 126 (6 meses)
k = daily_prices.shape[1] 
top_n = 20   # testar com 5, 10, 20
d = 3   # testar com 2 e 3

trading_periods = int(np.floor(daily_prices.shape[0] / (n_training + n_trading)))

results = {'Window': [], 'Type': [], 'Open': [], 'Sell Ticker': [], 'Sell Price': [], 'Buy Ticker': [], 'Buy Price': [], 'Close': [], 'Close Sell': [],  'Close Buy': [], 'PL' : [], 'Alpha': [], 'Returns': [] , 'Pair': []}
results_df = pd.DataFrame(results)  
return_window = {'Window': [],  'Avg Return': []}
return_window = pd.DataFrame(return_window)  

# Trading
for h in range(0, trading_periods):
    accumulated = {'Window': [],  'Pair': [], 'Cum Return': []}
    accumulated = pd.DataFrame(accumulated)
    ac = 0
    input_data = daily_prices.iloc[((n_training  + n_trading) * h):((n_training  + n_trading) * (h + 1)), :]
    training_data, training_data_normalised, trading_data, trading_data_normalised = utils.normalise_data(input_data, type, n_training, n_trading)
    sort_metric = []
    pair_metric = []
    if (type == "correlation"): 
        pair_metric = np.corrcoef(training_data_normalised, rowvar = False)
        sort_metric = np.sort(pair_metric[np.arange(0, k - 1), np.arange(1, k)])[::-1]

    if (type == "distances1" or type == "distances2"):
        pair_metric = np.zeros(shape = (k, k))
        for i in range(0, k):
            for j in range(i + 1, k):
                pair_metric[i][j] = sum((training_data_normalised.iloc[:, i] - training_data_normalised.iloc[:, j])**2)
        sort_metric = np.sort(pair_metric[np.arange(0, k - 1), np.arange(1, k)]) 
    
    if (type == "cointegration"):
        sort_index = []
        trace_statistics = []
        p_values = []
        for i in range(k):
            for j in range(i + 1, k):            
                p_valor_coint = ts.coint(training_data_normalised.iloc[:, i], training_data_normalised.iloc[:, j])[1]
                johansen_results = coint_johansen(training_data_normalised.iloc[:, [i,j]], det_order = 0, k_ar_diff = 1)
                if (p_valor_coint < 0.05 and johansen_results.trace_stat[0] > johansen_results.trace_stat_crit_vals[0,0]):
                    p_values.append(p_valor_coint)
                    pair_metric.append((i, j))
                    trace_statistics.append(johansen_results.trace_stat[0])
        sort_index = np.argsort(p_values) #[::-1]
        sort_metric = np.array(pair_metric)[[sort_index]]
        # top_n = len(sort_index) - 1
    
    # Selecting Pairs and Trading
    for l in range (0, top_n):
        trading_spread = []
        training_spread = []
        if (type == "correlation"):
            pairs = utils.find(sort_metric[l], pair_metric)
            model = sm.OLS(training_data_normalised.iloc[:, pairs[0]] , training_data_normalised.iloc[:, pairs[1]]).fit()
            training_spread = model.resid 
            mu = np.mean(training_spread)
            sigma = np.std(training_spread)
            trading_spread =  trading_data_normalised.iloc[:, pairs[0]] - model.predict(trading_data_normalised.iloc[:, pairs[1]])
        if (type == "distances1" or type == "distances2"):
            pairs = utils.find(sort_metric[l], pair_metric)
            training_spread = training_data_normalised.iloc[:, pairs[0]] - training_data_normalised.iloc[:, pairs[1]]
            mu = np.mean(training_spread)
            sigma = np.std(training_spread)
            trading_spread =  trading_data_normalised.iloc[:, pairs[0]] - trading_data_normalised.iloc[:, pairs[1]]
        if (type == "cointegration"):
            pairs = sort_metric[:,l].flatten()
            model = sm.OLS(training_data_normalised.iloc[:, pairs[0]], sm.add_constant(training_data_normalised.iloc[:, pairs[1]])).fit()
            training_spread = model.resid 
            mu = np.mean(training_spread)
            sigma = np.std(training_spread)
            trading_spread =  trading_data_normalised.iloc[:, pairs[0]] - model.predict(sm.add_constant(trading_data_normalised.iloc[:, pairs[1]]))
        
        #operation = []
        operation = utils.market_neutral_trading_rule(trading_spread, mu, sigma, n_trading, d, trading_data, pairs, h)
        
        if (operation.shape[0] > 0):
            accumulated.loc[ac, 'Window'] = h + 1
            accumulated.loc[ac, 'Pair'] = trading_data.columns[pairs[0]] + trading_data.columns[pairs[1]] 
            accumulated.loc[ac, 'Cum Return'] = np.cumprod(1 + operation.loc[:, 'Returns']).tail(1).iloc[0]
            ac = ac + 1
        
        results_df = pd.concat([results_df, operation])
    return_window.loc[h, 'Window'] = h + 1
    return_window.loc[h, 'Avg Return'] = (np.sum(accumulated.loc[:, 'Cum Return']) + (top_n - len(accumulated))) /top_n



# Retorno
Retorno = (np.cumprod(return_window.loc[:,'Avg Return']).tail(1) - 1)*100
# Retorno anualizado
Retorno_anualizado = (np.cumprod(return_window.loc[:,'Avg Return']).tail(1) - 1)*100/(len(daily_prices) - n_training)  * n_training  # ganho em porcentagem anualizado

print((results_df.shape[0]) - 1)
print(Retorno)
print(Retorno_anualizado)
results_df.to_csv("Pairs_" + type + ".csv", decimal='')

