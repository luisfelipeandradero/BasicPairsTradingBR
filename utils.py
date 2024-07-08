def find(element, matrix):
    for i, matrix_i in enumerate(matrix):
        for j, value in enumerate(matrix_i):
            if value == element:
                return (i, j)


def normalise_data(raw_data, type, n_training, n_trading):
    import pandas as pd
    import numpy as np
    training_data = raw_data.iloc[:n_training,:]
    trading_data = raw_data.iloc[-n_trading:, :]
    if (type == "correlation"):
        normalised_data = pd.DataFrame(np.diff(np.log(raw_data), axis = 0))
        training_data_normalised = normalised_data.iloc[:n_training - 1,:]
        trading_data_normalised = normalised_data.iloc[-n_trading:, :]
    if (type == "distances1"):
        mu =  np.mean(training_data, axis = 0)
        sigma = np.std(training_data, axis = 0)
        training_data_normalised =  (training_data - mu) / sigma
        trading_data_normalised =  (trading_data - mu) / sigma
    if (type == "distances2"):
        normalised_data = np.cumprod(1 + raw_data / raw_data.iloc[0, :])
        training_data_normalised = normalised_data.iloc[:n_training - 1,:]
        trading_data_normalised = normalised_data.iloc[-n_trading:, :]
    if (type == "cointegration"):
        training_data_normalised = training_data
        trading_data_normalised = trading_data 
    return training_data, training_data_normalised, trading_data, trading_data_normalised


def market_neutral_trading_rule(trading_spread, mu, sigma, n_trading, d, trading_data, pairs, h):
    import pandas as pd
    import numpy as np
    import math
    n = 0
    operation = {'Window': [], 'Type': [], 'Open': [], 'Sell Ticker': [], 'Sell Price': [], 'Buy Ticker': [], 'Buy Price': [], 'Close': [], 'Close Sell': [],  'Close Buy': [], 'PL' : [], 'Alpha': [], 'Returns': [] , 'Pair': []}
    operation = pd.DataFrame(operation)
    for m in range(0, n_trading - 1):
        #ticker_01 = trading_data.columns[pair[0]]
        #ticker_02 = trading_data.columns[pair[1]]
        if trading_spread.iloc[m] >= mu + d * sigma and m >= n: # and ticker_01[:4] == ticker_02[:4]:  #Sell short 1/beta dollar worth of stock 2 and buy 1 dollar worth of stock 1  #(H. Rad et al., 2016)
            # Opening trade Up
            operation.loc[m, 'Window'] = h + 1
            operation.loc[m, 'Type'] = "Short"
            operation.loc[m, 'Open'] = trading_data.index[m] 
            # Sell asset A and Buy asset B
            operation.loc[m, 'Sell Ticker'] = trading_data.columns[pairs[0]] 
            operation.loc[m, 'Sell Price'] = trading_data.iloc[m, pairs[0]]
            operation.loc[m, 'Buy Ticker'] = trading_data.columns[pairs[1]] 
            operation.loc[m, 'Buy Price'] = trading_data.iloc[m, pairs[1]]
            operation.loc[m, 'Alpha'] = operation.loc[m, 'Sell Price'] / operation.loc[m, 'Buy Price'] 
            operation.loc[m, 'Pair'] = trading_data.columns[pairs[0]] + trading_data.columns[pairs[1]]
            # Closing trade
            n = m + 1
            while math.isnan(operation.loc[m, 'Close'])  and n < n_trading - 1 and m < n_trading - 1:  
                if trading_spread.iloc[n] <= mu: # or trading_spread.iloc[n] >= mu + 3 * sigma): #or trading_spread.iloc[m] >= mu + 3 * sigma:
                    operation.loc[m, 'Close'] = trading_data.index[n] 
                    operation.loc[m, 'Close Sell'] = trading_data.iloc[n, pairs[0]]
                    operation.loc[m, 'Close Buy']  = trading_data.iloc[n, pairs[1]]
                n = n + 1
            
            if math.isnan(operation.loc[m, 'Close']):
                operation.loc[m, 'Close'] = trading_data.index[n_trading - 1] 
                operation.loc[m, 'Close Sell'] = trading_data.iloc[n_trading - 1, pairs[0]]
                operation.loc[m, 'Close Buy']  = trading_data.iloc[n_trading - 1, pairs[1]]
            operation.loc[m, 'PL'] = operation.loc[m, 'Alpha']* (operation.loc[m, 'Close Buy'] - operation.loc[m, 'Buy Price']) - (operation.loc[m, 'Close Sell'] -  operation.loc[m, 'Sell Price'])
            operation.loc[m, 'Returns'] = (operation.loc[m, 'Close Buy'] - operation.loc[m, 'Buy Price']) / operation.loc[m, 'Buy Price'] - (operation.loc[m, 'Close Sell'] - operation.loc[m, 'Sell Price']) / operation.loc[m, 'Sell Price']

        if trading_spread.iloc[m] <= mu - d * sigma and m >= n: #Buy 1 dollar worth of stock 2 and sell beta worth of stock 1 #(H. Rad et al., 2016)
            # Opening Trade Down
            operation.loc[m, 'Window'] = h + 1
            operation.loc[m, 'Type'] = "Long"
            operation.loc[m, 'Open'] = trading_data.index[m] 
            # Buy asset A and Sell asset B
            operation.loc[m, 'Buy Ticker'] = trading_data.columns[pairs[0]] 
            operation.loc[m, 'Buy Price'] = trading_data.iloc[m, pairs[0]]
            operation.loc[m, 'Sell Ticker'] = trading_data.columns[pairs[1]] 
            operation.loc[m, 'Sell Price'] = trading_data.iloc[m, pairs[1]]  
            operation.loc[m, 'Alpha'] = operation.loc[m, 'Sell Price'] / operation.loc[m, 'Buy Price'] 
            operation.loc[m, 'Pair'] = trading_data.columns[pairs[0]] + trading_data.columns[pairs[1]]              
            # Closing trade
            n = m + 1
            while math.isnan(operation.loc[m, 'Close']) and n < n_trading - 1 and m < n_trading - 1:
                if trading_spread.iloc[n] >= mu: # or trading_spread.iloc[n] <= mu - 3 * sigma): #  or trading_spread.iloc[m] >= mu - 3 * sigma:# - 0.5 * sigma :
                    operation.loc[m, 'Close'] = trading_data.index[n] 
                    operation.loc[m, 'Close Buy'] = trading_data.iloc[n, pairs[0]]
                    operation.loc[m, 'Close Sell']  = trading_data.iloc[n, pairs[1]]
                n = n + 1
            
            if math.isnan(operation.loc[m, 'Close']):
                operation.loc[m, 'Close'] = trading_data.index[n_trading - 1]
                operation.loc[m, 'Close Buy'] = trading_data.iloc[n_trading - 1, pairs[0]]
                operation.loc[m, 'Close Sell']  = trading_data.iloc[n_trading - 1, pairs[1]]
            operation.loc[m, 'PL'] = operation.loc[m, 'Alpha']* (operation.loc[m, 'Close Buy'] - operation.loc[m, 'Buy Price']) - (operation.loc[m, 'Close Sell'] -  operation.loc[m, 'Sell Price'])
            operation.loc[m, 'Returns'] = (operation.loc[m, 'Close Buy'] - operation.loc[m, 'Buy Price']) / operation.loc[m, 'Buy Price'] - (operation.loc[m, 'Close Sell'] - operation.loc[m, 'Sell Price']) / operation.loc[m, 'Sell Price']
    return operation

