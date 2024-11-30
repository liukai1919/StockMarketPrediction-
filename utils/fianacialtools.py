import pandas as pd

def calculate_rsi(data, window=14):
    # calculate the price change
    delta = data['Close'].diff()

    # separate the gain and loss
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # calculate the average gain and average loss
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # calculate the relative strength (RS)
    rs = avg_gain / avg_loss

    # calculate the RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi

def generate_signals(data):
    # calculate the RSI
    data['RSI'] = calculate_rsi(data)
    
    # overbought and oversold signals
    data['Buy_Signal'] = (data['RSI'] < 30).astype(int)  # RSI < 30 indicates oversold
    data['Sell_Signal'] = (data['RSI'] > 70).astype(int)  # RSI > 70 indicates overbought
    
    return data

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    # calculate the short and long exponential moving averages
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    
    # calculate the MACD line
    macd = short_ema - long_ema
    return macd

