from datetime import datetime
import streamlit as st
import pandas as pd
import torch
import yfinance as yf
import joblib
from transformers import pipeline
from statsmodels.tsa.seasonal import STL
import sys


# Load the models
#check if there is a GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model_sentiment = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", device=device)
model_trend = joblib.load('./app/AAPL_trend_seasonal_residual_data_sentiment_model.pkl')
model_multiple = joblib.load('./app/AAPL_multiple_parameter_model.pkl')

# Set the title
st.title('Stock Market Prediction')

# Add a header for stock selection
st.header('Stock Selection')
stock_list = ['AAPL']
stock_selected = st.selectbox('Select the Stock', stock_list)

# Add a header for news input
st.header('News Sentiment Analysis')
st.write("Enter the news content to analyze its sentiment and impact on stock prediction.")
news_content = st.text_area('Enter the news content')

# Add a horizontal line
st.markdown("---")

# Add a header for prediction
st.header('Prediction Results')

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
# Load the data
def load_data(stock_selected):
    today = datetime.now().strftime("%Y-%m-%d")
    df = yf.download(stock_selected, start="2020-01-01", end=today)[['Close']]
    if news_content:
        data = df['Close'].dropna()  # Remove NaN values
        sd = STL(data, period=7).fit()
        trend = pd.Series(sd.trend).dropna()
        seasonal = pd.Series(sd.seasonal).dropna()
        residual = pd.Series(sd.resid).dropna()
        data = data.loc[trend.index]
        combined = pd.concat([trend, seasonal, residual, data], axis=1)
        last_row = combined.iloc[-1]
        sentiment = model_sentiment(news_content)
        weight = 1 if sentiment[0]['label'] == 'positive' else -1 if sentiment[0]['label'] == 'negative' else 0
        last_row['weighted_score'] = sentiment[0]['score'] * weight
        last_row.loc[:, 'Label'] = sentiment[0]['label']
       
    else:
        df['EMA'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['MACD'] = calculate_macd(df)
        df = generate_signals(df)
        lookback_days = 10
        for feature in ['Close', 'EMA', 'MACD', 'RSI', 'Buy_Signal', 'Sell_Signal']:
            for lag in range(1, lookback_days + 1):
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        df.dropna(inplace=True)
        features = [col for col in df.columns if col not in ['target', 'Date']]
        X = df[features]
        last_row = X.iloc[-1]
    return last_row

def predict(last_row):
    if news_content:
        prediction = model_trend.predict(last_row.values.reshape(1, -1))
    else:
        prediction = model_multiple.predict(last_row.values.reshape(1, -1))
    return prediction

# Button
prediction = None
if st.button('Predict'):
    last_row = load_data(stock_selected)
    if 'Label' in last_row.index:
        st.write(f'The sentiment score is **{last_row["Label"]}** with a score of **{last_row["weighted_score"]:.2f}**')
        last_row.drop(index=['Label'], inplace=True)

    prediction = predict(last_row)
    change = prediction[0] - last_row['Close']
    st.write(f'The predicted stock price in {stock_selected} is **{prediction[0]:.2f}** with a change of **{change:.2f}**')

# Add a horizontal line at the end
st.markdown("---")
