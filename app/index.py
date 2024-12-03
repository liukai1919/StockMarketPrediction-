from datetime import datetime
import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
from transformers import pipeline
from statsmodels.tsa.seasonal import STL
import sys
sys.path.append('/Users/kailiu/StockMarketPrediction-') 
from utils.fianacialtools import calculate_macd, generate_signals

# Load the models
model_sentiment = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", device=0)
model_trend = joblib.load('models/AAPL_trend_seasonal_residual_data_sentiment_model.pkl')
model_multiple = joblib.load('models/AAPL_multiple_parameter_model.pkl')

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

# Load the data
def load_data(stock_selected):
    today = datetime.now().strftime("%Y-%m-%d")
    df = yf.download(stock_selected, start="2020-01-01", end=today)[['Close']]
    if news_content:
        data = df[['Close']].asfreq('D').interpolate()
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
        last_row['Label'] = sentiment[0]['label']
    else:
        df['EMA'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['MACD'] = calculate_macd(df)
        df = generate_signals(df)
        lookback_days = 10
        for feature in ['Close', 'EMA', 'MACD', 'RSI', 'Buy_Signal', 'Sell_Signal']:
            for lag in range(1, lookback_days + 1):
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        df.dropna(inplace=True)
        features = [col for col in df.columns if col not in ['Close', 'target', 'Date']]
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
    st.write(f'The predicted stock price in {stock_selected} is **{prediction[0]:.2f}** with a change of **{change[0]:.2f}**')

# Add a horizontal line at the end
st.markdown("---")