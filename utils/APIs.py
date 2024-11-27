import requests
import os
import yfinance as yf
import pandas as pd
#News API
def get_news(company_name, date, category='financial') -> dict:
    # get apikey from .env file
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    url = f"https://newsapi.org/v2/everything?q={company_name}&from={date}&sortBy=publishedAt&ca&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    return data
#Stock API
def get_stock_data(company_name) -> pd.DataFrame:
    stock = yf.Ticker(company_name)
    return stock
def download_stock_data(company_name, start_date, end_date) -> pd.DataFrame:
    data = yf.download(company_name, start=start_date, end=end_date)
    stock_data = data.reset_index()
    stock_data.columns = stock_data.columns.droplevel('Ticker')
    #set date as index
    stock_data.set_index('Date', inplace=True)
    return stock_data
