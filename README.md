# Stock Market Prediction

## Introduction
This project is a stock market prediction system that uses machine learning to predict the future price of a stock. It is a web application that allows users to input a stock symbol and get a prediction of the future price of the stock.

Follow the steps below to run the project:

## 1. Data Preparation

### 1.1 Stock Data
- Use the Yahoo Finance API to obtain stock data. You can utilize the Python package `yfinance`.

### 1.2 News Data
- The API used is available [here](https://rapidapi.com/apidojo/api/seeking-alpha/playground/apiendpoint_130e4a68-2511-46d0-8ada-25582c1ebb78). It offers a free tier with a limit of 500 requests per month.
- You can directly use the function provided in `APIs.py` to fetch the data.

### 1.3 Sentiment Data
- Download the sentiment data from [Kaggle](https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news).

## 2. Setting Up the Environment

### 2.1 Install Required Packages
- Create a new environment:
  ```bash
  conda create -n stockmarket python=3.11
  ```
- Activate the environment:
  ```bash
  conda activate stockmarket
  ```
- Install the required packages:
  ```bash
  pip install -r requirements.txt
  ```
- Create a `.env` file and add the following:
  ```plaintext
  API_KEY_NEWS=your_api_key_here
  ```
- To apply the API key, you can find it [here](https://rapidapi.com/apidojo/api/seeking-alpha/playground/apiendpoint_130e4a68-2511-46d0-8ada-25582c1ebb78).

## 3. Run the EDA Notebook
- Execute the EDA notebook to calculate the sentiment score and merge it with the stock data.

## 4. Run the Trend Machine Learning Notebook
- Use the `trend_machine_learning` notebook to build the model.

## 5. Project Structure
```
.
├── data
│   ├── AAPL_trend_seasonal_residual_data_sentiment.csv
│   └── ...
├── notebooks
│   ├── EDA.ipynb
│   ├── trend_machine_learning.ipynb
│   └── ...
├── test
│   ├── genstructure.ipynb
│   ├── APItest.ipynb
│   └── modeltest.ipynb
├── models
│   ├── AAPL_trend_seasonal_residual_data_sentiment_model.pkl
│   └── ...
├── utils
│   ├── APIs.py
│   └── ...
├── README.md
└── requirements.txt
```

## 6. Run the Web App
- Start the web app:
  ```bash
  streamlit run app/index.py
  ```
