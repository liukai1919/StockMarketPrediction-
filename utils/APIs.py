from datetime import datetime, time, timedelta

def convert_to_timestamp(readable_time, time_format="%Y-%m-%d %H:%M:%S", milliseconds=False):
    """
    transform the readable time to timestamp
    
    Args:
        readable_time (str): the readable time string, e.g. "2024-11-29 10:20:30"
        time_format (str): the format of the time string, default is "%Y-%m-%d %H:%M:%S"
        milliseconds (bool): whether to return the milliseconds timestamp, default is False (return the seconds timestamp)

    Returns:
        int: the timestamp (seconds or milliseconds)
    """
    try:
        # convert to datetime object
        dt = datetime.strptime(readable_time, time_format)
        # get the timestamp, seconds or milliseconds
        timestamp = int(dt.timestamp() * 1000) if milliseconds else int(dt.timestamp())
        return timestamp
    except ValueError as e:
        print(f"Error: {e}")
        return None

import requests

def get_seeking_alpha_news(symbol, since, until, size=20, number=1, api_key=None):
    """
    request the Seeking Alpha news API.

    Args:
        symbol (str): the stock symbol, e.g. 'TSLA'
        since (int): the start timestamp
        until (int): the end timestamp
        size (int): the number of news to return each time, default is 20
        number (int): the page number of news to request, default is 1
        api_key (str): the RapidAPI key

    Returns:
        dict: the JSON response content
    """
    url = f'https://seeking-alpha.p.rapidapi.com/news/v2/list-by-symbol?until={until}&since={since}&size={size}&number={number}&id={symbol}'

    headers = {
        'x-rapidapi-host': 'seeking-alpha.p.rapidapi.com',
        'x-rapidapi-key': api_key
    }

    try:
        # send the GET request
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # if the response code is not 2xx, raise an exception

        # return the JSON data
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


import pandas as pd

def process_seeking_alpha_data(data):
    """
    process the data from Seeking Alpha API, extract the part related to sentiment analysis or other useful information.

    Args:
        data (dict): the JSON data from Seeking Alpha API

    Returns:
        pd.DataFrame: the DataFrame contains the news data
    """
     # Check if data is None
    if data is None:
        print("No data received from the API.")
        return pd.DataFrame()
    # extract the news data part
    news_data = data.get('data', [])
    
    # if there is no news data
    if not news_data:
        return pd.DataFrame()

    # extract the related data
    processed_data = []

    for item in news_data:
        # get the news title and publish date
        title = item['attributes']['title']
        publish_on = item['attributes']['publishOn']
        author = item['relationships']['author']['data']['id']
        author_name = data.get('included', [])[0]['attributes']['nick'] if 'included' in data else None
        getty_image_url = item['attributes'].get('gettyImageUrl', None)

        # extract the sentiment data (currently empty, can be extracted based on actual data in the future)
        sentiments = item['relationships'].get('sentiments', {}).get('data', [])
        
        sentiment_score = None  # if there is no sentiment data
        sentiment_label = None  # if there is no sentiment data
        
        # if there is sentiment data, extract the sentiment score or label
        if sentiments:
            sentiment_score = sentiments[0].get('score')
            sentiment_label = sentiments[0].get('label')

        # add the data to the list
        processed_data.append({
            'title': title,
            'publish_on': publish_on,
            'author': author,
            'author_name': author_name,
            'getty_image_url': getty_image_url,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label
        })

    # return the DataFrame
    return pd.DataFrame(processed_data)

def get_data_by_two_week_intervals(symbol, api_key, start_date, end_date):
    """
    get the news data by two week intervals

    Args:
        symbol (str): the stock symbol
        api_key (str): the API key
        start_date (str): the start date (YYYY-MM-DD HH:MM:SS)
        end_date (str): the end date (YYYY-MM-DD HH:MM:SS)

    Returns:
        pd.DataFrame: 包含新闻相关数据的 DataFrame
    """
    # convert the date string to timestamp
    start_timestamp = convert_to_timestamp(start_date)
    end_timestamp = convert_to_timestamp(end_date)

    data_frames = []

    # calculate the two week intervals
    current_start_date = start_date
    while start_timestamp < end_timestamp:
        # calculate the current end date (14 days later)
        current_end_date = (datetime.strptime(current_start_date, "%Y-%m-%d %H:%M:%S") + timedelta(days=14)).strftime("%Y-%m-%d %H:%M:%S")
        if convert_to_timestamp(current_end_date) > end_timestamp:
            current_end_date = end_date  # ensure not exceed the end date
        print(f'grabbing data from {datetime.strptime(current_start_date, "%Y-%m-%d %H:%M:%S")} to {datetime.strptime(current_end_date, "%Y-%m-%d %H:%M:%S")}')
        # get the news data of the current interval
        data = get_seeking_alpha_news(symbol, convert_to_timestamp(current_start_date), convert_to_timestamp(current_end_date), api_key=api_key)
        print(f'grabbed data from {datetime.strptime(current_start_date, "%Y-%m-%d %H:%M:%S")} to {datetime.strptime(current_end_date, "%Y-%m-%d %H:%M:%S")} finshed')
        # process the data and save to the DataFrame
        df = process_seeking_alpha_data(data)
        data_frames.append(df)
        print(f'length of the DataFrame: {len(df)} news data grabbed')
        
        # update the start date, continue to get the next two week news
        current_start_date = (datetime.strptime(current_end_date, "%Y-%m-%d %H:%M:%S") + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        # update the start timestamp
        start_timestamp = convert_to_timestamp(current_start_date)

    # merge all the news data
    result_df = pd.concat(data_frames, ignore_index=True)

    return result_df

import os 

if __name__ == "__main__":
    api_key = os.getenv('NEWS_API_KEY_3') # use your own API key
    print(api_key)
    # get the data from 2010-01-01 to 2024-11-29
    start_date = '2008-05-12 00:00:00'
    end_date = '2020-01-01 00:00:00'
    df = get_data_by_two_week_intervals('AAPL', api_key, start_date, end_date)
    df.to_csv('/Users/kailiu/StockMarketPrediction-/data/AAPL_news_data_2015-01-01_2020-01-01.csv', index=False)
