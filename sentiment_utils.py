import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from textblob import TextBlob
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer

@st.cache_data(show_spinner=False)
def get_company_news(ticker, api_key, days=30):
    """
    Get company news from Finnhub API
    
    Args:
        ticker (str): Stock ticker symbol
        api_key (str): Finnhub API key
        days (int): Number of days to look back
        
    Returns:
        list: List of news articles
    """
    try:
        # Calculate date range (last month)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Make API request
        url = f'https://finnhub.io/api/v1/company-news'
        params = {
            'symbol': ticker,
            'from': start_date,
            'to': end_date,
            'token': api_key
        }
        
        response = requests.get(url, params=params)
        
        # Check if request was successful
        if response.status_code != 200:
            st.error(f"Error fetching news: {response.status_code}")
            return []
        
        # Parse response
        news_data = response.json()
        
        # Filter and clean news data
        filtered_news = []
        for news in news_data:
            # Process only if headline and datetime are available
            if news.get('headline') and news.get('datetime'):
                # Convert unix timestamp to datetime
                news_date = datetime.fromtimestamp(news['datetime'])
                
                # Create clean news item
                news_item = {
                    'date': news_date.strftime('%Y-%m-%d'),
                    'headline': news['headline'],
                    'summary': news.get('summary', ''),
                    'source': news.get('source', ''),
                    'url': news.get('url', '')
                }
                
                filtered_news.append(news_item)
        
        return filtered_news
    
    except Exception as e:
        st.error(f"Error fetching company news: {e}")
        return []

# def get_company_news(ticker, api_key, days=180):
#     """
#     Get company news using NewsAPI
    
#     Args:
#         ticker (str): Stock ticker or company name
#         api_key (str): NewsAPI key
#         days (int): Number of days to look back
        
#     Returns:
#         list: List of news articles
#     """
#     try:
#         # Calculate date range
#         end_date = datetime.now()
#         start_date = end_date - timedelta(days=days)

#         # Format dates for NewsAPI (YYYY-MM-DD)
#         from_date = start_date.strftime('%Y-%m-%d')
#         to_date = end_date.strftime('%Y-%m-%d')
        
#         # Make API request to NewsAPI
#         url = "https://newsapi.org/v2/everything"
#         params = {
#             'q': ticker,
#             'from': from_date,
#             'to': to_date,
#             'language': 'en',
#             'sortBy': 'publishedAt',
#             'apiKey': api_key,
#             'pageSize': 100
#         }

#         headers = {
#             'User-Agent': 'Mozilla/5.0'
#         }
        
#         response = requests.get(url, params=params,headers=headers)

#         # Check for response success
#         if response.status_code != 200:
#             st.error(f"Error fetching news: {response.status_code}")
#             return []

#         news_data = response.json().get('articles', [])
        
#         # Filter and clean news data
#         filtered_news = []
#         for article in news_data:
#             if article.get('title') and article.get('publishedAt'):
#                 news_date = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                
#                 news_item = {
#                     'date': news_date.strftime('%Y-%m-%d'),
#                     'headline': article['title'],
#                     'summary': article.get('description', ''),
#                     'source': article.get('source', {}).get('name', ''),
#                     'url': article.get('url', '')
#                 }
#                 filtered_news.append(news_item)
        
#         return filtered_news

#     except Exception as e:
#         st.error(f"Error fetching company news: {e}")
#         return []


def analyze_sentiment(news_data):
    """
    Analyze sentiment of news articles using TextBlob
    
    Args:
        news_data (list): List of news articles
        
    Returns:
        list: News articles with sentiment scores
    """
    news_with_sentiment = []
    
    for news in news_data:
        # Analyze headline sentiment using TextBlob
        headline = news['headline']
        summary = news.get('summary', '')
        
        # Combine headline and summary for better sentiment analysis
        text = f"{headline}. {summary}"
        
        # Get sentiment using TextBlob
        blob = TextBlob(text)
        polarity_blob = blob.sentiment.polarity

        sia = SentimentIntensityAnalyzer()
        sentiment_dict = sia.polarity_scores(text)
        polarity_vader = sentiment_dict['compound']

        polarity = (polarity_blob + polarity_vader) / 2
        
        # Categorize sentiment
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Add sentiment to news item
        news_with_sentiment.append({
            **news,
            'sentiment_score': polarity,
            'sentiment': sentiment
        })
    
    return news_with_sentiment

def merge_sentiment_with_stock_data(stock_data, news_with_sentiment):
    """
    Merge daily sentiment scores with stock data
    
    Args:
        stock_data (DataFrame): Historical stock data
        news_with_sentiment (list): News articles with sentiment scores
        
    Returns:
        DataFrame: Stock data with sentiment scores
    """
    # Create a dictionary to store daily sentiment scores
    daily_sentiment = {}
    
    # Group news by date and calculate average sentiment
    for news in news_with_sentiment:
        date = news['date']
        score = news['sentiment_score']
        
        if date in daily_sentiment:
            daily_sentiment[date].append(score)
        else:
            daily_sentiment[date] = [score]
    
    # Calculate average sentiment for each day
    avg_daily_sentiment = {}
    for date, scores in daily_sentiment.items():
        avg_daily_sentiment[date] = sum(scores) / len(scores)
    
    # Calculate overall average sentiment
    all_scores = [score for scores in daily_sentiment.values() for score in scores]
    overall_avg_sentiment = sum(all_scores) / len(all_scores) if all_scores else 0
    
    # Convert stock_data['Date'] to string format
    stock_data['Date_str'] = stock_data['Date'].dt.strftime('%Y-%m-%d')
    
    # Add sentiment scores to stock data
    stock_data['sentiment_score'] = stock_data['Date_str'].apply(
        lambda date: avg_daily_sentiment.get(date, overall_avg_sentiment)
    )
    
    # Drop Date_str column
    stock_data = stock_data.drop('Date_str', axis=1)
    
    return stock_data
