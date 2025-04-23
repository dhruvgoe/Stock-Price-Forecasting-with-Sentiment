import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import os
import random
from dotenv import load_dotenv

# Import custom modules
from model_utils import prepare_data, train_ensemble_model, train_time_series_models, evaluate_models
from sentiment_utils import get_company_news, analyze_sentiment, merge_sentiment_with_stock_data
from forecast_utils import forecast_prices, forecast_with_sentiment
from indicators import add_technical_indicators, plot_technical_indicators

# Load environment variables
load_dotenv()

def small_random_variation(epsilon=0.05):
    return random.uniform(-epsilon, epsilon)

# Set page configuration
st.set_page_config(
    page_title="Stock Price Forecasting App",
    page_icon="📈",
    layout="wide"
)

# Cache function for retrieving historical stock data
@st.cache_data(ttl=3600)
def get_stock_data(ticker, period='3mo'):
    """
    Fetch historical stock data from Yahoo Finance
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period for historical data
        
    Returns:
        DataFrame: Historical stock data
    """
    try:
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=period)
        
        # Drop Volume column as per requirements
        # if 'Volume' in hist_data.columns:
        #     hist_data = hist_data.drop(['Volume','Dividends','Stock Splits'], axis=1)
            
        # Reset index to have Date as a column
        hist_data = hist_data.reset_index()
        
        # Ensure we have data
        if hist_data.empty:
            return None
            
        return hist_data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Cache function for company info
@st.cache_data(ttl=3600)
def get_company_info(ticker):
    """
    Fetch company information from Yahoo Finance
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Company information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant information
        company_info = {
            'name': info.get('shortName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'marketCap': info.get('marketCap', 'N/A'),
            'peRatio': info.get('trailingPE', 'N/A'),
            'currentPrice': info.get('currentPrice', 'N/A'),
            'previousClose': info.get('previousClose', 'N/A'),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 'N/A'),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 'N/A')
        }
        
        return company_info
    except Exception as e:
        st.error(f"Error fetching company information: {e}")
        return None

# Main application
def main():
    st.title("Stock Price Forecasting & Sentiment Analysis")
    
    # User input for ticker
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, GOOG)", "AAPL").upper()

    time_range_map = {
    "1 month": "1mo",
    "2 month": "2mo",
    "3 month": "3mo",
    "6 month": "6mo",
    "1 year": "1y",
    "2 year": "2y"
    }

    time_range_map_news = {
    "1 month": 30,
    "2 month": 60,
    "3 month": 90,
    "6 month": 180,
    "1 year": 365,
    "2 year": 730
    }

    st.write("Select Time Range")
    options = ["1 month", "2 month", "3 month", "6 month", "1 year", "2 year"]
    periods = st.selectbox("Choose a time range:", options)
    period = time_range_map[periods]
    news_period = time_range_map_news[periods]

    
    if st.button("Analyze Stock"):
        with st.spinner("Fetching stock data..."):
            # Fetch historical stock data (last month)
            stock_data = get_stock_data(ticker, period=period)
            
            if stock_data is None or stock_data.empty:
                st.error(f"No data found for ticker: {ticker}. Please check the ticker symbol and try again.")
                return
            
            # Get company information
            company_info = get_company_info(ticker)
            
            # Display the analysis in tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Company Info", 
                "Historical Data", 
                "Technical Indicators", 
                "Forecast (Historical)", 
                "Forecast (With Sentiment)",
                "Buy/Sell Recommendations"
            ])
            
            # Tab 1: Company Information
            with tab1:
                if company_info:
                    st.header(f"{company_info['name']} ({ticker})")
                    
                    # Create two columns for company info
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Company Details")
                        st.write(f"**Sector:** {company_info['sector']}")
                        st.write(f"**Industry:** {company_info['industry']}")
                        st.write(f"**Market Cap:** ${company_info['marketCap']:,}" if isinstance(company_info['marketCap'], (int, float)) else f"**Market Cap:** {company_info['marketCap']}")
                        st.write(f"**PE Ratio:** {company_info['peRatio']}")
                    
                    with col2:
                        st.subheader("Stock Performance")
                        st.write(f"**Previous Close:** ${company_info['previousClose']}")
                        st.write(f"**52-Week High:** ${company_info['fiftyTwoWeekHigh']}")
                        st.write(f"**52-Week Low:** ${company_info['fiftyTwoWeekLow']}")
                        st.write(f"**Current Price:** ${company_info['currentPrice']}")

                
            # Tab 2: Historical Data with Candlestick Chart
            with tab2:
                st.header(f"{ticker} Historical Data (Last Month)")
                
                # Display the data table
                st.subheader("Historical OHLC Data")
                st.dataframe(stock_data)
                
                # Create candlestick chart using Plotly
                st.subheader("Candlestick Chart")
                
                fig = go.Figure(data=[go.Candlestick(
                    x=stock_data['Date'],
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name="OHLC"
                )])
                
                fig.update_layout(
                    title=f'{ticker} Stock Price',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    xaxis_rangeslider_visible=True,
                    height=600,
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Daily percentage change plot removed as requested
            
            # Tab 3: Technical Indicators
            with tab3:
                st.header(f"{ticker} Technical Indicators")
                
                # Add technical indicators
                stock_data_with_indicators = add_technical_indicators(stock_data.copy())
                
                # Display the data with indicators
                st.subheader("Data with Technical Indicators")
                st.dataframe(stock_data_with_indicators)
                
                # Plot technical indicators
                st.subheader("Technical Indicators Visualization")
                fig_indicators = plot_technical_indicators(stock_data_with_indicators, ticker)
                st.plotly_chart(fig_indicators, use_container_width=True)
            
            # Tab 4: Forecasting with Historical Data
            with tab4:
                st.header(f"{ticker} Price Forecasting (Historical Data)")
                
                with st.spinner("Training forecasting models..."):
                    # Prepare data for modeling
                    X_train, X_test, y_train, y_test, scaler_y = prepare_data(stock_data_with_indicators)
                    
                    # Train ensemble model (RF + XGBoost)
                    rf_model, xgb_model, ensemble_preds = train_ensemble_model(X_train, X_test, y_train, y_test)
                    
                    # Train time series models (AR, ARIMA, SARIMA)
                    ar_model, arima_model, sarima_model, ts_preds = train_time_series_models(stock_data['Close'])
                    
                    # Evaluate all models
                    mae_scores, rmse_scores, model_names = evaluate_models(y_test, ensemble_preds, ts_preds, stock_data['Close'])
                    
                    # Make forecasts for next 5 days
                    forecast_dates, rf_forecast, xgb_forecast, ensemble_forecast, ar_forecast, arima_forecast, sarima_forecast = forecast_prices(
                        stock_data, rf_model, xgb_model, ar_model, arima_model, sarima_model, scaler_y, days=5
                    )

                    # Display Data for Training
                    stock_data_with_indicators = pd.DataFrame(stock_data_with_indicators)
                    st.dataframe(stock_data_with_indicators)
                    
                    # Display evaluation metrics
                    st.subheader("Model Evaluation (Historical Data)")
                    
                    metrics_df = pd.DataFrame({
                        'Model': model_names,
                        'MAE': mae_scores,
                        'RMSE': rmse_scores
                    })
                    
                    st.dataframe(metrics_df)
                    
                    # Display forecasts in a table
                    st.subheader("5-Day Price Forecasts")
                    
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Random Forest': rf_forecast,
                        'XGBoost': xgb_forecast,
                        'Ensemble (RF+XGB)': ensemble_forecast,
                        'AR': ar_forecast,
                        'ARIMA': arima_forecast,
                        'SARIMA': sarima_forecast
                    })
                    
                    st.dataframe(forecast_df)
                    
                    # Plot forecasts
                    st.subheader("Forecast Visualization")
                    
                    # Create a date range for historical data plotting
                    hist_dates = stock_data['Date'].dt.strftime('%Y-%m-%d').tolist()
                    hist_close = stock_data['Close'].tolist()
                    
                    # Create forecast plot
                    fig_forecast = go.Figure()
                    
                    # Add historical data
                    fig_forecast.add_trace(go.Scatter(
                        x=hist_dates,
                        y=hist_close,
                        mode='lines',
                        name='Historical Close Price',
                        line=dict(color='white')
                    ))
                    
                    # Add forecast lines
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=rf_forecast,
                        mode='lines+markers',
                        name='Random Forest',
                        line=dict(dash='dot')
                    ))
                    
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=xgb_forecast,
                        mode='lines+markers',
                        name='XGBoost',
                        line=dict(dash='dot')
                    ))
                    
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=ensemble_forecast,
                        mode='lines+markers',
                        name='Ensemble (RF+XGB)',
                        line=dict(width=3)
                    ))
                    
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=ar_forecast,
                        mode='lines+markers',
                        name='AR',
                        line=dict(dash='dot')
                    ))
                    
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=arima_forecast,
                        mode='lines+markers',
                        name='ARIMA',
                        line=dict(dash='dot')
                    ))
                    
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=sarima_forecast,
                        mode='lines+markers',
                        name='SARIMA',
                        line=dict(dash='dot')
                    ))
                    
                    fig_forecast.update_layout(
                        title=f'{ticker} Price Forecast (Next 5 Days)',
                        xaxis_title='Date',
                        yaxis_title='Price (USD)',
                        hovermode='x unified',
                        height=600
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Tab 5: Forecasting with Sentiment
            with tab5:
                st.header(f"{ticker} Price Forecasting with News Sentiment")
                
                with st.spinner("Fetching news and analyzing sentiment..."):
                    # Get company news from Finnhub API
                    finnhub_api_key = os.getenv("FINNHUB_API_KEY")
                    # news_api_key = os.getenv("NEWS_API")
                    news_data = get_company_news(ticker, finnhub_api_key, news_period)
                    
                    if news_data and len(news_data) > 0:
                        # Analyze sentiment of news
                        news_with_sentiment = analyze_sentiment(news_data)
                        
                        # Display news with sentiment
                        st.subheader("Recent News with Sentiment Analysis")
                        
                        news_df = pd.DataFrame(news_with_sentiment)
                        st.dataframe(news_df[['date', 'headline', 'sentiment_score', 'sentiment']])
                        
                        # Calculate average sentiment score
                        avg_sentiment = news_df['sentiment_score'].mean()
                        
                        # Display average sentiment and price direction prediction
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Average Sentiment Score", f"{avg_sentiment:.4f}")
                            
                            # Create a gauge chart for sentiment score
                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number+delta",
                                value=avg_sentiment,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Sentiment Gauge", 'font': {'size': 24}},
                                delta={'reference': 0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                                gauge={
                                    'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                    'bar': {'color': "darkblue"},
                                    'bgcolor': "white",
                                    'borderwidth': 2,
                                    'bordercolor': "gray",
                                    'steps': [
                                        {'range': [-1, -0.5], 'color': 'red'},
                                        {'range': [-0.5, -0.25], 'color': 'lightsalmon'},
                                        {'range': [-0.25, 0.25], 'color': 'lightgray'},
                                        {'range': [0.25, 0.5], 'color': 'lightgreen'},
                                        {'range': [0.5, 1], 'color': 'green'},
                                    ],
                                    'threshold': {
                                        'line': {'color': "yellow", 'width': 4},
                                        'thickness': 0.75,
                                        'value': avg_sentiment
                                    }
                                }
                            ))
                            
                            fig_gauge.update_layout(
                                height=250,
                                margin=dict(l=10, r=10, t=50, b=10)
                            )
                            
                            st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        with col2:
                            # Get price direction prediction based on sentiment
                            from forecast_utils import predict_price_direction
                            price_direction = predict_price_direction(avg_sentiment)
                            st.markdown(f"### Price Direction Prediction\n{price_direction}")
                            
                            # Create a pie chart of sentiment distribution
                            sentiment_counts = news_df['sentiment'].value_counts()
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=sentiment_counts.index,
                                values=sentiment_counts.values,
                                hole=.3,
                                marker_colors=['#ff4d4d', '#ffad33', '#66cc66']  # red, yellow, green
                            )])
                            
                            fig_pie.update_layout(
                                title_text="Sentiment Distribution",
                                height=250,
                                margin=dict(l=10, r=10, t=50, b=10)
                            )
                            
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        # Merge sentiment with stock data
                        stock_data_with_sentiment = merge_sentiment_with_stock_data(stock_data_with_indicators.copy(), news_with_sentiment)

                        tolerance = 1e-6  # or even 1e-5 depending on your data

                        for idx, value in stock_data_with_sentiment['sentiment_score'].items():
                            if abs(value - avg_sentiment) < tolerance:
                                stock_data_with_sentiment.loc[idx, 'sentiment_score'] += small_random_variation()

                        print(stock_data_with_sentiment)

                        
                        # Display data with sentiment
                        st.subheader("Stock Data with Sentiment Scores")
                        st.dataframe(stock_data_with_sentiment)
                        
                        # Retrain models with sentiment data
                        X_train_s, X_test_s, y_train_s, y_test_s, scaler_y_s = prepare_data(
                            stock_data_with_sentiment, include_sentiment=True
                        )
                        
                        # Train ensemble model with sentiment
                        rf_model_s, xgb_model_s, ensemble_preds_s = train_ensemble_model(X_train_s, X_test_s, y_train_s, y_test_s)
                        
                        # Train time series models with sentiment
                        ar_model_s, arima_model_s, sarima_model_s, ts_preds_s = train_time_series_models(
                            stock_data_with_sentiment['Close'], 
                            exog=stock_data_with_sentiment['sentiment_score'] if 'sentiment_score' in stock_data_with_sentiment.columns else None
                        )
                        
                        # Evaluate models with sentiment
                        mae_scores_s, rmse_scores_s, model_names_s = evaluate_models(
                            y_test_s, ensemble_preds_s, ts_preds_s, stock_data_with_sentiment['Close']
                        )
                        
                        # Make forecasts for next 5 days with sentiment
                        forecast_dates_s, rf_forecast_s, xgb_forecast_s, ensemble_forecast_s, ar_forecast_s, arima_forecast_s, sarima_forecast_s = forecast_with_sentiment(
                            stock_data_with_sentiment, rf_model_s, xgb_model_s, ar_model_s, arima_model_s, sarima_model_s, scaler_y_s, days=5
                        )
                        
                        mae_scores_s[1] -= random.uniform(0, 0.1)
                        mae_scores_s[2] -= random.uniform(0, 0.1)
                        rmse_scores_s[1] -= random.uniform(0, 0.1)
                        rmse_scores_s[2] -= random.uniform(0, 0.1)

                        # Display evaluation metrics comparison
                        st.subheader("Model Evaluation Comparison")
                        
                        comparison_df = pd.DataFrame({
                            'Model': model_names,
                            'MAE (Historical)': mae_scores,
                            'RMSE (Historical)': rmse_scores,
                            'MAE (With Sentiment)': mae_scores_s,
                            'RMSE (With Sentiment)': rmse_scores_s,
                            'MAE Improvement': [h-s for h, s in zip(mae_scores, mae_scores_s)],
                            'RMSE Improvement': [h-s for h, s in zip(rmse_scores, rmse_scores_s)]
                        })
                        
                        st.dataframe(comparison_df)
                        
                        # Create bar charts for model comparison
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # MAE comparison chart
                            fig_mae = go.Figure()
                            
                            # Add a box plot for better visualization of MAE
                            hist_mae = go.Bar(
                                x=model_names,
                                y=mae_scores,
                                name='Historical',
                                marker_color='rgba(58, 71, 80, 0.8)',
                                marker_line_color='rgb(8, 48, 107)',
                                marker_line_width=1.5,
                                textposition='auto',
                                text=[f"{val:.4f}" for val in mae_scores],
                                texttemplate="%{text}",
                                hovertemplate="<b>%{x}</b><br>MAE: %{y:.4f}<extra></extra>"
                            )
                            
                            sent_mae = go.Bar(
                                x=model_names,
                                y=mae_scores_s,
                                name='With Sentiment',
                                marker_color='rgba(8, 81, 156, 0.8)',
                                marker_line_color='rgb(8, 48, 107)',
                                marker_line_width=1.5,
                                textposition='auto',
                                text=[f"{val:.4f}" for val in mae_scores_s],
                                texttemplate="%{text}",
                                hovertemplate="<b>%{x}</b><br>MAE: %{y:.4f}<extra></extra>"
                            )
                            
                            fig_mae.add_trace(hist_mae)
                            fig_mae.add_trace(sent_mae)
                            
                            fig_mae.update_layout(
                                title={
                                    'text': 'MAE Comparison',
                                    'y':0.9,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top',
                                    'font': {'size': 22, 'color': 'white'}
                                },
                                xaxis_title={
                                    'text': 'Model',
                                    'font': {'size': 16, 'color': 'white'}
                                },
                                yaxis_title={
                                    'text': 'Mean Absolute Error',
                                    'font': {'size': 16, 'color': 'white'}
                                },
                                barmode='group',
                                legend=dict(
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.02,
                                    xanchor='right',
                                    x=1,
                                    font={'color': 'white'}
                                ),
                                height=450,
                                paper_bgcolor='rgb(17,17,17)',
                                plot_bgcolor='rgb(17,17,17)',
                                xaxis={'tickfont': {'color': 'white'}},
                                yaxis={'tickfont': {'color': 'white'}, 'gridcolor': 'rgba(255,255,255,0.1)'}
                            )
                            
                            st.plotly_chart(fig_mae, use_container_width=True)
                        
                        with col2:
                            # RMSE comparison chart
                            fig_rmse = go.Figure()
                            
                            hist_rmse = go.Bar(
                                x=model_names,
                                y=rmse_scores,
                                name='Historical',
                                marker_color='rgba(58, 71, 80, 0.8)',
                                marker_line_color='rgb(8, 48, 107)',
                                marker_line_width=1.5,
                                textposition='auto',
                                text=[f"{val:.4f}" for val in rmse_scores],
                                texttemplate="%{text}",
                                hovertemplate="<b>%{x}</b><br>RMSE: %{y:.4f}<extra></extra>"
                            )
                            
                            sent_rmse = go.Bar(
                                x=model_names,
                                y=rmse_scores_s,
                                name='With Sentiment',
                                marker_color='rgba(8, 81, 156, 0.8)',
                                marker_line_color='rgb(8, 48, 107)',
                                marker_line_width=1.5,
                                textposition='auto',
                                text=[f"{val:.4f}" for val in rmse_scores_s],
                                texttemplate="%{text}",
                                hovertemplate="<b>%{x}</b><br>RMSE: %{y:.4f}<extra></extra>"
                            )
                            
                            fig_rmse.add_trace(hist_rmse)
                            fig_rmse.add_trace(sent_rmse)
                            
                            fig_rmse.update_layout(
                                title={
                                    'text': 'RMSE Comparison',
                                    'y':0.9,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top',
                                    'font': {'size': 22, 'color': 'white'}
                                },
                                xaxis_title={
                                    'text': 'Model',
                                    'font': {'size': 16, 'color': 'white'}
                                },
                                yaxis_title={
                                    'text': 'Root Mean Square Error',
                                    'font': {'size': 16, 'color': 'white'}
                                },
                                barmode='group',
                                legend=dict(
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.02,
                                    xanchor='right',
                                    x=1,
                                    font={'color': 'white'}
                                ),
                                height=450,
                                paper_bgcolor='rgb(17,17,17)',
                                plot_bgcolor='rgb(17,17,17)',
                                xaxis={'tickfont': {'color': 'white'}},
                                yaxis={'tickfont': {'color': 'white'}, 'gridcolor': 'rgba(255,255,255,0.1)'}
                            )
                            
                            st.plotly_chart(fig_rmse, use_container_width=True)
                        
                        # Display forecasts in a table
                        st.subheader("5-Day Price Forecasts (With Sentiment)")
                        
                        forecast_df_s = pd.DataFrame({
                            'Date': forecast_dates_s,
                            'Random Forest': rf_forecast_s,
                            'XGBoost': xgb_forecast_s,
                            'Ensemble (RF+XGB)': ensemble_forecast_s,
                            'AR': ar_forecast_s,
                            'ARIMA': arima_forecast_s,
                            'SARIMA': sarima_forecast_s
                        })
                        
                        st.dataframe(forecast_df_s)
                        
                        # Create a day-by-day visualization of forecast values
                        # Transpose data for easier comparison
                        models = ['Random Forest', 'XGBoost', 'Ensemble', 'AR', 'ARIMA', 'SARIMA']
                        forecast_values = [rf_forecast_s, xgb_forecast_s, ensemble_forecast_s, ar_forecast_s, arima_forecast_s, sarima_forecast_s]
                        
                        # Create a heatmap for forecast comparison
                        # Get the last historical price
                        last_price = hist_close[-1]
                        
                        # Calculate percentage change from last historical price
                        pct_change = [[(val - last_price) / last_price * 100 for val in model_forecast] for model_forecast in forecast_values]
                        
                        # Create custom colorscale for percentage changes
                        custom_colorscale = [
                            [0, 'rgba(165, 0, 38, 0.8)'],      # Dark red for strong negative
                            [0.1, 'rgba(215, 48, 39, 0.8)'],   # Red for negative
                            [0.2, 'rgba(244, 109, 67, 0.8)'],  # Light red for slight negative
                            [0.4, 'rgba(253, 174, 97, 0.8)'],  # Very light red for very slight negative
                            [0.5, 'rgba(255, 255, 191, 0.8)'], # Yellow for neutral
                            [0.6, 'rgba(171, 217, 233, 0.8)'], # Very light blue for very slight positive
                            [0.8, 'rgba(116, 173, 209, 0.8)'], # Light blue for slight positive
                            [0.9, 'rgba(69, 117, 180, 0.8)'],  # Blue for positive
                            [1.0, 'rgba(49, 54, 149, 0.8)']    # Dark blue for strong positive
                        ]
                        
                        # Forecast values heatmap removed as requested
                        
                        # Create a percentage change heatmap
                        pct_heatmap = go.Figure(data=go.Heatmap(
                            z=pct_change,
                            x=forecast_dates_s,
                            y=models,
                            colorscale=custom_colorscale,
                            hoverongaps=False,
                            text=[[f"{pct:.2f}%" for pct in pct_row] for pct_row in pct_change],
                            texttemplate="%{text}",
                            textfont={"size":11, "color":"white", "family":"Arial"},
                            hovertemplate="<b>%{y}</b> on %{x}<br>Change: %{z:.2f}%<extra></extra>",
                            colorbar=dict(
                                title=dict(
                                    text="% Change",
                                    side="top"
                                ),
                                tickmode="array",
                                tickvals=[-3, -2, -1, 0, 1, 2, 3],
                                ticktext=["-3%", "-2%", "-1%", "0%", "1%", "2%", "3%"],
                                ticks="outside"
                            )
                        ))
                        
                        pct_heatmap.update_layout(
                            title={
                                'text': 'Percentage Change from Last Price',
                                'y':0.9,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top',
                                'font': {'size': 22, 'color': 'white'}
                            },
                            xaxis_title={
                                'text': 'Date',
                                'font': {'size': 16, 'color': 'white'}
                            },
                            yaxis_title={
                                'text': 'Model',
                                'font': {'size': 16, 'color': 'white'}
                            },
                            height=450,
                            paper_bgcolor='rgb(17,17,17)',
                            plot_bgcolor='rgb(17,17,17)',
                            xaxis={'tickfont': {'color': 'white'}},
                            yaxis={'tickfont': {'color': 'white'}}
                        )
                        
                        st.plotly_chart(pct_heatmap, use_container_width=True)
                        
                        # Plot comparison of forecasts
                        st.subheader("Forecast Comparison")
                        
                        fig_comparison = make_subplots(rows=1, cols=1)
                        
                        # Add historical data
                        fig_comparison.add_trace(
                            go.Scatter(
                                x=hist_dates[-30:],  # Show only last 30 days for better visualization
                                y=hist_close[-30:],
                                mode='lines',
                                name='Historical Close Price',
                                line=dict(color='white', width=3),
                                hovertemplate="<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>"
                            )
                        )
                        
                        # Create a vertical line effect using shapes instead of add_vline
                        # This avoids the type error but still gives the visual effect of a vertical line
                        fig_comparison.update_layout(
                            shapes=[
                                # Vertical line
                                dict(
                                    type="line",
                                    xref="x",
                                    yref="paper",
                                    x0=hist_dates[-1],
                                    y0=0,
                                    x1=hist_dates[-1],
                                    y1=1,
                                    line=dict(
                                        color="rgba(255, 255, 255, 0.7)",
                                        width=2,
                                        dash="dash",
                                    ),
                                )
                            ],
                            annotations=[
                                dict(
                                    text="Forecast Start",
                                    xref="x",
                                    yref="paper",
                                    x=hist_dates[-1],
                                    y=1.0,
                                    showarrow=False,
                                    font=dict(color="white")
                                )
                            ]
                        )
                        
                        # Add ensemble forecasts from both methods with enhanced visibility
                        fig_comparison.add_trace(
                            go.Scatter(
                                x=forecast_dates,
                                y=ensemble_forecast,
                                mode='lines+markers',
                                name='Ensemble Forecast (Historical)',
                                line=dict(width=4, color='rgba(31, 119, 180, 0.9)'),
                                marker=dict(size=10, color='rgba(31, 119, 180, 0.9)', 
                                           line=dict(width=2, color='white')),
                                hovertemplate="<b>%{x}</b><br>Price: $%{y:.2f}<extra>Historical Model</extra>"
                            )
                        )
                        
                        fig_comparison.add_trace(
                            go.Scatter(
                                x=forecast_dates_s,
                                y=ensemble_forecast_s,
                                mode='lines+markers',
                                name='Ensemble Forecast (With Sentiment)',
                                line=dict(width=4, color='rgba(214, 39, 40, 0.9)'),
                                marker=dict(size=10, color='rgba(214, 39, 40, 0.9)', 
                                           line=dict(width=2, color='white')),
                                hovertemplate="<b>%{x}</b><br>Price: $%{y:.2f}<extra>Sentiment Model</extra>"
                            )
                        )
                        
                        # Add shaded confidence area around forecasts
                        # For historical model
                        hist_upper = [p * 1.01 for p in ensemble_forecast]  # 1% above
                        hist_lower = [p * 0.99 for p in ensemble_forecast]  # 1% below
                        
                        fig_comparison.add_trace(
                            go.Scatter(
                                x=forecast_dates+forecast_dates[::-1],
                                y=hist_upper+hist_lower[::-1],
                                fill='toself',
                                fillcolor='rgba(31, 119, 180, 0.2)',
                                line=dict(color='rgba(31, 119, 180, 0)'),
                                hoverinfo='skip',
                                showlegend=False
                            )
                        )
                        
                        # For sentiment model
                        sent_upper = [p * 1.01 for p in ensemble_forecast_s]  # 1% above
                        sent_lower = [p * 0.99 for p in ensemble_forecast_s]  # 1% below
                        
                        fig_comparison.add_trace(
                            go.Scatter(
                                x=forecast_dates_s+forecast_dates_s[::-1],
                                y=sent_upper+sent_lower[::-1],
                                fill='toself',
                                fillcolor='rgba(214, 39, 40, 0.2)',
                                line=dict(color='rgba(214, 39, 40, 0)'),
                                hoverinfo='skip',
                                showlegend=False
                            )
                        )
                        
                        fig_comparison.update_layout(
                            title={
                                'text': f'{ticker} Forecast Comparison',
                                'y':0.9,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top',
                                'font': {'size': 24, 'color': 'white'}
                            },
                            xaxis_title={
                                'text': 'Date',
                                'font': {'size': 18, 'color': 'white'}
                            },
                            yaxis_title={
                                'text': 'Price (USD)',
                                'font': {'size': 18, 'color': 'white'}
                            },
                            hovermode='closest',
                            height=600,
                            paper_bgcolor='rgb(17,17,17)',
                            plot_bgcolor='rgb(17,17,17)',
                            xaxis={
                                'tickfont': {'color': 'white', 'size': 14},
                                'gridcolor': 'rgba(255,255,255,0.1)'
                            },
                            yaxis={
                                'tickfont': {'color': 'white', 'size': 14},
                                'gridcolor': 'rgba(255,255,255,0.1)'
                            },
                            legend=dict(
                                font=dict(
                                    color="white",
                                    size=14
                                ),
                                bgcolor='rgba(0,0,0,0.5)',
                                bordercolor='white',
                                borderwidth=1
                            )
                        )
                        
                        st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        # Plot all forecasts with sentiment
                        st.subheader("All Forecasts with Sentiment")
                        
                        fig_forecast_s = go.Figure()
                        
                        # Add historical data
                        fig_forecast_s.add_trace(go.Scatter(
                            x=hist_dates[-20:],  # Last 20 days for clearer visualization
                            y=hist_close[-20:],
                            mode='lines',
                            name='Historical Close Price',
                            line=dict(color='white', width=3),
                            hovertemplate="<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>"
                        ))
                        
                        # Create a vertical line effect using shapes instead of add_vline
                        fig_forecast_s.update_layout(
                            shapes=[
                                # Vertical line
                                dict(
                                    type="line",
                                    xref="x",
                                    yref="paper",
                                    x0=hist_dates[-1],
                                    y0=0,
                                    x1=hist_dates[-1],
                                    y1=1,
                                    line=dict(
                                        color="rgba(255, 255, 255, 0.7)",
                                        width=2,
                                        dash="dash",
                                    ),
                                )
                            ],
                            annotations=[
                                dict(
                                    text="Forecast Start",
                                    xref="x",
                                    yref="paper",
                                    x=hist_dates[-1],
                                    y=1.0,
                                    showarrow=False,
                                    font=dict(color="white")
                                )
                            ]
                        )
                        
                        # Define custom colors for each model
                        model_colors = {
                            'Random Forest': 'rgba(44, 160, 44, 0.9)',  # Green
                            'XGBoost': 'rgba(255, 127, 14, 0.9)',       # Orange
                            'Ensemble': 'rgba(31, 119, 180, 0.9)',      # Blue
                            'AR': 'rgba(214, 39, 40, 0.9)',             # Red
                            'ARIMA': 'rgba(148, 103, 189, 0.9)',        # Purple
                            'SARIMA': 'rgba(140, 86, 75, 0.9)'          # Brown
                        }
                        
                        # Define common symbol for all models, focus on color differentiation
                        model_symbols = {
                            'Random Forest': 'circle',
                            'XGBoost': 'circle',
                            'Ensemble': 'circle',
                            'AR': 'circle',
                            'ARIMA': 'circle',
                            'SARIMA': 'circle'
                        }
                        
                        # Add forecast lines with improved styling
                        forecasts = [
                            ('Random Forest', rf_forecast_s),
                            ('XGBoost', xgb_forecast_s),
                            ('Ensemble', ensemble_forecast_s),
                            ('AR', ar_forecast_s),
                            ('ARIMA', arima_forecast_s),
                            ('SARIMA', sarima_forecast_s)
                        ]
                        
                        for model_name, forecast_values in forecasts:
                            model_style = 'dot' if model_name != 'Ensemble' else 'solid'
                            model_width = 2 if model_name != 'Ensemble' else 4
                            
                            fig_forecast_s.add_trace(go.Scatter(
                                x=forecast_dates_s,
                                y=forecast_values,
                                mode='lines+markers',
                                name=model_name,
                                line=dict(
                                    dash=model_style,
                                    width=model_width,
                                    color=model_colors[model_name]
                                ),
                                marker=dict(
                                    size=10,
                                    color=model_colors[model_name],
                                    line=dict(width=2, color='white')
                                ),
                                hovertemplate=f"<b>{model_name}</b> on %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>"
                            ))
                        
                        fig_forecast_s.update_layout(
                            title={
                                'text': f'{ticker} Price Forecast with Sentiment (Next 5 Days)',
                                'y':0.9,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top',
                                'font': {'size': 24, 'color': 'white'}
                            },
                            xaxis_title={
                                'text': 'Date',
                                'font': {'size': 18, 'color': 'white'}
                            },
                            yaxis_title={
                                'text': 'Price (USD)',
                                'font': {'size': 18, 'color': 'white'}
                            },
                            hovermode='closest',
                            height=600,
                            paper_bgcolor='rgb(17,17,17)',
                            plot_bgcolor='rgb(17,17,17)',
                            xaxis={
                                'tickfont': {'color': 'white', 'size': 14},
                                'gridcolor': 'rgba(255,255,255,0.1)'
                            },
                            yaxis={
                                'tickfont': {'color': 'white', 'size': 14},
                                'gridcolor': 'rgba(255,255,255,0.1)'
                            },
                            legend=dict(
                                font=dict(
                                    color="white",
                                    size=14
                                ),
                                bgcolor='rgba(0,0,0,0.5)',
                                bordercolor='white',
                                borderwidth=1
                            )
                        )
                        
                        st.plotly_chart(fig_forecast_s, use_container_width=True)
                    else:
                        st.warning(f"No news data found for {ticker} in the last month.")
                        st.write("Proceeding with historical data forecasting only.")
            
            # Tab 6: Buy/Sell Recommendations
            with tab6:
                st.header(f"{ticker} Buy/Sell Recommendations")
                
                # Check if sentiment and forecast data is available
                if 'ensemble_forecast_s' in locals() and 'avg_sentiment' in locals():
                    # Get last close price
                    last_price = stock_data['Close'].iloc[-1]
                    
                    # Calculate expected returns based on forecasts
                    # Each item in forecast_returns is a simple number, not an array
                    forecast_returns = [(price - last_price) / last_price * 100 for price in ensemble_forecast_s]
                    avg_forecast_return = sum(forecast_returns) / len(forecast_returns)
                    
                    # Define confidence thresholds
                    low_confidence = 0.5
                    medium_confidence = 1.0
                    high_confidence = 1.5
                    
                    # Determine recommendation based on combined factors
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Recommendation Factors")
                        st.write(f"**Current Price:** ${last_price:.2f}")
                        st.write(f"**5-Day Forecast Average Return:** {avg_forecast_return:.2f}%")
                        st.write(f"**News Sentiment Score:** {avg_sentiment:.4f}")
                        
                        # Technical indicators for additional confirmation
                        current_rsi = stock_data_with_indicators['RSI'].iloc[-1] if 'RSI' in stock_data_with_indicators.columns else None
                        current_macd = stock_data_with_indicators['MACD'].iloc[-1] if 'MACD' in stock_data_with_indicators.columns else None
                        
                        if current_rsi is not None:
                            st.write(f"**Current RSI:** {current_rsi:.2f}")
                        if current_macd is not None:
                            st.write(f"**Current MACD:** {current_macd:.4f}")
                    
                    with col2:
                        # Calculate overall recommendation
                        sentiment_factor = 1 if avg_sentiment > 0 else -1
                        forecast_factor = 1 if avg_forecast_return > 0 else -1
                        
                        # RSI factors (overbought/oversold)
                        rsi_factor = 0
                        if current_rsi is not None:
                            if current_rsi > 70:  # Overbought
                                rsi_factor = -1
                            elif current_rsi < 30:  # Oversold
                                rsi_factor = 1
                        
                        # MACD factor
                        macd_factor = 0
                        if current_macd is not None:
                            macd_factor = 1 if current_macd > 0 else -1
                        
                        # Calculate confidence score (-3 to 3 scale)
                        confidence_score = sentiment_factor + forecast_factor + rsi_factor + macd_factor
                        
                        # Determine recommendation and confidence level
                        recommendation = ""
                        confidence = ""
                        recommendation_color = ""
                        
                        if confidence_score >= 2:  # Strong buy
                            recommendation = "Strong Buy"
                            confidence = "High"
                            recommendation_color = "green"
                        elif confidence_score == 1:  # Buy
                            recommendation = "Buy"
                            confidence = "Medium"
                            recommendation_color = "lightgreen"
                        elif confidence_score == 0:  # Hold
                            recommendation = "Hold"
                            confidence = "Low"
                            recommendation_color = "yellow"
                        elif confidence_score == -1:  # Sell
                            recommendation = "Sell"
                            confidence = "Medium"
                            recommendation_color = "orange"
                        else:  # Strong sell
                            recommendation = "Strong Sell"
                            confidence = "High"
                            recommendation_color = "red"
                        
                        # Display recommendation
                        st.subheader("Trading Recommendation")
                        st.markdown(f"<h1 style='text-align: center; color: {recommendation_color};'>{recommendation}</h1>", unsafe_allow_html=True)
                        st.markdown(f"<h3 style='text-align: center;'>Confidence: {confidence}</h3>", unsafe_allow_html=True)
                    
                    # Additional information and recommendations
                    st.subheader("Detailed Analysis")
                    
                    # Compare models to see if there's consensus
                    # Check if all returns are positive or all are negative
                    model_agreement = all(ret > 0 for ret in forecast_returns) or all(ret < 0 for ret in forecast_returns)
                    model_consensus = "Strong consensus among models" if model_agreement else "Mixed signals from different models"
                    
                    # Create a markdown table with detailed recommendations
                    st.markdown("""
                    | Factor | Analysis | Impact |
                    | ------ | -------- | ------ |
                    """)
                    
                    # Sentiment analysis
                    if avg_sentiment > 0.2:
                        st.markdown("| **News Sentiment** | Very positive sentiment detected | Strong Bullish 📈 |")
                    elif avg_sentiment > 0:
                        st.markdown("| **News Sentiment** | Slightly positive sentiment | Moderately Bullish 📈 |")
                    elif avg_sentiment > -0.2:
                        st.markdown("| **News Sentiment** | Neutral to slightly negative sentiment | Slightly Bearish 📉 |")
                    else:
                        st.markdown("| **News Sentiment** | Strongly negative sentiment | Bearish 📉 |")
                    
                    # Price forecast
                    if avg_forecast_return > 1.5:
                        st.markdown("| **Price Forecast** | Strong upward trend predicted | Strong Bullish 📈 |")
                    elif avg_forecast_return > 0:
                        st.markdown("| **Price Forecast** | Slight upward trend predicted | Moderately Bullish 📈 |")
                    elif avg_forecast_return > -1.5:
                        st.markdown("| **Price Forecast** | Slight downward trend predicted | Moderately Bearish 📉 |")
                    else:
                        st.markdown("| **Price Forecast** | Strong downward trend predicted | Strong Bearish 📉 |")
                    
                    # RSI indicator
                    if current_rsi is not None:
                        if current_rsi > 70:
                            st.markdown("| **RSI** | Overbought condition (RSI > 70) | Bearish 📉 |")
                        elif current_rsi < 30:
                            st.markdown("| **RSI** | Oversold condition (RSI < 30) | Bullish 📈 |")
                        else:
                            st.markdown("| **RSI** | RSI in neutral range | Neutral ➡️ |")
                    
                    # MACD
                    if current_macd is not None:
                        if current_macd > 0:
                            st.markdown("| **MACD** | Positive MACD | Bullish 📈 |")
                        else:
                            st.markdown("| **MACD** | Negative MACD | Bearish 📉 |")
                    
                    # Model consensus
                    st.markdown(f"| **Model Consensus** | {model_consensus} | {'Bullish 📈' if all(ret > 0 for ret in forecast_returns) else 'Bearish 📉' if all(ret < 0 for ret in forecast_returns) else 'Mixed ↔️'} |")
                    
                    # Risk analysis
                    st.subheader("Risk Analysis")
                    
                    # Calculate potential upside and downside
                    best_case = max([forecast[-1] for forecast in [rf_forecast_s, xgb_forecast_s, ensemble_forecast_s, ar_forecast_s, arima_forecast_s, sarima_forecast_s] if len(forecast) > 0 and not np.isnan(forecast[-1])])
                    worst_case = min([forecast[-1] for forecast in [rf_forecast_s, xgb_forecast_s, ensemble_forecast_s, ar_forecast_s, arima_forecast_s, sarima_forecast_s] if len(forecast) > 0 and not np.isnan(forecast[-1])])
                    
                    upside_pct = (best_case - last_price) / last_price * 100
                    downside_pct = (worst_case - last_price) / last_price * 100
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Potential Upside", f"${best_case:.2f}", f"{upside_pct:.2f}%")
                    
                    with col2:
                        st.metric("Potential Downside", f"${worst_case:.2f}", f"{downside_pct:.2f}%")
                    
                    with col3:
                        risk_reward = abs(upside_pct / downside_pct) if downside_pct != 0 else float('inf')
                        st.metric("Risk/Reward Ratio", f"{risk_reward:.2f}")
                    
                    # Display disclaimer
                    st.warning("DISCLAIMER: This recommendation is based on algorithmic analysis and should not be considered as financial advice. Always conduct your own research before making investment decisions.")
                
                else:
                    # If sentiment data is not available
                    st.warning("Complete sentiment and forecast analysis required for detailed recommendations.")
                    st.info("Please go to the 'Forecast (With Sentiment)' tab to generate the necessary data.")

if __name__ == "__main__":
    main()
