import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe
    
    Args:
        df (DataFrame): Stock data with OHLC
        
    Returns:
        DataFrame: Stock data with technical indicators
    """
    # Make a copy of the dataframe
    df = df.copy()
    
    # Calculate Moving Averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA14'] = df['Close'].rolling(window=14).mean()
    
    # Calculate Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use expanding instead of rolling for first few entries to avoid NaNs
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    
    # Handle division by zero
    avg_loss = avg_loss.replace(0, 0.001)
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Moving Average Convergence Divergence (MACD)
    exp1 = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Calculate Bollinger Bands with minimum periods
    df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['BB_StdDev'] = df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_StdDev'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_StdDev'] * 2)
    
    # Fill any remaining NaN values with appropriate values
    # For MA and BB, use the Close price
    df['MA7'] = df['MA7'].fillna(df['Close'])
    df['MA14'] = df['MA14'].fillna(df['Close'])
    df['BB_Middle'] = df['BB_Middle'].fillna(df['Close'])
    df['BB_Upper'] = df['BB_Upper'].fillna(df['Close'] * 1.02)  # 2% above
    df['BB_Lower'] = df['BB_Lower'].fillna(df['Close'] * 0.98)  # 2% below
    df['BB_StdDev'] = df['BB_StdDev'].fillna(df['Close'] * 0.01)  # 1% of close
    
    # For RSI, use neutral value
    df['RSI'] = df['RSI'].fillna(50)
    
    # For MACD, use 0 or small values
    df['MACD'] = df['MACD'].fillna(0)
    df['MACD_Signal'] = df['MACD_Signal'].fillna(0)
    df['MACD_Hist'] = df['MACD_Hist'].fillna(0)
    
    return df

def plot_technical_indicators(df, ticker):
    """
    Plot technical indicators
    
    Args:
        df (DataFrame): Stock data with technical indicators
        ticker (str): Stock ticker
        
    Returns:
        Figure: Plotly figure object
    """
    # Create subplot figure
    fig = make_subplots(
        rows=4, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.18,
        subplot_titles=(
            "OHLC with Bollinger Bands and Moving Averages", 
            "RSI", 
            "MACD",
            "Volume"
        ),
        row_heights=[4,2,2,2]
    )
    
    # OHLC Chart
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Moving Averages
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['MA7'],
            name="MA (7)",
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['MA14'],
            name="MA (14)",
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['BB_Upper'],
            name="BB Upper",
            line=dict(color='rgba(250, 0, 0, 0.5)', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['BB_Middle'],
            name="BB Middle",
            line=dict(color='rgba(0, 0, 250, 0.5)', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['BB_Lower'],
            name="BB Lower",
            line=dict(color='rgba(250, 0, 0, 0.5)', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(200, 200, 255, 0.1)'
        ),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['RSI'],
            name="RSI",
            line=dict(color='purple', width=1)
        ),
        row=2, col=1
    )
    
    # Add overbought/oversold levels for RSI
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=[70] * len(df),
            name="Overbought",
            line=dict(color='red', width=1, dash='dash')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=[30] * len(df),
            name="Oversold",
            line=dict(color='green', width=1, dash='dash')
        ),
        row=2, col=1
    )
    
    # MACD
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['MACD'],
            name="MACD",
            line=dict(color='blue', width=1)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['MACD_Signal'],
            name="Signal",
            line=dict(color='red', width=1)
        ),
        row=3, col=1
    )
    
    # MACD Histogram
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['MACD_Hist'],
            name="Histogram",
            marker=dict(
                color=np.where(df['MACD_Hist'] >= 0, 'green', 'red'),
                line=dict(color='rgb(0, 0, 0)', width=1)
            )
        ),
        row=3, col=1
    )
    
    # Set y-axis titles
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Technical Indicators",
        height=900,
        xaxis4_title="Date",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        showlegend=True
    )
    
    return fig
