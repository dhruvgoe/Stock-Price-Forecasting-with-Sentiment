import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy import stats
import streamlit as st
import random

def apply_constraints(forecast_values, last_price, max_change_percent=0.01, model_type=''):
    """
    Apply constraints to forecasted values to prevent excessive deviations
    
    Args:
        forecast_values: Array of forecasted values
        last_price: Last observed price
        max_change_percent: Maximum allowed percentage change per day (default 1%)
        model_type: Type of model for slight variations in constraints
        
    Returns:
        Array of constrained forecasted values
    """
    import hashlib
    constrained_values = forecast_values.copy()
    
    # Calculate previous day price for each forecast day to limit day-to-day changes
    # Start with the last observed price
    prev_price = last_price
    
    # Set model-specific variations based on model type
    variation_factor = 1.0
    if model_type == 'rf':
        variation_factor = 0.95  # Random Forest slightly more conservative
        trend_weight = 0.3       # Less influenced by trend
    elif model_type == 'xgb':
        variation_factor = 1.05  # XGBoost slightly more aggressive (but still within limits)
        trend_weight = 0.7       # More influenced by trend
    elif model_type == 'ar':
        variation_factor = 0.97  # AR slightly different
        trend_weight = 0.5       # Moderately influenced by trend
    elif model_type == 'arima':
        variation_factor = 0.96  # ARIMA slightly more conservative
        trend_weight = 0.4       # Less influenced by trend
    elif model_type == 'sarima':
        variation_factor = 0.94  # SARIMA even more conservative
        trend_weight = 0.6       # More influenced by trend
    else:
        # Ensemble
        variation_factor = 1.0   # Balanced
        trend_weight = 0.5       # Moderately influenced by trend
    
    # Generate a seed based on model type and last_price
    # This ensures different stocks have different patterns
    seed_str = f"{model_type}_{int(last_price*100)}"
    seed_hash = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % 1000000
    random.seed(seed_hash)  # Set seed for reproducibility but unique to this stock/model
    
    # Generate a trend bias for this model and stock
    # This determines if predictions tend to go up or down
    trend_bias = random.uniform(-0.4, 0.4)  # -0.4 to +0.4 range
    
    # Adjust the base max change percent based on model type
    effective_max_change = max_change_percent * variation_factor
    
    # Ensure max_change doesn't exceed 1.5% in any case (worst case scenario)
    if effective_max_change > 0.015:
        effective_max_change = 0.015
    
    for i in range(len(constrained_values)):
        # Day-specific seed for consistent but unique randomness
        day_seed = seed_hash + i*1000
        random.seed(day_seed)
        
        # Add a controlled amount of randomness that increases with forecast length
        day_factor = (i + 1) / len(constrained_values)  # Ranges from 1/N to 1.0
        random_factor = 1 + (random.uniform(-0.2, 0.2) * 0.01 * day_factor)
        
        # Apply trend bias with day factor (increasing impact over time)
        trend_effect = trend_bias * trend_weight * day_factor * 0.01 * prev_price
        
        # Calculate max change allowed for this prediction (relative to previous price)
        day_max_change = prev_price * effective_max_change * random_factor
        
        # Get base constrained value
        base_value = constrained_values[i]
        
        # Limit the value compared to the previous day's price
        if base_value > prev_price + day_max_change:
            constrained_values[i] = prev_price + day_max_change
        elif base_value < prev_price - day_max_change:
            constrained_values[i] = prev_price - day_max_change
        
        # Apply trend effect while keeping within constraints
        constrained_values[i] += trend_effect
        
        # Do a final check to ensure it stays within hard limits (1.5%)
        hard_limit = prev_price * 0.015
        if constrained_values[i] > prev_price + hard_limit:
            constrained_values[i] = prev_price + hard_limit
        elif constrained_values[i] < prev_price - hard_limit:
            constrained_values[i] = prev_price - hard_limit
        
        # The current day's constrained value becomes the previous day's value for the next iteration
        prev_price = constrained_values[i]
            
    return constrained_values

def predict_price_direction(sentiment_score):
    """
    Predict price direction based on sentiment score
    
    Args:
        sentiment_score: Overall sentiment score
        
    Returns:
        String: Price direction prediction and confidence level
    """
    if sentiment_score > 0.2:
        confidence = min(abs(sentiment_score) * 100, 95)
        return f"⬆️ Price likely to rise (Confidence: {confidence:.1f}%)"
    elif sentiment_score < -0.05:
        confidence = min(abs(sentiment_score) * 100, 95)
        return f"⬇️ Price likely to fall (Confidence: {confidence:.1f}%)"
    else:
        return "➡️ Price likely to remain stable"

def forecast_prices(stock_data, rf_model, xgb_model, ar_model, arima_model, sarima_model, scaler_y, days=5):
    """
    Forecast stock prices for the next few days using different models
    
    Args:
        stock_data (DataFrame): Historical stock data
        rf_model: Trained Random Forest model
        xgb_model: Trained XGBoost model
        ar_model: Trained AutoRegressive model
        arima_model: Trained ARIMA model
        sarima_model: Trained SARIMA model
        scaler_y: Scaler for the target variable
        days (int): Number of days to forecast
        
    Returns:
        tuple: forecast_dates, rf_forecast, xgb_forecast, ensemble_forecast, ar_forecast, arima_forecast, sarima_forecast
    """
    # Define feature order to match training
    feature_order = [
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
        'price_change_1d', 'price_change_3d', 'price_change_5d',
        'rolling_mean_3d', 'rolling_mean_5d',
        'rolling_std_3d', 'rolling_std_5d'
    ]
    
    # Generate forecast dates
    last_date = stock_data['Date'].iloc[-1]
    forecast_dates = []
    
    # Get last observed price for constraints
    last_price = stock_data['Close'].iloc[-1]
    
    for i in range(1, days + 1):
        next_date = last_date + timedelta(days=i)
        # Skip weekends
        while next_date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
            next_date = next_date + timedelta(days=1)
        forecast_dates.append(next_date.strftime('%Y-%m-%d'))
    
    # Prepare data for machine learning forecasts
    last_row = stock_data.iloc[-1:].copy()
    
    # Initialize lists to store forecasts
    rf_forecast = []
    xgb_forecast = []
    ensemble_forecast = []
    
    # Generate features for each forecasted day
    for i in range(days):
        # Create features for the next day
        features = {}
        
        if i == 0:
            # For the first day, use the last actual values
            for j in range(1, 6):
                if j == 1:
                    features[f'lag_{j}'] = last_row['Close'].values[0]
                else:
                    features[f'lag_{j}'] = stock_data['Close'].iloc[-j]
            
            # Price changes
            features['price_change_1d'] = (last_row['Close'].values[0] - stock_data['Close'].iloc[-2]) / stock_data['Close'].iloc[-2]
            features['price_change_3d'] = (last_row['Close'].values[0] - stock_data['Close'].iloc[-4]) / stock_data['Close'].iloc[-4]
            features['price_change_5d'] = (last_row['Close'].values[0] - stock_data['Close'].iloc[-6]) / stock_data['Close'].iloc[-6]
            
            # Rolling means
            features['rolling_mean_3d'] = stock_data['Close'].iloc[-3:].mean()
            features['rolling_mean_5d'] = stock_data['Close'].iloc[-5:].mean()
            
            # Rolling std
            features['rolling_std_3d'] = stock_data['Close'].iloc[-3:].std()
            features['rolling_std_5d'] = stock_data['Close'].iloc[-5:].std()
        else:
            # For subsequent days, use the forecasted values
            for j in range(1, 6):
                if j <= i:
                    features[f'lag_{j}'] = ensemble_forecast[i-j]
                else:
                    features[f'lag_{j}'] = stock_data['Close'].iloc[-(j-i)]
            
            # Price changes
            if i == 1:
                features['price_change_1d'] = (ensemble_forecast[i-1] - last_row['Close'].values[0]) / last_row['Close'].values[0]
            else:
                features['price_change_1d'] = (ensemble_forecast[i-1] - ensemble_forecast[i-2]) / ensemble_forecast[i-2]
            
            if i >= 3:
                features['price_change_3d'] = (ensemble_forecast[i-1] - ensemble_forecast[i-4]) / ensemble_forecast[i-4]
            else:
                idx = 3 - i
                features['price_change_3d'] = (ensemble_forecast[i-1] - stock_data['Close'].iloc[-idx]) / stock_data['Close'].iloc[-idx]
            
            if i >= 5:
                features['price_change_5d'] = (ensemble_forecast[i-1] - ensemble_forecast[i-6]) / ensemble_forecast[i-6]
            else:
                idx = 5 - i
                features['price_change_5d'] = (ensemble_forecast[i-1] - stock_data['Close'].iloc[-idx]) / stock_data['Close'].iloc[-idx]
            
            # Rolling means and std
            if i < 3:
                last_vals = stock_data['Close'].iloc[-(3-i):].tolist()
                prev_forecasts = ensemble_forecast[:i]
                combined = last_vals + prev_forecasts
                features['rolling_mean_3d'] = np.mean(combined)
                features['rolling_std_3d'] = np.std(combined)
            else:
                features['rolling_mean_3d'] = np.mean(ensemble_forecast[i-3:i])
                features['rolling_std_3d'] = np.std(ensemble_forecast[i-3:i])
            
            if i < 5:
                last_vals = stock_data['Close'].iloc[-(5-i):].tolist()
                prev_forecasts = ensemble_forecast[:i]
                combined = last_vals + prev_forecasts
                features['rolling_mean_5d'] = np.mean(combined)
                features['rolling_std_5d'] = np.std(combined)
            else:
                features['rolling_mean_5d'] = np.mean(ensemble_forecast[i-5:i])
                features['rolling_std_5d'] = np.std(ensemble_forecast[i-5:i])
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([features])
        
        # Ensure features are in the same order as during training
        features_df = features_df[feature_order]
        
        # Make predictions with RF and XGBoost
        rf_pred = rf_model.predict(features_df)[0]
        xgb_pred = xgb_model.predict(features_df)[0]
        
        # Ensemble prediction
        ensemble_pred = (rf_pred + xgb_pred) / 2
        
        # Inverse transform to get actual price
        rf_pred_price = scaler_y.inverse_transform([[rf_pred]])[0][0]
        xgb_pred_price = scaler_y.inverse_transform([[xgb_pred]])[0][0]
        ensemble_pred_price = scaler_y.inverse_transform([[ensemble_pred]])[0][0]
        
        # Apply constraints to prevent unrealistic predictions
        max_change = last_price * 0.02 * (i+1)
        
        # Constrain RF prediction
        if rf_pred_price > last_price + max_change:
            rf_pred_price = last_price + max_change
        elif rf_pred_price < last_price - max_change:
            rf_pred_price = last_price - max_change
            
        # Constrain XGBoost prediction
        if xgb_pred_price > last_price + max_change:
            xgb_pred_price = last_price + max_change
        elif xgb_pred_price < last_price - max_change:
            xgb_pred_price = last_price - max_change
            
        # Constrain Ensemble prediction
        if ensemble_pred_price > last_price + max_change:
            ensemble_pred_price = last_price + max_change
        elif ensemble_pred_price < last_price - max_change:
            ensemble_pred_price = last_price - max_change
        
        # Add to forecast lists
        rf_forecast.append(rf_pred_price)
        xgb_forecast.append(xgb_pred_price)
        ensemble_forecast.append(ensemble_pred_price)
    
    # Apply model-specific constraints to ML model forecasts
    rf_forecast = apply_constraints(np.array(rf_forecast), last_price, model_type='rf')
    xgb_forecast = apply_constraints(np.array(xgb_forecast), last_price, model_type='xgb')
    ensemble_forecast = apply_constraints(np.array(ensemble_forecast), last_price)
    
    # Time series forecasts
    try:
        if ar_model is not None:
            ar_model_fit = sm.tsa.AutoReg(stock_data['Close'].values, lags=5).fit()
            ar_forecast = ar_model_fit.forecast(steps=days)
            # Apply constraints with model-specific factors
            ar_forecast = apply_constraints(ar_forecast, last_price, model_type='ar')
        else:
            ar_forecast = np.full(days, np.nan)
    except Exception as e:
        print(f"Error in AR model: {e}")
        ar_forecast = np.full(days, np.nan)
    
    try:
        if arima_model is not None:
            arima_model_fit = sm.tsa.ARIMA(stock_data['Close'].values, order=(5, 1, 0)).fit()
            arima_forecast = arima_model_fit.forecast(steps=days)
            # Apply constraints with model-specific factors
            arima_forecast = apply_constraints(arima_forecast, last_price, model_type='arima')
        else:
            arima_forecast = np.full(days, np.nan)
    except Exception as e:
        print(f"Error in ARIMA model: {e}")
        arima_forecast = np.full(days, np.nan)
    
    try:
        if sarima_model is not None:
            sarima_model_fit = sm.tsa.SARIMAX(
                stock_data['Close'].values, 
                order=(1, 1, 1), 
                seasonal_order=(1, 1, 1, 5)
            ).fit(disp=False)
            sarima_forecast = sarima_model_fit.forecast(steps=days)
            # Apply constraints with model-specific factors
            sarima_forecast = apply_constraints(sarima_forecast, last_price, model_type='sarima')
        else:
            sarima_forecast = np.full(days, np.nan)
    except Exception as e:
        print(f"Error in SARIMA model: {e}")
        sarima_forecast = np.full(days, np.nan)
    
    return forecast_dates, rf_forecast, xgb_forecast, ensemble_forecast, ar_forecast, arima_forecast, sarima_forecast

def forecast_with_sentiment(stock_data, rf_model, xgb_model, ar_model, arima_model, sarima_model, scaler_y, days=5):
    """
    Forecast stock prices for the next few days using different models with sentiment data
    
    Args:
        stock_data (DataFrame): Historical stock data with sentiment
        rf_model: Trained Random Forest model
        xgb_model: Trained XGBoost model
        ar_model: Trained AutoRegressive model
        arima_model: Trained ARIMA model
        sarima_model: Trained SARIMA model
        scaler_y: Scaler for the target variable
        days (int): Number of days to forecast
        
    Returns:
        tuple: forecast_dates, rf_forecast, xgb_forecast, ensemble_forecast, ar_forecast, arima_forecast, sarima_forecast
    """
    # Define feature order to match training
    feature_order = [
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
        'price_change_1d', 'price_change_3d', 'price_change_5d',
        'rolling_mean_3d', 'rolling_mean_5d',
        'rolling_std_3d', 'rolling_std_5d'
    ]
    
    # Check if sentiment is included in the features
    if 'sentiment_score' in stock_data.columns:
        feature_order.append('sentiment_score')
    
    # Generate forecast dates
    last_date = stock_data['Date'].iloc[-1]
    forecast_dates = []
    
    # Get last price for constraints
    last_price = stock_data['Close'].iloc[-1]
    
    for i in range(1, days + 1):
        next_date = last_date + timedelta(days=i)
        # Skip weekends
        while next_date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
            next_date = next_date + timedelta(days=1)
        forecast_dates.append(next_date.strftime('%Y-%m-%d'))
    
    # Prepare data for machine learning forecasts
    last_row = stock_data.iloc[-1:].copy()
    
    # Initialize lists to store forecasts
    rf_forecast = []
    xgb_forecast = []
    ensemble_forecast = []
    
    # Use the last sentiment score as the forecast sentiment
    last_sentiment = stock_data['sentiment_score'].iloc[-1]
    
    # Generate features for each forecasted day
    for i in range(days):
        # Create features for the next day
        features = {}
        
        if i == 0:
            # For the first day, use the last actual values
            for j in range(1, 6):
                if j == 1:
                    features[f'lag_{j}'] = last_row['Close'].values[0]
                else:
                    features[f'lag_{j}'] = stock_data['Close'].iloc[-j]
            
            # Price changes
            features['price_change_1d'] = (last_row['Close'].values[0] - stock_data['Close'].iloc[-2]) / stock_data['Close'].iloc[-2]
            features['price_change_3d'] = (last_row['Close'].values[0] - stock_data['Close'].iloc[-4]) / stock_data['Close'].iloc[-4]
            features['price_change_5d'] = (last_row['Close'].values[0] - stock_data['Close'].iloc[-6]) / stock_data['Close'].iloc[-6]
            
            # Rolling means
            features['rolling_mean_3d'] = stock_data['Close'].iloc[-3:].mean()
            features['rolling_mean_5d'] = stock_data['Close'].iloc[-5:].mean()
            
            # Rolling std
            features['rolling_std_3d'] = stock_data['Close'].iloc[-3:].std()
            features['rolling_std_5d'] = stock_data['Close'].iloc[-5:].std()
        else:
            # For subsequent days, use the forecasted values
            for j in range(1, 6):
                if j <= i:
                    features[f'lag_{j}'] = ensemble_forecast[i-j]
                else:
                    features[f'lag_{j}'] = stock_data['Close'].iloc[-(j-i)]
            
            # Price changes
            if i == 1:
                features['price_change_1d'] = (ensemble_forecast[i-1] - last_row['Close'].values[0]) / last_row['Close'].values[0]
            else:
                features['price_change_1d'] = (ensemble_forecast[i-1] - ensemble_forecast[i-2]) / ensemble_forecast[i-2]
            
            if i >= 3:
                features['price_change_3d'] = (ensemble_forecast[i-1] - ensemble_forecast[i-4]) / ensemble_forecast[i-4]
            else:
                idx = 3 - i
                features['price_change_3d'] = (ensemble_forecast[i-1] - stock_data['Close'].iloc[-idx]) / stock_data['Close'].iloc[-idx]
            
            if i >= 5:
                features['price_change_5d'] = (ensemble_forecast[i-1] - ensemble_forecast[i-6]) / ensemble_forecast[i-6]
            else:
                idx = 5 - i
                features['price_change_5d'] = (ensemble_forecast[i-1] - stock_data['Close'].iloc[-idx]) / stock_data['Close'].iloc[-idx]
            
            # Rolling means and std
            if i < 3:
                last_vals = stock_data['Close'].iloc[-(3-i):].tolist()
                prev_forecasts = ensemble_forecast[:i]
                combined = last_vals + prev_forecasts
                features['rolling_mean_3d'] = np.mean(combined)
                features['rolling_std_3d'] = np.std(combined)
            else:
                features['rolling_mean_3d'] = np.mean(ensemble_forecast[i-3:i])
                features['rolling_std_3d'] = np.std(ensemble_forecast[i-3:i])
            
            if i < 5:
                last_vals = stock_data['Close'].iloc[-(5-i):].tolist()
                prev_forecasts = ensemble_forecast[:i]
                combined = last_vals + prev_forecasts
                features['rolling_mean_5d'] = np.mean(combined)
                features['rolling_std_5d'] = np.std(combined)
            else:
                features['rolling_mean_5d'] = np.mean(ensemble_forecast[i-5:i])
                features['rolling_std_5d'] = np.std(ensemble_forecast[i-5:i])
        
        # Add sentiment score
        features['sentiment_score'] = last_sentiment
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([features])
        
        # Ensure features are in the same order as during training
        features_df = features_df[feature_order]
        
        # Make predictions with RF and XGBoost
        rf_pred = rf_model.predict(features_df)[0]
        xgb_pred = xgb_model.predict(features_df)[0]
        
        # Ensemble prediction
        ensemble_pred = (rf_pred + xgb_pred) / 2
        
        # Inverse transform to get actual price
        rf_pred_price = scaler_y.inverse_transform([[rf_pred]])[0][0]
        xgb_pred_price = scaler_y.inverse_transform([[xgb_pred]])[0][0]
        ensemble_pred_price = scaler_y.inverse_transform([[ensemble_pred]])[0][0]
        
        # Apply constraints to prevent unrealistic predictions
        max_change = last_price * 0.02 * (i+1)
        
        # Constrain RF prediction
        if rf_pred_price > last_price + max_change:
            rf_pred_price = last_price + max_change
        elif rf_pred_price < last_price - max_change:
            rf_pred_price = last_price - max_change
            
        # Constrain XGBoost prediction
        if xgb_pred_price > last_price + max_change:
            xgb_pred_price = last_price + max_change
        elif xgb_pred_price < last_price - max_change:
            xgb_pred_price = last_price - max_change
            
        # Constrain Ensemble prediction
        if ensemble_pred_price > last_price + max_change:
            ensemble_pred_price = last_price + max_change
        elif ensemble_pred_price < last_price - max_change:
            ensemble_pred_price = last_price - max_change
        
        # Add to forecast lists
        rf_forecast.append(rf_pred_price)
        xgb_forecast.append(xgb_pred_price)
        ensemble_forecast.append(ensemble_pred_price)
    
    # Apply model-specific constraints to ML model forecasts
    rf_forecast = apply_constraints(np.array(rf_forecast), last_price, model_type='rf')
    xgb_forecast = apply_constraints(np.array(xgb_forecast), last_price, model_type='xgb')
    ensemble_forecast = apply_constraints(np.array(ensemble_forecast), last_price)
    
    # Time series forecasts with sentiment as exogenous variable
    try:
        # Even though AR doesn't technically support exogenous variables,
        # we can still make a forecast based only on the historical data
        ar_model_fit = sm.tsa.AutoReg(stock_data['Close'].values, lags=5).fit()
        ar_forecast = ar_model_fit.forecast(steps=days)
        # Apply constraints with model-specific factors
        ar_forecast = apply_constraints(ar_forecast, last_price, model_type='ar')
    except Exception as e:
        print(f"Error in AR model with sentiment: {e}")
        ar_forecast = np.full(days, np.nan)
    
    try:
        # For ARIMA, use sentiment as an exogenous variable
        if arima_model is not None:
            # Create exogenous array with the last sentiment score
            exog_forecast = np.full(days, last_sentiment)
            
            arima_model_fit = sm.tsa.ARIMA(
                stock_data['Close'].values,
                order=(5, 1, 0),
                exog=stock_data['sentiment_score'].values
            ).fit()
            
            arima_forecast = arima_model_fit.forecast(steps=days, exog=exog_forecast)
            # Apply constraints with model-specific factors
            arima_forecast = apply_constraints(arima_forecast, last_price, model_type='arima')
        else:
            arima_forecast = np.full(days, np.nan)
    except Exception as e:
        print(f"Error in ARIMA model with sentiment: {e}")
        arima_forecast = np.full(days, np.nan)
    
    try:
        # For SARIMA, use sentiment as an exogenous variable
        if sarima_model is not None:
            # Create exogenous array with the last sentiment score
            exog_forecast = np.full(days, last_sentiment)
            
            sarima_model_fit = sm.tsa.SARIMAX(
                stock_data['Close'].values,
                exog=stock_data['sentiment_score'].values,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 5)
            ).fit(disp=False)
            
            sarima_forecast = sarima_model_fit.forecast(steps=days, exog=exog_forecast)
            # Apply constraints with model-specific factors
            sarima_forecast = apply_constraints(sarima_forecast, last_price, model_type='sarima')
        else:
            sarima_forecast = np.full(days, np.nan)
    except Exception as e:
        print(f"Error in SARIMA model with sentiment: {e}")
        sarima_forecast = np.full(days, np.nan)
    
    return forecast_dates, rf_forecast, xgb_forecast, ensemble_forecast, ar_forecast, arima_forecast, sarima_forecast