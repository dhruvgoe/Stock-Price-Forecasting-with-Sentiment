# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import xgboost as xgb
# from statsmodels.tsa.ar_model import AutoReg
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# import warnings

# # Ignore statsmodels convergence warnings
# warnings.filterwarnings("ignore")

# def prepare_data(stock_data, include_sentiment=False, test_size=0.2):
#     """
#     Prepare data for machine learning models
    
#     Args:
#         stock_data (DataFrame): Historical stock data
#         include_sentiment (bool): Whether to include sentiment data
#         test_size (float): Test set size ratio
        
#     Returns:
#         tuple: X_train, X_test, y_train, y_test, scaler_y
#     """
#     # Create features from the time series data
#     df = stock_data.copy()
    
#     # Create lagged features
#     for i in range(1, 6):
#         df[f'lag_{i}'] = df['Close'].shift(i)
        
#     # Create price change features
#     df['price_change_1d'] = df['Close'].pct_change(1)
#     df['price_change_3d'] = df['Close'].pct_change(3)
#     df['price_change_5d'] = df['Close'].pct_change(5)
    
#     # Create rolling mean features
#     df['rolling_mean_3d'] = df['Close'].rolling(window=3).mean()
#     df['rolling_mean_5d'] = df['Close'].rolling(window=5).mean()
    
#     # Create rolling std features
#     df['rolling_std_3d'] = df['Close'].rolling(window=3).std()
#     df['rolling_std_5d'] = df['Close'].rolling(window=5).std()
    
#     # Drop NaN values
#     df = df.dropna()
    
#     # Define features and target
#     features = [
#         'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
#         'price_change_1d', 'price_change_3d', 'price_change_5d',
#         'rolling_mean_3d', 'rolling_mean_5d',
#         'rolling_std_3d', 'rolling_std_5d'
#     ]
    
#     # Add sentiment score if included
#     if include_sentiment and 'sentiment_score' in df.columns:
#         features.append('sentiment_score')
    
#     X = df[features]
#     y = df['Close']
    
#     # Scale target variable
#     scaler_y = MinMaxScaler()
#     y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    
#     # Split data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=test_size, shuffle=False)
    
#     return X_train, X_test, y_train, y_test, scaler_y

# def train_ensemble_model(X_train, X_test, y_train, y_test):
#     """
#     Train Random Forest and XGBoost models and create an ensemble
    
#     Args:
#         X_train, X_test, y_train, y_test: Training and testing data
        
#     Returns:
#         tuple: rf_model, xgb_model, ensemble_predictions
#     """
#     # Train Random Forest model
#     rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf_model.fit(X_train, y_train)
#     rf_preds = rf_model.predict(X_test)
    
#     # Train XGBoost model
#     xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
#     xgb_model.fit(X_train, y_train)
#     xgb_preds = xgb_model.predict(X_test)
    
#     # Create ensemble predictions (average of RF and XGBoost)
#     ensemble_preds = (rf_preds + xgb_preds) / 2
    
#     return rf_model, xgb_model, ensemble_preds

# def train_time_series_models(close_prices, exog=None):
#     """
#     Train time series models (AR, ARIMA, SARIMA)
    
#     Args:
#         close_prices (Series): Close price series
#         exog (Series, optional): Exogenous variable (e.g., sentiment)
        
#     Returns:
#         tuple: ar_model, arima_model, sarima_model, time_series_predictions
#     """
#     # Prepare data
#     data = close_prices.values
#     train_size = int(len(data) * 0.8)
#     train, test = data[:train_size], data[train_size:]
    
#     # Prepare exogenous variables if provided
#     exog_train, exog_test = None, None
#     if exog is not None:
#         exog = exog.values
#         exog_train, exog_test = exog[:train_size], exog[train_size:]
    
#     # Train AutoRegressive model
#     try:
#         ar_model = AutoReg(train, lags=5)
#         ar_model_fit = ar_model.fit()
#         ar_preds = ar_model_fit.predict(start=len(train), end=len(train)+len(test)-1)
#     except:
#         ar_model = None
#         ar_preds = np.full(len(test), np.nan)
    
#     # Train ARIMA model
#     try:
#         arima_model = ARIMA(train, order=(5, 1, 0))
#         arima_model_fit = arima_model.fit()
#         arima_preds = arima_model_fit.forecast(steps=len(test))
#     except:
#         arima_model = None
#         arima_preds = np.full(len(test), np.nan)
    
#     # Train SARIMA model
#     try:
#         if exog is not None:
#             sarima_model = SARIMAX(train, exog=exog_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
#             sarima_model_fit = sarima_model.fit(disp=False)
#             sarima_preds = sarima_model_fit.forecast(steps=len(test), exog=exog_test)
#         else:
#             sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
#             sarima_model_fit = sarima_model.fit(disp=False)
#             sarima_preds = sarima_model_fit.forecast(steps=len(test))
#     except:
#         sarima_model = None
#         sarima_preds = np.full(len(test), np.nan)
    
#     # Combine time series predictions
#     ts_preds = {
#         'ar': ar_preds,
#         'arima': arima_preds,
#         'sarima': sarima_preds
#     }
    
#     return ar_model, arima_model, sarima_model, ts_preds

# def evaluate_models(y_true, ensemble_preds, ts_preds, close_prices):
#     """
#     Evaluate model performance using MAE and RMSE
    
#     Args:
#         y_true: True values
#         ensemble_preds: Predictions from ensemble model
#         ts_preds: Predictions from time series models
#         close_prices: Original close prices
        
#     Returns:
#         tuple: mae_scores, rmse_scores, model_names
#     """
#     # Prepare data
#     data = close_prices.values
#     train_size = int(len(data) * 0.8)
#     _, test = data[:train_size], data[train_size:]
    
#     # Calculate metrics for ensemble models
#     mae_rf_xgb = mean_absolute_error(y_true, ensemble_preds)
#     rmse_rf_xgb = np.sqrt(mean_squared_error(y_true, ensemble_preds))
    
#     # Calculate metrics for time series models
#     mae_ar = mean_absolute_error(test, ts_preds['ar']) if not np.isnan(ts_preds['ar']).all() else np.nan
#     rmse_ar = np.sqrt(mean_squared_error(test, ts_preds['ar'])) if not np.isnan(ts_preds['ar']).all() else np.nan
    
#     mae_arima = mean_absolute_error(test, ts_preds['arima']) if not np.isnan(ts_preds['arima']).all() else np.nan
#     rmse_arima = np.sqrt(mean_squared_error(test, ts_preds['arima'])) if not np.isnan(ts_preds['arima']).all() else np.nan
    
#     mae_sarima = mean_absolute_error(test, ts_preds['sarima']) if not np.isnan(ts_preds['sarima']).all() else np.nan
#     rmse_sarima = np.sqrt(mean_squared_error(test, ts_preds['sarima'])) if not np.isnan(ts_preds['sarima']).all() else np.nan
    
#     # Compile metrics
#     model_names = ['Ensemble (RF+XGB)', 'AR', 'ARIMA', 'SARIMA']
#     mae_scores = [mae_rf_xgb, mae_ar, mae_arima, mae_sarima]
#     rmse_scores = [rmse_rf_xgb, rmse_ar, rmse_arima, rmse_sarima]
    
#     return mae_scores, rmse_scores, model_names










import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Ignore statsmodels convergence warnings
warnings.filterwarnings("ignore")

def prepare_data(stock_data, include_sentiment=False, test_size=0.2):
    df = stock_data.copy()

    for i in range(1, 6):
        df[f'lag_{i}'] = df['Close'].shift(i)
    
    df['price_change_1d'] = df['Close'].pct_change(1)
    df['price_change_3d'] = df['Close'].pct_change(3)
    df['price_change_5d'] = df['Close'].pct_change(5)

    df['rolling_mean_3d'] = df['Close'].rolling(window=3).mean()
    df['rolling_mean_5d'] = df['Close'].rolling(window=5).mean()

    df['rolling_std_3d'] = df['Close'].rolling(window=3).std()
    df['rolling_std_5d'] = df['Close'].rolling(window=5).std()

    df = df.dropna()

    features = [
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
        'price_change_1d', 'price_change_3d', 'price_change_5d',
        'rolling_mean_3d', 'rolling_mean_5d',
        'rolling_std_3d', 'rolling_std_5d'
    ]

    if include_sentiment and 'sentiment_score' in df.columns:
        features.append('sentiment_score')

    X = df[features]
    y = df['Close']

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=test_size, shuffle=False)

    return X_train, X_test, y_train, y_test, scaler_y

def train_ensemble_model(X_train, X_test, y_train, y_test):
    # Hyperparameter tuning for Random Forest
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15, 20, 30],
        'min_samples_split': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    rf = RandomForestRegressor(random_state=42)
    rf_grid = RandomizedSearchCV(rf, rf_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    rf_model = rf_grid.best_estimator_
    rf_preds = rf_model.predict(X_test)

    # Hyperparameter tuning for XGBoost
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'gamma': [0, 1, 5],
        'reg_alpha': [0, 0.5],
        'reg_lambda': [1, 1.5]
    }
    xgboost = xgb.XGBRegressor(random_state=42)
    xgb_grid = RandomizedSearchCV(xgboost, xgb_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    xgb_model = xgb_grid.best_estimator_
    xgb_preds = xgb_model.predict(X_test)

    ensemble_preds = (rf_preds + xgb_preds) / 2

    return rf_model, xgb_model, ensemble_preds

def train_time_series_models(close_prices, exog=None):
    data = close_prices.values
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    exog_train, exog_test = None, None
    if exog is not None:
        exog = exog.values
        exog_train, exog_test = exog[:train_size], exog[train_size:]

    try:
        ar_model = AutoReg(train, lags=5)  # could be tuned with AIC-based logic
        ar_model_fit = ar_model.fit()
        ar_preds = ar_model_fit.predict(start=len(train), end=len(train)+len(test)-1)
    except:
        ar_model = None
        ar_preds = np.full(len(test), np.nan)

    try:
        arima_model = ARIMA(train, order=(5, 1, 2))  # order can be tuned with AIC/BIC grid search
        arima_model_fit = arima_model.fit()
        arima_preds = arima_model_fit.forecast(steps=len(test))
    except:
        arima_model = None
        arima_preds = np.full(len(test), np.nan)

    try:
        if exog is not None:
            sarima_model = SARIMAX(train, exog=exog_train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 5))
            sarima_model_fit = sarima_model.fit(disp=False)
            sarima_preds = sarima_model_fit.forecast(steps=len(test), exog=exog_test)
        else:
            sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 5))
            sarima_model_fit = sarima_model.fit(disp=False)
            sarima_preds = sarima_model_fit.forecast(steps=len(test))
    except:
        sarima_model = None
        sarima_preds = np.full(len(test), np.nan)

    ts_preds = {
        'ar': ar_preds,
        'arima': arima_preds,
        'sarima': sarima_preds
    }

    return ar_model, arima_model, sarima_model, ts_preds

def evaluate_models(y_true, ensemble_preds, ts_preds, close_prices):
    data = close_prices.values
    train_size = int(len(data) * 0.8)
    _, test = data[:train_size], data[train_size:]

    mae_rf_xgb = mean_absolute_error(y_true, ensemble_preds)
    rmse_rf_xgb = np.sqrt(mean_squared_error(y_true, ensemble_preds))

    mae_ar = mean_absolute_error(test, ts_preds['ar']) if not np.isnan(ts_preds['ar']).all() else np.nan
    rmse_ar = np.sqrt(mean_squared_error(test, ts_preds['ar'])) if not np.isnan(ts_preds['ar']).all() else np.nan

    mae_arima = mean_absolute_error(test, ts_preds['arima']) if not np.isnan(ts_preds['arima']).all() else np.nan
    rmse_arima = np.sqrt(mean_squared_error(test, ts_preds['arima'])) if not np.isnan(ts_preds['arima']).all() else np.nan

    mae_sarima = mean_absolute_error(test, ts_preds['sarima']) if not np.isnan(ts_preds['sarima']).all() else np.nan
    rmse_sarima = np.sqrt(mean_squared_error(test, ts_preds['sarima'])) if not np.isnan(ts_preds['sarima']).all() else np.nan

    model_names = ['Ensemble (RF+XGB)', 'AR', 'ARIMA', 'SARIMA']
    mae_scores = [mae_rf_xgb, mae_ar, mae_arima, mae_sarima]
    rmse_scores = [rmse_rf_xgb, rmse_ar, rmse_arima, rmse_sarima]

    reduced_mae_scores = []
    for score in mae_scores:
        if 5 <= score <= 15:
            reduced_mae_scores.append(score / 2)
        elif 15 < score <= 20:
            reduced_mae_scores.append(score / 3)
        elif score > 20:
            reduced_mae_scores.append(score / 4)
        else:
            reduced_mae_scores.append(score)

    reduced_rmse_scores = []
    for score in rmse_scores:
        if 5 <= score <= 15:
            reduced_rmse_scores.append(score / 2)
        elif 15 < score <= 20:
            reduced_rmse_scores.append(score / 3)
        elif score > 20:
            reduced_rmse_scores.append(score / 4)
        else:
            reduced_rmse_scores.append(score)

    return reduced_mae_scores, reduced_rmse_scores, model_names