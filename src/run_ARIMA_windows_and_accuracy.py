# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 09:15:02 2024

@author: Pierce Mullin & Yifan Zhang
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def run_arima_windows_univariate(data_train, data_forecast_actuals, p=0, d=0, q=0, trend=None, horizon=30):
    """
    Function to run ARIMA model on a univariate time series data window.
    If p = d = q = 0, forecasts are based on a random walk without drift.

    Args:
    - data_train: 2D numpy array, shape (input_window, n_windows) for the training data windows.
    - data_forecast_actuals: 2D numpy array, shape (forecast_window, n_windows) for the actual future data.
    - p, d, q: ARIMA model order parameters.
    - trend: Trend parameter for ARIMA model. For a random walk with drift, use 'c' (constant).
    - horizon: Forecast horizon, i.e., number of steps to forecast.

    Returns:
    - forecast_matrix: 2D numpy array with forecasts, shape (horizon, n_windows).
    - model_accuracy_table: DataFrame with accuracy metrics for each horizon step.
    """
    n_windows = data_train.shape[0]
    
    # Initialize forecast and error matrices
    forecast_matrix = np.full((horizon, n_windows), np.nan)
    error_matrix = np.full((horizon, n_windows), np.nan)
    
    if p == 0 and d == 0 and q == 0 and trend == None:
        # If p=d=q=0, we have a random walk (no drift).
        # Simply predict the last value in each training window for all forecast steps.
        forecast_matrix[:] = 0  # Set all forecast steps to 0 growth in RW no Drift
    else:
        # Run ARIMA model for each window
        for w in range(n_windows):
            model = ARIMA(data_train[w, :], order=(p, d, q), trend=trend)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=horizon)
            forecast_matrix[:, w] = forecast
            if w % 20 == 0:
                print(f"Window: {w} / {n_windows}")

    # Compute errors
    data_forecast_actuals_transpose = data_forecast_actuals.T
    error_matrix = forecast_matrix - data_forecast_actuals_transpose
    


    # Calculate accuracy metrics
    accuracy_matrix = np.zeros((5, horizon))

    for h in range(horizon):
        # Errors at step h
        horizon_errors = error_matrix[h, :]
        actuals_at_horizon = data_forecast_actuals_transpose[h, :]
        
        # Compute accuracy metrics for the unnormalized errors
        accuracy_matrix[0, h] = np.sqrt(np.mean(horizon_errors**2))  # RMSE
        accuracy_matrix[1, h] = np.mean(horizon_errors**2)  # MSE
        accuracy_matrix[2, h] = np.mean(horizon_errors)  # ME
        accuracy_matrix[3, h] = np.mean(np.abs(horizon_errors / actuals_at_horizon))  # MAPE
        accuracy_matrix[4, h] = np.mean(horizon_errors / actuals_at_horizon)  # MPE

    # Create DataFrame for accuracy table
    model_accuracy_table = pd.DataFrame(
        accuracy_matrix.T, columns=["RMSE", "MSE", "ME", "MAPE", "MPE"]
    )

    return forecast_matrix, model_accuracy_table

def run_arima_windows_univariate_forceRW(data_train, data_forecast_actuals, p=0, d=0, q=0, trend=None, horizon=30):
    """
    Function to run ARIMA model on a univariate time series data window.
    If p = d = q = 0, forecasts are based on a random walk without drift.

    Args:
    - data_train: 2D numpy array, shape (input_window, n_windows) for the training data windows.
    - data_forecast_actuals: 2D numpy array, shape (forecast_window, n_windows) for the actual future data.
    - p, d, q: ARIMA model order parameters.
    - trend: Trend parameter for ARIMA model. For a random walk with drift, use 'c' (constant).
    - horizon: Forecast horizon, i.e., number of steps to forecast.

    Returns:
    - forecast_matrix: 2D numpy array with forecasts, shape (horizon, n_windows).
    - model_accuracy_table: DataFrame with accuracy metrics for each horizon step.
    """
    n_windows = data_train.shape[0]
    
    # Initialize forecast and error matrices
    forecast_matrix = np.full((horizon, n_windows), np.nan)
    error_matrix = np.full((horizon, n_windows), np.nan)
    
    
    # Run ARIMA model for each window
    for w in range(n_windows):
        model = ARIMA(data_train[w, :], order=(p, d, q), trend=trend)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=horizon)
        forecast_matrix[:, w] = forecast
        if w % 20 == 0:
            print(f"Window: {w} / {n_windows}")

    # Compute errors
    data_forecast_actuals_transpose = data_forecast_actuals.T
    error_matrix = forecast_matrix - data_forecast_actuals_transpose
    


    # Calculate accuracy metrics
    accuracy_matrix = np.zeros((5, horizon))

    for h in range(horizon):
        # Errors at step h
        horizon_errors = error_matrix[h, :]
        actuals_at_horizon = data_forecast_actuals_transpose[h, :]
        
        # Compute accuracy metrics for the unnormalized errors
        accuracy_matrix[0, h] = np.sqrt(np.mean(horizon_errors**2))  # RMSE
        accuracy_matrix[1, h] = np.mean(horizon_errors**2)  # MSE
        accuracy_matrix[2, h] = np.mean(horizon_errors)  # ME
        accuracy_matrix[3, h] = np.mean(np.abs(horizon_errors / actuals_at_horizon))  # MAPE
        accuracy_matrix[4, h] = np.mean(horizon_errors / actuals_at_horizon)  # MPE

    # Create DataFrame for accuracy table
    model_accuracy_table = pd.DataFrame(
        accuracy_matrix.T, columns=["RMSE", "MSE", "ME", "MAPE", "MPE"]
    )

    return forecast_matrix, model_accuracy_table


def create_basic_accuracy_table(predictions, actuals):
    
    forecast_horizon = predictions.shape[1]
    accuracy_matrix = np.full((forecast_horizon, 6), np.nan)
    
    horizon_errors = predictions[:,:] - actuals[:,:]

    for h in range(forecast_horizon):

        # Compute accuracy metrics
        accuracy_matrix[h,0] = np.sqrt(np.mean(horizon_errors[:,h]**2))  # RMSE
        accuracy_matrix[h,1] = np.mean(horizon_errors[:,h]**2)  # MSE
        accuracy_matrix[h,2] = np.mean(horizon_errors[:,h])  # ME
        accuracy_matrix[h,3] = np.mean(np.abs(horizon_errors[:,h] / actuals[:,h]))  # MAPE
        accuracy_matrix[h,4] = np.mean(horizon_errors[:,h] / actuals[:,h])  # MPE
        
        ss_res = np.sum(horizon_errors[:,h]**2)  # Residual sum of squares
        ss_tot = np.sum((actuals[:,h] - np.mean(actuals[:,h]))**2)  # Total sum of squares
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0  # Handle edge case where ss_tot = 0
        accuracy_matrix[h, 5] = r_squared
    
    
    accuracy_table = pd.DataFrame(accuracy_matrix,
                                  columns=["RMSE", "MSE", "ME", "MAPE", "MPE", "R2"]) 

    return accuracy_table

def calculate_R2(predictions, actuals):
    
    forecast_horizon = predictions.shape[1]
    R2_matrix = np.full((forecast_horizon, 1), np.nan)
    
    #horizon_errors = predictions[:,:] - actuals[:,:]

    for h in range(forecast_horizon):

        # Compute R2 metrics     
        #ss_res = np.sum(horizon_errors[:,h]**2)  # Residual sum of squares
        #ss_tot = np.sum((actuals[:,h] - np.mean(actuals[:,h]))**2)  # Total sum of squares
        #r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0  # Handle edge case where ss_tot = 0
        
        corr = np.corrcoef(predictions[:,h],actuals[:,h])
        r_squared_temp = corr ** 2
        r_squared = r_squared_temp[0,1]
        R2_matrix[h, 0] = r_squared
    
        #print(r_squared_train)
        
    R2_table = pd.DataFrame(R2_matrix,
                                  columns=["R2"]) 

    return R2_table

def add_directional_accuracy(accuracy_table, inputs_actuals, predictions, targets):
    
    forecast_horizon = accuracy_table.shape[0]
    
    directional_accuracy_horizons = np.full((forecast_horizon,1),np.nan)

    for h in range(forecast_horizon):
        
        if h == 0:
            actual_one_period_change_h = targets[:,h] - inputs_actuals[:,(inputs_actuals.shape[1]-1),3]
            actual_one_period_direction_h = np.sign(actual_one_period_change_h)

            predictions_one_period_change_h = predictions[:,h] - inputs_actuals[:,(inputs_actuals.shape[1]-1),3]
            predictions_one_period_direction_h = np.sign(predictions_one_period_change_h)

            predictions_one_period_direction_match_h = predictions_one_period_direction_h == actual_one_period_direction_h
            predictions_one_period_direction_match_accuracy_h = (np.sum(predictions_one_period_direction_match_h,axis=0)/predictions_one_period_direction_match_h.shape[0])
        
            directional_accuracy_horizons[h,0] = predictions_one_period_direction_match_accuracy_h
            
        if h > 0:
            actual_one_period_change_h = targets[:,h] - targets[:,h-1]
            actual_one_period_direction_h = np.sign(actual_one_period_change_h)

            predictions_one_period_change_h = predictions[:,h] - predictions[:,h-1]
            predictions_one_period_direction_h = np.sign(predictions_one_period_change_h)

            predictions_one_period_direction_match_h = predictions_one_period_direction_h == actual_one_period_direction_h
            predictions_one_period_direction_match_accuracy_h = (np.sum(predictions_one_period_direction_match_h,axis=0)/predictions_one_period_direction_match_h.shape[0])
        
            directional_accuracy_horizons[h,0] = predictions_one_period_direction_match_accuracy_h
    
    directional_accuracy_horizons_output = directional_accuracy_horizons.flatten()
    accuracy_table["Direct. Acc."] = directional_accuracy_horizons_output
    
    return accuracy_table

