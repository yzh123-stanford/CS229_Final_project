# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 09:15:02 2024

@author: Pierce Mullin & Yifan Zhang
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot_input_data(btc_close, train_size, valid_size, test_size, input_window_length, forecast_horizon):
    
    btc_length = len(btc_close)
    combined_size = train_size + valid_size + test_size
    difference = btc_length - combined_size 
    # the difference is because the first 960 btc prices form the columns of the first sequence in X_train
    # the rest of the difference is from the forecast_horizon
    total_size = combined_size + input_window_length 
    train_size_adj = train_size + input_window_length
    
    if forecast_horizon > 1:
        btc_close_plot = btc_close[:-(forecast_horizon - 1)]
    else:
        btc_close_plot = btc_close
        
    train_color = mcolors.to_rgba('darkblue')  
    valid_color = mcolors.to_rgba('lightblue')  
    test_color = mcolors.to_rgba('turquoise')  
    
    plt.figure(figsize=(12, 8))
    plt.plot(range(train_size_adj), btc_close_plot[:train_size_adj], label="Train", color=train_color)
    plt.plot(range(train_size_adj, train_size_adj + valid_size), btc_close_plot[train_size_adj:(train_size_adj + valid_size)], label="Validation", color=valid_color)
    plt.plot(range(train_size_adj + valid_size, total_size), btc_close_plot[train_size_adj + valid_size:], label="Test", color=test_color)
    
    plt.xlabel("Time Period")
    plt.ylabel("Price ($)")
    plt.title("BTCUSDT Price Over Time")
    plt.legend()
    plt.grid(False)
    
    plt.show()
    
    
    
    

def plot_predictions_at_index(data_actuals_full, predictions, actuals, plot_index, history_zoom = 960):
    # Parameters for Chart
    #history_zoom = Number of periods to zoom in on historical data
    #plot_index = Index we want to see how the model performed on its predictions
    
    # Define historical and forecast ranges
    history_length = data_actuals_full.shape[1]  # Length of the actual historical data (number of time periods in full data)
    forecast_length = predictions.shape[1]  # Length of the forecast (from predictions)
    
    # Generate the zoomed in x-axis ranges
    X_actual = np.arange(history_length - history_zoom, history_length)  # Last N points of actual data
    X_forecast = np.arange(history_length, history_length + forecast_length)  
    
    # Select the data for plotting
    historical_data = data_actuals_full[plot_index, -history_zoom:, 3]  # Last `history_zoom` points of historical data
    forecast_data = predictions[plot_index, :]  # Forecasted data
    actual_future_data = actuals[plot_index, :]  # Actual future data for comparison
    
    plt.figure(figsize=(12, 6))
    plt.plot(X_actual, historical_data, label="Actual Historical", color="blue")
    plt.plot(X_forecast, forecast_data, label="Prediction (LSTM, in levels)", color="orange")
    plt.plot(X_forecast, actual_future_data, label="Actual Future", color="green")
    
    plt.xlabel("Time Period")
    plt.ylabel("Price ($)")
    plt.title(f"BTCUSDT Price Forecast in Levels (Sequence: {plot_index})")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

