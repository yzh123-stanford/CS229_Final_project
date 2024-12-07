# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 09:15:02 2024

@author: Pierce Mullin & Yifan Zhang
"""
import numpy as np

def create_windowed_sequences(data, window_size, forecast_horizon=5):
    """
    Create a set of sequences and target actuals for training and prediction
    
    Args:
        data: Multivariate aggregated data series (OHLC + normalized_dollar_volume)
        window_size: Length of sequence used in each observation
        forecast_horizon: Length of sequence to try and predict (i.e., actuals following input sequence)
    
    Returns:
        sequences_np: dataset of sequences, dimension (N, window_size, 5)
        targets_np: dataset of sequences, dimension (N, forecast_horizon, 5)
    
    """
    sequences = []
    targets = []
    data_array = data[['open_return', 'high_return', 'low_return', 'close_return','normalized_dollar_volume']].values
    for i in range(len(data_array) - window_size - forecast_horizon + 1):
        sequence = data_array[i:i+window_size]
        target = data_array[i + window_size:i + window_size + forecast_horizon]
        sequences.append(sequence)
        targets.append(target)
    sequences_np = np.array(sequences)
    targets_np = np.array(targets)
    return sequences_np, targets_np

def create_windowed_sequences_levels(data, window_size, forecast_horizon=5):
    """
    Create a set of sequences and target actuals for training and prediction
    
    Args:
        data: Multivariate aggregated data series (OHLC + normalized_dollar_volume)
        window_size: Length of sequence used in each observation
        forecast_horizon: Length of sequence to try and predict (i.e., actuals following input sequence)
    
    Returns:
        sequences_np: dataset of sequences, dimension (N, window_size, 5)
        targets_np: dataset of sequences, dimension (N, forecast_horizon, 5)
    
    """
    sequences = []
    targets = []
    data_array = data[['open', 'high', 'low', 'close','normalized_dollar_volume']].values
    for i in range(len(data_array) - window_size - forecast_horizon + 1):
        sequence = data_array[i:i+window_size]
        target = data_array[i + window_size:i + window_size + forecast_horizon]
        sequences.append(sequence)
        targets.append(target)
    sequences_np = np.array(sequences)
    targets_np = np.array(targets)
    return sequences_np, targets_np

