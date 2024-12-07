# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 09:15:02 2024

@author: Pierce Mullin & Yifan Zhang
"""
import pandas as pd


def load_and_clean_BTCUSDT_data(data_path, aggregation_frequency = '30T', data_clip = 0.2):
    """
    load, process and clean the data used in project model (BTCUSDT prices)
    
    Args:
        data_path: The path to the CSV datafile of 1-minute timestamp BTCUSDT prices
        aggregation_frequency: How much to aggregate the 1-minute data (due to computational / memory constraints). '30T' means 30-minute aggregations
        data_clip: The level of outlier handling used on the aggregated data (setting a max/min on periodic % returns)
    
    Returns:
        aggregated: The aggregated key data series
        aggregated_returns: The % returns format of aggregated to use in model training
    
    """

    data = pd.read_csv(data_path, dtype=str)

    #print("Column names:", data.columns)
    features = ['high', 'low', 'open', 'close', 'volume']

    df = data

    # Convert 'timestamp' to datetime and round any fractional seconds
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df['timestamp'] = df['timestamp'].dt.round('1s')
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    df = df.reset_index(drop=True)

    # Ensure key data columns are numeric
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create Dollar-Weighted Volume
    df["dollar_volume"] = df["volume"]*df["close"]
    #aggregated["dollar_volume"] = aggregated["volume"]*average_prices
    periods_per_30_days = 1*60*24*30
    df['30_day_dollar_ADTV'] = df['dollar_volume'].rolling(window=periods_per_30_days, min_periods=1).mean() 
    df['normalized_dollar_volume'] = df['dollar_volume'] / df['30_day_dollar_ADTV']

    # Create a column for the 30-minute groupings based on the timestamp floor
    df['time_group'] = df['timestamp'].dt.floor(aggregation_frequency)

    # Perform the aggregation within each 30-minute group
    aggregated = df.groupby('time_group').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'normalized_dollar_volume': 'mean'
        }).reset_index()

    # Calculate percentage returns for each column in `features`
    for col in features:
        aggregated[f'{col}_return'] = aggregated[col].pct_change()

    # Drop rows with NaN values resulting from percentage change calculation
    aggregated = aggregated.dropna().reset_index(drop=True)

    # Clip the returns at ±10% for each feature
    for col in features:
        aggregated[f'{col}_return'] = aggregated[f'{col}_return'].clip(lower=-data_clip, upper=data_clip)


    # Select only the return columns in the order of `features`
    return_features = [f'{col}_return' for col in features]
    features = ['high_return', 'low_return', 'open_return', 'close_return']
    aggregated_returns = aggregated[return_features]

    #print("Aggregated Data with Returns:")
    #print(aggregated)

    # Optional: View the final returns-only DataFrame if you need it
    #print("Aggregated Returns-Only DataFrame:")
    #print(aggregated_returns)

    return aggregated, aggregated_returns

