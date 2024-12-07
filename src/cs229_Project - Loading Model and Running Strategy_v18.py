# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 09:15:02 2024

@author: Pierce Mullin & Yifan Zhang
"""

#### Load Packages ####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim  
import random
from statsmodels.tsa.arima.model import ARIMA
import os
import json

#### Load Internal Files ####

from src.util.load_and_clean_BTCUSDT import load_and_clean_BTCUSDT_data
from src.util.create_windowed_series_and_targets import create_windowed_sequences
from src.util.create_windowed_series_and_targets import create_windowed_sequences_levels
from src.util.run_ARIMA_windows_and_accuracy import run_arima_windows_univariate
from src.util.run_ARIMA_windows_and_accuracy import run_arima_windows_univariate_forceRW
from src.util.run_ARIMA_windows_and_accuracy import create_basic_accuracy_table
from src.util.run_ARIMA_windows_and_accuracy import calculate_R2
from src.util.run_ARIMA_windows_and_accuracy import add_directional_accuracy
from src.util.model_hyperparameter_evaluation import model_evaluation_table
from src.util.plot_predictions import plot_predictions_at_index
from src.util.plot_predictions import plot_input_data
from src.util.run_trading_strategy import run_strategy_long_only
from src.util.run_trading_strategy import run_strategy_long_only_V2
from src.util.run_trading_strategy import run_strategy_long_short
from src.util.run_trading_strategy import run_strategy_long_short_V2
from src.util.run_trading_strategy import calculate_annual_sharpe_ratio
from src.util.run_trading_strategy import calculate_annual_sharpe_ratio_V2


#### Set Up Environment (if using cuda) ####
random.seed(229)
using_cuda = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if using_cuda == True:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    torch.cuda.empty_cache()

#### Loading and Processing Data ####
data_path = "data/BTCUSDT_data.csv" 
aggregation_frequency = '30T'
aggregated, aggregated_returns = load_and_clean_BTCUSDT_data(data_path = data_path,
                                                             aggregation_frequency = '30T',
                                                             data_clip = 0.1)

# Re-scale normalized_dollar_volume
print(np.mean(aggregated_returns,axis=0))
volume_scale_factor = np.mean(aggregated_returns["volume_return"]) / np.mean(aggregated_returns["close_return"])
aggregated_returns["volume_return"] = aggregated_returns["volume_return"] / np.abs(volume_scale_factor)
print(np.mean(aggregated_returns,axis=0))

#### Plot and view historical BTC prices ####
btc_close = aggregated['close']
plt.figure(figsize=(10, 6))
plt.plot(btc_close, label="BTCUSDT", color="blue")
plt.xlabel("Time Period")
plt.ylabel("$")
plt.title("BTCUSDT Price Over Time")
plt.legend()
plt.grid(False)
plt.show()

#### Processing the Data into a set of Sequences and Targets for LSTM model training and prediction ####
window_size = 2*24*20 # per-hour*hours-per-day*days
#forecast_horizon = 2*12*1 # 12 hours
forecast_horizon = 1 # single 30-min ahead


variables = ['open_return', 'high_return', 'low_return', 'close_return','normalized_dollar_volume']
variables_levels = ['open', 'high', 'low', 'close','normalized_dollar_volume']
df_model = aggregated[variables]
df_model_levels = aggregated[variables_levels]
print("Getting X, y")
X, y = create_windowed_sequences(df_model, window_size, forecast_horizon = forecast_horizon)
print("Getting X_prices, y_prices")
X_prices, y_prices = create_windowed_sequences_levels(df_model_levels, window_size, forecast_horizon = forecast_horizon)


#### Split Data into Train, Valid and Test sets ####
# Calculate exact split sizes
train_ratio, valid_ratio, test_ratio = 0.7, 0.15, 0.15
total_size = len(X)

# First, calculate training and validation sizes
train_size = int(total_size * train_ratio)
valid_size = int(total_size * valid_ratio)
test_size = total_size - (train_size + valid_size)

# Apply the split
X_train, y_train = X[:train_size], y[:train_size]
X_valid, y_valid = X[train_size:train_size + valid_size], y[train_size:train_size + valid_size]
X_test, y_test = X[train_size + valid_size:], y[train_size + valid_size:]

X_train_prices, y_train_prices = X_prices[:train_size], y_prices[:train_size]
X_valid_prices, y_valid_prices = X_prices[train_size:train_size + valid_size], y_prices[train_size:train_size + valid_size]
X_test_prices, y_test_prices = X_prices[train_size + valid_size:], y_prices[train_size + valid_size:]

combined_length = len(X_train) + len(X_valid) + len(X_test)

# Plot BTCUSDT Closing Price Over the Different Train/Valid/Test Periods
plot_input_data(btc_close, 
                train_size, 
                valid_size, 
                test_size,
                input_window_length = X_test_prices.shape[1],
                forecast_horizon = forecast_horizon,
                width = 12,
                height = 4)


#### Calculating the Benchmark for Model Accuracy ####
# Note: Using a Random Walk on closing prices (i.e., E(P_t+1|P_t) = P_t)
#       is grounded in the efficient market hypothesis (EMH) and provides a simple,
#       sensible benchmark to use in terms of comparison
# Note: Strong Form EMH would imply that forecast period growth rates are all 0%,
#       so we want our model's predicted growth rates to be closer to the target growth rates
#       than 0% would be

forecast_matrix_RW_train, model_accuracy_table_RW_train = run_arima_windows_univariate(X_train[:,:,3],
                                                                              y_train[:,:,3],
                                                                              p=0, d=0, q=0,
                                                                              trend = None,
                                                                              horizon=forecast_horizon)

forecast_matrix_RW_valid, model_accuracy_table_RW_valid = run_arima_windows_univariate(X_valid[:,:,3],
                                                                              y_valid[:,:,3],
                                                                              p=0, d=0, q=0,
                                                                              trend = None,
                                                                              horizon=forecast_horizon)

forecast_matrix_RW_test, model_accuracy_table_RW_test = run_arima_windows_univariate(X_test[:,:,3],
                                                                              y_test[:,:,3],
                                                                              p=0, d=0, q=0,
                                                                              trend = None,
                                                                              horizon=forecast_horizon)



# Saving Down Results
model_accuracy_table_RW_test.to_csv('results/model_accuracy_table_RW_test.csv', index=False)
np.save('results/forecast_matrix_RW_test.npy', forecast_matrix_RW_test)

# Loading Results
model_accuracy_table_RW_test = pd.read_csv('results/model_accuracy_table_RW_test.csv', na_values=["inf", "NaN"])
forecast_matrix_RW_test = np.load('results/forecast_matrix_RW_test.npy')
print(model_accuracy_table_RW_test)
print(forecast_matrix_RW_test)

#### Converting Data into Tensors for Model Training and Prediction ####
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_valid = torch.tensor(X_valid, dtype=torch.float32).to(device)
y_valid = torch.tensor(y_valid, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# =============================================================================
# 
# #### IMPORTANT NOTE: TRAINING WILL BE DONE IN A SEPARATE FILE AND WAS DONE IN GOOGLE CLOUD 
# 
# =============================================================================

#### Defining the LSTM ####

input_dim = 5
output_dim = (forecast_horizon*input_dim)

# Define the LSTM models in PyTorch
class SimpleLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(SimpleLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        x = self.fc(lstm_out[:, -1, :])
        return x.view(-1, forecast_horizon, input_dim)

# Define Complex LSTM structure
class ComplexLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(ComplexLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        x_fc = self.fc1(lstm_out[:, -1, :])
        x_fc = torch.relu(x_fc)
        x_fc = self.fc2(x_fc)
        return x_fc.view(-1, forecast_horizon, input_dim)

#### Loading Models and Model Results from Hyperparameter Selection ####

# Load Hyper Parameter Results
# Note: All models saved with similar naming conventions to loop across alternatives
results_dir = "results/"
model_suffix_template = "best_model_state_BTCUSDT_{loss_type}_{lstm_type}_{model_set}_{i}.pth"
metadata_suffix_template = "results_metadata_{loss_type}_{lstm_type}_{model_set}.json"
model_sets = ["original", "additional"]
loss_types = ["custom_loss", "smoothL1_loss"]
lstm_types = ["simple_LSTM", "complex_LSTM"]

# Reload models
loaded_results = []

for loss_type in loss_types:
    for lstm_type in lstm_types:
        for model_set in model_sets:

            # Load metadata
            metadata_file = os.path.join(results_dir, metadata_suffix_template.format(loss_type=loss_type, lstm_type=lstm_type, model_set=model_set))
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
    
                # Load models corresponding to the metadata
                for i, meta in enumerate(metadata):
                    model_file = os.path.join(results_dir, model_suffix_template.format(loss_type=loss_type, lstm_type=lstm_type, model_set=model_set, i=i))
                    if os.path.exists(model_file):
                        if torch.cuda.is_available():
                            model_state = torch.load(model_file)
                        else:
                            model_state = torch.load(model_file, map_location=torch.device('cpu'))
    
                            result = {
                                "hidden_dim": meta["hidden_dim"],
                                "num_layers": meta["num_layers"],
                                "batch_size": meta["batch_size"],
                                "dropout_rate": meta["dropout_rate"],
                                "best_val_loss": meta["best_val_loss"],
                                "best_val_loss_close": meta["best_val_loss_close"],
                                "train_losses": meta["train_losses"],
                                "val_losses": meta["val_losses"],
                                "train_losses_close": meta["train_losses_close"],
                                "val_losses_close": meta["val_losses_close"],
                                "best_model_state": model_state,
                                "loss_type": loss_type,
                                "lstm_type": lstm_type
                            }
                            
                            if loss_type == "custom_loss":
                                result.update({
                                    "gamma": meta.get("gamma"),
                                    "directional_lambda": meta.get("directional_lambda"),
                                    "closing_lambda": meta.get("closing_lambda")
                                })
        
                            loaded_results.append(result)

    
#### Plotting Accuracy of Models Across Epochs

ylim_max = 0.0003

for alt in range(len(loaded_results)):  # Loop over the hyperparameter alternatives
    result_alt = loaded_results[alt]
    train_losses_close = result_alt["train_losses_close"]

    val_losses_close = result_alt["val_losses_close"]
    
    # Create RW Benchmark Line for Validation Loss
    RW_benchmark_MSE_valid = model_accuracy_table_RW_valid['MSE'][0]  # Constant RW Benchmark (1-step ahead)
    RW_benchmark_MSE_line = np.full(len(train_losses_close), RW_benchmark_MSE_valid)  # Benchmark line

    # Plot losses and benchmark
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses_close, label='Training Loss (Close)', color='blue')
    plt.plot(val_losses_close, label='Validation Loss (Close)', color='orange')
    plt.plot(RW_benchmark_MSE_line, label='RW Benchmark', color='darkred', linestyle='--')
    
    # Formatting
    plt.title(f"Training and Validation Loss for Hyperparameter Set {alt+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Smooth L1 Loss")
    plt.legend()
    plt.ylim([0, ylim_max])  
    plt.grid(False)
    plt.show()


# Look at All Models in Hyperparameter Evaluation Table

evaluation_table = model_evaluation_table(loaded_results)

print(evaluation_table)

# Take the best model based on Validation Loss on closing price
best_result = min(loaded_results, key=lambda x: x["best_val_loss_close"])
#best_result = min(loaded_results, key=lambda x: x["best_val_loss"])

if best_result["lstm_type"] == "simple_LSTM":
    LSTMModel = SimpleLSTMModel
elif best_result["lstm_type"] == "complex_LSTM":
    LSTMModel = ComplexLSTMModel


best_model = LSTMModel(
    input_dim=input_dim,
    hidden_dim=best_result["hidden_dim"],
    output_dim=output_dim,
    num_layers=best_result["num_layers"],
    dropout_rate=best_result["dropout_rate"]
).to(device)

# Load the best model state
best_model.load_state_dict(best_result["best_model_state"])

if best_result["loss_type"] == "custom_loss":
    print(f"Loaded the best model with Hidden Dim: {best_result['hidden_dim']}, "
          f"Num Layers: {best_result['num_layers']}, "
          f"Batch Size: {best_result['batch_size']}, "
          f"Validation Loss (Close): {best_result['best_val_loss_close']:.10f}, "
          f"Gamma: {best_result['gamma']}, "
          f"Directional Lambda: {best_result['directional_lambda']}, "
          f"Closing Price Lambda: {best_result['closing_lambda']}")
    
elif best_result["loss_type"] == "smoothL1_loss":
    print(f"Loaded the best model with Hidden Dim: {best_result['hidden_dim']}, "
          f"Num Layers: {best_result['num_layers']}, "
          f"Batch Size: {best_result['batch_size']}, "
          f"Validation Loss (Close): {best_result['best_val_loss_close']:.10f}")

model = best_model

ylim_max = 0.0003

result_alt = loaded_results[13]
train_losses_close = result_alt["train_losses_close"]

val_losses_close = result_alt["val_losses_close"]

# Create RW Benchmark Line for Validation Loss
RW_benchmark_MSE_valid = model_accuracy_table_RW_valid['MSE'][0]  # Constant RW Benchmark (1-step ahead)
RW_benchmark_MSE_line = np.full(len(train_losses_close), RW_benchmark_MSE_valid)  # Benchmark line

# Plot losses and benchmark
plt.figure(figsize=(6, 6))
plt.plot(train_losses_close, label='Training Loss (Close)', color='blue')
plt.plot(val_losses_close, label='Validation Loss (Close)', color='orange')
plt.plot(RW_benchmark_MSE_line, label='RW Benchmark', color='darkred', linestyle='--')

# Formatting
plt.title(f"Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Smooth L1 Loss")
plt.legend()
plt.ylim([0, ylim_max])  
plt.grid(False)
plt.show()


#### Make Predictions Using Model ####

batch_size_predictions = 50 # Note: doing this in batches for memory purposes

# Function to make predictions in batches
def predict_in_batches(model, data, batch_size):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size].to(device)
            batch_predictions = model(batch_data).cpu().numpy()
            predictions.append(batch_predictions)
            if i % 1000 == 0:
                print(f"Iteration: {i} / {len(data)}")
    return np.concatenate(predictions, axis=0)

# Make predictions in batches
train_predictions = predict_in_batches(model, X_train, batch_size_predictions)
valid_predictions = predict_in_batches(model, X_valid, batch_size_predictions)
test_predictions = predict_in_batches(model, X_test, batch_size_predictions)

# Focus only on 'close' price (index 3)
train_predictions_close_percent = train_predictions[:, :, 3]
valid_predictions_close_percent = valid_predictions[:, :, 3]
test_predictions_close_percent = test_predictions[:, :, 3]

train_predictions_close_prices = np.full((train_predictions_close_percent.shape[0],forecast_horizon),np.nan)
valid_predictions_close_prices = np.full((valid_predictions_close_percent.shape[0],forecast_horizon),np.nan)
test_predictions_close_prices = np.full((test_predictions_close_percent.shape[0],forecast_horizon),np.nan)


#### Creating Model Accuracy Metrics for Comparison to Random Walk Benchmark
train_actuals_close_percent = y_train[:,:,3]
valid_actuals_close_percent = y_valid[:,:,3]
test_actuals_close_percent = y_test[:,:,3]

train_actuals_close_prices = y_train_prices[:,:,3]
valid_actuals_close_prices = y_valid_prices[:,:,3]
test_actuals_close_prices = y_test_prices[:,:,3]

n_cols = X_train_prices.shape[1]

for h in range(forecast_horizon):
    if h == 0:
        train_predictions_close_prices[:,h] = X_train_prices[:,(n_cols-1),3] * (1 + train_predictions_close_percent[:,h])
        valid_predictions_close_prices[:,h] = X_valid_prices[:,(n_cols-1),3] * (1 + valid_predictions_close_percent[:,h])
        test_predictions_close_prices[:,h] = X_test_prices[:,(n_cols-1),3] * (1 + test_predictions_close_percent[:,h])
    else:
        train_predictions_close_prices[:,h] = train_predictions_close_prices[:,h-1] * (1 + train_predictions_close_percent[:,h]) 
        valid_predictions_close_prices[:,h] = valid_predictions_close_prices[:,h-1] * (1 + valid_predictions_close_percent[:,h])
        test_predictions_close_prices[:,h] = test_predictions_close_prices[:,h-1] * (1 + test_predictions_close_percent[:,h])


RW_test_predictions_close_prices = X_test_prices[:,(X_test_prices.shape[1]-1),3]
RW_test_predictions_close_prices = RW_test_predictions_close_prices[:, np.newaxis] 
RW_test_predictions_close_prices = np.repeat(RW_test_predictions_close_prices, forecast_horizon, axis=1)

# Calculating Primary Accuracy Metrics
model_accuracy_table_train = create_basic_accuracy_table(predictions = train_predictions_close_prices,
                                                              actuals = train_actuals_close_prices)

model_accuracy_table_valid = create_basic_accuracy_table(predictions = valid_predictions_close_prices,
                                                              actuals = valid_actuals_close_prices)

model_accuracy_table_test = create_basic_accuracy_table(predictions = test_predictions_close_prices,
                                                              actuals = test_actuals_close_prices)

model_accuracy_table_test_RW = create_basic_accuracy_table(predictions = RW_test_predictions_close_prices,
                                                              actuals = test_actuals_close_prices)

# Calculatin R2 on % Returns (not meaningful in price space, ~1)

#X_train, y_train = X[:train_size], y[:train_size]
#X_valid, y_valid = X[train_size:train_size + valid_size], y[train_size:train_size + valid_size]
#X_test, y_test = X[train_size + valid_size:], y[train_size + valid_size:]

actuals_percent_train = y[:train_size]
actuals_percent_valid = y[train_size:train_size + valid_size]
actuals_percent_test = y[train_size + valid_size:]


R2_train_percent = calculate_R2(predictions = train_predictions_close_percent,
                                actuals = actuals_percent_train[:,:,3])
print(R2_train_percent)

R2_valid_percent = calculate_R2(predictions = valid_predictions_close_percent,
                                actuals = actuals_percent_valid[:,:,3])
print(R2_valid_percent)

R2_test_percent = calculate_R2(predictions = test_predictions_close_percent,
                                actuals = actuals_percent_test[:,:,3])
print(R2_test_percent)



R2_test_percent_RW = calculate_R2(predictions = forecast_matrix_RW_test.T,
                                actuals = actuals_percent_test[:,:,3])
print(R2_test_percent_RW) # By definition a 0% prediction would have no correlation


# Adding Directional Accuracy
model_accuracy_table_train = add_directional_accuracy(accuracy_table = model_accuracy_table_train,
                                                     inputs_actuals = X_train_prices, 
                                                     predictions = train_predictions_close_prices, 
                                                     targets = train_actuals_close_prices)

model_accuracy_table_valid = add_directional_accuracy(accuracy_table = model_accuracy_table_valid,
                                                     inputs_actuals = X_valid_prices, 
                                                     predictions = valid_predictions_close_prices, 
                                                     targets = valid_actuals_close_prices)

model_accuracy_table_test = add_directional_accuracy(accuracy_table = model_accuracy_table_test,
                                                     inputs_actuals = X_test_prices, 
                                                     predictions = test_predictions_close_prices, 
                                                     targets = test_actuals_close_prices)

model_accuracy_table_test_RW = add_directional_accuracy(accuracy_table = model_accuracy_table_test_RW,
                                                     inputs_actuals = X_test_prices, 
                                                     predictions = RW_test_predictions_close_prices, 
                                                     targets = test_actuals_close_prices)

pd.set_option('display.max_columns', None) 
pd.set_option('display.width', 200)        
pd.set_option('display.max_rows', None)

print(model_accuracy_table_train)
print(model_accuracy_table_valid)
print(model_accuracy_table_test)
print(model_accuracy_table_test_RW)


#### Plot Predictions at Select Periods to See How Model Performed ####

plot_predictions_at_index(data_actuals_full = X_test_prices,
                          predictions = test_predictions_close_prices,
                          actuals = test_actuals_close_prices,
                          plot_index = 1000,
                          history_zoom = 960)

plot_predictions_at_index(data_actuals_full = X_test_prices,
                          predictions = test_predictions_close_prices,
                          actuals = test_actuals_close_prices,
                          plot_index = 2000,
                          history_zoom = 960)

plot_predictions_at_index(data_actuals_full = X_test_prices,
                          predictions = test_predictions_close_prices,
                          actuals = test_actuals_close_prices,
                          plot_index = 3000,
                          history_zoom = 960)

plot_predictions_at_index(data_actuals_full = X_test_prices,
                          predictions = test_predictions_close_prices,
                          actuals = test_actuals_close_prices,
                          plot_index = 4000,
                          history_zoom = 960)

plot_predictions_at_index(data_actuals_full = X_test_prices,
                          predictions = test_predictions_close_prices,
                          actuals = test_actuals_close_prices,
                          plot_index = 4005,
                          history_zoom = 960)

plot_predictions_at_index(data_actuals_full = X_test_prices,
                          predictions = test_predictions_close_prices,
                          actuals = test_actuals_close_prices,
                          plot_index = 4010,
                          history_zoom = 960)

plot_predictions_at_index(data_actuals_full = X_test_prices,
                          predictions = test_predictions_close_prices,
                          actuals = test_actuals_close_prices,
                          plot_index = 4015,
                          history_zoom = 960)

plot_predictions_at_index(data_actuals_full = X_test_prices,
                          predictions = test_predictions_close_prices,
                          actuals = test_actuals_close_prices,
                          plot_index = 4020,
                          history_zoom = 960)

plot_predictions_at_index(data_actuals_full = X_test_prices,
                          predictions = test_predictions_close_prices,
                          actuals = test_actuals_close_prices,
                          plot_index = 4025,
                          history_zoom = 960)


plot_predictions_at_index(data_actuals_full = X_test_prices,
                          predictions = test_predictions_close_prices,
                          actuals = test_actuals_close_prices,
                          plot_index = 5000,
                          history_zoom = 960)

plot_predictions_at_index(data_actuals_full = X_test_prices,
                          predictions = test_predictions_close_prices,
                          actuals = test_actuals_close_prices,
                          plot_index = 6000,
                          history_zoom = 960)

plot_predictions_at_index(data_actuals_full = X_test_prices,
                          predictions = test_predictions_close_prices,
                          actuals = test_actuals_close_prices,
                          plot_index = 7000,
                          history_zoom = 960)




#### Run a Simple Long Only Trading Strategy ####

# consolidating Closing Price Data
predictions_combined = np.concatenate([train_predictions_close_prices,
                                       valid_predictions_close_prices,
                                       test_predictions_close_prices
                                       ], axis=0)
predictions_tol_seek = np.concatenate([train_predictions_close_prices
                                       ], axis=0)

actuals_combined = np.concatenate([X_train_prices[:, :, 3],  
                                   X_valid_prices[:, :, 3],
                                   X_test_prices[:, :, 3]
                                   ], axis=0)
actuals_tol_seek = np.concatenate([X_train_prices[:, :, 3]  
                                   ], axis=0)

data_frequency = 2*24*365 # BTC trades 24/7, so 2 30-min periods per hour x 24 hours x 365 days a year



total_wealth_long_short, buy_and_hold, trading_results_long_short = run_strategy_long_short(predictions = predictions_combined,
                                                                                            actuals = actuals_combined,
                                                                                            starting_time = 960,
                                                                                            starting_money = 1000,
                                                                                            rule_index = 0,
                                                                                            tol = 10,
                                                                                            verbose = True)



percent_tol_alternatives = [0.00,0.0001,0.0002,0.0003,0.00035,0.0004,0.00045,0.0005,0.00055,0.0006,0.0007,0.0008,0.0009,0.001,0.005]

train_valid_long_short_percent_tol_profit = []
train_valid_long_short_percent_tol_mean_return = []
train_valid_long_short_percent_tol_stddev = []

for tol_alt in percent_tol_alternatives:
        
    total_wealth_temp, buy_and_hold_temp, trading_results_temp = run_strategy_long_only_V2(predictions = predictions_tol_seek,
                                                                                             actuals = actuals_tol_seek,
                                                                                             starting_time = 960,
                                                                                             starting_money = 1000,
                                                                                             rule_index = 0,
                                                                                             tol_percent = tol_alt,
                                                                                             verbose = False)
    
    percent_returns_temp = np.diff(total_wealth_temp) / total_wealth_temp[:-1] 
    mean_annual_returns_temp = np.mean(percent_returns_temp) * data_frequency
    stddev_temp = np.std(percent_returns_temp)
    
    train_valid_long_short_percent_tol_profit.append(total_wealth_temp[-1])
    train_valid_long_short_percent_tol_mean_return.append(mean_annual_returns_temp)
    train_valid_long_short_percent_tol_stddev.append(stddev_temp)
    
# Ensure mean return and standard deviation are scalars (not arrays)
train_valid_long_short_percent_tol_mean_return_percent = [
    float(np.mean(mean_return)) * 100 for mean_return in train_valid_long_short_percent_tol_mean_return
]
train_valid_long_short_percent_tol_stddev_percent = [
    float(stddev) * 100 for stddev in train_valid_long_short_percent_tol_stddev
]

train_valid_long_short_sharpe_ratio = [
    (mean_return/100) / stddev if stddev != 0 else float('nan')  # Avoid division by zero
    for mean_return, stddev in zip(
        train_valid_long_short_percent_tol_mean_return_percent,
        train_valid_long_short_percent_tol_stddev_percent
    )
]

# Print results in a clean format
print("Results for Percent Tolerance Alternatives:")
print("-------------------------------------------------------------")
print(f"{'Tolerance':<15}{'Profit ($)':<15}{'Mean Return (%)':<20}{'Std Dev (%)':<15}{'Sharpe Ratio':<15}")
print("-------------------------------------------------------------")

for tol, profit, mean_return, stddev, sharpe in zip(
    percent_tol_alternatives,
    train_valid_long_short_percent_tol_profit,
    train_valid_long_short_percent_tol_mean_return_percent,
    train_valid_long_short_percent_tol_stddev_percent,
    train_valid_long_short_sharpe_ratio
):
    print(f"{tol:<15}{float(profit):<15.2f}{mean_return:<20.2f}{stddev:<15.4f}{sharpe:<15.3f}")


#0.0005
total_wealth_long_short_V2, buy_and_hold_V2, trading_results_long_short_V2 = run_strategy_long_short_V2(predictions = predictions_combined,
                                                                                            actuals = actuals_combined,
                                                                                            starting_time = 960,
                                                                                            starting_money = 1000,
                                                                                            rule_index = 0,
                                                                                            tol_percent = 0.0005,
                                                                                            verbose = True)


# Plot Results
plt.figure(figsize=(12, 6))
#plt.plot(total_wealth_long_only, label="Long Only Strategy", color="lightblue")
#plt.plot(total_wealth_long_only_V2, label="Long Only Strategy (% Tol)", color="lightgreen")
#plt.plot(total_wealth_long_short, label="Long Short Strategy", color="darkblue")
plt.plot(total_wealth_long_short_V2, label="Long Short Strategy", color="darkgreen")
plt.plot(buy_and_hold, label="Buy and Hold", color="orange")
plt.axhline(y=1000, color="darkred", linestyle="--", label="Starting Money ($)")
plt.xlabel("Time Period")
plt.ylabel("Portfolio Value ($)")
plt.title("Portfolio Value Over Trading Periods")
plt.legend()
plt.grid(False)
plt.show()

trading_results_long_short_V2.to_csv('results/trading_results_long_short_V2.csv', index=False)

correct_long = 0
incorrect_long = 0
correct_short = 0
incorrect_short = 0
held_cash_when_down = 0
held_cash_when_up = 0
total_up = 0
total_down = 0

total_long = trading_results_long_short_V2['Case / Action'].value_counts().get('Long', 0)
total_short = trading_results_long_short_V2['Case / Action'].value_counts().get('Short', 0)
total_held_cash = trading_results_long_short_V2['Case / Action'].value_counts().get('Hold Cash', 0)



for i in range(len(trading_results_long_short_V2) - 1):
    if trading_results_long_short_V2.loc[i + 1, 'Price_t'] > trading_results_long_short_V2.loc[i, 'Price_t']:
        total_up += 1
        if trading_results_long_short_V2.loc[i, 'Case / Action'] == 'Long' and trading_results_long_short_V2.loc[i + 1, 'Price_t'] > trading_results_long_short_V2.loc[i, 'Price_t']:
            correct_long += 1
        if trading_results_long_short_V2.loc[i, 'Case / Action'] == 'Hold Cash' and trading_results_long_short_V2.loc[i + 1, 'Price_t'] > trading_results_long_short_V2.loc[i, 'Price_t']:
            held_cash_when_up += 1
        if trading_results_long_short_V2.loc[i, 'Case / Action'] == 'Short' and trading_results_long_short_V2.loc[i + 1, 'Price_t'] > trading_results_long_short_V2.loc[i, 'Price_t']:
                incorrect_short += 1

    if trading_results_long_short_V2.loc[i + 1, 'Price_t'] < trading_results_long_short_V2.loc[i, 'Price_t']:
        total_down += 1
        if trading_results_long_short_V2.loc[i, 'Case / Action'] == 'Short' and trading_results_long_short_V2.loc[i + 1, 'Price_t'] < trading_results_long_short_V2.loc[i, 'Price_t']:
            correct_short += 1   
        if trading_results_long_short_V2.loc[i, 'Case / Action'] == 'Hold Cash' and trading_results_long_short_V2.loc[i + 1, 'Price_t'] < trading_results_long_short_V2.loc[i, 'Price_t']:
            held_cash_when_down += 1
        if trading_results_long_short_V2.loc[i, 'Case / Action'] == 'Long' and trading_results_long_short_V2.loc[i + 1, 'Price_t'] < trading_results_long_short_V2.loc[i, 'Price_t']:
            incorrect_long += 1
    
percent_correct_long = (correct_long / total_up) if total_long > 0 else 0
percent_incorrect_long = (incorrect_long / total_down) if total_long > 0 else 0
percent_correct_short = (correct_short / total_down) if total_short > 0 else 0
percent_incorrect_short = (incorrect_short / total_up) if total_short > 0 else 0
percent_held_cash_when_down = (held_cash_when_down / total_down) if total_held_cash > 0 else 0
percent_held_cash_when_up = (held_cash_when_up / total_up) if total_held_cash > 0 else 0


print("Correct Long Actions:")
print(f"Total: {correct_long}, Percentage: {percent_correct_long:.4f}%")

print("Held Cash When Price Increased:")
print(f"Total: {held_cash_when_up}, Percentage: {percent_held_cash_when_up:.4f}%")

print("Incorrect Long Actions:")
print(f"Total: {incorrect_long}, Percentage: {percent_incorrect_long:.4f}%")

print("Correct Short Actions:")
print(f"Total: {correct_short}, Percentage: {percent_correct_short:.4f}%")

print("Held Cash When Price Decreased:")
print(f"Total: {held_cash_when_down}, Percentage: {percent_held_cash_when_down:.4f}%")

print("Incorrect Short Actions:")
print(f"Total: {incorrect_short}, Percentage: {percent_incorrect_short:.4f}%")




#### Calculate Share Ratio and Annualized Returns ####

#percent_returns_long_only = np.diff(total_wealth_long_only) / total_wealth_long_short[:-1]  
percent_returns_long_short = np.diff(total_wealth_long_short) / total_wealth_long_short[:-1]  
percent_returns_long_short_V2 = np.diff(total_wealth_long_short_V2) / total_wealth_long_short_V2[:-1]  
percent_returns_buy_and_hold = np.diff(buy_and_hold) / buy_and_hold[:-1]  # Buy-and-hold returns

returns_long_short_V2 = np.diff(total_wealth_long_short_V2)
returns_buy_and_hold = np.diff(buy_and_hold)   # Buy-and-hold returns


data_frequency = 2*24*365 # BTC trades 24/7, so 2 30-min periods per hour x 24 hours x 365 days a year
#sharpe_long_only = calculate_annual_sharpe_ratio(percent_returns_long_only, annual_risk_free_rate = 0.0, data_frequency = data_frequency)
sharpe_long_short = calculate_annual_sharpe_ratio(percent_returns_long_short, annual_risk_free_rate = 0.0, data_frequency = data_frequency)
sharpe_long_short_V2 = calculate_annual_sharpe_ratio(percent_returns_long_short_V2, annual_risk_free_rate = 0.0, data_frequency = data_frequency)
sharpe_buy_and_hold = calculate_annual_sharpe_ratio(percent_returns_buy_and_hold, annual_risk_free_rate = 0.0, data_frequency = data_frequency)

sharpe_long_short_V2_notpercent = calculate_annual_sharpe_ratio(returns_long_short_V2, annual_risk_free_rate = 0.0, data_frequency = data_frequency)


#print(f"Sharpe Ratio (Long Only): {sharpe_long_only:.4f}")
print(f"Sharpe Ratio (Long Short): {sharpe_long_short:.4f}")
print(f"Sharpe Ratio (Long Short, % Tol): {sharpe_long_short_V2:.4f}")
print(f"Sharpe Ratio (Buy and Hold): {sharpe_buy_and_hold:.4f}")

#sharpe_long_only = calculate_annual_sharpe_ratio_V2(percent_returns_long_only, annual_risk_free_rate = 0.0, data_frequency = data_frequency)
sharpe_long_short = calculate_annual_sharpe_ratio_V2(percent_returns_long_short, annual_risk_free_rate = 0.0, data_frequency = data_frequency)
sharpe_buy_and_hold = calculate_annual_sharpe_ratio_V2(percent_returns_buy_and_hold, annual_risk_free_rate = 0.0, data_frequency = data_frequency)

#print(f"Sharpe Ratio (Long Only): {sharpe_long_only:.4f}")
print(f"Sharpe Ratio (Long Short): {sharpe_long_short:.4f}")
print(f"Sharpe Ratio (Buy and Hold): {sharpe_buy_and_hold:.4f}")

