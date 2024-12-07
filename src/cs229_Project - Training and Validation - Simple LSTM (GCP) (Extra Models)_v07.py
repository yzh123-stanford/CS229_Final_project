#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from collections import deque


# In[2]:


#### Load Internal Files ####

from util.load_and_clean_BTCUSDT import load_and_clean_BTCUSDT_data
from util.create_windowed_series_and_targets import create_windowed_sequences
from util.create_windowed_series_and_targets import create_windowed_sequences_levels
from util.run_ARIMA_windows_and_accuracy import run_arima_windows_univariate
from util.run_ARIMA_windows_and_accuracy import create_basic_accuracy_table
from util.run_ARIMA_windows_and_accuracy import add_directional_accuracy
from util.plot_predictions import plot_predictions_at_index
from util.run_trading_strategy import run_strategy_long_only
from util.run_trading_strategy import run_strategy_long_short


# In[3]:


#### Set Up Environment (if using cuda) ####
random.seed(229)
using_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if using_cuda == True:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    torch.cuda.empty_cache()


# In[4]:


#### Loading and Processing Data ####
data_path = "../data/BTCUSDT_data.csv" 
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


# In[5]:


#### Processing the Data into a set of Sequences and Targets for LSTM model training and prediction ####
window_size = 2*24*20 # per-hour*hours-per-day*days
forecast_horizon = 1 #2*12*1 # 12 hours

variables = ['open_return', 'high_return', 'low_return', 'close_return','normalized_dollar_volume']
variables_levels = ['open', 'high', 'low', 'close','normalized_dollar_volume']
df_model = aggregated[variables]
df_model_levels = aggregated[variables_levels]
print("Getting X, y")
X, y = create_windowed_sequences(df_model, window_size, forecast_horizon = forecast_horizon)
print("Getting X_prices, y_prices")
X_prices, y_prices = create_windowed_sequences_levels(df_model_levels, window_size, forecast_horizon = forecast_horizon)


# In[6]:


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

# Plot BTCUSDT Closing Price Over the Training Period
plt.figure(figsize=(10, 6))
plt.plot(btc_close[:train_size], label="BTCUSDT", color="blue")
plt.xlabel("Time Period")
plt.ylabel("$")
plt.title("BTCUSDT Price Over Time Train Period")
plt.legend()
plt.grid(False)
plt.show()


# In[7]:


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
model_accuracy_table_RW_test.to_csv('../results/model_accuracy_table_RW_test.csv', index=False)
np.save('../results/forecast_matrix_RW_test.npy', forecast_matrix_RW_test)

# Loading Results
model_accuracy_table_RW_test = pd.read_csv('../results/model_accuracy_table_RW_test.csv', na_values=["inf", "NaN"])
forecast_matrix_RW_test = np.load('../results/forecast_matrix_RW_test.npy')
print(model_accuracy_table_RW_test)
print(forecast_matrix_RW_test)


# In[8]:


#### Converting Data into Tensors for Model Training and Prediction ####
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_valid = torch.tensor(X_valid, dtype=torch.float32).to(device)
y_valid = torch.tensor(y_valid, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)


# In[9]:


#### Defining the LSTM ####

input_dim = 5
output_dim = (forecast_horizon*input_dim)

# Define the LSTM model in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        x = self.fc(lstm_out[:, -1, :])
        return x.view(-1, forecast_horizon, input_dim)
    
# Define Custom Loss Function
class CustomTrainingLoss(nn.Module):
    def __init__(self, gamma = 1.0, directional_lambda = 0.0, closing_lambda = 0.0, beta=1.0):
        """
        Initializes the CustomTrainingLoss module.
        
        The purpose of this custom loss function is to help the LSTM model train towards: 
            (1) good overall MSE, and
            (2) good MSE on Closing Prices (which will be the basis of trading strategies), and
            (3) good directional accuracy on closing prices, and
            (4) ability to weight accuracy in near vs. far-term forecasts, if desired
             
        This is to avoid flat predictions that just take the middle ground between up or down price moves.
        
        Args:
            gamma (float): Initial discount rate for the forecast horizon.
            directional_lambda (float): term that determines how relatively important directional accuracy should be to the total training loss
            closing_lambda (float): term that determines how relatively important closing price accuracy should be to the total training loss by adding additional penalty for that series' errors
            beta (float): Transition point for SmoothL1Loss.
        """
        super(CustomTrainingLoss, self).__init__()
        self.gamma = gamma
        self.directional_lambda = directional_lambda
        self.closing_lambda = closing_lambda
        self.beta = beta

    def set_gamma(self, gamma):
        self.gamma = gamma
        
    def set_directional_lambda(self, directional_lambda):
        self.directional_lambda = directional_lambda
        
    def set_closing_lambda(self, closing_lambda):
        self.closing_lambda = closing_lambda

    def forward(self, predictions, targets):
        """
        Compute the discounted SmoothL1Loss.
        Args:
            predictions (torch.Tensor): Predicted values of shape (batch_size, horizon, *dims).
            targets (torch.Tensor): True values of shape (batch_size, horizon, *dims).
        Returns:
            torch.Tensor: Custom Training loss.
        """
        horizon = predictions.size(1)  # Assumes shape (batch_size, horizon, ...)
        
        # Compute absolute difference
        diff = (predictions - targets)

        # Compute SmoothL1 loss for each forecast step
        smoothL1loss = torch.where(
            torch.abs(diff) < self.beta,
            0.5 * (diff**2) / self.beta,
            torch.abs(diff) - 0.5 * self.beta
        )

        # Apply discount factors
        discount_factors = torch.tensor([self.gamma**(h-1) for h in range(1, horizon + 1)], device=smoothL1loss.device)
        discount_factors = discount_factors.view(1, -1, *(1,) * (smoothL1loss.ndim - 2))  # Broadcast to match dims
        weighted_loss = smoothL1loss * discount_factors
        
        # Calculate Directional Errors
        closing_price_predictions = predictions[:, :, 3]  # Shape: (batch_size, horizon, series)...closing prices are the 4th series at index 3
        closing_price_targets = targets[:, :, 3]
        
        closing_price_diff = (predictions[:, :, 3] - targets[:, :, 3])
        
        directional_diff = torch.abs(torch.sign(closing_price_predictions) - torch.sign(closing_price_targets)) / 2 # Will give a 1 where there is directional mismatch
        
        directional_error_term = directional_diff * self.directional_lambda
        
        # Closing Price Specific Error
        smoothL1loss_close = torch.where(
            torch.abs(closing_price_diff) < self.beta,
            0.5 * (closing_price_diff**2) / self.beta,
            torch.abs(closing_price_diff) - 0.5 * self.beta
        )
        
        closing_error_term = smoothL1loss_close * self.closing_lambda

        
        # Combining to Create a Custom Training Loss
        
        custom_loss = weighted_loss.mean() + directional_error_term.mean() + closing_error_term.mean()

        return custom_loss  # Return custom loss


# In[12]:


# Set up Hyperparameter Alternatives for Model
n_alternatives = 9

input_dim = 5 #<--- Dont change this
hidden_dim = [20, 20, 20, 50, 100]
num_layers = [2, 2, 2, 2, 1]
output_dim = forecast_horizon * 5 #<--- Dont change this
num_epochs = 200 #<--- Dont change this
batch_size = [800, 800, 800, 800, 400] # may need smaller batches for the bigger models 
learning_rate = 0.0001 # 0.001
dropout_rate = [0.2, 0.2, 0.2, 0.3, 0.4] 
patience = 20 #<--- Dont change this
GAMMA = [1, 1, 1, 1, 1, 1] # Discount rate for errors across horizons to make near-term predictions more/less important
directional_lambda = [0.0, 0.0, 0.3, 0.0, 0.3]
closing_lambda = [0.0, 0.5, 0.3, 0.5, 0.3]


n_alternatives = 9


input_dim = 5 #<--- Dont change this
hidden_dim = [20, 20, 20, 50, 100, 20, 50, 50, 100]
num_layers = [2, 2, 2, 2, 1, 2, 2, 2, 1]
output_dim = forecast_horizon * 5 #<--- Dont change this
num_epochs = 200 #<--- Dont change this
batch_size = [800, 800, 800, 800, 400, 100, 100, 100, 100] # may need smaller batches for the bigger models 
learning_rate = 0.0001 # 0.001
dropout_rate = [0.2, 0.2, 0.2, 0.3, 0.4, 0.2, 0.3, 0.3, 0.4] 
patience = 25 #<--- Dont change this
GAMMA = [1, 1, 1, 1, 1, 1, 1, 1, 1] # Discount rate for errors across horizons to make near-term predictions more/less important
directional_lambda = [0.0, 0.0, 0.3, 0.0, 0.3, 0.3, 0.3, 0.3, 0.3]
closing_lambda = [0.0, 0.5, 0.3, 0.5, 0.3, 0.3, 1.0, 1.0, 1.0]

# Training loop with early stopping
best_val_loss = float('inf')
epochs_no_improve = 0

# Prepare DataLoader for training
train_data = TensorDataset(X_train, y_train)

# Making placeholders to hold model results
train_losses_master = []
train_losses_close_master = []
val_losses_master = []
val_losses_close_master = []
best_model_state_master = []
best_model_state_close_master = []
hyperparameter_results = []


# In[13]:


#### Model Training Loop and Hyperparameter Evaluation --- Custom Loss Function

for alt in range(5,8):
    
    print(f"Hyperparameter Alternative: [{(alt+1)}/{n_alternatives}], "
          f"Hidden Dimensions: {hidden_dim[alt]}, "
          f"Number of Layers: {num_layers[alt]}, "
          f"Batch Size: {batch_size[alt]}, "
          f"Dropout Rate: {dropout_rate[alt]}, "
          f"Gamma: {GAMMA[alt]}, "
          f"Directional Lambda: {directional_lambda[alt]}, "
          f"Closing Price Lambda: {closing_lambda[alt]}")
    
    # Initialize model, loss function, and optimizer
    model = LSTMModel(input_dim, hidden_dim[alt], output_dim, num_layers[alt], dropout_rate[alt]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_criterion = CustomTrainingLoss()
    training_criterion.set_gamma(GAMMA[alt])
    training_criterion.set_directional_lambda(directional_lambda[alt])
    training_criterion.set_closing_lambda(closing_lambda[alt])


    valid_criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare DataLoader for training
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size[alt])
    
    val_data = TensorDataset(X_valid, y_valid)   #
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size[alt], )  #

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_val_loss_close = float('inf')
    epochs_no_improve = 0

    # Create Lists to Track Error Across Epochs
    train_losses = []
    train_losses_close = []
    val_losses = []
    val_losses_close = []
    best_model_state = []
    best_model_state_close = []



    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_loss_close = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = training_criterion(outputs, targets)
            loss_close = valid_criterion(outputs[:,:,3], targets[:,:,3]) # Not a typo - I want comparable training loss and its only the gradient descent I want the discounted criterion
            epoch_train_loss += loss.item()
            epoch_train_loss_close += loss_close.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        avg_train_loss_close = epoch_train_loss_close / len(train_loader)
        train_losses_close.append(avg_train_loss_close)

        # Validation Loss
        model.eval()
        epoch_val_loss = 0
        epoch_val_loss_close = 0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model(val_inputs)
                val_loss = valid_criterion(val_outputs, val_targets)
                val_loss_close = valid_criterion(val_outputs[:,:,3], val_targets[:,:,3])
                epoch_val_loss += val_loss.item()
                epoch_val_loss_close += val_loss_close.item()
                
                #val_outputs = model(X_valid.to(device))
                #val_loss = criterion(val_outputs, y_valid.to(device))
                #val_losses.append(val_loss.item())
                #val_loss_close = criterion(val_outputs[:,:,3], y_valid[:,:,3].to(device))
                #val_losses_close.append(val_loss_close.item())
                
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        avg_val_loss_close = epoch_val_loss_close / len(val_loader)
        val_losses_close.append(avg_val_loss_close)


        print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Training Loss: {avg_train_loss:.10f}, "
          f"Validation Loss: {avg_val_loss:.10f}, "
          #f"Training Loss (Close): {avg_train_loss_close:.10f}, "
          f"Validation Loss (Close): {avg_val_loss_close:.10f}")


        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            
        if avg_val_loss_close < best_val_loss_close:
            best_val_loss_close = avg_val_loss_close
            epochs_no_improve_close = 0
            best_model_state_close = model.state_dict()
        else:
            epochs_no_improve_close += 1

        #if epochs_no_improve >= patience:
        #    print(f"Early stopping triggered after {epoch+1} epochs.")
        #    break
            
        if epochs_no_improve_close >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs (Based on Closing Price Returns).")
            break
            
    train_losses_master.append(train_losses)
    train_losses_close_master.append(train_losses_close)
    val_losses_master.append(val_losses)
    val_losses_close_master.append(val_losses_close)
    best_model_state_master.append(best_model_state.copy())
    
    result = {
        "hidden_dim": hidden_dim[alt],
        "num_layers": num_layers[alt],
        "batch_size": batch_size[alt],
        "dropout_rate": dropout_rate[alt],
        "best_val_loss": best_val_loss,
        "best_val_loss_close": best_val_loss_close,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_losses_close": train_losses_close,
        "val_losses_close": val_losses_close,
        "best_model_state": best_model_state,
        "best_model_state_close": best_model_state_close,
        "gamma": GAMMA[alt],
        "directional_lambda": directional_lambda[alt],
        "closing_lambda": closing_lambda[alt]



    }
    hyperparameter_results.append(result)





# In[14]:


#### Plotting Results
    
ylim_max = 0.0002

for alt in range(n_alternatives):  # Loop over the 6 hyperparameter alternatives
    #result_alt = hyperparameter_results_load[alt]
    train_losses_close = train_losses_close_master[alt]
    #train_losses_close = result_alt["train_losses_close"]

    val_losses_close = val_losses_close_master[alt]
    #val_losses_close = result_alt["val_losses_close"]
    
    # Create RW Benchmark Line for Validation Loss
    RW_benchmark_MSE_valid = model_accuracy_table_RW_valid['MSE'][0]  # Fixed RW Benchmark value
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
    plt.ylim([0, ylim_max])  # Adjust based on your loss range
    plt.grid(False)
    plt.show()


# In[15]:


#### Saving Models and Model Results --- Custom Loss ####

# Saving Hyperparameter Results
save_dir = "../results/"

results_metadata = [
    {
        "hidden_dim": result["hidden_dim"],
        "num_layers": result["num_layers"],
        "batch_size": result["batch_size"],
        "dropout_rate": result["dropout_rate"],
        "gamma": result["gamma"],
        "directional_lambda": result["directional_lambda"],
        "closing_lambda": result["closing_lambda"],
        "best_val_loss": result["best_val_loss"],
        "best_val_loss_close": result["best_val_loss_close"],
        "train_losses": result["train_losses"],
        "val_losses": result["val_losses"],
        "train_losses_close": result["train_losses_close"],
        "val_losses_close": result["val_losses_close"]
        #"best_model_state": result["best_model_state"],
        #"best_model_state_close": result["best_model_state_close"]

    }
    for result in hyperparameter_results
]

# Saving Results
metadata_path = os.path.join(save_dir, "results_metadata_custom_loss_simple_LSTM_additional.json")
with open(metadata_path, "w") as f:
    json.dump(results_metadata, f, indent=4)
    
# Saving Models
for i, result in enumerate(hyperparameter_results):
    model_path = os.path.join(save_dir, f"best_model_state_BTCUSDT_custom_loss_simple_LSTM_additional_{i}.pth")
    torch.save(result["best_model_state"], model_path)

print(f"Metadata saved to {metadata_path}. Model states saved to {save_dir}.")


# In[18]:


# Set up Hyperparameter Alternatives for Model
n_alternatives = 3

input_dim = 5 #<--- Dont change this
hidden_dim = [20, 50, 100]
num_layers = [2, 2, 1]
output_dim = forecast_horizon * 5 #<--- Dont change this
num_epochs = 200 #<--- Dont change this
batch_size = [800, 800, 400] # may need smaller batches for the bigger models 
learning_rate = 0.0001 # 0.001
dropout_rate = [0.2, 0.3, 0.4] 
patience = 20 #<--- Dont change this


n_alternatives = 7

input_dim = 5 #<--- Dont change this
hidden_dim = [20, 50, 100, 20, 50, 50, 100]
num_layers = [2, 2, 1, 2, 2, 2, 1]
output_dim = forecast_horizon * 5 #<--- Dont change this
num_epochs = 200 #<--- Dont change this
batch_size = [800, 800, 400, 100, 100, 100, 100] # may need smaller batches for the bigger models 
learning_rate = 0.0001 # 0.001
dropout_rate = [0.2, 0.3, 0.4, 0.2, 0.3, 0.3, 0.4] 
patience = 25 #<--- Dont change this

# Training loop with early stopping
best_val_loss = float('inf')
epochs_no_improve = 0

# Prepare DataLoader for training
train_data = TensorDataset(X_train, y_train)

# Making placeholders to hold model results
train_losses_master = []
train_losses_close_master = []
val_losses_master = []
val_losses_close_master = []
best_model_state_master = []
best_model_state_close_master = []
hyperparameter_results = []


# In[19]:


#### Model Training Loop and Hyperparameter Evaluation --- Regular SmoothL1Loss

for alt in range(3,6):
    
    print(f"Hyperparameter Alternative: [{(alt+1)}/{n_alternatives}], "
          f"Hidden Dimensions: {hidden_dim[alt]}, "
          f"Number of Layers: {num_layers[alt]}, "
          f"Batch Size: {batch_size[alt]}, "
          f"Dropout Rate: {dropout_rate[alt]}")
    
    # Initialize model, loss function, and optimizer
    model = LSTMModel(input_dim, hidden_dim[alt], output_dim, num_layers[alt], dropout_rate[alt]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_criterion = nn.SmoothL1Loss()

    valid_criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare DataLoader for training
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size[alt])
    
    val_data = TensorDataset(X_valid, y_valid)   #
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size[alt], )  #

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_val_loss_close = float('inf')
    epochs_no_improve = 0

    # Create Lists to Track Error Across Epochs
    train_losses = []
    train_losses_close = []
    val_losses = []
    val_losses_close = []
    best_model_state = []
    best_model_state_close = []



    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_loss_close = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = training_criterion(outputs, targets)
            loss_close = valid_criterion(outputs[:,:,3], targets[:,:,3]) # Not a typo - I want comparable training loss and its only the gradient descent I want the discounted criterion
            epoch_train_loss += loss.item()
            epoch_train_loss_close += loss_close.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        avg_train_loss_close = epoch_train_loss_close / len(train_loader)
        train_losses_close.append(avg_train_loss_close)

        # Validation Loss
        model.eval()
        epoch_val_loss = 0
        epoch_val_loss_close = 0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model(val_inputs)
                val_loss = valid_criterion(val_outputs, val_targets)
                val_loss_close = valid_criterion(val_outputs[:,:,3], val_targets[:,:,3])
                epoch_val_loss += val_loss.item()
                epoch_val_loss_close += val_loss_close.item()
                
                #val_outputs = model(X_valid.to(device))
                #val_loss = criterion(val_outputs, y_valid.to(device))
                #val_losses.append(val_loss.item())
                #val_loss_close = criterion(val_outputs[:,:,3], y_valid[:,:,3].to(device))
                #val_losses_close.append(val_loss_close.item())
                
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        avg_val_loss_close = epoch_val_loss_close / len(val_loader)
        val_losses_close.append(avg_val_loss_close)


        print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Training Loss: {avg_train_loss:.10f}, "
          f"Validation Loss: {avg_val_loss:.10f}, "
          #f"Training Loss (Close): {avg_train_loss_close:.10f}, "
          f"Validation Loss (Close): {avg_val_loss_close:.10f}")


        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            
        if avg_val_loss_close < best_val_loss_close:
            best_val_loss_close = avg_val_loss_close
            epochs_no_improve_close = 0
            best_model_state_close = model.state_dict()
        else:
            epochs_no_improve_close += 1

        #if epochs_no_improve >= patience:
        #    print(f"Early stopping triggered after {epoch+1} epochs.")
        #    break
            
        if epochs_no_improve_close >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs (Based on Closing Price Returns).")
            break
            
    train_losses_master.append(train_losses)
    train_losses_close_master.append(train_losses_close)
    val_losses_master.append(val_losses)
    val_losses_close_master.append(val_losses_close)
    best_model_state_master.append(best_model_state.copy())
    
    result = {
        "hidden_dim": hidden_dim[alt],
        "num_layers": num_layers[alt],
        "batch_size": batch_size[alt],
        "dropout_rate": dropout_rate[alt],
        "best_val_loss": best_val_loss,
        "best_val_loss_close": best_val_loss_close,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_losses_close": train_losses_close,
        "val_losses_close": val_losses_close,
        "best_model_state": best_model_state,
        "best_model_state_close": best_model_state_close



    }
    hyperparameter_results.append(result)





# In[20]:


#### Plotting Results
    
ylim_max = 0.0002

for alt in range(n_alternatives):  # Loop over the 6 hyperparameter alternatives
    #result_alt = hyperparameter_results_load[alt]
    train_losses_close = train_losses_close_master[alt]
    #train_losses_close = result_alt["train_losses_close"]

    val_losses_close = val_losses_close_master[alt]
    #val_losses_close = result_alt["val_losses_close"]
    
    # Create RW Benchmark Line for Validation Loss
    RW_benchmark_MSE_valid = model_accuracy_table_RW_valid['MSE'][0]  # Fixed RW Benchmark value
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
    plt.ylim([0, ylim_max])  # Adjust based on your loss range
    plt.grid(False)
    plt.show()


# In[21]:


#### Saving Models and Model Results --- Regular SmoothL1Loss ####

# Saving Hyperparameter Results
save_dir = "../results/"

results_metadata = [
    {
        "hidden_dim": result["hidden_dim"],
        "num_layers": result["num_layers"],
        "batch_size": result["batch_size"],
        "dropout_rate": result["dropout_rate"],
        "best_val_loss": result["best_val_loss"],
        "best_val_loss_close": result["best_val_loss_close"],
        "train_losses": result["train_losses"],
        "val_losses": result["val_losses"],
        "train_losses_close": result["train_losses_close"],
        "val_losses_close": result["val_losses_close"]
        #"best_model_state": result["best_model_state"],
        #"best_model_state_close": result["best_model_state_close"]

    }
    for result in hyperparameter_results
]

# Saving Results
metadata_path = os.path.join(save_dir, "results_metadata_smoothL1_loss_simple_LSTM_additional.json")
with open(metadata_path, "w") as f:
    json.dump(results_metadata, f, indent=4)
    
# Saving Models
for i, result in enumerate(hyperparameter_results):
    model_path = os.path.join(save_dir, f"best_model_state_BTCUSDT_smoothL1_loss_simple_LSTM_additional_{i}.pth")
    torch.save(result["best_model_state"], model_path)

print(f"Metadata saved to {metadata_path}. Model states saved to {save_dir}.")


# In[ ]:




