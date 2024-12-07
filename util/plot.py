import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(y_train_actual, y_valid_actual, y_test_actual,
                     train_predictions, valid_predictions, test_predictions):
    # Plot Training, Validation, and Testing Predictions
    plt.figure(figsize=(14, 7))
    plt.plot(np.arange(len(y_train_actual)), y_train_actual, label='Train Actual', color='purple')
    plt.plot(np.arange(len(y_train_actual)), train_predictions, label='LSTM Train Predict', color='cyan')
    plt.plot(np.arange(len(y_train_actual), len(y_train_actual) + len(y_valid_actual)), y_valid_actual, label='Valid Actual', color='green')
    plt.plot(np.arange(len(y_train_actual), len(y_train_actual) + len(y_valid_actual)), valid_predictions, label='LSTM Valid Predict', color='blue')
    plt.plot(np.arange(len(y_train_actual) + len(y_valid_actual), len(y_train_actual) + len(y_valid_actual) + len(y_test_actual)), y_test_actual, label='Test Actual', color='black')
    plt.plot(np.arange(len(y_train_actual) + len(y_valid_actual), len(y_train_actual) + len(y_valid_actual) + len(y_test_actual)), test_predictions, label='LSTM Test Predict', color='orange')
    plt.title('Bitcoin Price Prediction (Training, Validation, and Testing)')
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

def plot_valid_test_only_predictions(y_valid_actual, y_test_actual,
                                     valid_predictions, test_predictions):
    # Separate plot for Validation and Testing only (optional)
    plt.figure(figsize=(14, 7))
    plt.plot(np.arange(len(y_valid_actual)), y_valid_actual, label='Valid Actual', color='green')
    plt.plot(np.arange(len(y_valid_actual)), valid_predictions, label='LSTM Valid Predict', color='blue')
    plt.plot(np.arange(len(y_valid_actual), len(y_valid_actual) + len(y_test_actual)), y_test_actual, label='Test Actual', color='black')
    plt.plot(np.arange(len(y_valid_actual), len(y_valid_actual) + len(y_test_actual)), test_predictions, label='LSTM Test Predict', color='orange')
    plt.title('Bitcoin Price Prediction (Validation and Testing Only)')
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.show()