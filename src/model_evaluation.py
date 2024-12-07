import torch
from torchmetrics import MeanSquaredError, MeanAbsoluteError

class ModelEvaluator:
    def __init__(self):
        self.mse_metric = MeanSquaredError()
        self.mae_metric = MeanAbsoluteError()
    
    def calculate_mse(self, y_true, y_pred):
        # Ensuring tensors are compatible with reshape
        return self.mse_metric(torch.tensor(y_pred).reshape(-1), torch.tensor(y_true).reshape(-1)).item()
    
    def calculate_rmse(self, y_true, y_pred):
        mse = self.calculate_mse(y_true, y_pred)
        return torch.sqrt(torch.tensor(mse)).item()
    
    def calculate_mae(self, y_true, y_pred):
        return self.mae_metric(torch.tensor(y_pred).reshape(-1), torch.tensor(y_true).reshape(-1)).item()

