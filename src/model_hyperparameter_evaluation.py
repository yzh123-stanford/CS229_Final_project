# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:31:16 2024

@author: Pierce Mullin & Yifan Zhang
"""

import pandas as pd

def model_evaluation_table(loaded_results):

    evaluation_table = []
    
    # Loop through all loaded results
    for i, result in enumerate(loaded_results):
        # Gather common parameters
        row = {
            "Model Number": i + 1,
            "Hidden Dim": result["hidden_dim"],
            "Num Layers": result["num_layers"],
            "Batch Size": result["batch_size"],
            "Dropout Rate": result["dropout_rate"],
            "Loss Type": result["loss_type"],
            "LSTM Type": result["lstm_type"],
            "Best Validation Loss": result["best_val_loss"],
            "Best Validation Loss (Close)": result["best_val_loss_close"],
            "Gamma": result.get("gamma", "NA"),  # Use "NA" for SmoothL1 Loss because this parameter isnt present
            "Directional Lambda": result.get("directional_lambda", "NA"),  # Use "NA" for SmoothL1 Loss because this parameter isnt present
            "Closing Lambda": result.get("closing_lambda", "NA")  # Use "NA" for SmoothL1 Loss because this parameter isnt present
        }
        evaluation_table.append(row)
    
    evaluation_df = pd.DataFrame(evaluation_table)
    
    evaluation_df = evaluation_df.sort_values(by="Best Validation Loss (Close)", ascending=True)
    
    return evaluation_df
    