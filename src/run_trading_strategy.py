# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 09:15:02 2024

@author: Pierce Mullin & Yifan Zhang
"""

import pandas as pd
import numpy as np

def run_strategy_long_only(predictions, actuals, starting_time = 960, starting_money = 1000, rule_index = 1, tol = 0, verbose = True):
    
    n_periods = predictions.shape[0]
    results = np.full((n_periods,5),np.nan)
    actions_history = []


    # Simple Trading Rule --- Run Trading Rule
        
    coin_holdings = 0
    cash_history = []
    total_wealth_history = []
    buy_and_hold = []
    case_t = None
        
    for t in range(n_periods):
        
        price_t = actuals[t, (actuals.shape[1]-1)]
        prediction_t_plus_h = predictions[t,rule_index]
            
        if t == 0:
            cash = starting_money
            total_wealth = starting_money
            coin_0 = starting_money / price_t
        
        # Determine the Case
        if coin_holdings == 0 and prediction_t_plus_h > (price_t + tol):
            case_t = "Buy"
            
        if coin_holdings > 0 and prediction_t_plus_h > (price_t + tol):
            case_t = "Hold"
    
        if coin_holdings == 0 and prediction_t_plus_h < (price_t - tol):
            case_t = "Keep Cash"
            
        if coin_holdings > 0 and prediction_t_plus_h < (price_t - tol):
            case_t = "Sell"
            
        if prediction_t_plus_h <= (price_t + tol) and prediction_t_plus_h >= (price_t - tol):
            case_t = "Do Nothing"
    

        # Determine the Action
        if case_t == "Buy":
            coin_holdings = cash / price_t
            cash = 0
            total_wealth = coin_holdings * price_t
            
        if case_t == "Hold":
            total_wealth = coin_holdings * price_t
    
        if case_t == "Keep Cash":
            total_wealth = cash
            
        if case_t ==  "Sell":
            cash = coin_holdings * price_t
            coin_holdings = 0
            total_wealth = cash
            
        if case_t == "Do Nothing":
            pass
    
        if t == (n_periods-1):
            cash = max(coin_holdings * price_t, cash)
            coin_holdings = 0
            total_wealth = cash
            
        cash_history.append(cash)
        total_wealth_history.append(total_wealth)
        buy_and_hold.append(coin_0 * price_t)
        actions_history.append(case_t)
        
        results[t,0] = price_t
        results[t,1] = prediction_t_plus_h
        results[t,2] = cash
        results[t,3] = coin_holdings
        results[t,4] = total_wealth
        
        
            
        if verbose == True and t % 1000 == 0:
            print(f"Period: {t} / {n_periods}, Cash: {cash:.2f}, Coin Holdings: {coin_holdings:.4f}, Total Wealth: {total_wealth:.2f}, Price_t: {price_t:.2f}, E(Price_t+1): {prediction_t_plus_h:.2f}")
    
    print(f"Final Cash: {cash:.2f}")
        
    trading_results = pd.DataFrame(results,
                                   columns=["Price_t", "E(P_t+h)", "Cash", "Coin Holdings", "Total Wealth"])
    
    trading_results["Case / Action"] = actions_history  

    trading_results = trading_results[[
        "Price_t", "E(P_t+h)", "Case / Action", "Cash", "Coin Holdings", "Total Wealth"
    ]]

        
    return total_wealth_history, buy_and_hold, trading_results

def run_strategy_long_only_V2(predictions, actuals, starting_time = 960, starting_money = 1000, rule_index = 1, tol_percent = 0.0, verbose = True):
    
    n_periods = predictions.shape[0]
    results = np.full((n_periods,5),np.nan)
    actions_history = []


    # Simple Trading Rule --- Run Trading Rule
        
    coin_holdings = 0
    cash_history = []
    total_wealth_history = []
    buy_and_hold = []
    case_t = None
        
    for t in range(n_periods):
        
        price_t = actuals[t, (actuals.shape[1]-1)]
        prediction_t_plus_h = predictions[t,rule_index]
            
        if t == 0:
            cash = starting_money
            total_wealth = starting_money
            coin_0 = starting_money / price_t
        
        # Determine the Case
        if coin_holdings == 0 and prediction_t_plus_h > (price_t * (1 + tol_percent)):
            case_t = "Buy"
            
        if coin_holdings > 0 and prediction_t_plus_h > (price_t * (1 + tol_percent)):
            case_t = "Hold"
    
        if coin_holdings == 0 and prediction_t_plus_h < (price_t * (1 - tol_percent)):
            case_t = "Keep Cash"
            
        if coin_holdings > 0 and prediction_t_plus_h < (price_t * (1 - tol_percent)):
            case_t = "Sell"
            
        if prediction_t_plus_h <= (price_t * (1 + tol_percent)) and prediction_t_plus_h >= (price_t * (1 - tol_percent)):
            case_t = "Do Nothing"
    

        # Determine the Action
        if case_t == "Buy":
            coin_holdings = cash / price_t
            cash = 0
            total_wealth = coin_holdings * price_t
            
        if case_t == "Hold":
            total_wealth = coin_holdings * price_t
    
        if case_t == "Keep Cash":
            total_wealth = cash
            
        if case_t ==  "Sell":
            cash = coin_holdings * price_t
            coin_holdings = 0
            total_wealth = cash
            
        if case_t == "Do Nothing":
            pass
    
        if t == (n_periods-1):
            cash = max(coin_holdings * price_t, cash)
            coin_holdings = 0
            total_wealth = cash
            
        cash_history.append(cash)
        total_wealth_history.append(total_wealth)
        buy_and_hold.append(coin_0 * price_t)
        actions_history.append(case_t)
        
        results[t,0] = price_t
        results[t,1] = prediction_t_plus_h
        results[t,2] = cash
        results[t,3] = coin_holdings
        results[t,4] = total_wealth
        
        
            
        if verbose == True and t % 1000 == 0:
            print(f"Period: {t} / {n_periods}, Cash: {cash:.2f}, Coin Holdings: {coin_holdings:.4f}, Total Wealth: {total_wealth:.2f}, Price_t: {price_t:.2f}, E(Price_t+1): {prediction_t_plus_h:.2f}")
    
    print(f"Final Cash: {cash:.2f}")
        
    trading_results = pd.DataFrame(results,
                                   columns=["Price_t", "E(P_t+h)", "Cash", "Coin Holdings", "Total Wealth"])
    
    trading_results["Case / Action"] = actions_history  

    trading_results = trading_results[[
        "Price_t", "E(P_t+h)", "Case / Action", "Cash", "Coin Holdings", "Total Wealth"
    ]]

        
    return total_wealth_history, buy_and_hold, trading_results
    
def run_strategy_long_short(predictions, actuals, starting_time = 960, starting_money = 1000, rule_index = 1, tol = 0, verbose = True):
    
    n_periods = predictions.shape[0]
    results = np.full((n_periods,5),np.nan)
    actions_history = []


    # Simple Trading Rule --- Run Trading Rule
        
    coin_holdings = 0
    cash_history = []
    total_wealth_history = []
    buy_and_hold = []
    case_t = None
        
    for t in range(n_periods):
        
        price_t = actuals[t, (actuals.shape[1]-1)]
        prediction_t_plus_h = predictions[t,rule_index]
        if t > 0:
            price_t_minus_1 = actuals[t-1, (actuals.shape[1]-1)]

            
        if t == 0:
            cash = starting_money # start with money
            total_wealth = starting_money
            coin_holdings = starting_money / price_t # placeholder position, will be updated below
            coin_0 = starting_money / price_t
            cash = 0
        
        # Determine the Case
        if prediction_t_plus_h > (price_t + tol):
            case_t = "Long"
            
        if prediction_t_plus_h < (price_t - tol):
            case_t = "Short"
            
        if prediction_t_plus_h <= (price_t + tol) and prediction_t_plus_h >= (price_t - tol):
            case_t = "Hold Cash"
    

        # Determine the Action
        if case_t == "Long":
            if t > 0:
                total_wealth = total_wealth + (coin_holdings * (price_t - price_t_minus_1))
            cash = total_wealth
            coin_holdings = cash / price_t
            cash = 0
            
        if case_t ==  "Short":
            if t > 0:
                total_wealth = total_wealth + (coin_holdings * (price_t - price_t_minus_1))
            cash = total_wealth
            coin_holdings = -1*(cash / price_t)
            cash = 0
            
        if case_t == "Hold Cash":
            if t > 0:
                total_wealth = total_wealth + (coin_holdings * (price_t - price_t_minus_1))
            cash = total_wealth
            #previous_position = np.sign(coin_holdings)
            #coin_holdings = previous_position*(cash / price_t)
            coin_holdings = 0
            #cash = 0
    
        if t == (n_periods-1):
            cash = total_wealth
            coin_holdings = 0
            total_wealth = cash
            
        cash_history.append(cash)
        total_wealth_history.append(total_wealth)
        buy_and_hold.append(coin_0 * price_t)
        actions_history.append(case_t)
        
        results[t,0] = price_t
        results[t,1] = prediction_t_plus_h
        results[t,2] = cash
        results[t,3] = coin_holdings
        results[t,4] = total_wealth
        
        
            
        if verbose == True and t % 1 == 0:
            print(f"Period: {t} / {n_periods}, Cash: {cash:.2f}, Coin Holdings: {coin_holdings:.4f}, Total Wealth: {total_wealth:.2f}, Price_t: {price_t:.2f}, E(Price_t+1): {prediction_t_plus_h:.2f}")
    
    print(f"Final Cash: {cash:.2f}")
        
    trading_results = pd.DataFrame(results,
                                   columns=["Price_t", "E(P_t+h)", "Cash", "Coin Holdings", "Total Wealth"])
    
    trading_results["Case / Action"] = actions_history  

    trading_results = trading_results[[
        "Price_t", "E(P_t+h)", "Case / Action", "Cash", "Coin Holdings", "Total Wealth"
    ]]

        
    return total_wealth_history, buy_and_hold, trading_results

def run_strategy_long_short_V2(predictions, actuals, starting_time = 960, starting_money = 1000, rule_index = 1, tol_percent = 0, verbose = True):
    
    n_periods = predictions.shape[0]
    results = np.full((n_periods,5),np.nan)
    actions_history = []


    # Simple Trading Rule --- Run Trading Rule
        
    coin_holdings = 0
    cash_history = []
    total_wealth_history = []
    buy_and_hold = []
    case_t = None
        
    for t in range(n_periods):
        
        price_t = actuals[t, (actuals.shape[1]-1)]
        prediction_t_plus_h = predictions[t,rule_index]
        if t > 0:
            price_t_minus_1 = actuals[t-1, (actuals.shape[1]-1)]

            
        if t == 0:
            cash = starting_money # start with money
            total_wealth = starting_money
            coin_holdings = starting_money / price_t # placeholder position, will be updated below
            coin_0 = starting_money / price_t
            cash = 0
        
        # Determine the Case
        if prediction_t_plus_h > (price_t * (1 + tol_percent)):
            case_t = "Long"
            
        if prediction_t_plus_h < (price_t * (1 - tol_percent)):
            case_t = "Short"
            
        if prediction_t_plus_h <= (price_t * (1 + tol_percent)) and prediction_t_plus_h >= (price_t * (1 - tol_percent)):
            case_t = "Hold Cash"
    

        # Determine the Action
        if case_t == "Long":
            if t > 0:
                total_wealth = total_wealth + (coin_holdings * (price_t - price_t_minus_1))
            cash = total_wealth
            coin_holdings = cash / price_t
            cash = 0
            
        if case_t ==  "Short":
            if t > 0:
                total_wealth = total_wealth + (coin_holdings * (price_t - price_t_minus_1))
            cash = total_wealth
            coin_holdings = -1*(cash / price_t)
            cash = 0
            
        if case_t == "Hold Cash":
            if t > 0:
                total_wealth = total_wealth + (coin_holdings * (price_t - price_t_minus_1))
            cash = total_wealth
            #previous_position = np.sign(coin_holdings)
            #coin_holdings = previous_position*(cash / price_t)
            coin_holdings = 0
            #cash = 0
    
        if t == (n_periods-1):
            cash = total_wealth
            coin_holdings = 0
            total_wealth = cash
            
        cash_history.append(cash)
        total_wealth_history.append(total_wealth)
        buy_and_hold.append(coin_0 * price_t)
        actions_history.append(case_t)
        
        results[t,0] = price_t
        results[t,1] = prediction_t_plus_h
        results[t,2] = cash
        results[t,3] = coin_holdings
        results[t,4] = total_wealth
        
        
            
        if verbose == True and t % 1 == 0:
            print(f"Period: {t} / {n_periods}, Cash: {cash:.2f}, Coin Holdings: {coin_holdings:.4f}, Total Wealth: {total_wealth:.2f}, Price_t: {price_t:.2f}, E(Price_t+1): {prediction_t_plus_h:.2f}")
    
    print(f"Final Cash: {cash:.2f}")
        
    trading_results = pd.DataFrame(results,
                                   columns=["Price_t", "E(P_t+h)", "Cash", "Coin Holdings", "Total Wealth"])
    
    trading_results["Case / Action"] = actions_history  

    trading_results = trading_results[[
        "Price_t", "E(P_t+h)", "Case / Action", "Cash", "Coin Holdings", "Total Wealth"
    ]]

        
    return total_wealth_history, buy_and_hold, trading_results

    

def calculate_annual_sharpe_ratio(returns, annual_risk_free_rate=0.0, data_frequency = 365):
    """
    Calculate Sharpe Ratio.
    
    Args:
    - returns: Array of periodic returns.
    - risk_free_rate: Risk-free rate of return (expressed in annual terms).
    - data_frequency: number of periods in a year (default is daily data)
    
    Returns:
    - Sharpe Ratio (float).
    """
    
    frequency_adjusted_risk_free_rate = ((1+annual_risk_free_rate)**(1/data_frequency))-1
    
    excess_returns = returns - frequency_adjusted_risk_free_rate
    periodic_sharpe = np.mean(excess_returns) / np.std(returns)
    annual_sharpe = periodic_sharpe * np.sqrt(data_frequency)
    return annual_sharpe

def calculate_annual_sharpe_ratio_V2(returns, annual_risk_free_rate=0.0, data_frequency = 365):
    """
    Calculate Sharpe Ratio.
    
    Args:
    - returns: Array of periodic returns.
    - risk_free_rate: Risk-free rate of return (expressed in annual terms).
    - data_frequency: number of periods in a year (default is daily data)
    
    Returns:
    - Sharpe Ratio (float).
    """
    
    annual_returns = np.mean(returns) * data_frequency
    annual_excess_returns = annual_returns - annual_risk_free_rate

    annual_stdev = np.std(returns) * np.sqrt(data_frequency)
    
    annual_sharpe = annual_excess_returns / annual_stdev
    
    return annual_sharpe

