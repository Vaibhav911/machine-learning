#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:17:05 2020

@author: vaibhav
"""

import numpy as np
import pandas as pd

data = pd.read_csv("./../data/data_banknote_authentication.txt",
                   names = ['x1', 'x2', 'x3', 'x4' ,'y'])

# Normalize data, expecpt class
data[['x1', 'x2', 'x3', 'x4']] = (
    data[['x1', 'x2', 'x3', 'x4']] - data[['x1', 'x2', 'x3', 'x4']].mean()) / data[['x1', 'x2', 'x3', 'x4']].std()

# Split data into test and training set, 20% 80% resp.
mask = np.random.rand(len(data)) < 0.8

train_data = data[mask]
test_data = data[~mask]

train_data.reset_index(inplace=True, drop=True)
test_data.reset_index(inplace=True, drop=True)

def calc_grad(data, wts):
    """
    For a given set of weights and data,
    calculates the gradient.
    """
    # Initlalize gradient to [0, 0, ..., 0]
    grad = pd.DataFrame([0, 0, 0, 0], index=['x1', 'x2', 'x3', 'x4'])
    for index, row in data.iterrows():
        # Xn is the feature vector for nth training pnt.
        Xn = row[['x1', 'x2', 'x3', 'x4']]
        Xn = pd.DataFrame({0: Xn})
        # Yn is predicted value for Xn
        Yn = sigma(wts, Xn)
        grad += (Yn[0][0] - row['y']) * Xn
    return grad

def sigma(wts, X):
    """
    Returns 
                  1 
  -----------------------------------
   (1 + e^(-1 * transpose(wts) * X))
   
    """
    a = wts.transpose().dot(X)
    denom = 1 + np.exp(-1 * a)
    return 1 / (denom)
        
def calc_wts(data, eta):
    """
    Calculate weights using logistic regresssion's gradient descent
    after a specified number of iterations.
    """
    # wts are initial value of weights.
    wts = pd.DataFrame([0, 0, 0, 0], index=['x1', 'x2', 'x3', 'x4'])
    for iteration in range(300):
        # wts = wts - ( eta * gradient )
        wts -= eta * calc_grad(data, wts)
        print(wts, calc_accur(test_data, wts))
    return wts

def calc_accur(data, wts):
    """
    For a given set of data and weights,
    calculates the accuracy of weights.
    """
    # Initialize number of correct preditions to 0
    correct_pred = 0
    for index, row in data.iterrows():
        # Xn is feature vector
        Xn = row[['x1', 'x2', 'x3', 'x4']]
        Xn = pd.DataFrame({0: Xn})
        # val is predicted value for Xn and given weights.
        val = sigma(wts, Xn)[0][0]
        if val < 0.5:
            prediction = 0
        else:
            prediction = 1
        if prediction == row['y']:
            correct_pred += 1
    return correct_pred / len(data)

def main():    
    # Calculate weights, and measure their accuracy.
    wts = calc_wts(test_data, 0.01)
    accuracy = calc_accur(data, wts)
    print(accuracy)
    return 0

if __name__ == "__main__":
    main()