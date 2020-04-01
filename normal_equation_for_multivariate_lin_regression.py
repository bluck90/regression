# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:36:54 2020

@author: btgl1e14
"""

# Multiple Linear Regression using the Normal Equation to solve for values of theta

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset (using the dataset under "Multiple Linear Regression" from SuperDataScience)
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, 4].values

# Adding x[0] = 1 for each row
X = np.insert(X, [0], np.ones((50, 1)), axis=1)

# Applying normal equation to solve for theta
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Predicting y values from regression  (for test if splitting data)
y_pred = X.dot(theta)


"""
Xgraph = np.arange(1,51,1)

plt.scatter(Xgraph, y, color = 'red')
plt.scatter(Xgraph, y_pred, color = 'blue')
plt.show()
"""
