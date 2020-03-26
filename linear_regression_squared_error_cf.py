# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:08:23 2020

@author: btgl1e14
"""

# Linear Regression gradient descent with a squared error cost function

# Use find_theta_values to find the minimised parameters for the linear regression model h(x) = theta0 + theta1 * x

# Test variables
alpha = 0.06
x_values = [1,2,3,4,5]
y_values = [1,2,3,4,5]

# Define hypothesis for regression
def hypothesis(theta0, theta1, x):
    hypothesis = theta0 + theta1 * x
    return hypothesis

# Split up various parts  of the gradient descent algorithm
def sum_hypotheses_theta0(theta0, theta1, x_values, y_values):
    sum_hypotheses_theta0 = 0
    for i, x in enumerate(x_values):
        sum_hypotheses_theta0 = sum_hypotheses_theta0 + (hypothesis(theta0, theta1, x) - y_values[i])
    return sum_hypotheses_theta0
                                                         
def sum_hypotheses_theta1(theta0, theta1, x_values, y_values):
    sum_hypotheses_theta1 = 0
    for i, x in enumerate(x_values):
        sum_hypotheses_theta1 = sum_hypotheses_theta1 + (hypothesis(theta0, theta1, x) - y_values[i]) * x
    return sum_hypotheses_theta1        

def update_theta0(theta0, theta1, alpha, x_values, y_values):
    m = len(x_values)
    theta0_temp = theta0 - (alpha * (1 / m) * sum_hypotheses_theta0(theta0, theta1, x_values, y_values))
    return theta0_temp

def update_theta1(theta0, theta1, alpha, x_values, y_values):
    m = len(x_values)
    theta1_temp = theta1 - (alpha * (1 / m) * sum_hypotheses_theta1(theta0, theta1, x_values, y_values))
    return theta1_temp

# Iterating until convergence (if does not converge, alpha is too large)
def find_theta_values(x_values, y_values, alpha, theta0_init = 0, theta1_init = 0):
    theta0 = theta0_init
    theta1 = theta1_init
    # Just to initialise and ensure they are not equal to thetas
    extheta0 = theta0 - 1
    extheta1 = theta1 - 1
    theta0_temp = 0
    theta1_temp = 0
    #count = 0
    while theta0 != extheta0 and theta1 != extheta1:
        extheta0 = theta0
        extheta1 = theta1
        theta0_temp = update_theta0(theta0, theta1, alpha, x_values, y_values) 
        theta1_temp = update_theta1(theta0, theta1, alpha, x_values, y_values)
        theta0 = theta0_temp
        theta1 = theta1_temp
        # print(str(theta0) + " and " + str(theta1))
        #count += 1
        #print(count)
    return [theta0, theta1]

    
