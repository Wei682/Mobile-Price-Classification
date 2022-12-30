#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 15:33:47 2022
@author: wei
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 4))
    for i in range(len(y)):
        y_vect[i, int(y[i])] = 1
    return y_vect

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_deriv(z):
    return sigmoid(z) * (1 - sigmoid(z))

# The Xavier initialization
def Xavier_setup(nn_structure):
    W = {} #creating a dictionary i.e. a set of key: value pairs
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = np.random.random_sample((nn_structure[l], nn_structure[l-1])) * np.sqrt(1/nn_structure[l-1]) 
        b[l] = np.random.random_sample((nn_structure[l],))
    return W, b

def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b


def feed_forward(x, W, b):
    a = {1: x} # create a dictionary for holding the a values for all levels
    z = { } # create a dictionary for holding the z values for all the layers
    for l in range(1, len(W) + 1): # for each layer
        node_in = a[l]
        z[l+1] = W[l].dot(node_in) + b[l]  # z^(l+1) = W^(l)*a^(l) + b^(l)
        a[l+1] = sigmoid(z[l+1]) # a^(l+1) = f(z^(l+1))
    return a, z

def calculate_out_layer_delta(y, a_out, z_out):
    # delta^(nl) = -(y_i - a_i^(nl)) * f'(z_i^(nl))
    return -(y-a_out) * sigmoid_deriv(z_out) 


def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * sigmoid_deriv(z_l)

''' For L1 regularization'''
def calc_abs_der(W):
    res = np.zeros((len(W), len(W[0])))
    for i in range(len(W)):
        for j in range(len(W[0])):
            if W[i][j] < 0:
                res[i][j] = -1
            else:
                res[i][j] = 1
    return res

''' For L1 regularization'''
def calc_abs_sum(W):
    res = 0
    for i in range(len(W)):
        for j in range(len(W[0])):
            if W[i][j] < 0:
                res -= W[i][j]
            else:
                res += W[i][j]
    return res
                
def train_nn(nn_structure, X, y, iter_num=3000, alpha=2.5, regu_coeff = 0.00001):
    W, b = Xavier_setup(nn_structure)
    cnt = 0
    N = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        '''For regularization'''
        # J_weights tracks the regularization part of the cost function
        #J_weights = 0
        if cnt%100 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(N):
            delta = {}
            # perform the feed forward pass and return the stored a and z values, to be used in the
            # gradient descent step
            a, z = feed_forward(X[i, :], W, b)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], a[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-a[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(a[l][:,np.newaxis]))# np.newaxis increase the number of dimensions
                    tri_b[l] += delta[l+1]
        # perform the gradient descent step for the weights in each layer
        for l in range(len(nn_structure) - 1, 0, -1):
            '''For L2 regularization'''
            #W[l] += -alpha * (1.0/N * tri_W[l]) - alpha * regu_coeff * W[l]
            #b[l] += -alpha * (1.0/N * tri_b[l])
            #J_weights += sum(sum(W[l] ** 2)) * regu_coeff / 2
            '''For L1 regularization'''
            #W[l] += -alpha * (1.0/N * tri_W[l]) - alpha * regu_coeff * calc_abs_der(W[l])
            #b[l] += -alpha * (1.0/N * tri_b[l])
            #J_weights += calc_abs_sum(W[l]) * regu_coeff / 2
            ''' For no regularization'''
            W[l] += -alpha * (1.0/N * tri_W[l])
            b[l] += -alpha * (1.0/N * tri_b[l])
        # complete the average cost calculation
        '''For regularization'''
        #avg_cost = 1.0/N * avg_cost + J_weights
        ''' For no regularization'''
        avg_cost = 1.0/N * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func


def predict_y(W, b, X, n_layers):
    N = X.shape[0]
    y = np.zeros((N,))
    for i in range(N):
        a, z = feed_forward(X[i, :], W, b)
        y[i] = np.argmax(a[n_layers])
    return y

# Load the data
mobile_data_df = pd.read_csv('train.csv')
mobile_data_df.drop('three_g', inplace=True, axis=1)
mobile_data_arr = mobile_data_df.to_numpy()

''' No feature transformation '''
#X = mobile_data_arr[:, :(mobile_data_arr.shape[1] - 1)]
''' Squared feature transformation '''
#X = mobile_data_arr[:, :(mobile_data_arr.shape[1] - 1)] ** 2
''' Cubic feature transformation '''
X = mobile_data_arr[:, :(mobile_data_arr.shape[1] - 1)] ** 3
y = mobile_data_arr[:, (mobile_data_arr.shape[1] - 1):]
y = y.flatten()

# Perform feature scaling
X = preprocessing.StandardScaler().fit_transform(X)

# Train, test, validation set split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Convert to vectors
y_v_train = convert_y_to_vect(y_train)
y_v_val = convert_y_to_vect(y_val)
y_v_test = convert_y_to_vect(y_test)


training_accuracies = []
validation_accuracies = []

''' Three-layer'''
nn_structure = [19, 10, 4]
''' Four-layer'''
#nn_structure = [19, 13, 8, 4]
   
# train the NN
W, b = train_nn(nn_structure, X_train, y_v_train, 1500)

# plot the avg_cost_func
plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()

# print the prediction accuracies
y_train_pred = predict_y(W, b, X_train, len(nn_structure))
print('Prediction accuracy for training data is {}%'.format(accuracy_score(y_train, y_train_pred) * 100))
y_val_pred = predict_y(W, b, X_val, len(nn_structure))
print('Prediction accuracy for validation data is {}%'.format(accuracy_score(y_val, y_val_pred) * 100))