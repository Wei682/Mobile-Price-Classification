#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 09:38:55 2022

@author: wei
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

'''First, plot the accuracies of all the 18 cases'''    
# Plot the accuracies for no regularization
# Accuracies got from the models
training_accuracies = [0.991111,0.993333,0.942963,0.945185,0.908889,0.926667]
validation_accuracies = [0.964444,0.948889,0.92,0.906667,0.862222,0.875556]

barWidth = 0.25
fig = plt.subplots(figsize =(14, 8))
  
# Set position of bar on X axis
br1 = np.arange(len(training_accuracies))
br2 = [x + barWidth for x in br1]
  
# Make the plot
plt.bar(br1, training_accuracies, color ='skyblue', width = barWidth,
        edgecolor ='grey', label ='Training accuracy')
plt.bar(br2, validation_accuracies, color ='seagreen', width = barWidth,
        edgecolor ='grey', label ='Validation accuracy')

  
# Adding Xticks
plt.title('With no regularization')
plt.xlabel('Neural network', fontweight ='bold', fontsize = 18)
plt.ylabel('Accuracies', fontweight ='bold', fontsize = 18)
plt.ylim([0.8,1.0])
plt.xticks(ticks = [r + barWidth for r in range(len(training_accuracies))],
        labels=['No_FT, 3_layer', 'No_FT, 4_layer', 'Squared_FT, 3_layer', 'Squared_FT, 4_layer', 'Cubic_FT, 3_layer',' Cubic_FT, 3_layer'], fontsize = 12)
plt.yticks(fontsize = 12)
  
plt.legend()
plt.show()

# Plot the accuracies for L1 regularization
# Accuracies got from the models
training_accuracies = [0.991111,0.993333,0.942963,0.945185,0.908889,0.926667]
validation_accuracies = [0.964444,0.948889,0.920000,0.906667,0.862222,0.875556]

barWidth = 0.25
fig = plt.subplots(figsize =(14, 8))
  
# Set position of bar on X axis
br1 = np.arange(len(training_accuracies))
br2 = [x + barWidth for x in br1]
  
# Make the plot
plt.bar(br1, training_accuracies, color ='skyblue', width = barWidth,
        edgecolor ='grey', label ='Training accuracy')
plt.bar(br2, validation_accuracies, color ='seagreen', width = barWidth,
        edgecolor ='grey', label ='Validation accuracy')

  
# Adding Xticks
plt.title('With L1 regularization')
plt.xlabel('Neural network', fontweight ='bold', fontsize = 18)
plt.ylabel('Accuracies', fontweight ='bold', fontsize = 18)
plt.ylim([0.8,1.0])
plt.xticks(ticks = [r + barWidth for r in range(len(training_accuracies))],
        labels=['No_FT, 3_layer', 'No_FT, 4_layer', 'Squared_FT, 3_layer', 'Squared_FT, 4_layer', 'Cubic_FT, 3_layer',' Cubic_FT, 3_layer'], fontsize = 12)
plt.yticks(fontsize = 12)
  
plt.legend()
plt.show()

# Plot the accuracies for L2 regularization
# Accuracies got from the models
training_accuracies = [0.985926,0.996296,0.949630,0.963704,0.906667,0.914815]
validation_accuracies = [0.957778,0.951111,0.913333,0.9,0.855556,0.873333]

barWidth = 0.25
fig = plt.subplots(figsize =(14, 8))
  
# Set position of bar on X axis
br1 = np.arange(len(training_accuracies))
br2 = [x + barWidth for x in br1]
  
# Make the plot
plt.bar(br1, training_accuracies, color ='skyblue', width = barWidth,
        edgecolor ='grey', label ='Training accuracy')
plt.bar(br2, validation_accuracies, color ='seagreen', width = barWidth,
        edgecolor ='grey', label ='Validation accuracy')

  
# Adding Xticks
plt.title('With L2 regularization')
plt.xlabel('Neural network', fontweight ='bold', fontsize = 18)
plt.ylabel('Accuracies', fontweight ='bold', fontsize = 18)
plt.ylim([0.1,1.0])
plt.xticks(ticks = [r + barWidth for r in range(len(training_accuracies))],
        labels=['No_FT, 3_layer', 'No_FT, 4_layer', 'Squared_FT, 3_layer', 'Squared_FT, 4_layer', 'Cubic_FT, 3_layer',' Cubic_FT, 3_layer'], fontsize = 12)
plt.yticks(fontsize = 12)
  
plt.legend()
plt.show()

'''Next, use the case with the highest validation accuracy to predict the test data'''
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
        # J_weights tracks the regularization part of the cost function
        J_weights = 0
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
            W[l] += -alpha * (1.0/N * tri_W[l]) - alpha * regu_coeff * calc_abs_der(W[l])
            b[l] += -alpha * (1.0/N * tri_b[l])
            J_weights += calc_abs_sum(W[l]) * regu_coeff / 2
        # complete the average cost calculation
        avg_cost = 1.0/N * avg_cost + J_weights
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
X = mobile_data_arr[:, :(mobile_data_arr.shape[1] - 1)]
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
y_v_train_val = convert_y_to_vect(y_train_val)

training_accuracies = []
validation_accuracies = []

''' Three-layer'''
nn_structure = [19, 10, 4]

   
# train the NN
W, b, avg_cost = train_nn(nn_structure, X_train_val, y_v_train_val, 3000)

# print the new training accuracy
y_new_train_pred = predict_y(W, b, X_train_val, len(nn_structure))
print('Prediction accuracy for new training data is {}%'.format(accuracy_score(y_train_val, y_new_train_pred) * 100))

'''We got the training accuracy of 0.990556'''

# print the prediction accuracy
y_test_pred = predict_y(W, b, X_test, len(nn_structure))
print('Prediction accuracy for test data is {}%'.format(accuracy_score(y_test, y_test_pred) * 100))

'''We got the test accuracy of 0.975'''


