#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 12:22:48 2022

@author: wei
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load the data
mobile_data_df = pd.read_csv('train.csv')
mobile_data_df.drop('three_g', inplace=True, axis=1)
mobile_data_arr = mobile_data_df.to_numpy()

# Use the best model to predict the test data
''' If there is no feature transformation, L1 penalty, C = 100000000'''
X = mobile_data_arr[:, :(mobile_data_arr.shape[1] - 1)]
y = mobile_data_arr[:, (mobile_data_arr.shape[1] - 1):]
y = y.flatten()

# Perform feature scaling
X = preprocessing.StandardScaler().fit_transform(X)

# Train, test, validation set split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

logreg = LogisticRegression(penalty='l1', solver = "saga",C = 100000000)
logreg.fit(X_train_val, y_train_val)
y_hat_train_val = logreg.predict(X_train_val)
new_train_acc = accuracy_score(y_train_val, y_hat_train_val)
# We get accuracy of 0.986111
print("New train accuracy", new_train_acc)

y_hat_test = logreg.predict(X_test)
test_acc = accuracy_score(y_test, y_hat_test)
# We get accuracy of 0.985
print("Test accuracy", test_acc)