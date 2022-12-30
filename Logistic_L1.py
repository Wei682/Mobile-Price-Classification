#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 12:15:00 2022

@author: wei
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the data
mobile_data_df = pd.read_csv('train.csv')
mobile_data_df.drop('three_g', inplace=True, axis=1)
mobile_data_arr = mobile_data_df.to_numpy()

''' If there is no feature transformation'''
X = mobile_data_arr[:, :(mobile_data_arr.shape[1] - 1)]
y = mobile_data_arr[:, (mobile_data_arr.shape[1] - 1):]
y = y.flatten()

# Perform feature scaling
X = preprocessing.StandardScaler().fit_transform(X)

# Train, test, validation set split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

training_accuracies = []
validation_accuracies = []
# Choose different regularization weights
print("No feature transformation, C = 100000000")
logreg = LogisticRegression(penalty='l1',solver = "saga",C = 100000000)
logreg.fit(X_train, y_train)
y_hat_train = logreg.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)
training_accuracies.append(train_acc)
print(train_acc)
y_hat_val = logreg.predict(X_val)
val_acc = accuracy_score(y_val, y_hat_val)
validation_accuracies.append(val_acc)
print(val_acc)

print("No feature transformation, C = 1")
logreg = LogisticRegression(penalty='l1', solver = "saga", C = 1)
logreg.fit(X_train, y_train)
y_hat_train = logreg.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)
training_accuracies.append(train_acc)
print(train_acc)
y_hat_val = logreg.predict(X_val)
val_acc = accuracy_score(y_val, y_hat_val)
validation_accuracies.append(val_acc)
print(val_acc)

print("No feature transformation, C = 0.1")
logreg = LogisticRegression(penalty='l1', solver = "saga",C = 0.1)
logreg.fit(X_train, y_train)
y_hat_train = logreg.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)
training_accuracies.append(train_acc)
print(train_acc)
y_hat_val = logreg.predict(X_val)
val_acc = accuracy_score(y_val, y_hat_val)
validation_accuracies.append(val_acc)
print(val_acc)


'''If all feature values are squared'''
X = mobile_data_arr[:, :(mobile_data_arr.shape[1] - 1)] ** 2
y = mobile_data_arr[:, (mobile_data_arr.shape[1] - 1):]
y = y.flatten()

# Perform feature scaling
X = preprocessing.StandardScaler().fit_transform(X)

# Train, test, validation set split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Choose different regularization weights
print("Squared transformation, C = 100000000")
logreg = LogisticRegression(penalty='l1', solver = "saga",C = 100000000)
logreg.fit(X_train, y_train)
y_hat_train = logreg.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)
training_accuracies.append(train_acc)
print(train_acc)
y_hat_val = logreg.predict(X_val)
val_acc = accuracy_score(y_val, y_hat_val)
validation_accuracies.append(val_acc)
print(val_acc)


print("Squared transformation, C = 1")
logreg = LogisticRegression(penalty='l1', solver = "saga",C = 1)
logreg.fit(X_train, y_train)
y_hat_train = logreg.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)
training_accuracies.append(train_acc)
print(train_acc)
y_hat_val = logreg.predict(X_val)
val_acc = accuracy_score(y_val, y_hat_val)
validation_accuracies.append(val_acc)
print(val_acc)

print("Squared transformation, C = 0.1")
logreg = LogisticRegression(penalty='l1', solver = "saga",C = 0.1)
logreg.fit(X_train, y_train)
y_hat_train = logreg.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)
training_accuracies.append(train_acc)
print(train_acc)
y_hat_val = logreg.predict(X_val)
val_acc = accuracy_score(y_val, y_hat_val)
validation_accuracies.append(val_acc)
print(val_acc)
 

'''If cubic transformation'''
X = mobile_data_arr[:, :(mobile_data_arr.shape[1] - 1)] ** 3
y = mobile_data_arr[:, (mobile_data_arr.shape[1] - 1):]
y = y.flatten()

# Perform feature scaling
X = preprocessing.StandardScaler().fit_transform(X)

# Train, test, validation set split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Choose different regularization weights
print("Squared transformation, C = 100000000")
logreg = LogisticRegression(penalty='l1', solver = "saga",C = 100000000)
logreg.fit(X_train, y_train)
y_hat_train = logreg.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)
training_accuracies.append(train_acc)
print(train_acc)
y_hat_val = logreg.predict(X_val)
val_acc = accuracy_score(y_val, y_hat_val)
validation_accuracies.append(val_acc)
print(val_acc)


print("Squared transformation, C = 1")
logreg = LogisticRegression(penalty='l1', solver = "saga",C = 1)
logreg.fit(X_train, y_train)
y_hat_train = logreg.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)
training_accuracies.append(train_acc)
print(train_acc)
y_hat_val = logreg.predict(X_val)
val_acc = accuracy_score(y_val, y_hat_val)
validation_accuracies.append(val_acc)
print(val_acc)

print("Squared transformation, C = 0.1")
logreg = LogisticRegression(penalty='l1', solver = "saga",C = 0.1)
logreg.fit(X_train, y_train)
y_hat_train = logreg.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)
training_accuracies.append(train_acc)
print(train_acc)
y_hat_val = logreg.predict(X_val)
val_acc = accuracy_score(y_val, y_hat_val)
validation_accuracies.append(val_acc)
print(val_acc)


# Plot the accuracies
# creating the dataset
# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))
 
# Set position of bar on X axis
br1 = np.arange(len(training_accuracies))
br2 = [x + barWidth for x in br1]
 
# Make the plot
plt.bar(br1, training_accuracies, color ='skyblue', width = barWidth,
        edgecolor ='grey', label ='Training accuracy')
plt.bar(br2, validation_accuracies, color ='seagreen', width = barWidth,
        edgecolor ='grey', label ='Validation accuracy')

 
# Adding Xticks
plt.title("With L1 penalty")
plt.xlabel('Logistic Model', fontweight ='bold', fontsize = 15)
plt.ylabel('Accuracies', fontweight ='bold', fontsize = 15)
plt.ylim([0.8,1.0])
plt.xticks([r + barWidth for r in range(len(training_accuracies))],
        ['Case1', 'Case2', 'Case3', 'Case4', 'Case5', 'Case6', 'Case7', 'Case8', 'Case9'], fontsize = 12)
plt.yticks(fontsize = 12)
'''
Case 1 to 9:
No_FT,C=100000000', 'No_FT,C=1', 'No_FT,C=0.1', 'Squared FT,C=100000000', 'Squared FT,C=1', 'Squared FT,C=0.1', 'Cubic FT,C=100000000', 'Cubic FT,C=1', 'Cubuic FT,C=0.1'
'''
 
plt.legend()
plt.show()
