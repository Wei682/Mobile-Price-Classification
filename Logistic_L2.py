#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 23:10:13 2022
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
logreg = LogisticRegression(penalty='l2',solver = "saga",C = 100000000)
logreg.fit(X_train, y_train)
c1 = logreg.coef_[0][0:3]
y_hat_train = logreg.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)
training_accuracies.append(train_acc)
print(train_acc)
y_hat_val = logreg.predict(X_val)
val_acc = accuracy_score(y_val, y_hat_val)
validation_accuracies.append(val_acc)
print(val_acc)


print("No feature transformation, C = 1")
logreg = LogisticRegression(penalty='l2', solver = "saga", C = 1)
logreg.fit(X_train, y_train)
c2 = logreg.coef_[0][0:3]
y_hat_train = logreg.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)
training_accuracies.append(train_acc)
print(train_acc)
y_hat_val = logreg.predict(X_val)
val_acc = accuracy_score(y_val, y_hat_val)
validation_accuracies.append(val_acc)
print(val_acc)

print("No feature transformation, C = 0.1")
logreg = LogisticRegression(penalty='l2', solver = "saga",C = 0.1)
logreg.fit(X_train, y_train)
c3 = logreg.coef_[0][0:3]
y_hat_train = logreg.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)
training_accuracies.append(train_acc)
print(train_acc)
y_hat_val = logreg.predict(X_val)
val_acc = accuracy_score(y_val, y_hat_val)
validation_accuracies.append(val_acc)
print(val_acc)

df = pd.DataFrame({
      'C': [100000000,1,0.1],
      'weights1': [c1[0],c2[0],c3[0]],
      'weights2': [c1[1],c2[1],c3[1]],
      'weights3': [c1[2],c2[2],c3[2]],
  })

# plot
plt.title('Without feature transformation, L2 penalty')
plt.plot('C', 'weights1', data=df, linestyle='-', marker='o')
plt.plot('C', 'weights2', data=df, linestyle='-', marker='o')
plt.plot('C', 'weights3', data=df, linestyle='-', marker='o')
plt.xlabel('value of C')
plt.ylabel('value of first three weights in class 0')
plt.show()

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
logreg = LogisticRegression(penalty='l2', solver = "saga",C = 100000000)
logreg.fit(X_train, y_train)
c4 = logreg.coef_[0][0:3]
y_hat_train = logreg.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)
training_accuracies.append(train_acc)
print(train_acc)
y_hat_val = logreg.predict(X_val)
val_acc = accuracy_score(y_val, y_hat_val)
validation_accuracies.append(val_acc)
print(val_acc)


print("Squared transformation, C = 1")
logreg = LogisticRegression(penalty='l2', solver = "saga",C = 1)
logreg.fit(X_train, y_train)
c5 = logreg.coef_[0][0:3]
y_hat_train = logreg.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)
training_accuracies.append(train_acc)
print(train_acc)
y_hat_val = logreg.predict(X_val)
val_acc = accuracy_score(y_val, y_hat_val)
validation_accuracies.append(val_acc)
print(val_acc)

print("Squared transformation, C = 0.1")
logreg = LogisticRegression(penalty='l2', solver = "saga",C = 0.1)
logreg.fit(X_train, y_train)
c6 = logreg.coef_[0][0:3]
y_hat_train = logreg.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)
training_accuracies.append(train_acc)
print(train_acc)
y_hat_val = logreg.predict(X_val)
val_acc = accuracy_score(y_val, y_hat_val)
validation_accuracies.append(val_acc)
print(val_acc)

df = pd.DataFrame({
      'C': [100000000,1,0.1],
      'weights4': [c4[0],c5[0],c6[0]],
      'weights5': [c4[1],c5[1],c6[1]],
      'weights6': [c4[2],c5[2],c6[2]],
  })

# plot
plt.title('Squared transformation, L2 penalty')
plt.plot('C', 'weights4', data=df, linestyle='-', marker='o')
plt.plot('C', 'weights5', data=df, linestyle='-', marker='o')
plt.plot('C', 'weights6', data=df, linestyle='-', marker='o')
plt.xlabel('value of C')
plt.ylabel('value of first three weights in class 0')
plt.show()


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
print("Cubic transformation, C = 100000000")
logreg = LogisticRegression(penalty='l2', solver = "saga",C = 100000000)
logreg.fit(X_train, y_train)
y_hat_train = logreg.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)
training_accuracies.append(train_acc)
print(train_acc)
y_hat_val = logreg.predict(X_val)
val_acc = accuracy_score(y_val, y_hat_val)
validation_accuracies.append(val_acc)
print(val_acc)


print("Cubic transformation, C = 1")
logreg = LogisticRegression(penalty='l2', solver = "saga",C = 1)
logreg.fit(X_train, y_train)
y_hat_train = logreg.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)
training_accuracies.append(train_acc)
print(train_acc)
y_hat_val = logreg.predict(X_val)
val_acc = accuracy_score(y_val, y_hat_val)
validation_accuracies.append(val_acc)
print(val_acc)

print("Cubic transformation, C = 0.1")
logreg = LogisticRegression(penalty='l2', solver = "saga",C = 0.1)
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
plt.title("With L2 penalty")
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