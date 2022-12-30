#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 22:20:26 2022

@author: wei
"""

import pandas as pd
from scipy.stats import chi2_contingency


mobile_data = pd.read_csv("train.csv")
cont_table = pd.crosstab(mobile_data['three_g'], mobile_data['price_range'])
print("Contingency table: ")
print(cont_table)

chi, p_value, dof, expected = chi2_contingency(cont_table)
print("")
print("p_value: ", p_value)