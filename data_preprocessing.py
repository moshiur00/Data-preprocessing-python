# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 12:34:22 2018

@author: Md. Moshiur Rahman
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# Taking Care of Missing Data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])


#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_x = LabelEncoder()
x[:, 0] = label_encoder_x.fit_transform(x[:, 0])

onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()

label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)


# Spliting the Dataset into Training Set and Test Set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
