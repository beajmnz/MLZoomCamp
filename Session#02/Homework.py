# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:33:25 2021

@author: Bea Jimenez @beajmnz
"""

"""
Dataset

In this homework, we will use the New York City Airbnb Open Data. You can take 
it from Kaggle or download from here if you don't want to sign up to Kaggle.

The goal of this homework is to create a regression model for prediction 
apartment prices (column 'price').
"""

import numpy as np
import pandas as pd
import seaborn as sns



"""
EDA

    Load the data.
    Look at the price variable. Does it have a long tail?

"""

url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/AB_NYC_2019.csv"
houses = pd.read_csv(url)
houses.head()
houses.columns
sns.histplot(houses.price)

# Yes, it has a long tail

"""
Features

For the rest of the homework, you'll need to use only these columns:

    'latitude',
    'longitude',
    'price',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'calculated_host_listings_count',
    'availability_365'

Select only them.
"""

houses = houses[
    ['latitude',
    'longitude',
    'price',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'calculated_host_listings_count',
    'availability_365']
]


"""
Question 1

Find a feature with missing values. How many missing values does it have?
"""

houses.isnull().sum()

# reviews_per_month is the only feature with missing values. 10052.


"""
Question 2

What's the median (50% percentile) for variable 'minimum_nights'?
Split the data

    Shuffle the initial dataset, use seed 42.
    Split your data in train/val/test sets, with 60%/20%/20% distribution.
    Make sure that the target value ('price') is not in your dataframe.
    Apply the log transformation to the price variable using the np.log1p() function.

"""

houses.minimum_nights.median()

# 3


np.random.seed(42)
idx = np.arange(len(houses))
np.random.shuffle(idx)

n_train = int(len(houses) * .6)
n_val = int(len(houses) * .2)
n_test = int(len(houses)) - n_train - n_val

houses_train = houses.iloc[idx[n_train:]]
houses_val = houses.iloc[idx[n_train:n_train+n_val]]
houses_test  = houses.iloc[idx[n_train+n_val:]]

houses_train = houses_train.reset_index(drop=True)
houses_val = houses_val.reset_index(drop=True)
houses_test = houses_test.reset_index(drop=True)

y_train = np.log1p(houses_train.price.values)
y_val = np.log1p(houses_val.price.values)
y_test = np.log1p(houses_test.price.values)

del houses_train['price']
del houses_val['price']
del houses_test['price']


"""
Question 3

    We need to deal with missing values for the column from Q1.
    We have two options: fill it with 0 or with the mean of this variable.
    Try both options. For each, train a linear regression model without 
       regularization using the code from the lessons.
    For computing the mean, use the training only!
    Use the validation dataset to evaluate the models and compare the RMSE of each option.
    Round the RMSE scores to 2 decimal digits using round(score, 2)
    Which option gives better RMSE?

"""


