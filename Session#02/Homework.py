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
    Use the validation dataset to evaluate the models and compare the RMSE 
       of each option.
    Round the RMSE scores to 2 decimal digits using round(score, 2)
    Which option gives better RMSE?

"""

rev_mean = houses_train.reviews_per_month.mean()

houses_train2 = houses_train.copy()
houses_train3 = houses_train.copy()
houses_train2[houses_train2["reviews_per_month"].isnull()] = 0
houses_train3[houses_train3["reviews_per_month"].isnull()] = rev_mean


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]

w0, w = train_linear_regression(houses_train2.values, y_train)
y_pred = w0 + houses_train2.dot(w)

w0_, w_ = train_linear_regression(houses_train3.values, y_train)
y_pred2 = w0_ + houses_train3.dot(w_)


def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

round(rmse(y_train, y_pred),2)
round(rmse(y_train, y_pred2),2)

# they are the same (the feature is not relevant enough)


"""
Question 4

    Now let's train a regularized linear regression.
    For this question, fill the NAs with 0.
    Try different values of r from this list: [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10].
    Use RMSE to evaluate the model on the validation dataset.
    Round the RMSE scores to 2 decimal digits.
    Which r gives the best RMSE?

If there are multiple options, select the smallest r.
"""

houses_train[houses_train["reviews_per_month"].isnull()] = 0
houses_val[houses_val["reviews_per_month"].isnull()] = 0


def train_linear_regression_reg(X, y, r):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]

for r in [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    w0, w = train_linear_regression_reg(houses_train.values, y_train, r)
    y_pred = w0 + houses_val.dot(w)
    print(r, w0, round(rmse(y_val, y_pred),2))

# with two decimals I don't see any difference


"""
Question 5

    We used seed 42 for splitting the data. Let's find out how selecting the
      seed influences our score.
    Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
    For each seed, do the train/validation/test split with 60%/20%/20% 
       distribution.
    Fill the missing values with 0 and train a model without regularization.
    For each seed, evaluate the model on the validation dataset and collect the
       RMSE scores.
    What's the standard deviation of all the scores? To compute the standard 
       deviation, use np.std.
    Round the result to 3 decimal digits (round(std, 3))

    Note: Standard deviation shows how different the values are. If it's low, 
       then all values are approximately the same. If it's high, the values are
       different. If standard deviation of scores is low, then our model is 
       stable.

"""

scores = []

for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    np.random.seed(seed)
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
    
    houses_train[houses_train["reviews_per_month"].isnull()] = 0
    houses_val[houses_val["reviews_per_month"].isnull()] = 0
    
    w0, w = train_linear_regression(houses_train.values, y_train)
    y_pred = w0 + houses_val.dot(w)
    
    score = round(rmse(y_val, y_pred),3)
    print(seed, score)

    scores.append(score)

round(np.std(scores),3)

# 0.008

"""
Question 6

    Split the dataset like previously, use seed 9.
    Combine train and validation datasets.
    Fill the missing values with 0 and train a model with r=0.001.
    What's the RMSE on the test dataset?

"""

np.random.seed(9)
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

houses_full_train = pd.concat([houses_train,houses_val])
houses_full_train = houses_full_train.reset_index(drop=True)

houses_full_train[houses_full_train["reviews_per_month"].isnull()] = 0

y_full_train = np.concatenate([y_train,y_val])

w0, w = train_linear_regression_reg(houses_full_train.values, y_full_train, .001)
y_pred = w0 + houses_test.dot(w)
print(r, w0, round(rmse(y_test, y_pred),2))

# 0.65