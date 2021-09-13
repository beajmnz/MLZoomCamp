#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:55:22 2021

@author: Bea Jimenez <bjimenez@gmail.com>
"""

import numpy as np
import pandas as pd

# Question 1
#
# What's the version of NumPy that you installed?

print(np.__version__)

# Question 2
#
# What's the version of Pandas?

print(pd.__version__)

# Question 3
#
# What's the average price of BMW cars in the dataset?

url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv"
cars = pd.read_csv(url)

cars[cars.Make == "BMW"].MSRP.mean()

# Question 4

# Select a subset of cars after year 2015 (inclusive, i.e. 2015 and after). How many of them have missing values for Engine HP?

cars[cars.Year >= 2015]["Engine HP"].isnull().sum()

# Question 5

#   Calculate the average "Engine HP" in the dataset.
#    Use the fillna method and to fill the missing values in "Engine HP" with the mean value from the previous step.
#    Now, calcualte the average of "Engine HP" again.
#    Has it changed?

cars["Engine HP"].isnull().sum()  # 69 missing values

avgHP = round(cars["Engine HP"].mean())
cars[cars["Engine HP"].isnull()] = avgHP

cars["Engine HP"].isnull().sum()  # 0 missing values
round(cars["Engine HP"].mean()) == avgHP

# Question 6

#    Select all the "Rolls-Royce" cars from the dataset.
#    Select only columns "Engine HP", "Engine Cylinders", "highway MPG".
#    Now drop all duplicated rows using drop_duplicates method (you should get a dataframe with 7 rows).
#    Get the underlying NumPy array. Let's call it X.
#    Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T.
#        Let's call the result XTX.
#    Invert XTX.
#    What's the sum of all the elements of the result?

# Hint: if the result is negative, re-read the task one more time

cars[cars.Make == "Rolls-Royce"]
cars[cars.Make == "Rolls-Royce"][["Engine HP", "Engine Cylinders", "highway MPG"]]
X = cars[cars.Make == "Rolls-Royce"][
    ["Engine HP", "Engine Cylinders", "highway MPG"]
].drop_duplicates()
Xnp = np.array(X)
Xnp.T

XTX = np.multiply(Xnp, Xnp.T)

# Questions 7

#    Create an array y with values [1000, 1100, 900, 1200, 1000, 850, 1300].
#    Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
#    What's the value of the first element of w?.
