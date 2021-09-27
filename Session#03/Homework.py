# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 16:44:25 2021

@author: Bea Jimenez @beajmnz
https://twitter.com/beajmnz/status/1442163269670735874?s=20
"""

"""
Dataset

In this homework, we will continue the New York City Airbnb Open Data. You can 
take it from Kaggle or download from here if you don't want to sign up to Kaggle.

We'll keep working with the 'price' variable, and we'll transform it to a 
classification task.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/AB_NYC_2019.csv"
houses = pd.read_csv(url)
houses.head()
houses.columns


"""
Features

For the rest of the homework, you'll need to use the features from the previous 
homework with additional two 'neighbourhood_group' and 'room_type'. So the whole
 feature set will be set as follows:

    'neighbourhood_group',
    'room_type',
    'latitude',
    'longitude',
    'price',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'calculated_host_listings_count',
    'availability_365'

Select only them and fill in the missing values with 0.
"""

houses = houses[
    ['neighbourhood_group',
    'room_type',
    'latitude',
    'longitude',
    'price',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'calculated_host_listings_count',
    'availability_365']
]

houses.isnull().sum() # 10052 missing values in reviews_per_month
houses = houses.fillna(0)

"""
Question 1

What is the most frequent observation (mode) for the column 'neighbourhood_group'?
Split the data

    Split your data in train/val/test sets, with 60%/20%/20% distribution.
    Use Scikit-Learn for that (the train_test_split function) and set the seed to 42.
    Make sure that the target value ('price') is not in your dataframe.
    
"""

houses.neighbourhood_group.mode()  # Manhattan 
 
houses_full_train, houses_test = train_test_split(houses, test_size=.2, random_state=42)
houses_train, houses_val = train_test_split(houses_full_train, test_size=.25, random_state=42)
 
houses_train = houses_train.reset_index(drop=True)
houses_val = houses_val.reset_index(drop=True)
houses_test = houses_test.reset_index(drop=True)

y_train = houses_train.price.values
y_val = houses_val.price.values
y_test = houses_test.price.values

del houses_train['price']
del houses_val['price']
del houses_test['price']

 

"""
Question 2

    Create the correlation matrix for the numerical features of your train dataset.
        In a correlation matrix, you compute the correlation coefficient between 
        every pair of features in the dataset.
    What are the two features that have the biggest correlation in this dataset?

Example of a correlation matrix for the car price dataset:

Make price binary

    We need to turn the price variable from numeric into binary.
    Let's create a variable above_average which is 1 if the price is above 
       (or equal to) 152.

"""

correlation_matrix = houses_train.corr()

# 	latitude	longitude	minimum_nights	number_of_reviews	reviews_per_month	calculated_host_listings_count	availability_365
#longitude	0.08030088258319117	1.0	-0.060659674416984104	0.05508444903079403	0.13464215986450567	-0.11704068790956335	0.08366550097084335
#minimum_nights	0.02744122202734325	-0.060659674416984104	1.0	-0.07601980051963106	-0.12070278971361591	0.11864675413775373	0.13890125161844732
#number_of_reviews	-0.006245618672260882	0.05508444903079403	-0.07601980051963106	1.0	0.5903739015971663	-0.07316723252176341	0.17447711716588918
#reviews_per_month	-0.007159335582167729	0.13464215986450567	-0.12070278971361591	0.5903739015971663	1.0	-0.04876681579311577	0.16537591986288716
#calculated_host_listings_count	0.019375156668755916	-0.11704068790956335	0.11864675413775373	-0.07316723252176341	-0.04876681579311577	1.0	0.22591308547640596
#availability_365	-0.0058911861711353165	0.08366550097084335	0.13890125161844732	0.17447711716588918	0.16537591986288716	0.22591308547640596	1.0

correlation_matrix.unstack().sort_values(ascending=False)

# biggest correlation is between reviews_per_month and number_of_reviews

price_average = 152
above_average = np.where(y_train > price_average, 1, 0)
above_average_val = np.where(y_val > price_average, 1, 0)


"""
Question 3

    Calculate the mutual information score with the (binarized) price for the 
       two categorical variables that we have. Use the training set only.
    Which of these two variables has bigger score?
    Round it to 2 decimal digits using round(score, 2)

"""

round(mutual_info_score(houses_train.neighbourhood_group, above_average),2) # 0.05
round(mutual_info_score(houses_train.room_type, above_average),2) # 0.14


"""
Question 4

    Now let's train a logistic regression
    Remember that we have two categorical variables in the data. Include them 
       using one-hot encoding.
    Fit the model on the training dataset.
        To make sure the results are reproducible across different versions of 
        Scikit-Learn, fit the model with these parameters:
           model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
    Calculate the accuracy on the validation dataset and rount it to 2 decimal
       digits.
"""

dv = DictVectorizer(sparse=False)

train_dict = houses_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = houses_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
model.fit(X_train, above_average)

y_pred = model.predict_proba(X_val)[:,1]
model_acc = round((above_average_val == (y_pred >= .5)).mean(),2)  # .79


"""
Question 5

    We have 9 features: 7 numerical features and 2 categorical.
    Let's find the least useful one using the feature elimination technique.
    Train a model with all these features (using the same parameters as in Q4).
    Now exclude each feature from this set and train a model without it. Record
      the accuracy for each model.
    For each feature, calculate the difference between the original accuracy and
       the accuracy without the feature.
    Which of following feature has the smallest difference?
        neighbourhood_group
        room_type
        number_of_reviews
        reviews_per_month

    note: the difference doesn't have to be positive
"""

accuracies = []

for feat in houses_train.columns:
    houses_train_temp = houses_train.copy()
    del houses_train_temp[feat]
    
    houses_val_temp = houses_val.copy()
    del houses_val_temp[feat]
    
    train_dict = houses_train_temp.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    val_dict = houses_val_temp.to_dict(orient='records')
    X_val = dv.transform(val_dict)

    model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
    model.fit(X_train, above_average)

    y_pred = model.predict_proba(X_val)[:,1]
    acc = round((above_average_val == (y_pred >= .5)).mean(),4)
    accuracies.append(acc)

for i in range(len(houses_train.columns)):
    print (houses_train.columns[i], accuracies[i], round(model_acc - accuracies[i],3))
    

"""
Question 6

    For this question, we'll see how to use a linear regression model from 
       Scikit-Learn
    We'll need to use the original column 'price'. Apply the logarithmic 
       transformation to this column.
    Fit the Ridge regression model on the training data.
    This model has a parameter alpha. Let's try the following values: 
        [0, 0.01, 0.1, 1, 10]
    Which of these alphas leads to the best RMSE on the validation set? Round 
       your RMSE scores to 3 decimal digits.

If there are multiple options, select the smallest alpha.
"""


