#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 12:07:08 2021

@author: Bea Jimenez @beajmnz
https://twitter.com/beajmnz/status/1444313974464471040?s=20
"""

"""
4.10 Homework

Use this notebook (https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/04-evaluation/homework-4-starter.ipynb) as a starter

We'll use the credit scoring dataset:

    https://github.com/gastonstat/CreditScoring
    Also available here (https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-06-trees/CreditScoring.csv)

Preparation

    Execute the preparation code from the starter notebook

"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-06-trees/CreditScoring.csv"
df = pd.read_csv(url)
df.columns = df.columns.str.lower()
df.head()

""" 
Some of the features are encoded as numbers. Use the following code to de-code 
   them:
"""

status_values = {1: "ok", 2: "default", 0: "unk"}

df.status = df.status.map(status_values)


home_values = {
    1: "rent",
    2: "owner",
    3: "private",
    4: "ignore",
    5: "parents",
    6: "other",
    0: "unk",
}

df.home = df.home.map(home_values)

marital_values = {
    1: "single",
    2: "married",
    3: "widow",
    4: "separated",
    5: "divorced",
    0: "unk",
}

df.marital = df.marital.map(marital_values)

records_values = {1: "no", 2: "yes", 0: "unk"}

df.records = df.records.map(records_values)

job_values = {1: "fixed", 2: "partime", 3: "freelance", 4: "others", 0: "unk"}

df.job = df.job.map(job_values)

"""
Prepare the numerical variables:
"""

for c in ["income", "assets", "debt"]:
    df[c] = df[c].replace(to_replace=99999999, value=0)

"""
Remove clients with unknown default status
"""

df = df[df.status != "unk"].reset_index(drop=True)

"""
Create the target variable
"""

df["default"] = (df.status == "default").astype(int)
del df["status"]

"""
    Split the dataset into 3 parts: train/validation/test with 60%/20%/20% 
       distribution. Use train_test_split funciton for that with random_state=1
"""

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.default.values
y_val = df_val.default.values
y_test = df_test.default.values

del df_train["default"]
del df_val["default"]
del df_test["default"]


"""
Question 1

ROC AUC could also be used to evaluate feature importance of numerical variables.

Let's do that

    For each numerical variable, use it as score and compute AUC with the 
       "default" variable
    Use the training dataset for that

If your AUC is < 0.5, invert this variable by putting "-" in front

(e.g. -df_train['expenses'])

AUC can go below 0.5 if the variable is negatively correlated with the target 
varialble. You can change the direction of the correlation by negating this variable - then negative correlation becomes positive.

Which numerical variable (among the following 4) has the highest AUC?

    seniority
    time
    income
    debt
"""
scores = []
numeric_feat = ["seniority","time","age","expenses","income","assets","debt","amount","price"]

for feat in numeric_feat:
    fpr, tpr, thresholds = roc_curve(y_train,df_train[feat].values)
    feat_imp = auc(fpr, tpr)
    if feat_imp < 0.5:
        fpr, tpr, thresholds = roc_curve(y_train,-df_train[feat].values)
        feat_imp = auc(fpr, tpr)
        
    scores.append([feat, feat_imp])

df_scores = pd.DataFrame(scores, columns = ["feat","auc"])
df_scores.sort_values(by="auc", ascending=False)

# the highest AUC is in seniority

"""

Training the model

From now on, use these columns only:

['seniority', 'income', 'assets', 'records', 'job', 'home']

Apply one-hot-encoding using DictVectorizer and train the logistic regression 
with these parameters:

LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)

"""

model_cols = ['seniority', 'income', 'assets', 'records', 'job', 'home']

dv = DictVectorizer(sparse=False)

train_dict = df_train[model_cols].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[model_cols].to_dict(orient='records')
X_val = dv.transform(val_dict)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:,1]



"""
Question 2

What's the AUC of this model on the validation dataset? (round to 3 digits)

    0.512
    0.612
    0.712
    0.812
"""

model_auc = roc_auc_score(y_val, y_pred)

print(model_auc.round(3))

# AUC is 0.812

"""
Question 3

Now let's compute precision and recall for our model.

    Evaluate the model on all thresholds from 0.0 to 1.0 with step 0.01
    For each threshold, compute precision and recall
    Plot them

At which threshold precision and recall curves intersect?

    0.2
    0.4
    0.6
    0.8
"""

scores = []

thresholds = np.linspace(0, 1, 101)

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    
    scores.append((t, tp, fp, fn, tn))

columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores['prec'] = df_scores.tp / (df_scores.tp + df_scores.fp)
df_scores['rec'] = df_scores.tp / (df_scores.tp + df_scores.fn)

plt.plot(df_scores.threshold, df_scores['prec'], label='Precision')
plt.plot(df_scores.threshold, df_scores['rec'], label='Recall')
plt.legend()

# they intersect at around 0.4

"""
Question 4

Precision and recall are conflicting - when one grows, the other goes down. 
  That's why they are often combined into the F1 score - a metrics that takes 
  into account both

This is the formula for computing F1:

F1 = 2 * P * R / (P + R)

Where P is precision and R is recall.

Let's compute F1 for all thresholds from 0.0 to 1.0 with increment 0.01

At which threshold F1 is maximal?

    0.1
    0.3
    0.5
    0.7
"""

df_scores['F1'] = 2 * df_scores.prec * df_scores.rec / (df_scores.prec + df_scores.rec)

df_scores.sort_values(by="F1", ascending=False).threshold

# maximum F1 is at threshold 0.3

"""
Question 5

Use the KFold class from Scikit-Learn to evaluate our model on 5 different folds:

KFold(n_splits=5, shuffle=True, random_state=1)

    Iterate over different folds of df_full_train
    Split the data into train and validation
    Train the model on train with these parameters: 
        LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    Use AUC to evaluate the model on validation

How large is standard devidation of the AUC scores across different folds?

    0.001
    0.014
    0.09
    0.14

"""

def train(df_train, y_train, C=1.0):
    dicts = df_train[model_cols].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model

def predict(df, dv, model):
    dicts = df[model_cols].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.default.values
    y_val = df_val.default.values

    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))

# 0.826 +- 0.000
# 0.826 +- 0.000
# 0.813 +- 0.017
# 0.816 +- 0.016
# 0.814 +- 0.015

# The closest answer is 0.014 

"""
Question 6

Now let's use 5-Fold cross-validation to find the best parameter C

    Iterate over the following C values: [0.01, 0.1, 1, 10]
    Initialize KFold with the same parameters as previously
    Use these parametes for the model: 
        LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    Compute the mean score as well as the std (round the mean and std to 3 
                                               decimal digits)

Which C leads to the best mean score?

    0.01
    0.1
    1
    10

If you have ties, select the score with the lowest std. If you still have ties,
  select the smallest C
"""

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for C in [0.01, 0.1, 1, 10]:

    scores = []
    
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]
    
        y_train = df_train.default.values
        y_val = df_val.default.values
    
        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)
    
        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))
    
    
#C=0.01 0.814 +- 0.015
#C=0.1 0.814 +- 0.015
#C=1 0.814 +- 0.015
#C=10 0.814 +- 0.015

# they are all the same, so the lowest C is 0.01
