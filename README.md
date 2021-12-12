# Project Title

This is the Algorithmic Trading Application for Challenge 14! The project starts with a lot of the data prep already done for me.
I import a CSV file and slice it to just have the date as an index and daily closes.  I create a percentage change column
and call it actual returns.  After that I create two columns that contain the 4 day and 100 day rolling simple moving averages.
I call them SMA_Fast and SMA_Slow, respectively.  I then create a signal column that is 1 when the actual return column for date i is greater than 0, and -1 when it is less than 0.  I create a strategy returns column by multiplying actual returns by the signal column that I created previously.  Finally, I create a plot of the cumulative returns by running a cumsum() like so:
(1 + signals_df['Strategy Returns']).cumprod().plot().

After the above is done I create a training and testing dataset.  Most important here I then specify the beginning and end times
for both the training and dataset.  The training timeframe I define as the beginning of the data plus three months.
I then tweak this training time period and try beginning plus 6 months. 

After the timeframes are defined I scale the data using StandardScaler.  I fit and transform the data, and then the fun begins.


## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge.

* Explain the purpose of the analysis.
    
    The idea behind this challenge is to see how the machine learning algorithms will do compared to when we use the small training time period to when we unleash them on the testing time period.  We then want to compare them against each other by tweaking training time periods and SMA windows.  I begin with a SVM style learning method, and then I do a logistic regression model. After the initial results I do the analysis again but changing the training dataframe time period. I go from 3 months of training data to 6 months of training data.  This changes the number of data points for the testing data points as well.  I also change the SMA windows from 4 and 100, to 8 and 50.  For my final tweak I change both the SMA windows to 8 and 50, and I also extend the training period to 6 months from the original 3 months. The comparisons and the results are below.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

Just to define real quick what is Accuracy, precision, and recall I think would be helpful.

Accuracy = (number of True Positives + number of True Negatives) / Number of all datapoints (True and False Positives and Negatives)

Precision = (number of true positives) / (number of true positives + number of false positives)
So this number answers of all the positives you got, how many were correctly identified as positives?

Recall = (number of true positives) / (number of true positives + number of false negatives)
This metric asks of all the real world positives that were in the dataset, how many of them did the model identify?

* Original Baseline Model (3 months of training)

              precision    recall  f1-score   support

        -1.0       0.43      0.04      0.07      1804
         1.0       0.56      0.96      0.71      2288

    accuracy                           0.55      4092
   macro avg       0.49      0.50      0.39      4092
weighted avg       0.50      0.55      0.43      4092

* First update of baseline using LogisticRegression (still 3 months of training).

                precision    recall  f1-score   support

        -1.0       0.44      0.33      0.38      1804
         1.0       0.56      0.66      0.61      2288

    accuracy                           0.52      4092
   macro avg       0.50      0.50      0.49      4092
weighted avg       0.51      0.52      0.51      4092


* Original Baseline Model but with 6 months of training

                 precision    recall  f1-score   support

        -1.0       0.44      0.02      0.04      1732
         1.0       0.56      0.98      0.71      2211

    accuracy                           0.56      3943
   macro avg       0.50      0.50      0.38      3943
weighted avg       0.51      0.56      0.42      3943



* LogisticRegression Model but now using 6 months of training data.

                precision    recall  f1-score   support

        -1.0       0.52      0.03      0.06      1732
         1.0       0.56      0.98      0.71      2211

    accuracy                           0.56      3943
   macro avg       0.54      0.50      0.39      3943
weighted avg       0.54      0.56      0.43      3943

* Baseline Model with 3 months of training but new SMA windows of 8 and 50
(The model seemed to break! I got a division by zero error and nothing would plot. You can see the -1 row is unhappy with 0s)

                precision    recall  f1-score   support

        -1.0       0.00      0.00      0.00      1826
         1.0       0.56      1.00      0.72      2321

    accuracy                           0.56      4147
   macro avg       0.28      0.50      0.36      4147
weighted avg       0.31      0.56      0.40      4147

* LogisticRegression Model with 3 months of training but new SMA windows of 8 and 50

                precision    recall  f1-score   support

        -1.0       0.44      0.23      0.30      1826
         1.0       0.56      0.77      0.65      2321

    accuracy                           0.53      4147
   macro avg       0.50      0.50      0.47      4147
weighted avg       0.50      0.53      0.49      4147

* Baseline Model with 6 months of training but new SMA windows of 8 and 50
(The model seemed to break again! I got a division by zero error and nothing would plot. You can see the -1 row is unhappy with 0s)
You can see I have fewer data at 4001 vs 4147 above with only 3 months of training data

                 precision    recall  f1-score   support

        -1.0       0.00      0.00      0.00      1757
         1.0       0.56      1.00      0.72      2244

    accuracy                           0.56      4001
   macro avg       0.28      0.50      0.36      4001
weighted avg       0.31      0.56      0.40      4001

* LogisticRegression Model with 6 months of training but new SMA windows of 8 and 50

                 precision    recall  f1-score   support

        -1.0       0.43      0.06      0.11      1757
         1.0       0.56      0.94      0.70      2244

    accuracy                           0.55      4001
   macro avg       0.49      0.50      0.40      4001
weighted avg       0.50      0.55      0.44      4001

## Summary

*  We had eight scenarios of the machine learning models:
1. Baseline model with 3 months of training
2. Baseline model with 6 months of training
3. Baseline model with 3 months of training and new SMA windows of 8 and 50
4. Baseline model with 6 months of training and new SMA windows of 8 and 50
5. Logistic model with 3 months of training
6. Logistic model with 6 months of training
7. Logistic model with 3 months of training and new SMA windows of 8 and 50
8. Logistic model with 6 months of training and new SMA windows of 8 and 50

We can right away disqualify the Baseline models with new SMA windows. Both of these broke and couldn't handle
the shorts of the strategy from the error messages I got.  The logistic model with 6 months of training with and without new 
SMAs did very poorly. The Logistic model with 3 months of training and the original SMAs did well at first but then seemed to lose
money at the tail end of the dataframe. It was a bit erratic. The best models were the Baseline with 3 and 6 months of training, and the Logistic Model with 3 months of training and new SMAs.  Fascinating Challenge indeed!
---

## Technologies

I am using python version 3.7.10 and am importing the following from the built-in libraries and from functions i've created myself:

import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report

---

## Installation Guide

I have python version 3.7.10 and git version 2.33.0.windows.2 installed on a laptop running windows 10 pro.

I launch jupyter lab and run machine_learning_trading_bot.ipynb and that's it!


---

## Usage

Just upload the machine_learning_trading_bot.ipynb notebook and run the code. User can feel free to change any training or test timeframes used for charts or slicing if they want to study anything else.  They can also change the SMA windows if they like.


---

## Contributors
Just me, Paul Lopez.


---

## License
No licenses required. Just install everything for free, pull from my repository, and enjoy!
