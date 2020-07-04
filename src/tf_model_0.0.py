#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 22:36:17 2020

Python 3.7.6

@author: mlongstreth
"""

# Import libraries
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import sys
import timeit

# Start timer to evaluate script efficiency
start = timeit.default_timer()

# Set file location
file = './model_data/acc_gyro_data.csv'

# Import data
df = pd.read_csv(file,
                 index_col=0)

# Create dataframe from timestamp information
new_index = df['timestamp'].to_frame()

# Retrieve number of rows per timestamp
timestamp_count = new_index['timestamp'].value_counts().iloc[0]

# Add column to new index containing row count by timestamp
new_index = new_index.assign(index=np.arange(len(new_index)) % timestamp_count)

# Re-index data with timestamp and timestamp row
df.index = pd.MultiIndex.from_frame(new_index)

# Clean up environment
del timestamp_count, new_index

# Drop timestamp column as it is now in the index
df = df.drop('timestamp', 
             axis=1)

# Set seed for reproducibility
np.random.seed(123)

# Shuffle the dataframe
df = shuffle(df)

# Initialize categorical variable encoder
encoder = LabelEncoder()

# Add label code using encoder to dataframe
df['label_code'] = encoder.fit_transform(df['label'])

# Retrieve encoding key
encoding_key = df[['label', 'label_code']].drop_duplicates().reset_index(drop=True)

# Drop original label column, as it is now encoded
df = df.drop('label',
             axis=1)

# Create series from label encoding and drop it from dataframe
targets = df.pop('label_code')

# Create training and testing data
X_train, X_test, y_train, y_test = train_test_split(df, 
                                                    targets, 
                                                    test_size= 0.20)

Cs = np.logspace(-6, 3, 10)

parameters = [{'kernel': ['rbf'], 'C': Cs},
              {'kernel': ['linear'], 'C': Cs}]

svc = SVC()

clf = GridSearchCV(estimator = svc, 
                   param_grid = parameters, 
                   cv = 5, 
                   n_jobs = -1)

clf.fit(X_train, y_train)



# End timer
stop = timeit.default_timer()

# Calculate total time
total_time = stop - start

# Output running time in a nice format.
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)

sys.stdout.write("Total running time: %d:%d:%d.\n" % (hours, mins, secs))

# Clean up environment
del hours, mins, secs, start, stop, total_time