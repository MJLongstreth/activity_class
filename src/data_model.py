#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 00:03:41 2020

Python 3.7.6

@author: mlongstreth
"""


# Import libraries
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import dump
import sys
import timeit

# Start timer to evaluate script efficiency
start = timeit.default_timer()

# Set file location
file = './data/merged/acc_gyro_data.csv'

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
del timestamp_count, new_index, file

# Drop timestamp column as it is now in the index
df = df.drop('timestamp', 
             axis=1)

# Set seed for reproducibility
np.random.seed(123)

# Shuffle the dataframe
df = shuffle(df)

# Initialize categorical variable encoder
encoder = LabelEncoder()

# Fit encoder to labels
encoder.fit(df['label'])

# Save encoder to decode labels after training/testing
dump(encoder, './encoders/labelencoder/labelencoder.joblib')

# Add label code using encoder to dataframe
df['label_code'] = encoder.transform(df['label'])

# Retrieve encoding key
encoding_keys = df[['label', 'label_code']].drop_duplicates().reset_index(drop=True)
#encoding_key = encoding_key.set_index('label_code')
encoding_keys.to_csv('./encoders/labelencoder/encoding_keys.csv')

# Drop original label column, as it is now encoded
df = df.drop('label',
             axis=1)

# Create series from label encoding and drop it from dataframe
targets = df.pop('label_code')

# Create minmaxscaler
scaler = MinMaxScaler()

# Fit scaler to df
scaler.fit(df)

# Save scaler to unscale data after training
dump(scaler, './scalers/minmaxscaler/minmaxscaler.joblib')

# Transform df with scaler and save scaled data to dataframe
df_scaled = pd.DataFrame(scaler.transform(df))

# Initialize empty list for df_scaled column names
df_scaled_columns = []

# Loop through original dataframe columns and append '_scaled' to end of each \
    # column name
for i in df.columns:
    i = i + '_scaled'
    df_scaled_columns.append(i)

# Add column names to scaled dataframe    
df_scaled.columns = df_scaled_columns

# Add original index to scaled dataframe
df_scaled.index = df.index

# Create training and testing data with 20% of data for training
X_train, X_test, y_train, y_test = train_test_split(df_scaled, 
                                                    targets, 
                                                    test_size= 0.20)

# Create training and testing data with 20% of data for training
X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                  y_train, 
                                                  test_size= 0.10)

# Export training, validataion and test datasets
df.to_csv('./data/model/df.csv')
df_scaled.to_csv('./data/model/df_scaled.csv')
X_train.to_csv('./data/model/X_train.csv')
y_train.to_csv('./data/model/y_train.csv')
X_val.to_csv('./data/model/X_val.csv')
y_val.to_csv('./data/model/y_val.csv')
X_test.to_csv('./data/model/X_test.csv')
y_test.to_csv('./data/model/y_test.csv')

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