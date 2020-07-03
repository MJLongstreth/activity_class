#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 23:58:35 2020

Python 3.7.6

@author: mlongstreth
"""


# Import necessary packages
import os
import glob
import pandas as pd
import sys
import timeit

# Start timer to evaluate script efficiency
start = timeit.default_timer()

# Create lists containing data by sensor type
acc_paths = glob.glob('model_data/*acc_stream.csv')
gyro_paths = glob.glob('model_data/*gyro_stream.csv')
magn_paths = glob.glob('model_data/*magn_stream.csv')

# Intialize empty list for loop
acc_dfs = []

# Loop through file paths to data and load file from csv to dataframe
for i in acc_paths:
    acc_dfs.append(pd.read_csv(i))

# Intialize empty list for loop
gyro_dfs = []

# Loop through file paths to data and load file from csv to dataframe
for i in gyro_paths:
    gyro_dfs.append(pd.read_csv(i))
  
# Intialize empty list for loop
magn_dfs = []

# Loop through file paths to data and load file from csv to dataframe
for i in gyro_paths:
    magn_dfs.append(pd.read_csv(i))

# Clean up environment
del i, acc_paths, gyro_paths, magn_paths

# Initialize empty data frame for loop
acc_df = pd.DataFrame()

# Loop through list containing dataframes for sensor type, drop the first \
    # 1000 rows and drop the last 1000 rows, due to self collection
for i in acc_dfs:
    i = i.loc[1000:(len(i)-1001)]
    acc_df = acc_df.append(i, ignore_index=True)

# Drop uneeded column
acc_df = acc_df.drop(['Unnamed: 0'], axis=1)

# Initialize empty data frame for loop
gyro_df = pd.DataFrame()

# Loop through list containing dataframes for sensor type, drop the first \
    # 1000 rows and drop the last 1000 rows, due to self collection
for i in gyro_dfs:
    i = i.loc[1000:(len(i)-1001)]
    gyro_df = gyro_df.append(i, ignore_index=True)

# Drop uneeded column
gyro_df = gyro_df.drop(['Unnamed: 0'], axis=1)

# Initialize empty data frame for loop
magn_df = pd.DataFrame()

# Loop through list containing dataframes for sensor type, drop the first \
    # 1000 rows and drop the last 1000 rows, due to self collection
for i in magn_dfs:
    i = i.loc[1000:(len(i)-1001)]
    magn_df = magn_df.append(i, ignore_index=True)

# Drop uneeded column
magn_df = magn_df.drop(['Unnamed: 0'], axis=1)

# Split source file path information to extract label, sensor type and collection date
acc_df[['delete', 'folder', 'label', 'sensor_type', 'collection_date', 'file']] = acc_df['source'].str.split('/', expand=True)
gyro_df[['delete', 'folder', 'label', 'sensor_type', 'collection_date', 'file']] = gyro_df['source'].str.split('/', expand=True)
magn_df[['delete', 'folder', 'label', 'sensor_type', 'collection_date', 'file']] = magn_df['source'].str.split('/', expand=True)

# Clean up environment
del i,acc_dfs, gyro_dfs, magn_dfs

# Drop uneeded columns from each dataframe
acc_df = acc_df.drop(['source', 'delete', 'folder', 'sensor_type', 'file', 'collection_date'], axis=1)
gyro_df = gyro_df.drop(['source', 'delete', 'folder', 'sensor_type', 'file', 'collection_date'], axis=1)
magn_df = magn_df.drop(['source', 'delete', 'folder','sensor_type', 'file', 'collection_date'], axis=1)

# Rename columns for each data frome with sensor type suffix
acc_df.columns = ['x_acc', 'y_acc', 'z_acc', 'Timestamp_acc', 'label_acc']
gyro_df.columns = ['x_gyro', 'y_gyro', 'z_gyro', 'Timestamp_gyro', 'label_gyro']
magn_df.columns = ['x_magn', 'y_magn', 'z_magn', 'Timestamp_magn', 'label_magn']


# Perform value counts on timestamp information to filter out omissions
acc_time_counts = acc_df['Timestamp_acc'].value_counts().to_frame()
gyro_time_counts = gyro_df['Timestamp_gyro'].value_counts().to_frame()
magn_time_counts = magn_df['Timestamp_magn'].value_counts().to_frame()

# Calculate differences in timestamps between all sensor information
acc_gyro_diff = list(set(acc_time_counts.index).difference(gyro_time_counts.index))
gyro_acc_diff = list(set(gyro_time_counts.index).difference(acc_time_counts.index))
acc_magn_diff = list(set(acc_time_counts.index).difference(magn_time_counts.index))
magn_acc_diff = list(set(magn_time_counts.index).difference(acc_time_counts.index))
gyro_magn_diff = list(set(gyro_time_counts.index).difference(magn_time_counts.index))
magn_gyro_diff = list(set(magn_time_counts.index).difference(gyro_time_counts.index))

# Filter out differences from dataframes
acc_df = acc_df[~acc_df['Timestamp_acc'].isin(acc_gyro_diff)]
gyro_df = gyro_df[~gyro_df['Timestamp_gyro'].isin(gyro_acc_diff)]
magn_df = magn_df[~magn_df['Timestamp_magn'].isin(magn_acc_diff)]
acc_df = acc_df[~acc_df['Timestamp_acc'].isin(acc_magn_diff)]
gyro_df = gyro_df[~gyro_df['Timestamp_gyro'].isin(gyro_magn_diff)]
magn_df = magn_df[~magn_df['Timestamp_magn'].isin(magn_gyro_diff)]

# Clean up environment
del acc_time_counts, gyro_time_counts, magn_time_counts, acc_gyro_diff
del gyro_acc_diff, acc_magn_diff, magn_acc_diff, gyro_magn_diff, magn_gyro_diff

# Sort all dataframes by timestamp
acc_df = acc_df.sort_values('Timestamp_acc', ascending=True)
gyro_df = gyro_df.sort_values('Timestamp_gyro', ascending=True)
magn_df = magn_df.sort_values('Timestamp_magn', ascending=True)

# Reset indices on all dataframes
acc_df = acc_df.reset_index(drop=True)
gyro_df = gyro_df.reset_index(drop=True)
magn_df = magn_df.reset_index(drop=True)

# Join all data frames
acc_gyro_data = acc_df.join(gyro_df)
acc_gyro_data = acc_gyro_data.drop(['Timestamp_gyro', 'label_gyro'], axis=1)
acc_gyro_data = acc_gyro_data[['Timestamp_acc', 'x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro', 'label_acc']]
acc_gyro_data.columns = ['timestamp', 'x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro', 'label']
all_sensor_data = acc_df.join(gyro_df).join(magn_df)
all_sensor_data = all_sensor_data.drop(['Timestamp_gyro', 'label_gyro', 'Timestamp_magn', 'label_magn'], axis=1)
all_sensor_data = all_sensor_data[['Timestamp_acc', 'x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro', 'x_magn', 'y_magn', 'z_magn', 'label_acc']]
all_sensor_data.columns = ['timestamp', 'x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro', 'x_magn', 'y_magn', 'z_magn', 'label']

# Export merged dataframes to directory for model build
acc_df.to_csv('./model_data/acc_df.csv')
gyro_df.to_csv('./model_data/gyro_df.csv')
magn_df.to_csv('./model_data/magn_df.csv')
acc_gyro_data.to_csv('./model_data/acc_gyro_data.csv')
all_sensor_data.to_csv('./model_data/all_sensor_data.csv')

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