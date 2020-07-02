#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 22:54:00 2020

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

acc_paths = glob.glob('model_data/*acc_stream.csv')
gyro_paths = glob.glob('model_data/*gyro_stream.csv')
magn_paths = glob.glob('model_data/*magn_stream.csv')

acc_dfs = []

for i in acc_paths:
    acc_dfs.append(pd.read_csv(i))

gyro_dfs = []

for i in acc_paths:
    gyro_dfs.append(pd.read_csv(i))
    
magn_dfs = []

for i in acc_paths:
    magn_dfs.append(pd.read_csv(i))

del i, acc_paths, gyro_paths, magn_paths

acc_df = pd.DataFrame()

for i in acc_dfs:
    acc_df = acc_df.append(i, 
                           ignore_index=True)

acc_df = acc_df.drop(['Unnamed: 0'], 
                     axis=1)

gyro_df = pd.DataFrame()

for i in gyro_dfs:
    gyro_df = gyro_df.append(i, 
                             ignore_index=True)

gyro_df = gyro_df.drop(['Unnamed: 0'], 
                       axis=1)

magn_df = pd.DataFrame()

for i in magn_dfs:
    magn_df = magn_df.append(i, 
                             ignore_index=True)

magn_df = magn_df.drop(['Unnamed: 0'], 
                       axis=1)

acc_df[['delete', 'folder', 'label', 'sensor_type', 'collection_date', 'file']] = acc_df['source'].str.split('/', expand=True)
gyro_df[['delete', 'folder', 'label', 'sensor_type', 'collection_date', 'file']] = gyro_df['source'].str.split('/', expand=True)
magn_df[['delete', 'folder', 'label', 'sensor_type', 'collection_date', 'file']] = magn_df['source'].str.split('/', expand=True)

del i,acc_dfs, gyro_dfs, magn_dfs

acc_df = acc_df.drop(['source', 'delete', 'folder'], axis=1)
gyro_df = gyro_df.drop(['source', 'delete', 'folder'], axis=1)
magn_df = magn_df.drop(['source', 'delete', 'folder'], axis=1)

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

