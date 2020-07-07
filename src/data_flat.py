#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 21:47:02 2020

Python 3.7.6

@author: mlongstreth
"""

# Import necessary packages
import os
import pandas as pd
import json
import sys
import timeit

# Start timer to evaluate script efficiency
start = timeit.default_timer()

# Initialize empty list to store json file paths
data_files = []

# Search working directory for json files and append path to data files list
for dirpath, subdirs, files in os.walk('.'):
    for x in files:
        if x.endswith(".json"):
            data_files.append(os.path.join(dirpath, x))
    
# Delete variable no longer needed           
del dirpath, files, subdirs, x

# Loop to read each file in data files and extract dictionary contents to \
    # dataframe
for i in range(len(data_files)):
    
    # Each json file contains x number of dictionaries, read each dictionary \
        # into a list
    data = [json.loads(line) for line in open(data_files[i], 'r')]
    
    # Retrieve dictionary key value
    for item in data[i].keys():
        item
    
    # Retrieve dictionary data from key
    x = list(map(lambda x: x[item], data))
    
    # Retrieve dictionary key for next loop
    for item in x[0].keys():
        item
    
    # Initialize empty data frame
    df = pd.DataFrame()
    
    # Loop through extracted dictionaries and extract array information to \
        # separate lines keeping the 'Timestamp'
    for z in x:
        temp_df = pd.DataFrame(z[item])
        temp_df['Timestamp'] = z['Timestamp']
        df = df.append(temp_df, ignore_index=True)
    
    # Create column in dataframe indicating the source file
    df['source'] = data_files[i]
    
    # Create file name for export from original file name, replacing JSON \
        # with csv
    file_name = data_files[i].split('/')[-1].replace('.json', '.csv')
    
    # Export each JSON file that has been converted to a dataframe as a csv
    df.to_csv('./data/flat/' + file_name)

# Clean up environment
del data, data_files, df, file_name, i, item, temp_df, x, z
    
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