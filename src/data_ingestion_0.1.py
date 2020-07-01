#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 22:51:15 2020

Python 3.7.6

@author: mlongstreth
"""

# Import necessary packages
import os
import pandas as pd
import json

data_files = []
for dirpath, subdirs, files in os.walk('.'):
    for x in files:
        if x.endswith(".json"):
            data_files.append(os.path.join(dirpath, x))

# Delete variable no longer needed    
del dirpath, files, x, subdirs

# Read file paths into a dataframe
df = pd.DataFrame(data_files)

# Rename column to path
df.columns = ['path']

# Split path to extract labels, sensor type, date, filename and then join file path
df = pd.DataFrame(df.apply(lambda x: x.str.split('/'))['path'].to_list(),
                  columns=['delete', 'folder', 'label', 'sensor_type', 'collection_date', 'file']).join(df).drop(['delete', 'folder'], 
                                                                                       axis=1)                                                                                                       

# Initialize empty list to store data from json files                                                                                                   
data = []

# Loop over data files paths and add json file dictionary to list
for file in data_files:
    x = pd.read_json(file,
                     lines=True)
    data.append(x)

# Add data to dataframe
df['data'] = data

# Delete variable no longer needed 
del data, data_files, x, file

# Working section

# Split DF into dataframes by sensor type
acc_data = df[df['sensor_type'] == 'acc']
gyro_data = df[df['sensor_type'] == 'gyro']


# Unpack first level of dictionary
df_1 = acc_data['data'].iloc[0].apply(pd.Series)

temp_1 = []

for index, row in df_1.iterrows():
    temp_1.append(row.apply(pd.Series))
    
temp_2 = []

for i in temp_1:
    for index, row in i.iterrows():
        #row = row.drop('Timestamp')
        row = row.apply(pd.Series)
        temp_2.append(row)
    
temp_3 = []
    
for i in temp_2:
    y = i.stack().apply(pd.Series).mean()
    temp_3.append(y)
    
temp_4 = []

for i in temp_3:
    x = pd.DataFrame(i).transpose()
    temp_4.append(x)
    
empty_df = pd.DataFrame()

for i in temp_4:
    empty_df = empty_df.append(i, ignore_index=True)

"""    
# Split DF into 3 dataframes by type of data
acc_data = df[df['sensor_type'] == 'acc']
gyro_data = df[df['sensor_type'] == 'gyro']
magn_data = df[df['sensor_type'] == 'magn']

df_2 = df_1.apply(pd.Series)

df_3 = df_2.apply(pd.Series)

# Create index from the timestamp
df_2.index = df_2['Timestamp']

# Drop the timestamp column
df_2 = df_2.drop(['Timestamp'],
                 axis=1)




"""






