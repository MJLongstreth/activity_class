#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 22:51:15 2020

Python 3.7.6

@author: mlongstreth
"""

# Rewrite line 12 & 13 to bash script
# brew install tree
# tree -fi data/*/*/*/*.json > data/data_file.txt

# Import necessary packages
import pandas as pd
import json

# Load data from data_file.txt file
with open('data/data_files.txt') as f:
    data_files = f.readlines()

# Delete variable no longer needed    
del f

data_files = data_files[0:len(data_files) - 2]

df = []

for file in data_files:
    df.append(file.split(" ")[0])

# Read file paths into a dataframe
df = pd.DataFrame(df)

# Delete variable no longer needed  
del data_files, file

# Rename column to path
df.columns = ['path']

# Split path to extract labels, sensor type, date, filename and then join file path
df = pd.DataFrame(df.apply(lambda x: x.str.split('/'))['path'].to_list(),
                  columns=['folder', 'label', 'sensor_type', 'date', 'file']).join(df).drop('folder', 
                                                                                       axis=1)

#df['path'] = '/' + df['path']                                                                                       
                                                                                            
                                                                                            
# Split DF into 3 dataframes by type of data
acc_data = df[df['sensor_type'] == 'acc']
gyro_data = df[df['sensor_type'] == 'gyro']
magn_data = df[df['sensor_type'] == 'magn']

temp = acc_data['path'].to_list()

acc_paths = []

for file in temp:
    data = pd.read_json(file)
    #acc_path.append(data)

#df['dict'] = df.apply(lambda x: pd.read_json(x, lines=True))

temp1=temp[0]

# Load accelerometer data
df_1 = pd.read_json(temp[3],
                    lines=True)


# Load accelerometer data
df_1 = pd.read_json('data/prone/acc/20206030/20200630T061012Z_182730000190_acc_stream.json',
                    lines=True)











