#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 01:35:36 2020

Python 3.7.6

@author: mlongstreth
"""
import pandas as pd

acc_data = x

# Unpack first level of dictionary

test = acc_data['data'].to_list()

temp = []
temp_2 = []
temp_3 = []
temp_4 = []

for i in test:    
    for index, row in i.iterrows():
        temp.append(row.apply(pd.Series))
        for i in temp:
            for index, row in i.iterrows():
                #row = row.drop('Timestamp')
                row = row.apply(pd.Series)
                temp_2.append(row)
        
"""
x = acc_data['data'].iloc[index].apply(pd.Series)



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