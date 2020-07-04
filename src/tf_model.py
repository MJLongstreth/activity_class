# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 19:35:49 2020

Python Version: 3.7.6

@author: micha
"""

# Import libraries
import pandas as pd
import numpy as np
import tensorflow as tf


# Import data
df = pd.read_csv('./model_data/acc_gyro_data.csv',
                 index_col=0)

# Set seed for reproducibility
np.random.seed(123)

# Begin to prepare data for tensorflow
target = df.pop('cluster_labels')

# Build dataset for tensorflow
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

# Inspect first 5 observations for dataset
for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))

def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

train_dataset = dataset.batch(1)

model = get_compiled_model()
model.fit(train_dataset, epochs=15)
