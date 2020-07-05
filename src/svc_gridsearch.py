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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.svm import SVC
from joblib import dump, load
from sklearn.metrics import confusion_matrix, accuracy_score
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
#encoding_key = encoding_key.set_index('label_code')
encoding_key.to_csv('./models/svc_gridsearch/clf.best_estimator_/encoding_key.csv')

# Drop original label column, as it is now encoded
df = df.drop('label',
             axis=1)

# Create series from label encoding and drop it from dataframe
targets = df.pop('label_code')

# Create minmaxscaler
scaler = MinMaxScaler()

# Fit scaler to df
scaler.fit(df)

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

# Create regularization parameters
Cs = np.logspace(-6, 3, 10)

# Create parameters for GridSearchCV using regularization parameters and two \
    # different kernels
parameters = [{'kernel': ['rbf'], 'C': Cs},
              {'kernel': ['linear'], 'C': Cs}]

# Get total number of models to be trained
total_models = len(ParameterGrid(parameters)) * 5

# Print total number of models to be trained
print(f"The total number of parameters-combinations is: {total_models}")

# Skip lines 120-146 if no need to re-runGridSearch

# Initialize support vector model
svc = SVC(probability=True)

# Apply GrridSeachCV using svc, parameters, 5 cross-folds and all available cores
clf = GridSearchCV(estimator = svc, 
                   param_grid = parameters, 
                   cv = 5, 
                   n_jobs = -1)

# Fit GridSearchCV object to training data
clf.fit(X_train, y_train)

# Inspect results from GridSearch
clf_results_df = pd.DataFrame(clf.cv_results_)

# Save results from GridSearch to csv
clf_results_df.to_csv('./models/svc_gridsearch/clf.best_estimator_/clf_results_df.csv')

# Retrieve the best estimating model from GridSearch
model = clf.best_estimator_

# Create model_name to save model
model_name = './models/svc_gridsearch/clf.best_estimator_/svc_gridsearch_best_model.joblib'

# Save best estimating model for future use
dump(model, model_name)

# Test load model
model = load(model_name)

# Get predictions on testing data
predictions = model.predict(X_test)

# Create confusion matrix from predictions and testing targets
predictions_confusion_matrix = pd.DataFrame(confusion_matrix(y_test, predictions))

predictions_confusion_matrix.index = encoding_key.sort_values('label_code')['label']

predictions_confusion_matrix.columns = list(encoding_key.sort_values('label_code')['label'])

predictions_confusion_matrix.to_csv('./models/svc_gridsearch/clf.best_estimator_/predictions_confusion_matrix.csv')

# Get model accuracy on testing data
test_accuracy = accuracy_score(y_test, predictions)
np.savetxt('./models/svc_gridsearch/clf.best_estimator_/test_accuracy.txt', np.atleast_1d(test_accuracy))

#predictions_proba = model.predict_proba(X_test)[:,1]

# Unscale X_test
results = pd.DataFrame(scaler.inverse_transform(X_test))

# Set results index to match X_test
results.index = X_test.index

# Set results columns to original columns
results.columns = df.columns

# Add testing labels to X_test
results = results.join(y_test)

# Add predictions to to X_test
results['predictions'] = predictions

# Temporarily add index as columns for merge
results = results.reset_index()

# Merge encoding keys to get un-encoded labels for original labels
results = results.merge(encoding_key,
                        left_on='label_code',
                        right_on='label_code',
                        suffixes=('_temp', '_predictions'))

# Merge encoding keys to get un-encoded labels for predictions
results = results.merge(encoding_key,
                        left_on='predictions',
                        right_on='label_code',
                        suffixes=(None, '_predictions')).drop('label_code_predictions', axis=1)

# Set index to original index
results = results.set_index(['timestamp', 'index'])

results['error'] = results['label'] != results['label_predictions']

# Export results to csv
results.to_csv('./models/svc_gridsearch/clf.best_estimator_/results.csv')

# Clean up environment
del Cs, X_test, X_train, clf, clf_results_df, df, df_scled, df_scled columns
del encoding_key, file, i, model, model_name, parameters, predictions
del predictions_confusion_matrix, results, scaler, svc, targets, test_accuracy
del total, y_test, y_trian

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
