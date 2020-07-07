#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 02:08:17 2020

Python 3.7.6

@author: mlongstreth
"""


# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.svm import SVC
from joblib import dump, load
from sklearn.metrics import confusion_matrix, accuracy_score
import sys
import timeit

# Start timer to evaluate script efficiency
start = timeit.default_timer()

# Import data
X_train = pd.read_csv('./data/model/X_train.csv', index_col=['timestamp', 'index'])
y_train = pd.read_csv('./data/model/y_train.csv', index_col=['timestamp', 'index'])
X_val = pd.read_csv('./data/model/X_val.csv', index_col=['timestamp', 'index'])
y_val = pd.read_csv('./data/model/y_val.csv', index_col=['timestamp', 'index'])
X_test = pd.read_csv('./data/model/X_test.csv', index_col=['timestamp', 'index'])
y_test = pd.read_csv('./data/model/y_test.csv', index_col=['timestamp', 'index'])

# Set seed for reproducibility
np.random.seed(123)

# Create regularization parameters
Cs = np.logspace(-6, 3, 10)

# Create parameters for GridSearchCV using regularization parameters and two \
    # different kernels
parameters = [{'kernel': ['rbf'], 'C': Cs, 'cache_size': [400.0]},
              {'kernel': ['linear'], 'C': Cs, 'cache_size': [400.0]}]

# Get total number of models to be trained
total_models = len(ParameterGrid(parameters)) * 5

# Print total number of models to be trained
print(f"The total number of parameters-combinations is: {total_models}")

# Initialize support vector model
svc = SVC(probability=True)

# Apply GrridSeachCV using svc, parameters, 5 cross-folds and all available cores
svc_clf = GridSearchCV(estimator = svc, 
                       param_grid = parameters, 
                       cv = 5, 
                       n_jobs = -1)

# Fit GridSearchCV object to training data
svc_clf.fit(X_val, y_val.values.ravel())

# Inspect results from GridSearch
svc_clf_results_df = pd.DataFrame(svc_clf.cv_results_)

# Save results from GridSearch to csv
svc_clf_results_df.to_csv('./models/skl_svm_svc/skl_svm_svc_grid_search_results.csv')

# Retrieve the best estimating model from GridSearch
model = svc_clf.best_estimator_.fit(X_train, y_train.values.ravel())

# Save scaler to unscale data after training
dump(model, './models/skl_svm_svc/skl_svm_svc_model.joblib')

# Get predictions on testing data
predictions = model.predict(X_test)

# Create confusion matrix from predictions and testing targets
predictions_confusion_matrix = pd.DataFrame(confusion_matrix(y_test, predictions))

# Import encoding keys
encoding_keys = pd.read_csv('./encoders/labelencoder/encoding_keys.csv')

# Add index from encoding key to confusion matrix
predictions_confusion_matrix.index = encoding_keys.sort_values('label_code')['label']

# Add column names from encoding key to confusion matrix
predictions_confusion_matrix.columns = list(encoding_keys.sort_values('label_code')['label'])

# Export confusion matric to csv
predictions_confusion_matrix.to_csv('./models/skl_svm_svc/skl_svm_scv__confusion_matrix.csv')

# Get model accuracy on testing data
test_accuracy = accuracy_score(y_test, predictions)

# Print the model accuracy on the test set
print(f"The model accuracy: {test_accuracy}")

# Save the model accuracy on the test set to directory
np.savetxt('./models/skl_svm_svc/skl_svm_scv_test_accuracy.txt',
           np.atleast_1d(test_accuracy))

# Get probablity of predictions
predictions_proba = model.predict_proba(X_test)[:,1]

# Load scaler used to scale the data
scaler = load('./scalers/minmaxscaler/minmaxscaler.joblib')

# Unscale X_test
results = pd.DataFrame(scaler.inverse_transform(X_test))

# Set results index to match X_test
results.index = X_test.index

# Set results columns to original columns
results.columns = X_test.columns

# Create column names for results df
columns = []

for i in X_test.columns:
    i = i.replace('_scaled', '')
    columns.append(i)
    
results.columns = columns

# Add testing labels to X_test
results = results.join(y_test)

# Load scaler used to scale the data
encoder = load('./encoders/labelencoder/labelencoder.joblib')

# Decode original labels
results['label'] = encoder.inverse_transform(results['label_code'])

# Add predictions to to X_test
results['prediction_code'] = predictions

# Decode predicted labels
results['prediction_label'] = encoder.inverse_transform(results['prediction_code'])

# Create column indicating if record errored
results['error'] = results['label'] != results['prediction_label']

# Add prediction probabilities to results
results['predictions_probability'] = predictions_proba

# Export results to csv
results.to_csv('./models/skl_svm_svc/skl_svm_scv_results.csv')

del Cs, X_test, X_train, X_val, columns, encoder, encoding_keys, i, model
del parameters, predictions, predictions_confusion_matrix, predictions_proba
del results, scaler, svc, svc_clf, svc_clf_results_df, test_accuracy, total_models
del y_test, y_train, y_val

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
