#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 17:08:29 2024

@author: kittapadmeesonk
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, f1_score
import time

# Start the timer
start_time = time.time()  

# 1. Load data
df = pd.read_csv("Loandata_SMOTENC_STD_XG_PCC.csv")
X = df.drop("DefaultStatus", axis=1)
y = df["DefaultStatus"]


# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Hyperparameter Grid
param_grid = {
    'C': [0.1, 1, 10],  # Inverse of regularization strength
    'penalty': ['l1', 'l2'],  # Regularization type
    'solver': ['liblinear', 'saga']         # Optimization algorithm
}

# 4. Model with Grid Search
model = LogisticRegression(max_iter=1000)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# 5. Best Model and Evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# 6. Evaluation
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# Stop the timer 
end_time = time.time()
runtime = end_time - start_time

print("Best Hyperparameters:", grid_search.best_params_)
print("Accuracy:", accuracy)
print("Classification Report:\n", class_report)
print("ROC-AUC:", roc_auc)
print(f"\nTotal Runtime: {runtime:.2f} seconds") 