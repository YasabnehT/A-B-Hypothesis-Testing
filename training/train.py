import sys
import mlflow.sklearn
import mlflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import numpy as np
import pandas as pd
import sys
import warnings
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from util import get_categorical_columns, get_numerical_columns
from ml_utils import cross_validation, plot_result, label_encoder


df = pd.read_csv('data/AdSmartClean_data.csv')
main_df = df.copy()
# print(df.head())
numerical_cols = get_numerical_columns(df)
categorical_cols = get_categorical_columns(df)
# print(categorical_cols)
categorical_cols.remove('auction_id')
categorical_cols_encoded = label_encoder(df)
categorical_cols_encoded.drop(columns=['auction_id'], inplace=True)


X = df.copy()
#Dropping duplicate column names
X.drop(['yes', 'no', 'platform_os', 'hour'], axis=1, inplace=True)

X.drop(categorical_cols, axis=1, inplace=True)

X = pd.concat([X, categorical_cols_encoded], axis=1)

X['target'] = 1

X.loc[X['no'] == 1, 'target'] = 0

y = X['target']
X.drop(['target'], axis=1, inplace=True)
X.drop(['yes', 'no'], axis=1, inplace=True)

# print(X.head())

"""
LogisticRegression
"""
model = LogisticRegression()

model_result = cross_validation(model, X, y, 5)


with open("training/logistic_regression_metrics.txt", 'w') as outfile:
    outfile.write(
        f"Training data accuracy: {model_result['Training Accuracy scores'][0]}\n")
    outfile.write(
        f"Validation data accuracy: {model_result['Validation Accuracy scores'][0]}")


model_name = "Logistic Regression"
plot_result(model_name, "Accuracy", "Accuracy scores in 5 Folds",
               model_result["Training Accuracy scores"],
               model_result["Validation Accuracy scores"],
               'training/logistic_regression_accuracy.png')

# Precision Results

# Plot Precision Result
plot_result(model_name, "Precision", "Precision scores in 5 Folds",
               model_result["Training Precision scores"],
               model_result["Validation Precision scores"],
               'training/logistic_regression_preicision.png')

# Recall Results plot

# Plot Recall Result
plot_result(model_name, "Recall", "Recall scores in 5 Folds",
               model_result["Training Recall scores"],
               model_result["Validation Recall scores"],
               'training/logistic_regression_recall.png')


# f1 Score Results

# Plot F1-Score Result
plot_result(model_name, "F1", "F1 Scores in 5 Folds",
               model_result["Training F1 scores"],
               model_result["Validation F1 scores"],
               'training/logistic_regression_f1_score.png')

"""
RandomForestClassifier
"""
model = RandomForestClassifier(max_depth=20)

model_result = cross_validation(model, X, y, 5)


with open("training/random_forest_classifier_metrics.txt", 'w') as outfile:
    outfile.write(
        f"Training data accuracy: {model_result['Training Accuracy scores'][0]}\n")
    outfile.write(
        f"Validation data accuracy: {model_result['Validation Accuracy scores'][0]}")


model_name = "Random Forest Classifier"
plot_result(model_name, "Accuracy", "Accuracy scores in 5 Folds",
               model_result["Training Accuracy scores"],
               model_result["Validation Accuracy scores"],
               'training/random_forest_classifier_accuracy.png')

# Precision Results

# Plot Precision Result
plot_result(model_name, "Precision", "Precision scores in 5 Folds",
               model_result["Training Precision scores"],
               model_result["Validation Precision scores"],
               'training/random_forest_classifier_preicision.png')

# Recall Results plot

# Plot Recall Result
plot_result(model_name, "Recall", "Recall scores in 5 Folds",
               model_result["Training Recall scores"],
               model_result["Validation Recall scores"],
               'training/random_forest_classifier_recall.png')


# f1 Score Results

# Plot F1-Score Result
plot_result(model_name, "F1", "F1 Scores in 5 Folds",
               model_result["Training F1 scores"],
               model_result["Validation F1 scores"],
               'training/random_forest_classifier_f1_score.png')



"""
XGBClassifier
"""
model = XGBClassifier()

model_result = cross_validation(model, X, y, 5)


with open("training/xgb_classifier_metrics.txt", 'w') as outfile:
    outfile.write(
        f"Training data accuracy: {model_result['Training Accuracy scores'][0]}\n")
    outfile.write(
        f"Validation data accuracy: {model_result['Validation Accuracy scores'][0]}")


model_name = "XGBoost Classifier"
plot_result(model_name, "Accuracy", "Accuracy scores in 5 Folds",
               model_result["Training Accuracy scores"],
               model_result["Validation Accuracy scores"],
               'training/xgb_classifier_accuracy.png')

# Precision Results

# Plot Precision Result
plot_result(model_name, "Precision", "Precision scores in 5 Folds",
               model_result["Training Precision scores"],
               model_result["Validation Precision scores"],
               'training/xgb_classifier_preicision.png')

# Recall Results plot

# Plot Recall Result
plot_result(model_name, "Recall", "Recall scores in 5 Folds",
               model_result["Training Recall scores"],
               model_result["Validation Recall scores"],
               'training/xgb_classifier_recall.png')


# f1 Score Results

# Plot F1-Score Result
plot_result(model_name, "F1", "F1 Scores in 5 Folds",
               model_result["Training F1 scores"],
               model_result["Validation F1 scores"],
               'training/xgb_classifier_f1_score.png')


"""
DecisionTreeClassifier
"""
model = DecisionTreeClassifier()

model_result = cross_validation(model, X, y, 5)


with open("training/decision_tree_classifier_metrics.txt", 'w') as outfile:
    outfile.write(
        f"Training data accuracy: {model_result['Training Accuracy scores'][0]}\n")
    outfile.write(
        f"Validation data accuracy: {model_result['Validation Accuracy scores'][0]}")


model_name = "Decision Tree Classifier"
plot_result(model_name, "Accuracy", "Accuracy scores in 5 Folds",
               model_result["Training Accuracy scores"],
               model_result["Validation Accuracy scores"],
               'training/decision_tree_classifier_accuracy.png')

# Precision Results

# Plot Precision Result
plot_result(model_name, "Precision", "Precision scores in 5 Folds",
               model_result["Training Precision scores"],
               model_result["Validation Precision scores"],
               'training/decision_tree_classifier_preicision.png')

# Recall Results plot

# Plot Recall Result
plot_result(model_name, "Recall", "Recall scores in 5 Folds",
               model_result["Training Recall scores"],
               model_result["Validation Recall scores"],
               'training/decision_tree_classifier_recall.png')


# f1 Score Results

# Plot F1-Score Result
plot_result(model_name, "F1", "F1 Scores in 5 Folds",
               model_result["Training F1 scores"],
               model_result["Validation F1 scores"],
               'training/decision_tree_classifier_f1_score.png')


