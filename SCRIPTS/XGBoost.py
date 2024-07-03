"""
Samuel Rea
last modified: Samuel Rea july 1, 2024
Description: THis program builds a XGBoost decision tree

This creates a good comparison between bagging (random forest) and
boosting (XGBoost). both models use decision trees as their base but they way in which they utilize them 
varies drastically.
"""
#import packages needed
from skopt import gp_minimize
from skopt.space import Integer, Real
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# Load and preprocess your data
data = pd.read_csv("Clean_BC_Data.csv")
y = data["diagnosis"].values
x = data.drop(columns="diagnosis").values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Define the objective function to optimize (accuracy in this case) for XGBoost
def xgb_objective(params):
    n_estimators, max_depth, learning_rate, subsample = params

    # Initialize the XGBoost model with the current hyperparameters
    xgb_model = xgb.XGBClassifier(n_estimators=int(n_estimators),
                                  max_depth=int(max_depth),
                                  learning_rate=learning_rate,
                                  subsample=subsample,
                                  random_state=3,
                                  use_label_encoder=False,
                                  eval_metric='logloss')

    # Evaluate using cross-validation
    scores = cross_val_score(xgb_model, x_train, y_train, cv=5, scoring='accuracy')
    return -np.mean(scores)  # minimize negative accuracy

# Define the search space for XGBoost
xgb_space = [
    Integer(50, 500, name='n_estimators'),
    Integer(3, 50, name='max_depth'),
    Real(0.01, 0.3, prior='log-uniform', name='learning_rate'),
    Real(0.5, 1.0, name='subsample')
]

# Perform Bayesian Optimization for XGBoost
xgb_result = gp_minimize(xgb_objective, xgb_space, n_calls=25, random_state=3)

# Print best hyperparameters found for XGBoost
best_xgb_params = xgb_result.x
print("Best XGBoost Hyperparameters:")
print(f"n_estimators: {best_xgb_params[0]}")
print(f"max_depth: {best_xgb_params[1]}")
print(f"learning_rate: {best_xgb_params[2]}")
print(f"subsample: {best_xgb_params[3]}")

# Train final XGBoost model with best hyperparameters
best_xgb_model = xgb.XGBClassifier(n_estimators=best_xgb_params[0],
                                   max_depth=best_xgb_params[1],
                                   learning_rate=best_xgb_params[2],
                                   subsample=best_xgb_params[3],
                                   random_state=3,
                                   use_label_encoder=False,
                                   eval_metric='logloss')
best_xgb_model.fit(x_train, y_train)

#saving the xgboost model to be analysed later
joblib.dump(best_xgb_model, 'xgboost_model_BC.pkl')
