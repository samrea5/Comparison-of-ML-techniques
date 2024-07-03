"""
Samuel Rea
last modified: Samuel Rea July 2, 2024
Description: This program contains the research on hyperparameters(Bayesian Optimization) as well as the best
basic random forest based on those hyperparameters. There is then a boosted tree to see if this increases performance
when compared to the basic fine-tuned random forest. The random forest identifies whether a tumor is malignant (1) or
benign (0) depending on a series of factors describing the cell makeup.
"""
from skopt import gp_minimize
from skopt.space import Integer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib

# Load and preprocess your data
data = pd.read_csv("Clean_BC_Data.csv")
y = data["diagnosis"].values
x = data.drop(columns="diagnosis").values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Define the objective function to optimize (accuracy in this case)
def objective(params):
    n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = params

    # Initialize the model with the current hyperparameters
    rf_model = RandomForestClassifier(n_estimators=int(n_estimators),
                                      max_depth=int(max_depth),
                                      min_samples_split=int(min_samples_split),
                                      min_samples_leaf=int(min_samples_leaf),
                                      max_features=int(max_features),
                                      random_state=3)

    # Evaluate using cross-validation
    scores = cross_val_score(rf_model, x_train, y_train, cv=5, scoring='accuracy')
    return -np.mean(scores)  # minimize negative accuracy


# Define the search space
space = [
    Integer(50, 500, name='n_estimators'),
    Integer(3, 50, name='max_depth'),
    Integer(2, 10, name='min_samples_split'),
    Integer(1, 4, name='min_samples_leaf'),
    Integer(2, x.shape[1], name='max_features')
]

# Perform Bayesian Optimization
result = gp_minimize(objective, space, n_calls=25, random_state=3)

# Print best hyperparameters found
best_params = result.x
print("Best Hyperparameters:")
print(f"n_estimators: {best_params[0]}")
print(f"max_depth: {best_params[1]}")
print(f"min_samples_split: {best_params[2]}")
print(f"min_samples_leaf: {best_params[3]}")
print(f"max_features: {best_params[4]}")

# Train final model with best hyperparameters
best_rf_model = RandomForestClassifier(n_estimators=best_params[0],
                                       max_depth=best_params[1],
                                       min_samples_split=best_params[2],
                                       min_samples_leaf=best_params[3],
                                       max_features=best_params[4],
                                       random_state=3)
best_rf_model.fit(x_train, y_train)

#saving the model so analysis can be done on it
joblib.dump(best_rf_model, 'random_forest_model_BC.pkl')
