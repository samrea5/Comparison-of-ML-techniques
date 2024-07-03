"""
Samuel Rea
last modified: Samuel Rea july 2, 2024
Description: THis program build the logistic regression model
"""

import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load and preprocess your data
data = pd.read_csv("Clean_BC_Data.csv")
y = data["diagnosis"].values
x = data.drop(columns="diagnosis").values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# Define the objective function
def objective(trial):
    # Suggest values for the hyperparameters
    C = trial.suggest_loguniform('C', 1e-5, 1e2)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
    max_iter = trial.suggest_int('max_iter', 100, 5000)

    # Ensure solver and penalty are compatible
    if penalty == 'l1' and solver not in ['liblinear', 'saga']:
        return float('inf')
    if penalty == 'l2' and solver not in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
        return float('inf')

    # Create the logistic regression model
    model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter)

    # Perform cross-validation
    score = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy').mean()

    return score


# Create a study and optimize it
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get the best parameters
best_params = study.best_params
best_score = study.best_value

print("Best Parameters:", best_params)
print("Best Cross-Validation Accuracy:", best_score)

# Train the final model with the best parameters
#**just unpacks dictionary as arguments
logistic_Regression_model = LogisticRegression(**best_params)
logistic_Regression_model.fit(x_train, y_train)

#saving logistic regression to be used in analysis
joblib.dump(logistic_Regression_model, 'logistic_regression_model_BC.pkl')