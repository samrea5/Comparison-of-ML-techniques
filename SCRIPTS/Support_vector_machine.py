"""
Samuel Rea
last modified: Samuel Rea july 2, 2024
Description: THis program builds a support vector machine
"""

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
import joblib

# Load and preprocess your data
data = pd.read_csv("Clean_BC_Data.csv")
y = data["diagnosis"].values
x = data.drop(columns="diagnosis").values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define the SVM model with parameters to optimize
svm = SVC()

# Define search space for hyperparameters
search_space = {
    'C': (0.1, 10.0, 'log-uniform'),
    'gamma': (1e-6, 1e+1, 'log-uniform'),
    'kernel': ['linear', 'poly', 'rbf']
}

# Initialize BayesSearchCV for SVM with F1 scoring
#always use random state three to replicate results in Github
opt = BayesSearchCV(svm, search_space, scoring='f1', cv=5, n_iter=20, random_state=3)
svm_model_best =opt.fit(x_train, y_train)
# Save the best SVM model found by BayesSearchCV
joblib.dump(svm_model_best, 'svm_model_BC.joblib')