# Comparison-of-ML-techniques

### Introduction
This project showcases basic machine learning techniques to analyze the Wisconsin Breast Cancer dataset. The machine learning models classify tumor cells as either malignant (1) or benign (0) based on a myriad of factors including cell size, concavity, texture, perimeter, and roughly 25 other predictors. Exploratory data analysis (EDA) was performed on each factor to examine correlations and distributions. The data was standardized and split into training and test sets using random seed #3. This foundation allowed for the development of four machine learning models: Logistic Regression, Random Forest, XGBoost, and Support Vector Machine (SVM). All models achieved high accuracy (>0.9) and precision (>0.9).

### Methods
**A detailed review of the methods used in this project**
**Data Cleaing**

The project began with data cleaning and exploration to build a solid foundation for model development. Only two columns (id, Unnamed: 32) were removed. The diagnosis column was modified to label malignant diagnoses as 1 and benign diagnoses as 0. Further data cleaning could include identifying and focusing on the most significant factors.

**Exploritory Data Analysis**

* Basic statistics were gathered for each variable to identify outliers.
* Violin plots were created to visualize the distribution of each variable.
* Correlation between predictors and the diagnosis was examined using Pearson's Correlation Coefficient, and results were visualized with a heat map.
* The data was standardized and split into training and test sets using random seed #3

**Logistic Regression**

A logistic regression model was built using the scikit-learn package in Python. Hyperparameters were optimized using Bayesian optimization with the Optuna package, ensuring efficient and computationally feasible tuning. The final logistic regression model achieved an accuracy of 0.9737 and a precision of 0.9744 on the test set. The model was saved using the joblib package for later analysis and comparison.

**Random Forest**

The random forest model followed a similar process. Bayesian optimization was used to find the best hyperparameters, employing the BayesianCV package. Once the model was built, it was saved with joblib for later use. The random forest achieved an accuracy of 0.9298 and a precision of 0.9 on the test set.

**XGBoost Decision Tree**

The XGBoost algorithm was included to compare boosting (XGBoost) versus bagging (Random Forest). The model was built using the XGBoost package, which interfaces efficiently with Torch for faster model building. Hyperparameters were optimized using Bayesian optimization. The model was then saved using joblib. The XGBoost model achieved an accuracy of 0.9474 and a precision of 0.925 on the test set.

**Support Vector Machines**

The final machine learning algorithm was the Support Vector Machine (SVM), which creates a hyperplane in n-dimensions to classify data. A Radial Basis Function (RBF) kernel was chosen due to the multifaceted and complex non-linear relationships in the data. Hyperparameters were optimized with Bayesian optimization, and the model was saved using joblib. The SVM achieved an accuracy of 0.9737 and a precision of 0.9744 on the test set.

### Navigating this Repository

This repository contains three main folders:
* SCRIPTS: Contains scripts used to build the models and explore the data.
* OUTPUT: Contains charts, graphs, a short conclusion and saved models for users interested in the project's conclusions.
* DATA: Contains the dataset used in this project and a link to the original dataset.
