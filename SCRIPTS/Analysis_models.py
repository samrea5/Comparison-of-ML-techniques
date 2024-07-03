"""
Samuel Rea
last modified: Samuel Rea july 2, 2024
Description: This contains the analysis for the machine learning models made in the other python scripts.
"""

#importing necessary packages
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score
import matplotlib.pyplot as plt
#loading in the machine learning models
xgb_model = joblib.load('xgboost_model_BC.pkl')
rf_model = joblib.load('random_forest_model_BC.pkl')
logistic_regression_model = joblib.load('logistic_regression_model_BC.pkl')
svm_model = joblib.load('svm_model_BC.joblib')
#loading in the data and splitting it again into train and test sets. Due to the set random key this should function
#just the same as the way the models were trained. IE the data split is the same.
# Load and preprocess your data
data = pd.read_csv("Clean_BC_Data.csv")
y = data["diagnosis"].values
x = data.drop(columns="diagnosis").values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#rf analysis
# Evaluate on test set
# Generate predictions and confusion matrix
y_rf_pred = rf_model.predict(x_test)
rf_cm = confusion_matrix(y_test, y_rf_pred)

# Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=['Benign', 'Malignant'])
disp.plot(cmap='Blues', colorbar=False)
plt.show()

#XGB analysis
# Generate predictions and confusion matrix for XGBoost
y_xgb_pred = xgb_model.predict(x_test)
xgb_cm = confusion_matrix(y_test, y_xgb_pred)

# Confusion Matrix for XGBoost
xgb_disp = ConfusionMatrixDisplay(confusion_matrix=xgb_cm, display_labels=['Benign', 'Malignant'])
xgb_disp.plot(cmap='Blues', colorbar=False)
plt.title('XGBoost Confusion Matrix')
plt.show()

#analysis of the logistic regression
# Generate predictions and confusion matrix
y_lr_pred = logistic_regression_model.predict(x_test)
lr_cm = confusion_matrix(y_test, y_lr_pred)

# Confusion Matrix
lr_disp = ConfusionMatrixDisplay(confusion_matrix=lr_cm, display_labels=['Benign', 'Malignant'])
lr_disp.plot(cmap='Blues', colorbar=False)
plt.title('logistic Regression Confusion Matrix')
plt.show()


#SVM analysis
# Generate predictions and confusion matrix
y_svm_pred = svm_model.predict(x_test)
svm_cm = confusion_matrix(y_test, y_svm_pred)

# Confusion Matrix
svm_disp = ConfusionMatrixDisplay(confusion_matrix=svm_cm, display_labels=['Benign', 'Malignant'])
svm_disp.plot(cmap='Blues', colorbar=False)
plt.title('Support Vector Machine Confusion Matrix')
plt.show()

#comparison between rf(bagging), XGB(boosting), logistic regression
# Calculate overall accuracy and precision for Random Forest and XGBoost
rf_accuracy = accuracy_score(y_test, y_rf_pred)
rf_precision = precision_score(y_test, y_rf_pred)
xgb_accuracy = accuracy_score(y_test, y_xgb_pred)
xgb_precision = precision_score(y_test, y_xgb_pred)
lr_accuracy = accuracy_score(y_test, y_lr_pred)
lr_precision = precision_score(y_test, y_lr_pred)
svm_accuracy = accuracy_score(y_test, y_svm_pred)
svm_precision = precision_score(y_test, y_svm_pred)

# Print overall accuracy and precision for both models
print(f"Random Forest Overall Accuracy: {rf_accuracy}")
print(f"Random Forest Overall Precision: {rf_precision}")

print(f"XGBoost Overall Accuracy: {xgb_accuracy}")
print(f"XGBoost Overall Precision: {xgb_precision}")

print(f"Logistic Regression Overall Accuracy: {lr_accuracy}")
print(f"Logistic Regression Overall Precision: {lr_precision}")

print(f"SVM Overall Accuracy: {svm_accuracy}")
print(f"SVM Overall Precision: {svm_precision}")

# Plotting
labels = ['Random Forest Accuracy', 'Random Forest Precision', 'XGBoost Accuracy', 'XGBoost Precision',
          "LogR Accuracy", "LogR Precision","SVM Accuracy", "SVM Precision"
          ]
scores = [rf_accuracy, rf_precision, xgb_accuracy, xgb_precision, lr_accuracy, lr_precision,
          svm_accuracy, svm_precision
          ]

plt.figure(figsize=(10, 6))
plt.bar(labels, scores, color=['blue', 'blue', 'green', 'green',"orange","orange","purple","purple"])
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Comparison of Random Forest, XGBoost, Logistic Regression and SVM')
plt.ylim(0.8, 1.0)  # Adjust ylim based on your score range
plt.xticks(rotation=15)
for i, v in enumerate(scores):
    plt.text(i, v + 0.005, str(round(v, 4)), ha='center', va='bottom', fontsize=10)
plt.show()