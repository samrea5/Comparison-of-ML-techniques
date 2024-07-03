"""
Samuel Rea
last modified: Samuel Rea july 1, 2024
Description: This file does exploratory data and graph analysis of the data set and saves the most important
graphs as pdfs.
"""

# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

# Load the cleaned dataset
file_path = 'Clean_BC_Data.csv'
data = pd.read_csv(file_path)

#selecting the features i want and the target variable for clarity and reproducability
features = ["radius_mean", "texture_mean", "perimeter_mean",
    "area_mean", "smoothness_mean", "compactness_mean",
    "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "fractal_dimension_worst","symmetry_worst", "concave points_worst", "concavity_worst",
    "compactness_worst", "smoothness_worst", "area_worst", "perimeter_worst", "texture_worst",
    "radius_worst", "fractal_dimension_se", "symmetry_se", "concave points_se", "concavity_se",
    "compactness_se", "smoothness_se", "area_se", "perimeter_se", "texture_se", "radius_se"
            ]
target = "diagnosis"

#summary statisitcs
summary = data.describe()
summary_df = pd.DataFrame(summary)
#print(summary_df)
summary_df.to_csv("Summary_Stats_BC",index=True)


#histograms of all of the features
data[features].hist(bins=15, figsize=(15, 10), layout=(len(features)//3 + 1, 3))
plt.tight_layout()
plt.show()

# Correlation heatmap
columns_of_interest = features + ['diagnosis']
data_heatmap=data[columns_of_interest]
corr_matrix = data_heatmap.corr()
sorted_features = corr_matrix.loc[features, 'diagnosis'].sort_values()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix.loc[sorted_features.index, ['diagnosis']], annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation between Features and Diagnosis')
plt.show()

# Violin plots for each feature grouped by the target variable
"""
for feature in features:
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=target, y=feature, data=data)
    plt.title(f'Violin Plot of {feature} by {target}')
    plt.show()
    """