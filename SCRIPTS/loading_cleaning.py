"""
Samuel Rea
last modified: Samuel Rea july 1, 2024
Description: This file loads and cleans the data set so it is ready to pass into the machine learning
models in the next script
"""


#importing packages
import pandas as pd

#loading in the data
#https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
file_path = 'data.csv'
data=pd.read_csv(file_path)

#Malignant=1 and Benign=0
mapping = {"M":1,"B":0}
data['diagnosis'] = data['diagnosis'].replace(mapping)
#checking the data frame
print(data)
data=data.drop(columns=["id","Unnamed: 32"])
# Display basic information about the dataset
print(data.info())

#saving the new data frame for later use
cleaned_file_path = 'Clean_BC_Data.csv'
data.to_csv(cleaned_file_path, index=False)
print(f"Cleaned data saved to {cleaned_file_path}")

