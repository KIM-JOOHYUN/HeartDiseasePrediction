import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

df = pd.read_csv('PreprocessedData.csv')

#divide data  by input feature and  target attribute
x = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

#Standard Scaler
standard_scaler = StandardScaler()
std_x = standard_scaler.fit_transform(x)

#Robust Scaler
robust_scaler = RobustScaler()
rob_x = robust_scaler.fit_transform(x)

# Create Random Forest model 
random_forest_model = RandomForestClassifier()

#use StratifiedShuffleSplit for K-Fold(K = 10)
stratified_shuffle_split = StratifiedShuffleSplit(
    train_size=0.7, test_size=0.3, n_splits=10)

#10-fold cross validataion
std_scores = cross_val_score(random_forest_model, std_x, y, cv=stratified_shuffle_split)
std_avg_score = std_scores.mean()

rob_scores = cross_val_score(random_forest_model, rob_x, y, cv=stratified_shuffle_split)
rob_avg_score = rob_scores.mean()

#print result
print('Standard Average score :', round(std_avg_score, 2))
print('Scores :', std_scores)
print('\n')
print('Robust Average score :', round(rob_avg_score, 2))
print('Scores :', rob_scores)
