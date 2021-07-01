import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings(action='ignore')

df=pd.read_csv('PreprocessedData.csv')

X=df.drop(columns=['TenYearCHD'])
y=df['TenYearCHD'].values


# MinMaxScaler
mmscaler=MinMaxScaler()
mmX=mmscaler.fit_transform(X)

# StandardScaler
stdscaler=StandardScaler()
stdX=stdscaler.fit_transform(X)


# RobustScaler
robscaler=RobustScaler()
robX=robscaler.fit_transform(X)


#train/test
stratified_shuffle_split = StratifiedShuffleSplit(train_size=0.7, test_size=0.3, n_splits=10)

knn_cv = KNeighborsClassifier (n_neighbors = 31)
 #10-fold cross validation
mm_scores = cross_val_score (knn_cv, mmX, y, cv = stratified_shuffle_split,scoring='accuracy') 
mm_avg_score = mm_scores.mean()
    
std_scores=cross_val_score (knn_cv, stdX, y, cv = stratified_shuffle_split,scoring='accuracy') 
std_avg_score = std_scores.mean()
    
rob_scores=cross_val_score (knn_cv, robX, y, cv = stratified_shuffle_split,scoring='accuracy') 
rob_avg_score = rob_scores.mean()

print('MinMax Average score :', round(mm_avg_score, 2))
print('Scores :', mm_scores)
print()

print('Standard Average score :', round(std_avg_score, 2))
print('Scores :', std_scores)
print()

print('Robust Average score :', round(rob_avg_score, 2))
print('Scores :', rob_scores)
