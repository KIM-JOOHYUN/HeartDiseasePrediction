import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# load the preprocessed dataset
df = pd.read_csv('PreprocessedData.csv')

#divide the dataset into input feature and target attribute
x = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

# Scaling
standard_scaler = StandardScaler()
x = standard_scaler.fit_transform(x)

# Splitting the dataset into Training and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Parameter candidate
C = np.logspace(-3, 3, 7)

penalty = ["l2"]

param_grid = dict(C=C, penalty=penalty)

# Create Logistic Regression model 
logistic_model = LogisticRegression()

# Find a model which has the best parameter by Grid Search
grid_search_model = GridSearchCV(logistic_model, param_grid, cv=5)

# fit the data to the best model(training)
best_model = grid_search_model.fit(x_train, y_train)

# Measure accuracy of this model
best_score = grid_search_model.best_score_

# print the result(best accuracy, the best parameter)
print('Best Score : ', best_score)
print('Best C :', best_model.best_estimator_.get_params()['C'])
print('Best Penalty : ', best_model.best_estimator_.get_params()['penalty'])

#######################################

# predict y_test values by x_test
y_pred = grid_search_model.predict(x_test)

# make confusion matrix by y_test and y_pred
cnf_metrix = metrics.confusion_matrix(y_test, y_pred)

# calculate accurancy by confusion matrix
total = np.sum(cnf_metrix)

TP = cnf_metrix[0][0]
TN = cnf_metrix[1][1]
FP = cnf_metrix[1][0]
FN = cnf_metrix[0][1]

confusion_accurancy = round((TP + TN) / total, 2)


#the model's score
best_score = round(grid_search_model.best_score_, 2)

#display the score of model and the accurancy which calculated by confusion matrix
print('Confusion_matrix_score :', confusion_accurancy)
print('Model_score : ', best_score)


# draw heatmap by  confusion matrix
sns.heatmap(pd.DataFrame(cnf_metrix), annot=True, cmap='YlGnBu', fmt='g')
plt.title("Confusion matrix")
plt.ylabel("Actual label")
plt.xlabel("Predict label")
plt.show()




