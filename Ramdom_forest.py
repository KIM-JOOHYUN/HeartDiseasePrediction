import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Read preprocessed data from csv file
df = pd.read_csv('PreprocessedData.csv')

# divide the dataset into input feature and targe attribute
x = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

# Scaling
standard_scaler = StandardScaler()
x = standard_scaler.fit_transform(x)

# Split the dataset into train and test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Create random forest model
random_forest_model = RandomForestClassifier()


# fit the data to random forest model
ensemble_model = random_forest_model.fit(x_train, y_train)

# predict target value with test input data
y_pred = ensemble_model.predict(x_test)

# Make confusion matrix by predict result
cnf_metrix = metrics.confusion_matrix(y_test, y_pred)

# calcualte accuracy by confusion matrix
total = np.sum(cnf_metrix)

TP = cnf_metrix[0][0]
TN = cnf_metrix[1][1]
FP = cnf_metrix[1][0]
FN = cnf_metrix[0][1]

confusion_accurancy = round((TP + TN) / total, 2)

# display the result(the model's accuracy)
print('Confusion_matrix_score :', confusion_accurancy)

# Draw heatmap with confusion matrix
sns.heatmap(pd.DataFrame(cnf_metrix), annot=True, cmap='YlGnBu', fmt='g')
plt.title("Confusion matrix", y=1.1)
plt.ylabel("Actual label")
plt.xlabel("Predict label")
plt.show()
