import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
warnings.filterwarnings(action='ignore')

df=pd.read_csv('PreprocessedData.csv')

X=df.drop(columns=['TenYearCHD'])
y=df['TenYearCHD'].values

#Scaling data(MinMax Scaler)
scaler=MinMaxScaler()
X=scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#k value candidate
k_list = range(1,50)
accuracies = []  
max=0  #the max accuracy
index=0   #the k value which has max accuracy

#Find the best K value
for k in k_list:
    knn= KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train,y_train)
    accuracies.append(knn.score(x_test,y_test))
    if(max<knn.score(x_test,y_test)):
        max=knn.score(x_test,y_test)
        index=k

#display best k value
print("The best k value :",index)
#display best accuracy
print("** The accuracy of prediction: " , max)

#Draw a plot showing the accuracy of each K value.
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Classifier Accuracy")
plt.show()





