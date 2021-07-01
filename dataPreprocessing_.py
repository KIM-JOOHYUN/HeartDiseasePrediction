import pandas as pd
import numpy as np

# Function to return a column which has max correlation.
def maxCorr(dataset, missing_col) :

    max_corr_col = dataset[missing_col].idxmax()

    return max_corr_col

#Function to sort the data by max_corr_col and fill missing values in missing_col by ffill
def sortCol(dataset, missing_col, max_corr_col) :

    dataset.sort_values(by = max_corr_col, inplace = True)

    dataset[missing_col].fillna(method = 'ffill',inplace = True)

    return dataset

#Function to preprocessing
def preprocessing(df) :
    # a dataframe for the absolute value of the correlation of each column. 
    corr = df.corr().abs()

    # Replace the correlation of the same column with 0
    corr = corr.replace(1, 0)

    #dataset's columns.
    df_col = df.columns

    # check every column
    for i in df_col :

        # if it has missing data fill them.
        if (df[i].isnull().sum() > 0) :

            # find a column which has max correaltion value
            max_corr_col = maxCorr(corr, i)

            # fill missing values by sortCol function
            df = sortCol(df, i, max_corr_col)
    
    return df

# Load the dataset from csv file
df = pd.read_csv('heart_disease.csv')
df.drop(columns = ['education'], inplace = True)

#check
print(df.head(10))

#check the missing data
print(df.isnull().sum())

#fill NaN values by preprocessing function
df = preprocessing(df)

#sort the dataset by index
df.sort_index(inplace = True)

#check that all missing data is filled
print(df.isnull().sum())


#store preprocessed data by csv file
df.to_csv('PreprocessedData.csv', index = False)

