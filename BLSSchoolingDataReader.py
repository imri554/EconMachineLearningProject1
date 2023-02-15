import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler


class BLSSchoolingDataReader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        # Load the data from the CSV file
        df = pd.read_csv(self.data_path)
    
        # imputation; a lot of the data had no response and so dropping
        #NA's would cut too much of the dataset

        #should drop the column if there's too much missing

        #impute if there is enough data in the column

        df = df.replace(to_replace=[-1, -2, -3, -4, -5], value=pd.NaT)

        num_rows = df.shape[0]
        num_cols = df.shape[1]
        threshold = 0.5

        num_cols_missing = (df.isna().sum() > threshold * num_rows).sum()

        print("Number of columns with more than 50% missing data:", num_cols_missing)
        print(num_cols_missing / num_cols)
        
        #we should drop columns that have more than 50% missing data

        # get list of columns with more than 50% of values missing
        cols_to_drop = df.columns[df.isna().sum() > threshold * num_rows]

        # drop columns from DataFrame
        df = df.drop(columns=cols_to_drop)

        print("Number of columns after dropping:", df.shape[1])
        #df = df.fillna(df.mode().iloc[0])
        #print(df.isnull().sum())

        #check to make sure that we have at least 50% of values in each column
        #if not, drop the column

        #now, imputation
        #impute with the most common value
        df = df.fillna(df.mode().iloc[0])

        return df


test = BLSSchoolingDataReader('proj1data/schoolingInfo/schoolingInfo.csv').load_data()
print(test)
summary = test.describe()
print(summary)

#Testing to make sure that there are no null values left, and that the msot common value is used for imputation
# print("done")
# print(test.isnull().sum())

# for col in test.columns[:10]:
#     print(col)
#     print(test[col].value_counts())