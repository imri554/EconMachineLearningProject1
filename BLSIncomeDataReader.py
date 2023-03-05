import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler


class BLSIncomeDataReader:
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

       
        #impute with the most common value
        df = df.fillna(df.mode().iloc[0])

        return df


test = BLSIncomeDataReader('data/yearlyIncome/yearlyIncome.csv').load_data()
print(test)
summary = test.describe()
print(summary)
