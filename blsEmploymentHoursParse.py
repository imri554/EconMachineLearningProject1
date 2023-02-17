
import re
import pandas as pd
import openpyxl

class BLSEmploymentHoursReader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        df = pd.read_csv(self.data_path)
    
        # Load the Excel file
        df = pd.read_csv('data/employerHoursWithTitle/employerHoursWithTitle.csv')
        df = df.replace(to_replace=[-1, -2, -3, -4, -5], value=pd.NaT)

        # Create a dictionary to hold the sums for each year
        year_sums = {}

        # Loop over the columns in the DataFrame
        for col in df.columns:
            # Use regex to extract the year from the column header
            match = re.search(r'(\d{4})', col)
            if match:
                year = match.group(1)
                if year not in year_sums:
                    year_sums[year] = df[col]
                else:
                    year_sums[year] += df[col]

        # Convert the year_sums dictionary to a DataFrame
        sums_df = pd.DataFrame(year_sums)

        # Transpose the DataFrame so that the years become the column headers
        #sums_df = sums_df.transpose()

        # Write the DataFrame to a new Excel file
       

        # #drop and impute
        # num_rows = sums_df.shape[0]
        # num_cols = sums_df.shape[1]
        # threshold = 0.8

        # num_cols_missing = (sums_df.isna().sum() > threshold * num_rows).sum()

        # print("Number of columns with more than 80% missing data:", num_cols_missing)
        # print(num_cols_missing / num_cols)
        
        # #we should drop columns that have more than 50% missing data

        # # get list of columns with more than 50% of values missing
        # cols_to_drop = sums_df.columns[sums_df.isna().sum() > threshold * num_rows]

        # # drop columns from DataFrame
        # sums_df = sums_df.drop(columns=cols_to_drop)

        # print("Number of columns after dropping:", sums_df.shape[1])
        #df = df.fillna(df.mode().iloc[0])
        #print(df.isnull().sum())

        #check to make sure that we have at least 50% of values in each column
        #if not, drop the column

        #now, imputation
        #impute with the most common value
        sums_df = sums_df.fillna(sums_df.mode().iloc[0])
        #sums_df.to_excel('consolidated.xlsx')

        return sums_df





test = BLSEmploymentHoursReader('data/employerHoursWithTitle/employerHoursWithTitle.csv').load_data()
print(test)