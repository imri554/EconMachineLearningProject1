
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

        #replace survey values that were not answered by participant with Nan
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
        #now, imputation
        #impute with the most common value
        sums_df = sums_df.fillna(sums_df.mode().iloc[0])
        #sums_df.to_excel('consolidated.xlsx')

        return sums_df





test = BLSEmploymentHoursReader('data/employerHoursWithTitle/employerHoursWithTitle.csv').load_data()
print(test)