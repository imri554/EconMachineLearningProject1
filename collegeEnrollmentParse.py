
import re
import pandas as pd
import openpyxl

# Load the Excel file
df = pd.read_csv('proj1/data/employerHoursWithTitle/employerHoursWithTitle.csv')
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
sums_df.to_excel('consolidated.xlsx')

