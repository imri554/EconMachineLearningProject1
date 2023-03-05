import re
import pandas as pd
import openpyxl

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor


class AnnualFundamentalsReader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        df = pd.read_excel(self.data_path)

        #Make dummy variables for the tickers
        dummy_df = pd.get_dummies(df[['Ticker Symbol']])

        df = pd.concat([df, dummy_df], axis=1)

        df = df.drop(['Ticker Symbol'], axis=1)

        print(df.head())

        # Calculate the percentage of missing values in each column
        missing_values = df.isna().sum() / len(df)

        df = df.drop(columns=missing_values[missing_values > 0.5].index)

        #imputation
        imputer = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=2))

        ##TODO: ADD THE TICKERS
        columns_for_pca = [
        'Accounts Payable and Accrued Liabilities - Increase/(Decrease)',
        'Acquisitions',
        'Capital Expenditures',
        'Capital Expend Property, Plant and Equipment Schd V',
        'Cash and Cash Equivalents - Increase/(Decrease)',
        'Current Debt - Changes',
        'Data Year - Fiscal',
        'Long-Term Debt - Issuance',
        'Long-Term Debt - Reduction',
        'Depreciation and Amortization (Cash Flow)',
        'Total Debt Including Current',
        'Cash Dividends (Cash Flow)',
        'Earnings Per Share (Diluted) - Including Extraordinary Items',
        'Financing Activities - Net Cash Flow',
        'Gross Profit (Loss)',
        'Interest Paid - Net',
        'Inventory - Decrease (Increase)',
        'Increase in Investments',
        'Investing Activities - Net Cash Flow',
        'Short-Term Investments - Change',
        'Operating Activities - Net Cash Flow',
        'Purchase of Common and Preferred Stock',
        'Retained Earnings',
        'Accounts Receivable - Decrease (Increase)',
        'Revenue - Total',
        'Sale of Investments',
        'Sale of Property',
        'Sale of Property, Plant and Equipment and Investments - Gain (Loss)',
        'Sale of Common and Preferred Stock',
        'Stockholders Equity - Total',
        'Income Taxes - Accrued - Increase/(Decrease)',
        'Excess Tax Benefit Stock Options - Cash Flow Operating',
        'Excess Tax Benefit of Stock Options - Cash Flow Financing',
        'Deferred Taxes (Cash Flow)',
        'Income Taxes Paid',
        'Research and Development Expense',
        'Research & Development - Prior', 
        #INCLUDES THE DATES
        ]
        columns_for_pca = columns_for_pca + list(dummy_df.columns)
        # select data for PCA
        data_for_pca = df[columns_for_pca]

        data_for_pca = pd.DataFrame(imputer.fit_transform(data_for_pca), columns=columns_for_pca)

        #aggregating the 50 companies financials, by year by summation
        imputed_yearly = data_for_pca.groupby('Data Year - Fiscal').sum()

        imputed_data = data_for_pca

        imputed_data.to_excel('processedFundamentalsImputed.xlsx')

        ### PCA

        #standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(imputed_yearly)

        # perform PCA
        pca = PCA(n_components=2)
        pca.fit(scaled_data)
        pca_data = pca.transform(scaled_data)

        # create new dataframe with PCA results
        pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])

        # # add Ticker Symbol and Company Name to PCA dataframe
        # pca_df['Ticker Symbol'] = df['Ticker Symbol'].values
        # pca_df['Company Name'] = df['Company Name'].values

        # display PCA dataframe
        print(pca_df.head())
        pca_df.to_excel('pcaFundamentals.xlsx')

        # variance ratios of each principal component
        variance_ratios = pca.explained_variance_ratio_

        # print variance ratios
        print("Variance ratios of each principal component:")
        for i, ratio in enumerate(variance_ratios):
            print(f"PC{i+1}: {ratio:.2f}")

        data_transformed = pca.transform(imputed_yearly)

        #replace the original columns with the principal components
        data_transformed_df = pd.DataFrame(data_transformed, columns=['PC1', 'PC2'])

        data_transformed_df.to_excel('pcaTransformedFundamentals.xlsx')

        return df


#DROP I AND J AND M AND K P Q R S T 



test = AnnualFundamentalsReader('/Users/imrihaggin1/Library/CloudStorage/GoogleDrive-imri_haggin@brown.edu/My Drive/Brown Work/junior year/machinelearning/proj1/EconMachineLearningProject1/data/data2.0/creating an AI Index/annualFundamentals.xlsx').load_data()
print(test)