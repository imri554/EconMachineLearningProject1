import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import sys

sys.path.append('/Users/imrihaggin1/Library/CloudStorage/GoogleDrive-imri_haggin@brown.edu/My Drive/Brown Work/junior year/machinelearning/proj1')

from BLSIncomeDataReader import BLSIncomeDataReader
from blsEmploymentHoursParse import BLSEmploymentHoursReader

#bring in the data
incomeData = BLSIncomeDataReader('data/yearlyIncome/yearlyIncome.csv').load_data()

incomeData.drop(columns=['R0000100'], axis=1)
#incomeData['measure'] = "income"

employmentHours = BLSEmploymentHoursReader('data/employerHoursWithTitle/employerHoursWithTitle.csv').load_data()


investmentData = pd.read_excel('data/aiInvestmentData.xlsx')

gdpData = pd.read_excel('data/gdpAlone.xlsx')
gdpData = gdpData.transpose()
new_header = gdpData.iloc[0] #grab the first row for the header
gdpData = gdpData[1:] #take the data less the header row
gdpData.columns = new_header #set the header row as the df header
gdpData = gdpData.drop(columns=[gdpData.columns[0]])
gdpData = pd.concat([gdpData]*8984)

incomeData = incomeData.drop(columns=["R0000100"])
incomeCols = incomeData.columns
employmentHours = employmentHours.reindex(columns=incomeCols)
gdpData = gdpData.reindex(columns=incomeCols)


### need to reshape the sets to be 3d tensor that allows for the ltsm to work through the time steps
#data = pd.concat([incomeData, employmentHours, gdpData], keys = ["income", "employmentHours", "gdp"])
data = pd.concat([gdpData, employmentHours, incomeData], axis = 1)
#data = pd.concat([data, gdpData], keys = ["gdp"])
data = data.to_numpy()
time_steps = gdpData.shape[1]
features = 3


#need to duplicate the GDP line so that it matches the other data sets
#8983 times it should be



# x = data.drop(columns=['R0536402']).values
# y = data['R0536402'].values

#1961 - 2021


numdata = data.reshape((-1, time_steps, features))

y_oh = tf.keras.utils.to_categorical(y - y.min())

#lets try to predict income from wages and salary in the past year 2019
#U4282300
input_shape = (num_samples, num_timesteps, num_features)

###is this important? seems to take a way a lot of the data
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(data)
# scaled_data = pd.DataFrame(scaled_data)

#print(data.columns)
#test train split
x_train, x_test, y_train, y_test = train_test_split(x_3d, y, test_size=0.2, random_state=42)


# Define the LSTM model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=128, input_shape=input_shape, return_sequences = True))
model.add(tf.keras.layers.LSTM(64))
model.add(tf.keras.layers.Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error', 'accu'])

# Train the model
history = model.fit(x_train, y_train, batch_size=32, epochs=50)

# Evaluate the model
loss, mae = model.evaluate(x_test, y_test)

print(f'Test loss: {loss:.4f}, Test MAE: {mae:.4f}')



# Make predictions using the trained model
#y_pred = model.predict(x_test)
