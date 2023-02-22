import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler

sys.path.append('/Users/imrihaggin1/Library/CloudStorage/GoogleDrive-imri_haggin@brown.edu/My Drive/Brown Work/junior year/machinelearning/proj1')


# #bring in the data
# incomeData = BLSIncomeDataReader('data/yearlyIncome/yearlyIncome.csv').load_data()

# incomeData.drop(columns=['R0000100'], axis=1)
# #incomeData['measure'] = "income"

# employmentHours = BLSEmploymentHoursReader('data/employerHoursWithTitle/employerHoursWithTitle.csv').load_data()


# investmentData = pd.read_excel('data/aiInvestmentData.xlsx')

# gdpData = pd.read_excel('data/gdpAlone.xlsx')
# gdpData = gdpData.transpose()
# gdpData.reset_index(drop=True, inplace=True)
# new_header = gdpData.iloc[0] #grab the first row for the header
# gdpData = gdpData[1:] #take the data less the header row
# gdpData.columns = new_header #set the header row as the df header
# gdpData = gdpData.drop(columns=[gdpData.columns[0]])
# gdpData = pd.concat([gdpData]*8984)
# #gdpData.to_excel("gdpDataFormatted.xlsx")

# incomeData = incomeData.drop(columns=["R0000100"])
# incomeCols = incomeData.columns
#incomeData.to_excel("incomeDataFormatted.xlsx")

#employmentHours.to_excel("employmentHoursFormatted.xlsx")
#gdpData = gdpData.reindex(columns=incomeCols)

### need to reshape the sets to be 3d tensor that allows for the ltsm to work through the time steps
#data = pd.concat([incomeData, employmentHours, gdpData], keys = ["income", "employmentHours", "gdp"])
data = pd.read_excel('data/collatedData_copy.xlsx')
data = data.drop(columns=['ID', 'RecordType'])

data = data.to_numpy()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
scaled_data = pd.DataFrame(scaled_data)


data = scaled_data.to_numpy()


time_steps = data.shape[1]
features = 3


data = data.reshape((-1, time_steps, features))

incomeData = pd.read_excel('incomeDataFormatted.xlsx')

yValues = incomeData["2019"].values
yValues = np.reshape(yValues, (8984, 1))

###is this important? seems to take a way a lot of the data
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(data)
# scaled_data = pd.DataFrame(scaled_data)

#print(data.columns)
#test train split
x_train, x_test, y_train, y_test = train_test_split(data, yValues, test_size=0.2, random_state=42)

x_train = np.array(x_train).astype('float32')
y_train = np.array(y_train).astype('float32')
x_test = np.array(x_test).astype('float32')
y_test = np.array(y_test).astype('float32')


# scaler = MinMaxScaler(feature_range=(0, 1))
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# Define the LSTM model
model = tf.keras.Sequential()
optimizer = tf.keras.optimizers.RMSprop(clipnorm=1.0)

model.add(tf.keras.layers.LSTM(units=128, input_shape=(15, 3), return_sequences = True))
model.add(tf.keras.layers.LSTM(128))
model.add(tf.keras.layers.Dense(1))

# Compile the model
model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])

# Train the model
history = model.fit(x_train, y_train, batch_size=32, epochs=10)

# Evaluate the model
loss, mae = model.evaluate(x_test, y_test)

print(f'Test loss: {loss:.4f}, Test MAE: {mae:.4f}')



# Make predictions using the trained model
#y_pred = model.predict(x_test)
