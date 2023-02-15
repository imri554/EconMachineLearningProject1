import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import sys

sys.path.append('/Users/imrihaggin1/Library/CloudStorage/GoogleDrive-imri_haggin@brown.edu/My Drive/Brown Work/junior year/machinelearning/proj1')

#birth year column: R0536402

from BLSSchoolingDataReader import BLSSchoolingDataReader
from BLSIncomeDataReader import BLSIncomeDataReader

#bring in the data
schoolData = BLSSchoolingDataReader('proj1data/schoolingInfo/schoolingInfo.csv').load_data()
incomeData = BLSIncomeDataReader('proj1data/shortIncomeBLS/shortIncome.csv').load_data()

### need to reshape the sets to be 3d tensor that allows for the ltsm to work through the time steps

data = pd.concat([schoolData, incomeData], axis=1)
data = pd.DataFrame(data)

x = data.drop(columns=['R0536402']).values
y = data['R0536402'].values

num_timesteps = len(data['R0536402'].unique())
num_features = x.shape[1]
num_samples = x.shape[0]

x_3d = x.reshape(num_samples, num_timesteps, num_features)

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
