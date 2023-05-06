import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import requests
import datetime as dt
import csv

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM

# access data from the binance api
url = 'https://api.binance.com/api/v3/klines'
req = requests.get(url, params={'symbol': 'ETHUSDT', 'interval': '1h', 'limit': 1000})
csv_fields = ['Kline open time', 'Open price', 'High price', 'Low price', 'Close price', 'Volume', 'Kline Close time', 
              'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Unused']
data = req.json()

# save the data to csv file
with open('data.csv', 'w') as outcsv:   
    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow(csv_fields)
    for item in data:
        writer.writerow([item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8], item[9], item[10], item[11]])

# format the csv file into dataframe
df = pd.read_csv('data.csv')
df['Kline open time'] = pd.to_datetime(df['Kline open time'], unit='ms')

# prepare the data 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df['Close price'].values.reshape(-1,1))
span = 60

# first 400 values for training, the last 50 are for testing
x_train, y_train = [], []

for x in range(span, len(scaled_data-200)):
    x_train.append(scaled_data[x-span:x, 0])
    y_train.append(scaled_data[x, 0])

# convert into numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # add one additional dimension

# build the model
model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=128))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #prediction of the closing value

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=64)

# test the model accuracy on existing data
model_inputs = df[len(df) - 200 - span:]
model_inputs = model_inputs['Close price'].values.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

# make prediction on test data
x_test = []

for x in range(span, len(model_inputs)):
    x_test.append(model_inputs[x-span:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# plot the test predictions
plt.plot(df['Kline open time'][800:], df['Close price'][800:], color='black', label='actual company price')
plt.plot(df['Kline open time'][800:], predicted_prices, color='green', label='predicted price')
plt.title('Ether share price')
plt.xlabel('Kline open time')
plt.ylabel('Price, $')
plt.show()