import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pandas_datareader as data
# from keras.layers import Dense, Dropout, LSTM
# from keras.models import Sequential
import streamlit as st


start = '2010-01-01'
end = '2022-06-24'

st.title("Stock Price Prediction")

user_input = st.text_input("Enter Stock Symbol", "SBIN.NS")

df = data.DataReader('SBIN.NS', 'yahoo', start, end)


#  Describing Data

st.subheader("Data From 2010-2022")
st.write(df.describe())

#  Visualisations

st.subheader("Closing Price vs Time Chart")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

# Splitting data into Training and testing vectors
l = len(df)
data_training = pd.DataFrame(df['Close'][:int(l*0.70)]) # 0:2156
data_testing = pd.DataFrame(df['Close'][int(l*0.70):]) # 2156:3080


print(data_training.shape)
print(data_testing.shape)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1)) 

data_training_array = scaler.fit_transform(data_training)


#  Load my models from
from keras.models import load_model

model = load_model('keras_model.h5')

# Testing part

past_100_days = data_training.tail(100)

final_df = past_100_days.append(data_testing, ignore_index = True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

y_predicted.shape

scaler.scale_

scale_factor = 1/scaler.scale_[0]

y_predicted = y_predicted*scale_factor

#  Final output

plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.title('Original Price and Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
