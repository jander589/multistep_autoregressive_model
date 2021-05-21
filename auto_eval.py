import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time

#Load LSTM model
lstm_model = tf.keras.models.load_model('autoregressive_model')

#Read csv file
df = pd.read_csv(
    'SensorValues.csv',
    parse_dates=['Time'],
    index_col='Time'
)
#Scale the valeus
scaler = MinMaxScaler(feature_range=(-1,1))
value = scaler.fit_transform(df.values)

#Select last 90% and split into x and y
val_size = int(len(value) * 0.9)
value = value[val_size:]
x_val = value[:-1]
y_val = value[1:]

#Create prediction datasets
x, y = [], []
history_size = 6*24*3 #One day of values, size of history
target_size = 6*24 #number of values to predict, one day
for i in range(history_size, len(value)-target_size):
    indices = range(i-history_size, i)
    x.append(x_val[indices])
    y.append(y_val[i:i+target_size])
x_val, y_val = np.asarray(x), np.asarray(y)

###########################MEASURE PERFORMANCE###########################
times = []
temp_error = []
hum_error = []
press_error = []
gas_error = []
lux_error = []
co_error = []

#Time measurement
y_predict = []
y_out = []
for k in range(100):
    predict = x_val[k]
    predict = predict.reshape(1, predict.shape[0], predict.shape[1])
    t0 = time.time()
    for i in range(target_size):
        yhat = lstm_model.predict(predict) #Predict new value
        y_predict.append(yhat) #Save predicted value
        #Append predicted value to predict
        temp = []
        temp.append(predict[0,1:,:])
        temp[0] = np.append(temp[0], yhat)
        predict = np.array(temp)
        predict = predict.reshape(predict.shape[0], history_size, 6)
    y_out = np.array(y_predict)
    t1 = time.time()
    times.append(t1-t0)

#Error measurement
for i in range(x_val.shape[0]):
    x_in = x_val[i]
    x_in = x_in.reshape(1, x_in.shape[0], x_in.shape[1])
    y_predict = []
    y_in = scaler.inverse_transform(y_val[i])
    for j in range(target_size):
        yhat = lstm_model.predict(x_in) #Predict new value
        yhat = scaler.inverse_transform(yhat)
        #Calculate the errors
        temp_error.append(abs(y_in[j,0]-yhat[0,0]))
        hum_error.append(abs(y_in[j,1]-yhat[0,1]))
        press_error.append(abs(y_in[j,2]-yhat[0,2]))
        gas_error.append(abs(y_in[j,3]-yhat[0,3]))
        lux_error.append(abs(y_in[j,4]-yhat[0,4]))
        co_error.append(abs(y_in[j,5]-yhat[0,5]))

#Calculate mean of all errors
temp_mean = np.mean(temp_error)
hum_mean = np.mean(hum_error)
press_mean = np.mean(press_error)
gas_mean = np.mean(gas_error)
lux_mean = np.mean(lux_error)
co_mean = np.mean(co_error)

#Calculate standard deviation of all errors
temp_std = np.std(temp_error)
hum_std = np.std(hum_error)
press_std = np.std(press_error)
gas_std = np.std(gas_error)
lux_std = np.std(lux_error)
co_std = np.std(co_error)

#Calculate average time and std to produce a prediction
time_mean = np.mean(times)
time_std = np.std(times)

#Print all values
print(f'Time mean={time_mean} std={time_std}')
print(f'Temp mean={temp_mean} std={temp_std}')
print(f'Hum mean={hum_mean} std={hum_std}')
print(f'Press mean={press_mean} std={press_std}')
print(f'Gas mean={gas_mean} std={gas_std}')
print(f'Lux mean={lux_mean} std={lux_std}')
print(f'co2 mean={co_mean} std={co_std}')