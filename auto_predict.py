import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#Load LSTM model
lstm_model = tf.keras.models.load_model('autoregressive_model')

#Read csv file
df = pd.read_csv(
    'SensorValues.csv',
    parse_dates=['Time'],
    index_col='Time'
)

#Scale the values
scaler = MinMaxScaler(feature_range=(-1,1))
value = scaler.fit_transform(df.values)

#Select last 90% of values and spliut into x and y values
val_size = int(len(value) * 0.9)
test = value[val_size:]
x_test = test[:-1]
y_test = test[1:]

#Create prediction dataset
history_size = 6*24*3 #window size
target_size = 6*24 #number of values to predict, 24 hours 
predict = []
test_l = len(x_test)
predict.append(x_test[range(history_size)])
predict = np.array(predict)
history = predict

#Predict values
y_predict = []
for i in range(target_size):
    yhat = lstm_model.predict(predict) #Predict new value
    y_predict.append(yhat) #Save predicted value
    #Append predicted value to predict
    temp = []
    temp.append(predict[0,1:,:])
    temp[0] = np.append(temp[0], yhat)
    predict = np.array(temp)
    predict = predict.reshape(predict.shape[0], history_size, 6)
y_predict = np.array(y_predict)

#Inverse scale the values
for i in range(y_predict.shape[1]):
    y_predict[:, i, :] = scaler.inverse_transform(y_predict[:, i, :])
y_test = scaler.inverse_transform(y_test)

#Plot the values with the corresponding observed values
font = {'family': 'normal', 'weight' : 'bold', 'size' : 15}
plt.rc('font', **font)
fig, axs = plt.subplots(3,2)
axs[0,0].plot(y_test[history_size:history_size+target_size, 0], color='blue')
axs[0,0].plot(y_predict[:, :, 0], 'r+')
axs[0,0].set_title('Temperature')
axs[0,0].set_ylim([19, 23])
axs[0,1].plot(y_test[history_size:history_size+target_size, 1], color='blue')
axs[0,1].plot(y_predict[:, :, 1], 'r+')
axs[0,1].set_title('Humidity')
axs[0,1].set_ylim([18, 50])
axs[1,0].plot(y_test[history_size:history_size+target_size, 2], color='blue')
axs[1,0].plot(y_predict[:, :, 2], 'r+')
axs[1,0].set_title('Pressure')
axs[1,0].set_ylim([942, 1000])
axs[1,1].plot(y_test[history_size:history_size+target_size, 3], color='blue')
axs[1,1].plot(y_predict[:, :, 3], 'r+')
axs[1,1].set_title('gas/1000')
axs[1,1].set_ylim([-100, 1600])
axs[2,0].plot(y_test[history_size:history_size+target_size, 4], color='blue')
axs[2,0].plot(y_predict[:, :, 4], 'r+')
axs[2,0].set_title('Lux')
axs[2,0].set_ylim([-10, 75])
axs[2,1].plot(y_test[history_size:history_size+target_size, 5], color='blue', label='Actual value')
axs[2,1].plot(y_predict[:, :, 5], 'r+', label='Predicted value')
axs[2,1].set_title('co2')
axs[2,1].set_ylim([300, 1200])
plt.subplots_adjust(hspace=0.5)
plt.legend(loc='upper center', bbox_to_anchor=(-0.1, 4.5))
plt.show()