import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

#Read csv file
df = pd.read_csv(
    'SensorValues.csv',
    parse_dates=['Time'],
    index_col='Time'
)
#Scale values to be between -1 and 1
scaler = MinMaxScaler(feature_range=(-1,1))
value = scaler.fit_transform(df.values)

#Divide into train, validation and test sets
train_size = int(len(value) * 0.8)
train = value[:train_size]
val = value[train_size:]

#Divide into input and output
x_train = train[:-1]
y_train = train[1:]
x_val = val[:-1]
y_val = val[1:]

#Create batches with moving window
#Training batch
history_size = 6*24*3
x, y = [], []
for i in range(history_size, len(train)-1):
    indices = range(i-history_size, i)
    x.append(x_train[indices])
    y.append(y_train[i])
x_train, y_train = np.asarray(x), np.asarray(y)
train_batch = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_batch = train_batch.batch(32).repeat()

#Validation batch
x, y = [], []
for i in range(history_size, len(val)-1):
    indices = range(i-history_size, i)
    x.append(x_val[indices])
    y.append(y_val[i])
x_val, y_val = np.asarray(x), np.asarray(y)
val_batch = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_batch = val_batch.batch(32).repeat()

#Define and compile model
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=32, return_sequences=True),
    tf.keras.layers.Dropout(0.01),
    tf.keras.layers.LSTM(units=32),
    tf.keras.layers.Dense(units=6)
])
lstm_model.compile(loss='mean_squared_error', optimizer='adam')

#Train the model
history = lstm_model.fit(
    train_batch,
    epochs=100,
    steps_per_epoch=1000,
    batch_size=32,
    validation_data=val_batch,
    validation_steps=100,
    shuffle=True
)
#Save model
lstm_model.save('autoregressive_model')