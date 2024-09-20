import pandas as pd
import os
from utils import analyze_csv_files
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
from keras.layers import Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# Set up logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'output_{timestamp}.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

# Directory containing CSV files
# csv_directory = './.data/'
# analyze_csv_files(csv_directory)

TEMP_FILE_PATH = './.data/BelagaviTemperature.csv'

df = pd.read_csv(TEMP_FILE_PATH)

df = df.loc[2:,' Sulebhavi'].reset_index()
# print(df.head(10))
# print(df.dtypes)
# print(df.shape)
df = df.rename(columns={' Sulebhavi': 'temperature'}).drop(columns=['index'])
# print(df.head(10))
# print(df.dtypes)
# print(df.shape)


logging.info('start normalizing')

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['temperature']])

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

seq_length = 24 * 7  # Use one week of hourly data to predict the next hour
X, y = create_sequences(scaled_data, seq_length)

# Reshape input for LSTM [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Log GPU information
logging.info(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
logging.info(f"Is GPU available: {tf.test.is_gpu_available()}")
logging.info(f"Tensorflow built with CUDA: {tf.test.is_built_with_cuda()}")

# Create and compile the model
model = Sequential([
    LSTM(50, activation='tanh', 
         recurrent_activation='sigmoid', 
         input_shape=(seq_length, 1)),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='mse')

logging.info('start training')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform predictions and actual values
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions - y_test)**2))
logging.info(f"Root Mean Squared Error: {rmse}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title('Temperature Prediction')
plt.xlabel('Time')
plt.ylabel('Temperature')

# Save the plot as an image file
plot_filename = f'temperature_prediction_{timestamp}.png'
plt.savefig(plot_filename)
logging.info(f"Plot saved as {plot_filename}")

# Optionally, still show the plot if running interactively
# plt.show()

# Log training history
history_filename = f'training_history_{timestamp}.png'
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(history_filename)
logging.info(f"Training history plot saved as {history_filename}")

# ... rest of your code ...