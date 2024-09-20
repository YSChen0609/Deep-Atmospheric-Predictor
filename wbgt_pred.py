import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import logging
import os
import tensorflow as tf
from datetime import datetime

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'wbgt_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

# Set up plot directory
plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)

# Load the data
df = pd.read_csv('./.data/Sulebhavi_processed.csv')
print(df.head())

# 1. Split WBGT column from the dataframe
wbgt_column = [col for col in df.columns if 'wbgt' in col.lower()][0]
target = df[wbgt_column]
features = df.drop(columns=[wbgt_column])

logging.info(f"Target column: {wbgt_column}")
logging.info(f"Number of features: {features.shape[1]}")

# 2. Prepare the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# 3. Create sequences for LSTM
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 5. Scale the data
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# 6. Create sequences
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, time_steps)

logging.info(f"Training data shape: {X_train_seq.shape}")
logging.info(f"Testing data shape: {X_test_seq.shape}")

# 7. Build the LSTM model
model = Sequential([
    LSTM(50, activation='tanh', 
         recurrent_activation='sigmoid', 
         input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), 
         return_sequences=True),
    LSTM(50, activation='tanh', recurrent_activation='sigmoid'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 8. Train the model
history = model.fit(
    X_train_seq, y_train_seq,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 9. Evaluate the model
loss = model.evaluate(X_test_seq, y_test_seq, verbose=0)
logging.info(f'Test loss: {loss}')

# 10. Make predictions
y_pred = model.predict(X_test_seq)

# 11. Inverse transform the predictions and actual values
y_pred = scaler_y.inverse_transform(y_pred)
y_test_2d = y_test_seq.reshape(-1, 1)  # Reshape y_test to 2D array
y_test = scaler_y.inverse_transform(y_test_2d)

# 12. Calculate and log RMSE
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
logging.info(f'Root Mean Squared Error: {rmse}')

# 13. Plot and save actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('WBGT: Actual vs Predicted')
plt.savefig(os.path.join(plot_dir, 'wbgt_actual_vs_predicted.png'))
plt.close()

# 14. Plot and save training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(plot_dir, 'wbgt_training_history.png'))
plt.close()

logging.info("Model training and evaluation completed. Check the logs and plots directories for results.")