import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, add, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import yfinance as yf
import time
import matplotlib.pyplot as plt
from numba import jit

# Numba-optimized dataset creation
@jit(nopython=True)
def create_dataset_numba(dataset, time_step):
    """
    Numba-accelerated function to create a sliding window dataset.
    """
    n = len(dataset) - time_step - 1
    dataX = np.zeros((n, time_step))
    dataY = np.zeros(n)
    
    for i in range(n):
        dataX[i] = dataset[i:(i + time_step), 0]
        dataY[i] = dataset[i + time_step, 0]
    
    return dataX, dataY

# Numba-optimized RMSE calculation
@jit(nopython=True)
def calculate_rmse_numba(y_true, y_pred):
    """
    Numba-accelerated RMSE calculation.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Download and prepare the dataset
df = yf.download('SPY', start='2000-01-01', end='2025-01-01')
data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Parameters for dataset creation
time_step = 100
training_size = int(len(data_scaled) * 0.9)
test_size = len(data_scaled) - training_size
train_data = data_scaled[0:training_size, :]
test_data = data_scaled[training_size:len(data_scaled), :]

# Create datasets (X: sequences, y: next value)
X_train, y_train = create_dataset_numba(train_data, time_step)
X_test, y_test = create_dataset_numba(test_data, time_step)
# Note: For N‑BEATS we work with 2D input of shape (samples, time_step).

# Define an N‑BEATS block.
def nbeats_block(x, n_layers, hidden_units, input_size, output_size):
    """
    A single block for the N‑BEATS architecture.
    The block uses fully connected layers to produce a set of parameters (theta)
    that is split into a backcast and a forecast. The backcast is subtracted from the input
    to form a residual for the next block.
    """
    block_input = x
    for _ in range(n_layers):
        x = Dense(hidden_units, activation='relu')(x)
    # The theta vector has length equal to (input_size + output_size)
    theta = Dense(input_size + output_size)(x)
    # Backcast: first 'input_size' elements
    backcast = Lambda(lambda t: t[:, :input_size])(theta)
    # Forecast: remaining 'output_size' elements
    forecast = Lambda(lambda t: t[:, input_size:])(theta)
    # Update the residual by subtracting the backcast from the block input
    residual = Lambda(lambda tensors: tensors[0] - tensors[1])([block_input, backcast])
    return residual, forecast

# Build the complete N‑BEATS model.
def build_nbeats_model(input_size, output_size, n_blocks, n_layers, hidden_units):
    """
    Constructs an N‑BEATS model composed of multiple blocks.
    Each block receives a residual from the previous block.
    The final forecast is the sum of the forecasts from all blocks.
    """
    inputs = Input(shape=(input_size,))
    residual = inputs
    forecast_list = []
    
    # Stacking multiple N‑BEATS blocks
    for _ in range(n_blocks):
        residual, forecast = nbeats_block(residual, n_layers, hidden_units, input_size, output_size)
        forecast_list.append(forecast)
    
    # Sum the forecasts from all blocks to form the final output
    if len(forecast_list) > 1:
        final_forecast = add(forecast_list)
    else:
        final_forecast = forecast_list[0]
    
    model = Model(inputs=inputs, outputs=final_forecast)
    return model

# Model parameters
input_size = time_step       # Same as the length of the sliding window
output_size = 1              # Forecasting the next time step
n_blocks = 3
n_layers = 4
hidden_units = 128

model = build_nbeats_model(input_size, output_size, n_blocks, n_layers, hidden_units)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="mean_squared_error")

model.summary()

# Set callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")

# Make predictions
train_predict = model.predict(X_train, batch_size=32)
test_predict = model.predict(X_test, batch_size=32)

# Inverse transform the predictions and ground truth to original scale
train_predict_inv = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict_inv = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE using the Numba-optimized function
train_rmse = calculate_rmse_numba(y_train_inv.ravel(), train_predict_inv.ravel())
test_rmse = calculate_rmse_numba(y_test_inv.ravel(), test_predict_inv.ravel())

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

# Prepare predictions for plotting
trainPredictPlot = np.empty_like(data_scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_step:len(train_predict_inv)+time_step, :] = train_predict_inv

testPredictPlot = np.empty_like(data_scaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict_inv)+time_step:len(train_predict_inv)+time_step+len(test_predict_inv), :] = test_predict_inv

# Plotting the predictions against the actual stock prices
plt.figure(figsize=(15, 7))
plt.plot(scaler.inverse_transform(data_scaled),
         label='Actual Stock Price', 
         alpha=0.7,
         linewidth=2)
plt.plot(trainPredictPlot, 
         label='Training Predictions', 
         alpha=0.6,
         linewidth=2)
plt.plot(testPredictPlot, 
         label='Testing Predictions', 
         alpha=0.6,
         linewidth=2)
plt.title('SPY Stock Price Prediction using N‑BEATS Model',
         fontsize=14,
         pad=20)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Stock Price (USD)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot training history
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training', fontsize=14, pad=20)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()