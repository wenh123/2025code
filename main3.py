import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D
from tensorflow.keras.layers import LSTM, Concatenate
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from numba import jit, float64, int64
from numba.typed import List
import yfinance as yf
import time

# Numba-optimized dataset creation
@jit(nopython=True)
def create_dataset_numba(dataset, time_step):
    """
    Numba-accelerated function to create time series dataset.
    Uses JIT compilation for faster array operations.
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
    Faster than using sklearn's mean_squared_error for large arrays.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Load and prepare the dataset
df = yf.download('SPY', start='2000-01-01', end='2025-01-01')
data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

start_time = time.time()
# Parameters
time_step = 100
training_size = int(len(data_scaled) * 0.9)
test_size = len(data_scaled) - training_size
train_data = data_scaled[0:training_size,:]
test_data = data_scaled[training_size:len(data_scaled),:]

# Use Numba-optimized dataset creation
X_train, y_train = create_dataset_numba(train_data, time_step)
X_test, y_test = create_dataset_numba(test_data, time_step)

# Reshape input for the model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Transformer Block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Standard transformer encoder block with improved parameters.
    """
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

# LSTM Block
def lstm_block(inputs, units, dropout=0):
    """
    LSTM block with residual connection
    """
    x = LSTM(units, return_sequences=True)(inputs)
    x = Dropout(dropout)(x)
    return x

# Hybrid Model Definition
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# Transformer branch
transformer_output = transformer_encoder(inputs, head_size=256, num_heads=4, ff_dim=4, dropout=0.1)

# LSTM branch
lstm_output = lstm_block(inputs, units=128, dropout=0.1)

# Combine branches
combined = Concatenate()([transformer_output, lstm_output])
x = GlobalAveragePooling1D(data_format='channels_first')(combined)
x = Dropout(0.1)(x)
x = Dense(32, activation="relu")(x)
x = Dense(16, activation="relu")(x)
outputs = Dense(1, activation="linear")(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mean_squared_error"
)

# Model Summary
model.summary()

# Train the model with early stopping and reduced learning rate on plateau
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Make predictions
train_predict = model.predict(X_train, batch_size=32)
test_predict = model.predict(X_test, batch_size=32)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE using Numba-optimized function
train_rmse = calculate_rmse_numba(
    scaler.inverse_transform(y_train.reshape(-1, 1)).ravel(),
    train_predict.ravel()
)
test_rmse = calculate_rmse_numba(
    scaler.inverse_transform(y_test.reshape(-1, 1)).ravel(),
    test_predict.ravel()
)

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")

# Plotting the results
trainPredictPlot = np.empty_like(data_scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_step:len(train_predict)+time_step, :] = train_predict

testPredictPlot = np.empty_like(data_scaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+time_step:len(train_predict)+time_step+len(test_predict), :] = test_predict

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
plt.title('SPY Stock Price Prediction using Hybrid Transformer-LSTM Model',
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