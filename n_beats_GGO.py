import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import yfinance as yf
import time
import matplotlib.pyplot as plt
from numba import jit
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------
# 1. DATA PREPARATION
# ------------------------------------------------------------

# Download SPY data from Yahoo Finance
df = yf.download('SPY', start='2000-01-01', end='2025-01-01')
data = df[['Close']].values

# Scale the data to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Define a fixed sliding window length (time_step)
time_step = 100
training_size = int(len(data_scaled) * 0.9)
train_data = data_scaled[:training_size, :]
test_data = data_scaled[training_size:, :]

def create_dataset(data, time_step):
    """
    Creates a sliding window time series dataset.
    Each sample is a window of `time_step` values and the target is the next value.
    """
    n = len(data) - time_step - 1
    X = np.zeros((n, time_step))
    y = np.zeros(n)
    for i in range(n):
        X[i] = data[i:(i + time_step), 0]
        y[i] = data[i + time_step, 0]
    return X, y

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# ------------------------------------------------------------
# 2. N‑BEATS MODEL FUNCTIONS
# ------------------------------------------------------------
def nbeats_block(x, n_layers, hidden_units, input_size, output_size):
    """
    A single block for the N‑BEATS architecture.
    The block passes the input through several fully connected layers,
    produces a theta vector that is split into a backcast and a forecast.
    The backcast is subtracted from the input to form the residual for the next block.
    """
    block_input = x
    for _ in range(n_layers):
        x = Dense(hidden_units, activation='relu')(x)
    theta = Dense(input_size + output_size)(x)
    backcast = Lambda(lambda t: t[:, :input_size])(theta)
    forecast = Lambda(lambda t: t[:, input_size:])(theta)
    residual = Lambda(lambda tensors: tensors[0] - tensors[1])([block_input, backcast])
    return residual, forecast

def build_nbeats_model(input_size, output_size, n_blocks, n_layers, hidden_units):
    """
    Constructs the full N‑BEATS model by stacking multiple N‑BEATS blocks.
    The final forecast is computed as the sum of the forecasts from all blocks.
    """
    inputs = Input(shape=(input_size,))
    residual = inputs
    forecast_list = []
    
    for _ in range(n_blocks):
        residual, forecast = nbeats_block(residual, n_layers, hidden_units, input_size, output_size)
        forecast_list.append(forecast)
        
    if len(forecast_list) > 1:
        final_forecast = add(forecast_list)
    else:
        final_forecast = forecast_list[0]
    
    model = Model(inputs=inputs, outputs=final_forecast)
    return model

# ------------------------------------------------------------
# 3. CANDIDATE EVALUATION FUNCTION
# ------------------------------------------------------------
def evaluate_candidate(config):
    """
    Builds and trains an N‑BEATS model using the candidate hyperparameters.
    Returns the RMSE on the test set (after inverse scaling).
    
    config: list or array-like [n_blocks, n_layers, hidden_units].
            Note: values are continuous and will be rounded to integer.
    """
    n_blocks = int(round(config[0]))
    n_layers = int(round(config[1]))
    hidden_units = int(round(config[2]))
    
    # Clear any previous Keras graph.
    tf.keras.backend.clear_session()
    
    model = build_nbeats_model(input_size=time_step, output_size=1, 
                               n_blocks=n_blocks, n_layers=n_layers, hidden_units=hidden_units)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), 
                  loss='mean_squared_error')
    
    # Train for a small number of epochs for a fast evaluation.
    _ = model.fit(X_train, y_train, 
                  validation_data=(X_test, y_test),
                  epochs=5, 
                  batch_size=32,
                  verbose=0)
    
    # Make predictions on the test set.
    preds = model.predict(X_test, batch_size=32)
    
    # Inverse transform predictions and true values.
    preds_inv = scaler.inverse_transform(preds)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    rmse = np.sqrt(np.mean((y_test_inv - preds_inv) ** 2))
    return rmse

# ------------------------------------------------------------
# 4. GREYLAG GOOSE OPTIMIZATION (GGO)
# ------------------------------------------------------------
# Search space bounds for the hyperparameters to be tuned.
bounds = {
    'n_blocks': (1, 5),      # Number of blocks: integer between 1 and 5
    'n_layers': (2, 6),      # Number of layers per block: integer between 2 and 6
    'hidden_units': (32, 256)  # Hidden units per layer: integer between 32 and 256
}

# GGO parameters
population_size = 10  # Number of candidate solutions (geese)
max_iter = 5          # Number of iterations for the optimization

# Initialize the population with random candidates.
# Each candidate is represented by a vector: [n_blocks, n_layers, hidden_units]
population = []
for _ in range(population_size):
    candidate = [
        np.random.uniform(bounds['n_blocks'][0], bounds['n_blocks'][1]),
        np.random.uniform(bounds['n_layers'][0], bounds['n_layers'][1]),
        np.random.uniform(bounds['hidden_units'][0], bounds['hidden_units'][1])
    ]
    population.append(candidate)
population = np.array(population)

# Evaluate the initial population.
print("Evaluating initial population...")
fitness = []
for candidate in population:
    f = evaluate_candidate(candidate)
    fitness.append(f)
    print(f"Candidate {np.around(candidate,2)} -> RMSE: {f:.4f}")
fitness = np.array(fitness)

# Identify the global best candidate.
best_idx = np.argmin(fitness)
global_best = population[best_idx].copy()
global_best_fitness = fitness[best_idx]
print(f"\nInitial best candidate: {np.around(global_best,2)} with RMSE: {global_best_fitness:.4f}")

# Optimization loop using a simple Greylag Goose strategy.
# Note: This is a simplified demo version of a nature‐inspired metaheuristic.
for iteration in range(max_iter):
    print(f"\nIteration {iteration+1}/{max_iter}:")
    for i in range(population_size):
        candidate = population[i]
        # Generate a random update vector.
        r = np.random.rand(3)
        # Update candidate by moving a fraction r toward the global best plus a small Gaussian perturbation.
        candidate_new = candidate + r * (global_best - candidate) + np.random.normal(scale=0.1, size=3)
        
        # Clip the candidate to respect the search bounds.
        candidate_new[0] = np.clip(candidate_new[0], bounds['n_blocks'][0], bounds['n_blocks'][1])
        candidate_new[1] = np.clip(candidate_new[1], bounds['n_layers'][0], bounds['n_layers'][1])
        candidate_new[2] = np.clip(candidate_new[2], bounds['hidden_units'][0], bounds['hidden_units'][1])
        
        # Evaluate the new candidate.
        new_fitness = evaluate_candidate(candidate_new)
        
        # If the new candidate performs better, update the population.
        if new_fitness < fitness[i]:
            population[i] = candidate_new
            fitness[i] = new_fitness
            print(f"  Updated candidate {i}: {np.around(candidate_new,2)} -> RMSE: {new_fitness:.4f}")
            # Update the global best if needed.
            if new_fitness < global_best_fitness:
                global_best = candidate_new.copy()
                global_best_fitness = new_fitness
                print(f"  ** New global best candidate: {np.around(global_best,2)} with RMSE: {global_best_fitness:.4f}")
    print(f"End of iteration {iteration+1}: Global best candidate: {np.around(global_best,2)} with RMSE: {global_best_fitness:.4f}")

print("\nOptimization completed!")
print("Best configuration found:")
best_n_blocks = int(round(global_best[0]))
best_n_layers = int(round(global_best[1]))
best_hidden_units = int(round(global_best[2]))
print("  n_blocks    =", best_n_blocks)
print("  n_layers    =", best_n_layers)
print("  hidden_units=", best_hidden_units)
print("  Fitness (RMSE) =", global_best_fitness)

# ------------------------------------------------------------
# 5. FINAL MODEL TRAINING WITH BEST CONFIGURATION
# ------------------------------------------------------------
tf.keras.backend.clear_session()

final_model = build_nbeats_model(input_size=time_step, output_size=1, 
                                 n_blocks=best_n_blocks, n_layers=best_n_layers, hidden_units=best_hidden_units)
final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss="mean_squared_error")
final_model.summary()

# Callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

start_time = time.time()
history = final_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
end_time = time.time()
print(f"Final model training time: {end_time - start_time:.2f} seconds")

# Make predictions with the final model.
train_predict = final_model.predict(X_train, batch_size=32)
test_predict = final_model.predict(X_test, batch_size=32)

# Inverse transform predictions and true values.
train_predict_inv = scaler.inverse_transform(train_predict)
test_predict_inv = scaler.inverse_transform(test_predict)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse_train = np.sqrt(np.mean((y_train_inv - train_predict_inv)**2))
rmse_test = np.sqrt(np.mean((y_test_inv - test_predict_inv)**2))
print(f"Final model Train RMSE: {rmse_train:.4f}")
print(f"Final model Test RMSE: {rmse_test:.4f}")

# ------------------------------------------------------------
# 6. PLOTTING THE RESULTS
# ------------------------------------------------------------
# Prepare the predictions so they align with the original dataset.
trainPredictPlot = np.empty_like(data_scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_step:len(train_predict_inv)+time_step, :] = train_predict_inv

testPredictPlot = np.empty_like(data_scaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict_inv)+time_step:len(train_predict_inv)+time_step+len(test_predict_inv), :] = test_predict_inv

# Plot the actual and predicted stock prices.
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
plt.title('SPY Stock Price Prediction using N‑BEATS with GGO-Tuned Architecture',
         fontsize=14,
         pad=20)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Stock Price (USD)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot the training loss history.
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Final Model Loss During Training', fontsize=14, pad=20)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()