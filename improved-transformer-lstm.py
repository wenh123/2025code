import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, LayerNormalization, MultiHeadAttention, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    stock_data_calc = stock_data.copy()
    
    # Basic indicators
    stock_data_calc['Ups_and_Downs'] = stock_data_calc['Close'].diff().fillna(0)
    stock_data_calc['Turnover'] = stock_data_calc['Volume'] * stock_data_calc['Close']
    stock_data_calc['Change_Percent'] = stock_data_calc['Close'].pct_change().fillna(0) * 100
    
    # RSI
    delta = stock_data_calc['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_data_calc['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    stock_data_calc['MA5'] = stock_data_calc['Close'].rolling(window=5).mean()
    stock_data_calc['MA20'] = stock_data_calc['Close'].rolling(window=20).mean()
    
    # MACD
    exp1 = stock_data_calc['Close'].ewm(span=12, adjust=False).mean()
    exp2 = stock_data_calc['Close'].ewm(span=26, adjust=False).mean()
    stock_data_calc['MACD'] = exp1 - exp2
    stock_data_calc['Signal_Line'] = stock_data_calc['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    bb_window = 20
    rolling_mean = stock_data_calc['Close'].rolling(window=bb_window).mean()
    rolling_std = stock_data_calc['Close'].rolling(window=bb_window).std()
    stock_data_calc['BB_middle'] = rolling_mean
    stock_data_calc['BB_upper'] = rolling_mean + (rolling_std * 2)
    stock_data_calc['BB_lower'] = rolling_mean - (rolling_std * 2)
    
    return stock_data_calc.dropna()

def prepare_sequences(data, time_steps=20, future_steps=5):
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(time_steps, len(data_scaled) - future_steps + 1):
        X.append(data_scaled[i-time_steps:i])
        y.append(data_scaled[i:i+future_steps, 3])  # Getting next 5 days' closing prices
        
    return np.array(X), np.array(y), scaler

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class EnhancedTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(EnhancedTransformerBlock, self).__init__()
        self.att1 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.att2 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dropout(rate),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output1 = self.att1(inputs, inputs)
        attn_output1 = self.dropout1(attn_output1, training=training)
        out1 = self.layernorm1(inputs + attn_output1)
        
        attn_output2 = self.att2(out1, out1)
        attn_output2 = self.dropout2(attn_output2, training=training)
        out2 = self.layernorm2(out1 + attn_output2)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)

def build_enhanced_model(time_steps, num_features, future_steps):
    embed_dim = 128
    num_heads = 8
    ff_dim = 256
    
    input_layer = Input(shape=(time_steps, num_features))
    
    x = Conv1D(filters=64, kernel_size=3, padding='same')(input_layer)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = tf.nn.leaky_relu(x)
    
    x = Conv1D(filters=128, kernel_size=3, padding='same')(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = tf.nn.leaky_relu(x)
    
    skip = x
    x = Dense(embed_dim)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = tf.nn.leaky_relu(x)
    x = x + skip
    
    x = PositionalEncoding(time_steps, embed_dim)(x)
    
    for _ in range(3):
        skip = x
        x = EnhancedTransformerBlock(embed_dim, num_heads, ff_dim, rate=0.1)(x)
        x = x + skip
    
    lstm_out = LSTM(embed_dim, return_sequences=True)(x)
    attention = MultiHeadAttention(num_heads=4, key_dim=32)(lstm_out, lstm_out)
    x = LayerNormalization(epsilon=1e-6)(attention + lstm_out)
    
    x = LSTM(embed_dim)(x)
    
    skip = Dense(128)(x)
    x = Dense(128)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = tf.nn.leaky_relu(x)
    x = x + skip
    
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(future_steps, activation='linear')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Main execution
ticker_tw = "0050.TW"
start_date = "1991-07-01"
end_date = "2020-08-31"
data_tw = get_stock_data(ticker_tw, start_date, end_date)

# Get market indicators
tickers_market = {
    'sp500': '^GSPC',
    'nasdaq': '^IXIC',
    'dxy': 'DX-Y.NYB',
    'vix': '^VIX'
}

market_data = {}
for name, ticker in tickers_market.items():
    try:
        temp_data = yf.download(ticker, start=start_date, end=end_date)
        temp_data['Return'] = temp_data['Close'].pct_change().fillna(0)
        market_data[name] = temp_data[['Return']]
    except Exception as e:
        print(f"Error downloading {name}: {e}")

# Merge all data
data = data_tw.copy()
for name, mkt_data in market_data.items():
    data = data.merge(mkt_data[['Return']], left_index=True, right_index=True, how='left', suffixes=('', f'_{name}'))

# Fill missing values
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

# Prepare sequences
time_steps = 20
future_steps = 5
X, y, scaler = prepare_sequences(data, time_steps=time_steps, future_steps=future_steps)

# Split data
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# Build and compile model
model = build_enhanced_model(time_steps, X.shape[2], future_steps)
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='huber',
    metrics=['mae', 'mse']
)

# Training callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
]

# Training configuration
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.Huber(delta=1.0),
    metrics=['mae', 'mse']
)

# Enhanced callbacks for better training
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
]

# Progressive training strategy
batch_sizes = [32, 16, 8]
epochs_per_stage = [30, 30, 40]

for batch_size, epochs in zip(batch_sizes, epochs_per_stage):
    print(f"\nTraining with batch size {batch_size}")
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

# Evaluate and visualize results
y_pred = model.predict(X_test)

# Function to rescale predictions
def rescale_predictions(y_scaled, scaler):
    y_reshaped = y_scaled.reshape(-1, 1)
    y_rescaled = scaler.inverse_transform(y_reshaped)
    return y_rescaled.reshape(y_scaled.shape)

# Calculate metrics
y_test_rescaled = y_test * scaler.scale_[3] + scaler.center_[3]
y_pred_rescaled = y_pred * scaler.scale_[3] + scaler.center_[3]

for i in range(future_steps):
    mae = mean_absolute_error(y_test_rescaled[:, i], y_pred_rescaled[:, i])
    rmse = np.sqrt(mean_squared_error(y_test_rescaled[:, i], y_pred_rescaled[:, i]))
    r2 = r2_score(y_test_rescaled[:, i], y_pred_rescaled[:, i])
    print(f"Day {i+1} Predictions:")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# Visualization
plt.figure(figsize=(15, 8))
for i in range(future_steps):
    plt.subplot(future_steps, 1, i+1)
    plt.plot(y_test_rescaled[:100, i], label=f'True Values Day {i+1}', color='blue')
    plt.plot(y_pred_rescaled[:100, i], label=f'Predictions Day {i+1}', color='red')
    plt.title(f'Day {i+1} Predictions vs Actual')
    plt.legend()
plt.tight_layout()
plt.show()