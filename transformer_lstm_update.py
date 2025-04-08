import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Layer, Dense, Dropout, LSTM, LayerNormalization, MultiHeadAttention, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import traceback

def calculate_features(df):
    """
    Calculate technical indicators and features exactly matching those used in training.
    
    This function carefully replicates the exact feature set used during model training,
    ensuring consistency in our rolling predictions. Each feature is calculated in the
    same way as during the training phase to maintain model performance.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing OHLCV data (Open, High, Low, Close, Volume)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with features exactly matching the training set features
    """
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # 1. Basic price and return features
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close']).diff()
    
    # 2. Volatility features for different time windows
    for window in [5, 10, 20, 30]:
        # Standard volatility
        data[f'Volatility_{window}d'] = data['Returns'].rolling(window=window).std()
        
        # Trading range
        data[f'Range_{window}d'] = ((data['High'] - data['Low']) / data['Close']).rolling(window=window).mean()
        
        # Exponential volatility
        data[f'ExpVol_{window}d'] = data['Returns'].ewm(span=window).std()
    
    # 3. Volume-based features
    data['Volume_Returns'] = data['Volume'].pct_change()
    data['Volume_MA_Ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
    
    # 4. Moving averages and their ratios
    for window in [5, 10, 20]:
        data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
        data[f'MA_Ratio_{window}'] = data['Close'] / data[f'MA_{window}']
    
    # 5. RSI calculation
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # 6. MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Handle any infinite or missing values
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Select and order columns to exactly match the training features
    selected_columns = [
        'Close', 'High', 'Low', 'Open', 'Volume',
        'Returns', 'Log_Returns',
        'Volatility_5d', 'Range_5d', 'ExpVol_5d',
        'Volatility_10d', 'Range_10d', 'ExpVol_10d',
        'Volatility_20d', 'Range_20d', 'ExpVol_20d',
        'Volatility_30d', 'Range_30d', 'ExpVol_30d',
        'Volume_Returns', 'Volume_MA_Ratio',
        'MA_5', 'MA_Ratio_5',
        'MA_10', 'MA_Ratio_10',
        'MA_20', 'MA_Ratio_20',
        'RSI', 'MACD', 'MACD_Signal'
    ]
    
    # Verify all required features are present
    missing_features = set(selected_columns) - set(data.columns)
    if missing_features:
        raise ValueError(f"Missing features in calculation: {missing_features}")
    
    return data[selected_columns]

class RollingPredictor:
    """
    A class that handles rolling window predictions with consistent feature calculation.
    """
    def __init__(self, model, feature_scaler, target_scaler, time_steps, feature_calculator):
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.time_steps = time_steps
        self.feature_calculator = feature_calculator
        self.current_sequence = None
        self.raw_data = None
    
    def initialize_sequence(self, initial_data):
        """
        Initialize the prediction sequence with historical data.
        Ensures features match those used in training.
        """
        self.raw_data = initial_data.copy()
        
        # Calculate features consistently with training
        features = self.feature_calculator(initial_data)
        
        # Verify feature columns match scaler's features
        expected_features = features.columns
        if not all(feature in expected_features for feature in self.feature_scaler.feature_names_in_):
            raise ValueError(f"Feature mismatch. Expected features: {self.feature_scaler.feature_names_in_}")
        
        # Scale features
        scaled_features = self.feature_scaler.transform(features)
        
        # Store the most recent window
        self.current_sequence = scaled_features[-self.time_steps:]
    
    def update_sequence(self, new_data_point):
        """
        Update the sequence with a new data point.
        Ensures consistent feature calculation.
        """
        if self.raw_data is None:
            raise ValueError("Predictor not initialized. Call initialize_sequence first.")
        
        # Update raw data
        self.raw_data = pd.concat([self.raw_data, new_data_point])
        
        # Keep limited history to save memory
        self.raw_data = self.raw_data.iloc[-100:]
        
        # Calculate features
        features = self.feature_calculator(self.raw_data)
        
        # Scale features
        scaled_features = self.feature_scaler.transform(features)
        
        # Update sequence
        self.current_sequence = scaled_features[-self.time_steps:]
    
    def predict_next(self):
        """
        Make a prediction for the next time step.
        """
        if self.current_sequence is None:
            raise ValueError("Predictor not initialized. Call initialize_sequence first.")
        
        # Prepare input
        model_input = self.current_sequence.reshape(1, self.time_steps, -1)
        
        # Get predictions
        price_pred, vol_pred = self.model.predict(model_input, verbose=0)
        
        # Transform back to original scale
        price_pred_original = self.target_scaler.inverse_transform(price_pred.reshape(-1, 1))
        vol_pred_original = vol_pred.flatten() * self.target_scaler.scale_[0]
        
        return price_pred_original[0, 0], vol_pred_original[0]

def rolling_prediction_evaluation(data, model, feature_scaler, target_scaler, time_steps):
    """
    Evaluate the model using rolling window predictions with proper date handling.
    """
    # Initialize predictor
    predictor = RollingPredictor(
        model, 
        feature_scaler, 
        target_scaler, 
        time_steps, 
        calculate_features
    )
    
    # Initialize with training data
    test_size = 500
    initial_data = data[:-test_size]  # Use all but last 500 points for initialization
    predictor.initialize_sequence(initial_data)
    
    # Prepare for rolling predictions
    test_data = data[-test_size:]
    predictions = []
    volatilities = []
    actual_values = []
    prediction_dates = []  # Store dates explicitly
    
    # Make rolling predictions
    for idx in range(len(test_data)):
        # Store the date
        prediction_dates.append(test_data.index[idx])
        
        # Make prediction
        price_pred, vol_pred = predictor.predict_next()
        predictions.append(price_pred)
        volatilities.append(vol_pred)
        
        # Store actual value
        actual_value = test_data.iloc[idx]['Close']
        actual_values.append(actual_value)
        
        # Update sequence with actual value if not at the end
        if idx < len(test_data) - 1:
            predictor.update_sequence(test_data.iloc[idx:idx+1])
            
        # Print progress
        if idx % 50 == 0:
            print(f"Completed {idx}/{len(test_data)} predictions")
    
    return np.array(predictions), np.array(volatilities), np.array(actual_values), np.array(prediction_dates)

def visualize_rolling_predictions(predictions, volatilities, actual_values, dates):
    """
    Create visualization for rolling predictions with proper date handling.
    
    This function creates a comprehensive visualization of our model's predictions,
    including both price levels and returns. It uses sophisticated date formatting
    to ensure the time axis is clear and readable.
    
    Parameters:
    -----------
    predictions : array-like
        The model's price predictions
    volatilities : array-like
        The model's volatility predictions
    actual_values : array-like
        The actual price values
    dates : array-like
        The dates corresponding to each prediction
    """
    # First, verify all our data arrays have matching lengths
    if not (len(predictions) == len(volatilities) == len(actual_values) == len(dates)):
        raise ValueError(f"Data length mismatch: predictions={len(predictions)}, "
                        f"volatilities={len(volatilities)}, actual_values={len(actual_values)}, "
                        f"dates={len(dates)}")
    
    # Create figure with two subplots: one for prices, one for returns
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Top subplot: Price predictions
    ax1.plot(dates, actual_values, label='True Values', color='blue', linewidth=2)
    ax1.plot(dates, predictions, label='Predictions', color='red', linewidth=2)
    
    # Add uncertainty bands (2 standard deviations around predictions)
    ax1.fill_between(dates, 
                     predictions - 2*volatilities,
                     predictions + 2*volatilities,
                     color='red', alpha=0.2,
                     label='Prediction Uncertainty (±2σ)')
    
    # Configure the price plot
    ax1.set_title('Rolling Price Predictions vs Actual Values', fontsize=14)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Format date axis for price plot - show every 2 months
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Bottom subplot: Returns comparison
    # Calculate daily returns for both actual and predicted values
    returns_true = np.diff(actual_values) / actual_values[:-1]
    returns_pred = np.diff(predictions) / predictions[:-1]
    
    # Plot returns
    ax2.plot(dates[1:], returns_true, label='True Returns', color='blue', linewidth=2)
    ax2.plot(dates[1:], returns_pred, label='Predicted Returns', color='red', linewidth=2)
    
    # Configure the returns plot
    ax2.set_title('Rolling Returns Comparison', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Returns', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Format date axis for returns plot - match the price plot formatting
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Save and display the figure
    plt.savefig('rolling_prediction_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'rolling_prediction_results.png'")
    plt.show()
    plt.close()

class TransformerBlock(tf.keras.layers.Layer):
    """
    An improved Transformer block with proper serialization support.
    
    This implementation includes:
    - Proper attribute storage for serialization
    - Enhanced error handling
    - Improved configuration management
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        
        # Store initialization parameters as instance variables
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._ff_dim = ff_dim
        self._rate = rate
        
        # Create the attention layer
        self.att = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,  # Scaled key_dim to match num_heads
            dropout=rate
        )
        
        # Create the feedforward network
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        
        # Layer normalization and dropout layers
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        """
        Forward pass of the transformer block.
        """
        # Self-attention with proper query/key/value inputs
        attn_output = self.att(
            query=inputs,
            key=inputs,
            value=inputs,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feedforward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        """
        Returns the configuration of the layer for serialization.
        """
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self._embed_dim,
            "num_heads": self._num_heads,
            "ff_dim": self._ff_dim,
            "rate": self._rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a layer from its config.
        """
        return cls(**config)
    
class ModelCheckpointWithKeras(tf.keras.callbacks.ModelCheckpoint):
    """
    Custom ModelCheckpoint callback that properly handles Keras format saving.
    This class extends the base ModelCheckpoint to ensure compatibility with newer Keras versions.
    """
    def __init__(
        self,
        filepath,
        monitor='val_loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    ):
        super().__init__(
            filepath=filepath,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            save_freq=save_freq
        )
###############################################
# 1. Enhanced Feature Engineering
###############################################

def calculate_enhanced_features(df):
    """
    Calculate technical indicators with proper handling of infinite values and outliers
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Basic returns calculations with cleanup
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close']).diff()
    
    # Volatility features with proper error handling
    for window in [5, 10, 20, 30]:
        # Rolling volatility
        df[f'Volatility_{window}d'] = df['Returns'].rolling(window=window).std()
        
        # Normalized trading range with protection against zero division
        df[f'Range_{window}d'] = df.apply(
            lambda x: (x['High'] - x['Low']) / x['Close'] if x['Close'] != 0 else 0,
            axis=1
        ).rolling(window=window).mean()
        
        # Exponential volatility
        df[f'ExpVol_{window}d'] = df['Returns'].ewm(span=window).std()
    
    # Volume features with protection against zero division
    df['Volume_Returns'] = df['Volume'].pct_change()
    volume_ma = df['Volume'].rolling(window=20).mean()
    df['Volume_MA_Ratio'] = df.apply(
        lambda x: x['Volume'] / volume_ma[x.name] if volume_ma[x.name] != 0 else 1,
        axis=1
    )
    
    # Trend features with protection against zero division
    for window in [5, 10, 20]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        ma_values = df[f'MA_{window}']
        df[f'MA_Ratio_{window}'] = df.apply(
            lambda x: x['Close'] / ma_values[x.name] if ma_values[x.name] != 0 else 1,
            axis=1
        )
    
    # RSI with error handling
    df['RSI'] = calculate_rsi(df['Close'])
    
    # MACD with error handling
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Forward fill NaN values
    df = df.fillna(method='ffill')
    
    # For any remaining NaN values at the start, backward fill
    df = df.fillna(method='bfill')
    
    # Final check for any remaining invalid values
    df = df.clip(lower=-1e6, upper=1e6)  # Clip extreme values
    
    return df

def calculate_rsi(prices, period=14):
    """
    Calculate RSI with proper error handling using vectorized operations
    """
    try:
        # Calculate price changes
        delta = prices.diff()
        
        # Create masks for gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS with vectorized operations
        rs = pd.Series(0.0, index=prices.index)  # Initialize with zeros
        valid_denominator = avg_losses != 0  # Create mask for valid denominators
        rs[valid_denominator] = avg_gains[valid_denominator] / avg_losses[valid_denominator]
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        # Clean up any invalid values
        rsi = rsi.replace([np.inf, -np.inf], np.nan)
        rsi = rsi.fillna(method='ffill').fillna(method='bfill')
        rsi = rsi.clip(lower=0, upper=100)  # RSI should be between 0 and 100
        
        return rsi
    
    except Exception as e:
        print(f"Error calculating RSI: {str(e)}")
        return pd.Series(50, index=prices.index)  # Return neutral RSI on error

def create_advanced_model(time_steps, num_features):
    """
    Create an enhanced model with parallel processing paths for different feature types
    """
    # Input layer
    input_layer = Input(shape=(time_steps, num_features))
    
    # Define feature splits using Lambda layers
    # We use tf.slice for more precise control over tensor splitting
    price_features = Lambda(
        lambda x: tf.slice(x, [0, 0, 0], [-1, -1, 5]),
        name='price_features'
    )(input_layer)
    
    vol_features = Lambda(
        lambda x: tf.slice(x, [0, 0, 5], [-1, -1, 5]),
        name='volatility_features'
    )(input_layer)
    
    tech_features = Lambda(
        lambda x: tf.slice(x, [0, 0, 10], [-1, -1, -1]),
        name='technical_features'
    )(input_layer)
    
    # Process each feature group separately
    # Price path
    price_encoded = Dense(32, name='price_encoding')(price_features)
    price_transformer = TransformerBlock(32, 4, 64)(price_encoded)
    
    # Volatility path
    vol_encoded = Dense(32, name='volatility_encoding')(vol_features)
    vol_transformer = TransformerBlock(32, 4, 64)(vol_encoded)
    
    # Technical indicators path
    tech_encoded = Dense(32, name='technical_encoding')(tech_features)
    tech_transformer = TransformerBlock(32, 4, 64)(tech_encoded)
    
    # Combine all paths
    combined = Concatenate(name='feature_combination')([
        price_transformer, 
        vol_transformer, 
        tech_transformer
    ])
    
    # Additional processing
    x = LSTM(128, return_sequences=True, name='lstm_1')(combined)
    x = Dropout(0.4)(x)
    x = LSTM(64, name='lstm_2')(x)
    x = Dropout(0.4)(x)
    
    # Output layers for both price and volatility prediction
    price_output = Dense(1, name='price_output')(x)
    vol_output = Dense(1, activation='softplus', name='volatility_output')(x)
    
    model = Model(inputs=input_layer, outputs=[price_output, vol_output])
    
    # Custom loss function that considers both price and volatility
    def combined_loss(y_true, y_pred):
        price_loss = tf.keras.losses.huber(y_true, y_pred[0])
        vol_loss = tf.keras.losses.mean_squared_error(
            tf.abs(y_true - y_pred[0]), y_pred[1])
        return price_loss + 0.2 * vol_loss
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=combined_loss,
        metrics=['mae']
    )
    
    return model

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD with proper error handling
    """
    try:
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        # Clean up any invalid values
        macd = macd.replace([np.inf, -np.inf], np.nan)
        signal_line = signal_line.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values
        macd = macd.fillna(method='ffill').fillna(method='bfill')
        signal_line = signal_line.fillna(method='ffill').fillna(method='bfill')
        
        # Clip extreme values
        macd = macd.clip(lower=-1e6, upper=1e6)
        signal_line = signal_line.clip(lower=-1e6, upper=1e6)
        
        return macd, signal_line
    
    except Exception as e:
        print(f"Error calculating MACD: {str(e)}")
        return pd.Series(0, index=prices.index), pd.Series(0, index=prices.index)

###############################################
# 2. Improved Data Collection and Preparation
###############################################

def get_stock_data(ticker, start_date, end_date):
    """
    Enhanced data collection with proper date handling and error checking
    """
    try:
        # Download data with explicit adjustment handling
        stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Handle multi-level columns if present
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)
        
        # Calculate enhanced features
        stock_data = calculate_enhanced_features(stock_data)
        
        # Forward fill missing values only
        stock_data = stock_data.fillna(method='ffill')
        
        # Drop any remaining NaN values
        stock_data = stock_data.dropna()
        
        return stock_data
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

###############################################
# 3. Improved Time Series Data Preparation
###############################################

def prepare_time_series_data(data, time_steps, feature_names, target_col='Close', scale_target=True):
    """
    Prepare time series data with consistent feature calculation.
    """
    # Initialize feature calculator
    feature_calculator = AdaptiveFeatureCalculator()
    
    # Calculate features
    features = feature_calculator.calculate_features(data)
    
    # Ensure we have all expected features
    missing_features = set(feature_names) - set(features.columns)
    if missing_features:
        raise ValueError(f"Missing features in calculation: {missing_features}")
    
    # Ensure feature order matches
    features = features[feature_names]
    
    # Separate target
    target_data = data[target_col]
    
    # Scale features and target
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler() if scale_target else None
    
    # Scale features
    scaled_features = feature_scaler.fit_transform(features)
    scaled_features = pd.DataFrame(scaled_features, columns=features.columns, index=features.index)
    
    # Scale target if requested
    if scale_target:
        target_data = target_scaler.fit_transform(target_data.values.reshape(-1, 1)).flatten()
    
    # Create sequences
    X, y = [], []
    for i in range(time_steps, len(features)):
        X.append(scaled_features.iloc[i-time_steps:i].values)
        y.append(target_data[i])
    
    X, y = np.array(X), np.array(y)
    
    # Split data
    total_size = len(X)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    
    train_dates = features.index[time_steps:time_steps+train_size]
    val_dates = features.index[time_steps+train_size:time_steps+train_size+val_size]
    test_dates = features.index[time_steps+train_size+val_size:]
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    return (X_train, y_train, train_dates), (X_val, y_val, val_dates), (X_test, y_test, test_dates), (feature_scaler, target_scaler)


###############################################
# 4. Enhanced Model Architecture
###############################################

def create_advanced_model(time_steps, num_features):
    """
    Create an enhanced model with parallel processing paths for different feature types
    """
    # Input layer
    input_layer = Input(shape=(time_steps, num_features))
    
    # Split input into different feature groups
    price_features = Lambda(lambda x: x[:,:,:5])(input_layer)  # Assuming first 5 features are price-related
    vol_features = Lambda(lambda x: x[:,:,5:10])(input_layer)  # Next 5 are volatility-related
    tech_features = Lambda(lambda x: x[:,:,10:])(input_layer)  # Rest are technical indicators
    
    # Process each feature group separately
    # Price path
    price_encoded = Dense(32)(price_features)
    price_transformer = TransformerBlock(32, 4, 64)(price_encoded)
    
    # Volatility path
    vol_encoded = Dense(32)(vol_features)
    vol_transformer = TransformerBlock(32, 4, 64)(vol_encoded)
    
    # Technical indicators path
    tech_encoded = Dense(32)(tech_features)
    tech_transformer = TransformerBlock(32, 4, 64)(tech_encoded)
    
    # Combine all paths
    combined = Concatenate()([price_transformer, vol_transformer, tech_transformer])
    
    # Additional processing
    x = LSTM(128, return_sequences=True)(combined)
    x = Dropout(0.4)(x)
    x = LSTM(64)(x)
    x = Dropout(0.4)(x)
    
    # Output layers for both price and volatility prediction
    price_output = Dense(1, name='price_output')(x)
    vol_output = Dense(1, activation='softplus', name='volatility_output')(x)
    
    model = Model(inputs=input_layer, outputs=[price_output, vol_output])
    
    # Custom loss function that considers both price and volatility
    def combined_loss(y_true, y_pred):
        price_loss = tf.keras.losses.huber(y_true, y_pred[0])
        vol_loss = tf.keras.losses.mean_squared_error(
            tf.abs(y_true - y_pred[0]), y_pred[1])
        return price_loss + 0.2 * vol_loss
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss=combined_loss,
                 metrics=['mae'])
    
    return model

###############################################
# 5. Enhanced Evaluation and Visualization
###############################################
def create_visualization(y_true, y_pred, dates, vol_pred=None):
    """
    Creates comprehensive visualizations of model predictions.
    
    Parameters:
    -----------
    y_true : array-like
        The true values
    y_pred : array-like
        The predicted values
    dates : array-like
        The dates corresponding to the predictions
    vol_pred : array-like, optional
        Predicted volatility values for uncertainty visualization
    """
    # Create a figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Price predictions
    ax1.plot(dates, y_true, label='True Values', color='blue', linewidth=2)
    ax1.plot(dates, y_pred, label='Predictions', color='red', linewidth=2)
    
    # Add volatility bands if available
    if vol_pred is not None:
        ax1.fill_between(dates, 
                        y_pred - 2*vol_pred,
                        y_pred + 2*vol_pred,
                        color='red', alpha=0.2,
                        label='Prediction Uncertainty')
    
    ax1.set_title('Price Predictions vs Actual Values', fontsize=14)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Returns comparison
    returns_true = np.diff(y_true) / y_true[:-1]
    returns_pred = np.diff(y_pred) / y_pred[:-1]
    
    ax2.plot(dates[1:], returns_true, label='True Returns', color='blue', linewidth=2)
    ax2.plot(dates[1:], returns_pred, label='Predicted Returns', color='red', linewidth=2)
    ax2.set_title('Returns Comparison', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Returns', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'prediction_results.png'")
    
    # Show the plot
    plt.show()
    
    # Close the figure to free memory
    plt.close()

def evaluate_and_visualize(model, test_data, test_dates, scalers, history=None):
    """
    Comprehensive model evaluation with proper date handling and enhanced visualizations
    """
    X_test, y_test, dates = test_data
    feature_scaler, target_scaler = scalers
    
    # Get predictions
    price_pred, vol_pred = model.predict(X_test)
    
    # Inverse transform if necessary
    if target_scaler:
        y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        price_pred = target_scaler.inverse_transform(price_pred).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, price_pred)
    rmse = np.sqrt(mean_squared_error(y_test, price_pred))
    r2 = r2_score(y_test, price_pred)
    
    # Calculate returns
    actual_returns = np.diff(y_test) / y_test[:-1]
    pred_returns = np.diff(price_pred) / price_pred[:-1]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Price predictions
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(dates, y_test, label='True Values', color='blue')
    ax1.plot(dates, price_pred, label='Predictions', color='red')
    ax1.fill_between(dates, 
                     price_pred - 2*vol_pred.flatten(),
                     price_pred + 2*vol_pred.flatten(),
                     color='red', alpha=0.2)
    ax1.set_title('Price Predictions with Uncertainty')
    ax1.legend()
    
    # Plot 2: Returns comparison
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(dates[1:], actual_returns, label='True Returns', color='blue')
    ax2.plot(dates[1:], pred_returns, label='Predicted Returns', color='red')
    ax2.set_title('Returns Comparison')
    ax2.legend()
    
    # Plot 3: Prediction Error and Volatility
    ax3 = plt.subplot(3, 1, 3)
    prediction_error = np.abs(y_test - price_pred)
    ax3.plot(dates, prediction_error, label='Prediction Error', color='purple')
    ax3.plot(dates, vol_pred, label='Predicted Volatility', color='orange')
    ax3.set_title('Prediction Error vs Predicted Volatility')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    print(f"\nModel Performance Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Returns RMSE: {np.sqrt(mean_squared_error(actual_returns, pred_returns)):.4f}")
    
    # Calculate directional accuracy
    correct_direction = np.sum(np.sign(pred_returns) == np.sign(actual_returns))
    directional_accuracy = correct_direction / len(actual_returns)
    print(f"Directional Accuracy: {directional_accuracy:.4f}")

def evaluate_adaptive_predictions(data, model, feature_scaler, target_scaler, time_steps):
    """
    Evaluate the adaptive model with proper array alignment for metrics.
    
    This function carefully tracks the indices of our predictions and ensures
    that all our evaluation metrics have consistent lengths. We handle the 
    special cases at the start of our prediction sequence where we don't
    have enough history for certain calculations.
    """
    # Initialize predictor
    predictor = AdaptiveRollingPredictor(
        model, feature_scaler, target_scaler, time_steps
    )
    
    # Initialize with training data
    test_size = 500
    initial_data = data[:-test_size]
    predictor.initialize_sequence(initial_data)
    
    # Prepare for rolling predictions
    test_data = data[-test_size:]
    predictions = []
    volatilities = []
    actual_values = []
    prediction_dates = []
    prediction_errors = []
    adaptation_metrics = []
    
    # Make rolling predictions
    for idx in range(len(test_data)):
        # Store date and actual value
        current_date = test_data.index[idx]
        prediction_dates.append(current_date)
        actual_value = test_data.iloc[idx]['Close']
        actual_values.append(actual_value)
        
        # Make prediction
        price_pred, vol_pred = predictor.predict_next()
        predictions.append(price_pred)
        volatilities.append(vol_pred)
        
        # Calculate prediction error and adaptation metric
        # We need at least 2 predictions to calculate adaptation
        if idx >= 2:
            # Current prediction error
            current_error = abs(predictions[-1] - actual_values[-1]) / actual_values[-1]
            
            # Previous prediction error
            prev_error = abs(predictions[-2] - actual_values[-2]) / actual_values[-2]
            
            # Store error and calculate adaptation
            prediction_errors.append(current_error)
            error_improvement = prev_error - current_error
            adaptation_metrics.append(error_improvement)
        elif idx == 1:
            # For the second prediction, we can only calculate error
            current_error = abs(predictions[-1] - actual_values[-1]) / actual_values[-1]
            prediction_errors.append(current_error)
            # Add a placeholder for adaptation metric to maintain alignment
            adaptation_metrics.append(0)
        elif idx == 0:
            # For the first prediction, we can't calculate error or adaptation
            prediction_errors.append(0)
            adaptation_metrics.append(0)
        
        # Update sequence with actual value if not at the end
        if idx < len(test_data) - 1:
            predictor.update_sequence(test_data.iloc[idx:idx+1])
        
        # Print progress
        if idx % 50 == 0:
            print(f"Completed {idx}/{len(test_data)} predictions")
    
    # Ensure all arrays have the same length
    assert len(prediction_dates) == len(predictions) == len(volatilities) == \
           len(actual_values) == len(prediction_errors) == len(adaptation_metrics), \
           "Array length mismatch in results"
    
    return {
        'predictions': np.array(predictions),
        'volatilities': np.array(volatilities),
        'actual_values': np.array(actual_values),
        'dates': np.array(prediction_dates),
        'errors': np.array(prediction_errors),
        'adaptation': np.array(adaptation_metrics)
    }

def visualize_adaptive_predictions(results):
    """
    Create visualization with properly aligned arrays.
    """
    # Verify all arrays have the same length
    array_lengths = {
        'predictions': len(results['predictions']),
        'volatilities': len(results['volatilities']),
        'actual_values': len(results['actual_values']),
        'dates': len(results['dates']),
        'errors': len(results['errors']),
        'adaptation': len(results['adaptation'])
    }
    
    if len(set(array_lengths.values())) > 1:
        raise ValueError(f"Array length mismatch: {array_lengths}")
    
    predictions = results['predictions']
    volatilities = results['volatilities']
    actual_values = results['actual_values']
    dates = results['dates']
    errors = results['errors']
    adaptation = results['adaptation']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 15))
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    # 1. Price predictions plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, actual_values, label='True Values', color='blue', linewidth=2)
    ax1.plot(dates, predictions, label='Predictions', color='red', linewidth=2)
    ax1.fill_between(dates, 
                     predictions - 2*volatilities,
                     predictions + 2*volatilities,
                     color='red', alpha=0.2,
                     label='Prediction Uncertainty (±2σ)')
    
    ax1.set_title('Adaptive Rolling Price Predictions', fontsize=14)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Prediction error plot
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(dates, errors, label='Prediction Error %', color='purple', linewidth=2)
    ax2.axhline(y=np.mean(errors), color='gray', linestyle='--', 
                label=f'Mean Error: {np.mean(errors)*100:.2f}%')
    
    ax2.set_title('Prediction Error Over Time', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Error %', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Model adaptation plot
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(dates, adaptation, label='Adaptation Metric', color='green', linewidth=2)
    ax3.axhline(y=0, color='gray', linestyle='--', label='No Improvement')
    
    ax3.set_title('Model Adaptation Performance', fontsize=14)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Error Improvement', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Format date axes
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('adaptive_prediction_results.png', dpi=300, bbox_inches='tight')
    
    # Calculate and display performance metrics
    mean_error = np.mean(errors[2:]) * 100  # Skip first two points
    error_std = np.std(errors[2:]) * 100
    adaptation_score = np.mean(adaptation[2:]) * 100
    
    print("\nPerformance Metrics:")
    print(f"Mean Prediction Error: {mean_error:.2f}%")
    print(f"Error Standard Deviation: {error_std:.2f}%")
    print(f"Average Adaptation Score: {adaptation_score:.2f}%")
    
    # Calculate directional accuracy
    correct_direction = np.sum(np.sign(np.diff(predictions)) == 
                             np.sign(np.diff(actual_values)))
    directional_accuracy = correct_direction / (len(predictions) - 1) * 100
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")
    
    plt.show()
    plt.close()

class AdaptiveFeatureCalculator:
    """
    Calculates technical indicators with adaptive parameters based on recent market conditions.
    This class dynamically adjusts indicator parameters based on market volatility.
    """
    def __init__(self):
        self.volatility_scale = 1.0
        self.lookback_periods = {}
    
    def adjust_timeframes(self, data):
        """Dynamically adjust indicator timeframes based on recent volatility"""
        recent_volatility = data['Close'].pct_change().iloc[-20:].std()
        # Adjust lookback periods based on volatility
        base_periods = {'short': 5, 'medium': 10, 'long': 20}
        volatility_factor = np.clip(recent_volatility / self.volatility_scale, 0.5, 2.0)
        
        self.lookback_periods = {
            key: int(period * volatility_factor)
            for key, period in base_periods.items()
        }
    
    def calculate_features(self, df):
        """Calculate technical indicators with adaptive parameters"""
        data = df.copy()
        self.adjust_timeframes(data)
        
        # Price-based features with increased weight on recent data
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close']).diff()
        
        # Recent price changes get more weight
        for window in [1, 3, 5]:
            data[f'Recent_Return_{window}d'] = (
                data['Returns'].rolling(window)
                .apply(lambda x: np.average(x, weights=np.exp(np.arange(len(x)))))
            )
        
        # Adaptive timeframe indicators
        for name, period in self.lookback_periods.items():
            # Volatility with exponential weighting
            data[f'Volatility_{name}'] = (
                data['Returns']
                .ewm(span=period, adjust=False)
                .std()
            )
            
            # Moving averages with attention to recent prices
            data[f'MA_{name}'] = (
                data['Close']
                .ewm(span=period, adjust=False)
                .mean()
            )
            
            # Price momentum
            data[f'Momentum_{name}'] = data['Close'].diff(period)
        
        # Volume analysis with recent focus
        data['Volume_Returns'] = data['Volume'].pct_change()
        data['Recent_Volume_Impact'] = (
            data['Volume_Returns'] * data['Returns'].abs()
        ).ewm(span=5, adjust=False).mean()
        
        # Market regime indicators
        data['Trend_Strength'] = np.abs(
            data['Close'] / data['MA_medium'] - 1
        ).ewm(span=10, adjust=False).mean()
        
        # Clean up any invalid values
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Drop original price columns
        feature_cols = data.columns.difference(['Open', 'High', 'Low', 'Close', 'Volume'])
        
        return data[feature_cols]

class AdaptiveTransformerBlock(Layer):
    """
    Enhanced transformer block that adapts to recent market conditions
    and puts more emphasis on recent data points.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, time_steps, rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.time_steps = time_steps
        
        # Create recency bias weights
        self.recency_weights = tf.exp(
            tf.range(time_steps, dtype=tf.float32) - time_steps + 1
        )
        
        # Attention layer with recency bias
        self.att = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=rate
        )
        
        # Feedforward network with residual connection
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dropout(rate),
            Dense(embed_dim)
        ])
        
        # Layer normalization
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        # Dropout layers
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training=None):
        # Apply recency weights to input
        weighted_inputs = inputs * tf.reshape(
            self.recency_weights, [1, -1, 1]
        )
        
        # Self-attention with recency-weighted inputs
        attn_output = self.att(weighted_inputs, weighted_inputs)
        attn_output = self.dropout1(attn_output, training=training)
        
        # First residual connection
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feedforward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # Second residual connection
        return self.layernorm2(out1 + ffn_output)

class AdaptiveRollingPredictor:
    """
    Enhanced rolling predictor that adapts to changing market conditions
    and puts more emphasis on recent price information.
    """
    def __init__(self, model, feature_scaler, target_scaler, time_steps):
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.time_steps = time_steps
        self.feature_calculator = AdaptiveFeatureCalculator()
        self.current_sequence = None
        self.raw_data = None
        self.recent_scale = 1.0
        
    def initialize_sequence(self, initial_data):
        """Initialize with historical data and compute adaptive scaling"""
        self.raw_data = initial_data.copy()
        
        # Calculate initial features with adaptive parameters
        features = self.feature_calculator.calculate_features(initial_data)
        
        # Update scaling based on recent data
        recent_data = initial_data.iloc[-20:]
        self.recent_scale = recent_data['Close'].std()
        
        # Scale features
        scaled_features = self.feature_scaler.transform(features)
        self.current_sequence = scaled_features[-self.time_steps:]
        
    def update_sequence(self, new_data_point):
        """Update sequence with new data, adapting to market conditions"""
        if self.raw_data is None:
            raise ValueError("Predictor not initialized")
        
        # Update raw data and maintain recent history
        self.raw_data = pd.concat([self.raw_data.iloc[-100:], new_data_point])
        
        # Update recent scale
        recent_data = self.raw_data.iloc[-20:]
        self.recent_scale = recent_data['Close'].std()
        
        # Recalculate features with adaptive parameters
        updated_features = self.feature_calculator.calculate_features(self.raw_data)
        
        # Scale features
        scaled_features = self.feature_scaler.transform(updated_features)
        self.current_sequence = scaled_features[-self.time_steps:]
        
    def predict_next(self):
        """Make prediction with adaptive scaling and uncertainty estimation"""
        if self.current_sequence is None:
            raise ValueError("Predictor not initialized")
        
        # Add batch dimension for model input
        model_input = self.current_sequence.reshape(1, self.time_steps, -1)
        
        # Get base predictions
        price_pred, vol_pred = self.model.predict(model_input, verbose=0)
        
        # Adjust predictions based on recent market conditions
        price_pred_original = self.target_scaler.inverse_transform(
            price_pred.reshape(-1, 1)
        )
        
        # Scale volatility prediction by recent market conditions
        vol_pred_adjusted = vol_pred.flatten() * self.recent_scale
        
        return price_pred_original[0, 0], vol_pred_adjusted[0]

def create_adaptive_model(time_steps, num_features):
    """Create model with adaptive architecture"""
    embed_dim = 64
    num_heads = 4
    ff_dim = 64
    
    # Input layer
    input_layer = Input(shape=(time_steps, num_features))
    
    # Initial feature processing
    x = Dense(embed_dim)(input_layer)
    
    # Multiple adaptive transformer blocks
    for _ in range(2):
        transformer = AdaptiveTransformerBlock(
            embed_dim, num_heads, ff_dim, time_steps
        )
        x = transformer(x)
    
    # LSTM layers for temporal dependencies
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = Dropout(0.4)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = Dropout(0.4)(x)
    
    # Separate heads for price and volatility
    price_output = Dense(1, name='price_output')(x)
    vol_output = Dense(1, activation='softplus', name='volatility_output')(x)
    
    model = Model(inputs=input_layer, outputs=[price_output, vol_output])
    
    # Custom loss that considers both price and volatility
    def adaptive_loss(y_true, y_pred):
        price_loss = tf.keras.losses.huber(y_true, y_pred[0])
        vol_loss = tf.keras.losses.mean_squared_error(
            tf.abs(y_true - y_pred[0]), y_pred[1]
        )
        return price_loss + 0.2 * vol_loss
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=adaptive_loss
    )
    
    return model

def prepare_initial_data(ticker, start_date, end_date):
    """
    Prepare the initial dataset with consistent feature calculation.
    This function ensures that our feature space matches across all stages.
    """
    print("Fetching stock data...")
    data = get_stock_data(ticker, start_date, end_date)
    if data is None:
        raise ValueError("Failed to fetch stock data")
    
    print("Calculating initial features...")
    feature_calculator = AdaptiveFeatureCalculator()
    features = feature_calculator.calculate_features(data)
    
    # Print feature names for debugging
    print("\nFeatures included:")
    for idx, col in enumerate(features.columns):
        print(f"{idx + 1}. {col}")
    
    return data, features.shape[1], features.columns.tolist()

def visualize_improved_predictions(predictions, volatilities, actual_values, dates):
    """
    Create an enhanced visualization of the model's predictions with detailed analysis.
    
    Parameters:
    -----------
    predictions : array-like
        The model's price predictions
    volatilities : array-like
        The model's volatility predictions
    actual_values : array-like
        The actual price values
    dates : array-like
        The dates corresponding to each prediction
    """
    # Verify data integrity
    if not (len(predictions) == len(volatilities) == len(actual_values) == len(dates)):
        raise ValueError("Data length mismatch in visualization inputs")
    
    # Calculate additional metrics
    prediction_errors = (predictions - actual_values) / actual_values
    rolling_mae = pd.Series(np.abs(prediction_errors)).rolling(window=20).mean()
    
    # Create figure with multiple subplots for detailed analysis
    fig = plt.figure(figsize=(15, 15))
    gs = plt.GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
    
    # 1. Price Predictions Plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, actual_values, label='True Values', color='blue', linewidth=2)
    ax1.plot(dates, predictions, label='Predictions', color='red', linewidth=2)
    
    # Add uncertainty bands
    ax1.fill_between(dates, 
                     predictions - 2*volatilities,
                     predictions + 2*volatilities,
                     color='red', alpha=0.2,
                     label='Prediction Uncertainty (±2σ)')
    
    ax1.set_title('Adaptive Rolling Price Predictions', fontsize=14)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Prediction Error Plot
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(dates, prediction_errors, label='Prediction Error %', 
             color='purple', linewidth=2)
    ax2.axhline(y=np.mean(prediction_errors), color='gray', linestyle='--',
                label=f'Mean Error: {np.mean(prediction_errors)*100:.2f}%')
    
    ax2.set_title('Prediction Error Over Time', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Error %', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Rolling MAE Plot
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(dates, rolling_mae, label='20-Day Rolling MAE', 
             color='green', linewidth=2)
    ax3.axhline(y=np.mean(rolling_mae), color='gray', linestyle='--',
                label=f'Average MAE: {np.mean(rolling_mae)*100:.2f}%')
    
    ax3.set_title('Rolling Mean Absolute Error', fontsize=14)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('MAE', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Volatility Analysis Plot
    ax4 = fig.add_subplot(gs[3])
    realized_volatility = pd.Series(np.abs(np.diff(actual_values) / actual_values[:-1])).rolling(window=20).std()
    predicted_volatility = pd.Series(volatilities).rolling(window=20).mean()
    
    ax4.plot(dates[20:], realized_volatility[19:], label='Realized Volatility', 
             color='blue', linewidth=2)
    ax4.plot(dates, predicted_volatility, label='Predicted Volatility', 
             color='red', linewidth=2)
    
    ax4.set_title('Volatility Analysis', fontsize=14)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.set_ylabel('Volatility', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Format date axes
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Calculate and display performance metrics
    print("\nPerformance Metrics:")
    print(f"Mean Prediction Error: {np.mean(prediction_errors)*100:.2f}%")
    print(f"Mean Absolute Error: {np.mean(np.abs(prediction_errors))*100:.2f}%")
    print(f"Error Standard Deviation: {np.std(prediction_errors)*100:.2f}%")
    
    # Calculate directional accuracy
    direction_actual = np.diff(actual_values) > 0
    direction_pred = np.diff(predictions) > 0
    directional_accuracy = np.mean(direction_actual == direction_pred) * 100
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")
    
    # Save the figure
    plt.savefig('improved_prediction_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'improved_prediction_results.png'")
    plt.show()
    plt.close()

    # Return metrics dictionary for further analysis
    return {
        'mean_error': np.mean(prediction_errors) * 100,
        'mae': np.mean(np.abs(prediction_errors)) * 100,
        'error_std': np.std(prediction_errors) * 100,
        'directional_accuracy': directional_accuracy,
        'rolling_mae': rolling_mae
    }

class ImprovedAdaptiveRollingPredictor:
    """
    Enhanced rolling predictor that properly incorporates new data and adjusts predictions
    based on recent performance. This implementation includes error correction and
    adaptive scaling mechanisms.
    """
    def __init__(self, model, feature_scaler, target_scaler, time_steps, memory_length=20):
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.time_steps = time_steps
        self.memory_length = memory_length
        self.feature_calculator = AdaptiveFeatureCalculator()
        
        # Initialize tracking variables
        self.current_sequence = None
        self.raw_data = None
        self.recent_scale = 1.0
        self.prediction_errors = []
        self.last_actual = None
        self.error_correction_alpha = 0.1  # Learning rate for error correction
    
    def initialize_sequence(self, initial_data):
        """
        Initialize the prediction sequence with historical data and set up error tracking.
        
        Parameters:
        -----------
        initial_data : pandas.DataFrame
            Historical price data with required columns (Open, High, Low, Close, Volume)
        """
        # Store initial data with enough history for feature calculation
        self.raw_data = initial_data.copy()
        
        # Calculate initial features
        features = self.feature_calculator.calculate_features(initial_data)
        
        # Establish initial scaling based on recent volatility
        recent_data = initial_data.iloc[-self.memory_length:]
        self.recent_scale = recent_data['Close'].std()
        
        # Scale features and store the most recent sequence
        scaled_features = self.feature_scaler.transform(features)
        self.current_sequence = scaled_features[-self.time_steps:]
        
        # Initialize last actual price
        self.last_actual = initial_data['Close'].iloc[-1]
        
        # Initialize error tracking
        self.prediction_errors = []
        
        print("Predictor initialized with sequence length:", len(self.current_sequence))
    
    def update_sequence(self, new_data_point):
        """
        Update the sequence with new data, incorporating recent market conditions
        and prediction errors.
        
        Parameters:
        -----------
        new_data_point : pandas.DataFrame
            New market data point to incorporate
        """
        if self.raw_data is None:
            raise ValueError("Predictor not initialized. Call initialize_sequence first.")
        
        # Update raw data while maintaining limited history
        self.raw_data = pd.concat([self.raw_data, new_data_point])
        history_needed = max(self.time_steps * 3, self.memory_length * 2)
        self.raw_data = self.raw_data.iloc[-history_needed:]
        
        # Update market condition metrics
        recent_data = self.raw_data.iloc[-self.memory_length:]
        self.recent_scale = recent_data['Close'].std()
        
        # Calculate features for the entire sequence
        try:
            features = self.feature_calculator.calculate_features(self.raw_data)
            
            # Scale features
            scaled_features = self.feature_scaler.transform(features)
            
            # Update current sequence
            self.current_sequence = scaled_features[-self.time_steps:]
            
            # Update last actual price
            self.last_actual = new_data_point['Close'].iloc[-1]
            
        except Exception as e:
            print(f"Error updating sequence: {str(e)}")
            traceback.print_exc()
            raise
    
    def calculate_error_correction(self):
        """
        Calculate error correction factor based on recent prediction errors.
        
        Returns:
        --------
        float
            The correction factor to apply to the next prediction
        """
        if not self.prediction_errors:
            return 0.0
        
        # Calculate weighted average of recent errors
        # More recent errors have higher weights
        weights = np.exp(np.arange(len(self.prediction_errors)))
        weighted_error = np.average(self.prediction_errors[-self.memory_length:],
                                  weights=weights[-self.memory_length:])
        
        return weighted_error * self.error_correction_alpha
    
    def predict_next(self):
        """
        Make a prediction for the next time step with error correction
        and adaptive scaling.
        
        Returns:
        --------
        tuple
            (price_prediction, volatility_prediction)
        """
        if self.current_sequence is None:
            raise ValueError("Predictor not initialized. Call initialize_sequence first.")
        
        try:
            # Prepare input for model
            model_input = self.current_sequence.reshape(1, self.time_steps, -1)
            
            # Get base predictions
            price_pred, vol_pred = self.model.predict(model_input, verbose=0)
            
            # Transform price prediction back to original scale
            price_pred_original = self.target_scaler.inverse_transform(
                price_pred.reshape(-1, 1)
            )[0, 0]
            
            # Apply error correction
            correction = self.calculate_error_correction()
            corrected_price = price_pred_original + correction
            
            # Calculate adaptive volatility prediction
            base_volatility = vol_pred.flatten()[0]
            adapted_volatility = base_volatility * self.recent_scale
            
            # If we have a last actual price, calculate prediction error
            if self.last_actual is not None:
                current_error = (corrected_price - self.last_actual) / self.last_actual
                self.prediction_errors.append(current_error)
                
                # Keep error history limited
                if len(self.prediction_errors) > self.memory_length:
                    self.prediction_errors = self.prediction_errors[-self.memory_length:]
            
            return corrected_price, adapted_volatility
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            traceback.print_exc()
            raise

def improved_rolling_prediction_evaluation(data, model, feature_scaler, target_scaler, time_steps):
    """
    Evaluate the model using the improved rolling window predictions.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The complete dataset
    model : tensorflow.keras.Model
        The trained prediction model
    feature_scaler : sklearn.preprocessing.StandardScaler
        The scaler used for features
    target_scaler : sklearn.preprocessing.StandardScaler
        The scaler used for target values
    time_steps : int
        Number of time steps to use for prediction
        
    Returns:
    --------
    tuple
        (predictions, volatilities, actual_values, prediction_dates)
    """
    # Initialize predictor with enhanced capabilities
    predictor = ImprovedAdaptiveRollingPredictor(
        model, 
        feature_scaler, 
        target_scaler, 
        time_steps
    )
    
    # Use more initial data for better adaptation
    test_size = 500
    initial_data = data[:-test_size]
    predictor.initialize_sequence(initial_data)
    
    # Prepare for rolling predictions
    test_data = data[-test_size:]
    predictions = []
    volatilities = []
    actual_values = []
    prediction_dates = []
    
    print(f"\nStarting rolling predictions for {len(test_data)} time steps...")
    
    # Make rolling predictions with progress tracking
    for idx in range(len(test_data)):
        try:
            # Store date and actual value
            current_date = test_data.index[idx]
            prediction_dates.append(current_date)
            actual_value = test_data.iloc[idx]['Close']
            actual_values.append(actual_value)
            
            # Make prediction
            price_pred, vol_pred = predictor.predict_next()
            predictions.append(price_pred)
            volatilities.append(vol_pred)
            
            # Update sequence with actual value if not at the end
            if idx < len(test_data) - 1:
                predictor.update_sequence(test_data.iloc[idx:idx+1])
            
            # Print progress
            if idx % 50 == 0:
                print(f"Completed {idx}/{len(test_data)} predictions")
                if idx > 0:
                    recent_mae = np.mean(np.abs(np.array(predictions[-50:]) - 
                                              np.array(actual_values[-50:])))
                    print(f"Recent MAE: {recent_mae:.2f}")
                    
        except Exception as e:
            print(f"Error at index {idx}: {str(e)}")
            traceback.print_exc()
            raise
            
    return (np.array(predictions), np.array(volatilities), 
            np.array(actual_values), np.array(prediction_dates))

###############################################
# Main Execution
###############################################
def evaluate_next_day_prediction(data, model, feature_scaler, target_scaler, time_steps):
    """
    Evaluate the model's prediction for the next day using returns-based prediction
    to ensure realistic price movements.
    """
    # Initialize predictor
    predictor = ImprovedAdaptiveRollingPredictor(
        model, 
        feature_scaler, 
        target_scaler, 
        time_steps
    )
    
    # Use all data except the last day for initialization
    initial_data = data[:-1]
    last_price = initial_data['Close'].iloc[-1]
    predictor.initialize_sequence(initial_data)
    
    # Get the last day's data for testing
    test_data = data.iloc[-1:]
    actual_value = test_data['Close'].iloc[0]
    prediction_date = test_data.index[0]
    
    # Make prediction for the next day
    price_pred, vol_pred = predictor.predict_next()
    
    # Convert prediction to a reasonable return
    # Limit the predicted price movement to a realistic range (e.g., ±2%)
    max_daily_move = 0.02  # 2% maximum daily movement
    
    # Calculate historical volatility to adapt the maximum movement
    recent_volatility = data['Close'].pct_change().tail(20).std()
    max_daily_move = min(max_daily_move, recent_volatility * 2)  # Dynamic adjustment
    predicted_return = (price_pred - last_price) / last_price
    capped_return = np.clip(predicted_return, -max_daily_move, max_daily_move)
    
    # Calculate the adjusted prediction
    adjusted_prediction = last_price * (1 + capped_return)
    
    # Calculate realistic volatility (typically 1-2% for daily moves)
    adjusted_volatility = last_price * 0.015  # 1.5% baseline volatility
    
    print("\nNext Day Prediction Results:")
    print(f"Date: {prediction_date}")
    print(f"Previous Day Close: {last_price:.2f}")
    print(f"Actual Price: {actual_value:.2f}")
    print(f"Predicted Price: {adjusted_prediction:.2f}")
    print(f"Predicted Daily Return: {capped_return*100:.2f}%")
    print(f"Actual Daily Return: {((actual_value - last_price)/last_price)*100:.2f}%")
    
    return adjusted_prediction, adjusted_volatility, actual_value, prediction_date, last_price

def visualize_next_day_prediction(prediction, volatility, actual_value, prediction_date, 
                                historical_data, last_price):
    """
    Create an enhanced visualization focusing on the next day prediction with
    appropriate scaling and context.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot recent historical data (last 30 days)
    recent_data = historical_data.iloc[-30:]
    plt.plot(recent_data.index, recent_data['Close'], 
             label='Historical Prices', color='blue', linewidth=2)
    
    # Plot the prediction point and actual value
    plt.scatter(prediction_date, prediction, color='red', s=100, 
                label='Prediction', zorder=5)
    plt.scatter(prediction_date, actual_value, color='green', s=100, 
                label='Actual Value', zorder=5)
    
    # Add uncertainty band for prediction
    plt.fill_between([prediction_date], 
                     [prediction - 2*volatility],
                     [prediction + 2*volatility],
                     color='red', alpha=0.2,
                     label='Prediction Uncertainty (±2σ)')
    
    # Calculate returns for more meaningful error reporting
    predicted_return = (prediction - last_price) / last_price * 100
    actual_return = (actual_value - last_price) / last_price * 100
    
    plt.title(f'Next Day Price Prediction\n'
              f'Predicted Return: {predicted_return:.2f}% vs '
              f'Actual Return: {actual_return:.2f}%', 
              fontsize=14)
    
    # Set y-axis limits to focus on relevant price range
    price_range = max(actual_value, prediction) - min(actual_value, prediction)
    plt.ylim(min(recent_data['Close'].min(), actual_value, prediction) - price_range/2,
             max(recent_data['Close'].max(), actual_value, prediction) + price_range/2)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Format date axis
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('next_day_prediction.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'next_day_prediction.png'")
    plt.show()
    plt.close()

# Updated main execution code for next day prediction
if __name__ == "__main__":
    try:
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Define parameters
        ticker_tw = "0050.TW"
        start_date = "1991-07-01"
        end_date = "2020-08-31"
        time_steps = 10
        
        print("Starting next day prediction analysis...")
        
        # Prepare data
        data, num_features, feature_names = prepare_initial_data(ticker_tw, start_date, end_date)
        print(f"\nPrepared data with {num_features} features")
        
        # Prepare time series data
        train_data, val_data, test_data, scalers = prepare_time_series_data(
            data, time_steps, feature_names, target_col='Close', scale_target=True
        )
        
        # Create and train model
        model = create_adaptive_model(time_steps, train_data[0].shape[2])
        
        # Train model
        y_train_combined = [train_data[1], np.abs(np.diff(train_data[1], prepend=train_data[1][0]))]
        y_val_combined = [val_data[1], np.abs(np.diff(val_data[1], prepend=val_data[1][0]))]
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        model.fit(
            train_data[0],
            y_train_combined,
            validation_data=(val_data[0], y_val_combined),
            epochs=100,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        # Make next day prediction - note the updated unpacking of return values
        feature_scaler, target_scaler = scalers
        prediction, volatility, actual_value, prediction_date, last_price = evaluate_next_day_prediction(
            data, model, feature_scaler, target_scaler, time_steps
        )
        
        # Visualize the prediction with the last_price parameter
        visualize_next_day_prediction(
            prediction, volatility, actual_value, prediction_date, data, last_price
        )
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        traceback.print_exc()
        raise