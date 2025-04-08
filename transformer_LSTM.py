import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, LayerNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

###############################################
# 1. 資料取得與合併
###############################################

# (1) 取得台股技術指標數據
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Ups_and_Downs'] = stock_data['Close'].diff().fillna(0)
    stock_data['Turnover'] = stock_data['Volume'] * stock_data['Close']
    stock_data['Change_Percent'] = stock_data['Close'].pct_change().fillna(0) * 100
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume',
                             'Turnover', 'Ups_and_Downs', 'Change_Percent']].dropna()
    return stock_data

ticker_tw = "0050.TW"
start_date = "1991-07-01"
end_date = "2020-08-31"
data_tw = get_stock_data(ticker_tw, start_date, end_date)

# (2) 取得國際市場數據：S&P500 與美元指數
ticker_sp500 = "^GSPC"
ticker_dxy   = "DX-Y.NYB"

sp500_data = yf.download(ticker_sp500, start=start_date, end=end_date)
usd_data   = yf.download(ticker_dxy, start=start_date, end=end_date)

# 計算報酬率（以 Close 欄位計算）
sp500_data['Return'] = sp500_data['Close'].pct_change().fillna(0)
usd_data['Return']   = usd_data['Close'].pct_change().fillna(0)

# (3) 設定 index 為 datetime 型態（若尚未轉換）
data_tw.index = pd.to_datetime(data_tw.index)
sp500_data.index = pd.to_datetime(sp500_data.index)
usd_data.index = pd.to_datetime(usd_data.index)

# (4) 合併資料：以台股資料日期為主，左右合併國際市場數據
data = data_tw.merge(sp500_data[['Return']], left_index=True, right_index=True, how='left')
data = data.merge(usd_data[['Return']], left_index=True, right_index=True, how='left')

# 避免名稱重複
data.rename(columns={'Return_x': 'Return_sp500', 'Return_y': 'Return_usd'}, inplace=True)

# 若合併後有缺失值，使用前向填補及後向填補
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

print("合併後的資料預覽：")
print(data.head())

###############################################
# 2. 資料前處理與時序資料構建
###############################################

# (1) 標準化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# (2) 構建時序資料：以前10天資料預測第11天的「收盤價」
time_steps = 10
X, y = [], []
for i in range(time_steps, len(data_scaled)):
    X.append(data_scaled[i-time_steps:i])
    y.append(data_scaled[i, 3])

X, y = np.array(X), np.array(y)

# (3) 分割訓練與測試資料
train_size = 6627
test_size = 500
X_train, X_test = X[:train_size], X[-test_size:]
y_train, y_test = y[:train_size], y[-test_size:]

###############################################
# 3. 定義 Positional Encoding 與 Transformer 區塊
###############################################

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

###############################################
# 4. 建立多模態 Transformer-LSTM 模型
###############################################


embed_dim = 64
num_heads = 4
ff_dim = 64
num_features = X.shape[2]

input_layer = Input(shape=(time_steps, num_features))
x = Dense(embed_dim)(input_layer)
x = PositionalEncoding(time_steps, embed_dim)(x)
x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
# 接 LSTM 層（使用兩層 LSTM 與 Dropout）
x = LSTM(128, activation='tanh', return_sequences=True)(x)
x = Dropout(0.3)(x)
x = LSTM(64, activation='tanh')(x)
# 最後輸出預測結果（預測收盤價）
output_layer = Dense(1, activation='linear')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(), loss='mean_absolute_error')
model.summary()

###############################################
# 5. 訓練模型
###############################################

history = model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1, validation_data=(X_test, y_test))

###############################################
# 6. 預測與評估
###############################################

# 預測結果
y_pred = model.predict(X_test)

mean_close = scaler.mean_[3]
std_close = scaler.scale_[3]

y_test_rescaled = y_test * std_close + mean_close
y_pred_rescaled = y_pred.flatten() * std_close + mean_close


mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='True Values', color='blue')
plt.plot(y_pred_rescaled, label='Predictions', color='red')
plt.title('True Values vs Predictions')
plt.legend()
plt.show()
