import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf

# ----------------- MODEL DEFINITION -----------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        out = self.fc(x)
        return out.squeeze()

# ----------------- FUNCTIONS -----------------
def get_data(ticker="AAPL"):
    df = yf.download(ticker, start="2022-01-01")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2

    df.dropna(inplace=True)
    return df

def scale_data(df, feature_cols):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    return scaled, scaler

def create_sequences(data, window_size=30):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
    return torch.tensor(X, dtype=torch.float32)

# ----------------- STREAMLIT APP -----------------
st.title("📈 Transformer Stock Price Predictor by Yohannes A.")

ticker = st.text_input("Enter Stock Ticker:", value="AAPL")
df = get_data(ticker)

feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_20', 'EMA_50', 'RSI', 'MACD']
scaled_data, scaler = scale_data(df, feature_cols)
seq_input = create_sequences(scaled_data)

# Load Model
model = TimeSeriesTransformer(input_dim=len(feature_cols))
#model.load_state_dict(torch.load("C://Users//abunie//Desktop//DC//transformer_model.pth", map_location=torch.device('cpu')))
model.load_state_dict(torch.load("transformer_model.pth", map_location=torch.device('cpu')))

model.eval()

with torch.no_grad():
    preds = model(seq_input).numpy()

# Inverse transform predictions
def inverse_close(pred_scaled):
    dummy = np.zeros((len(pred_scaled), len(feature_cols)))
    dummy[:, 3] = pred_scaled  # only 'Close'
    return scaler.inverse_transform(dummy)[:, 3]

#pred_prices = inverse_close(preds)
pred_prices = inverse_close(preds).flatten()

#true_prices = df["Close"].values[-len(pred_prices):]
true_prices = df["Close"].values[-len(pred_prices):].flatten()


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

mse = mean_squared_error(true_prices, pred_prices)
rmse = np.sqrt(mse)
mae = mean_absolute_error(true_prices, pred_prices)

st.subheader("📏 Model Accuracy")
col1, col2 = st.columns(2)
col1.metric("RMSE", f"${rmse:.2f}")
col2.metric("MAE", f"${mae:.2f}")


# Plot
st.subheader("📊 Predicted vs Actual Close Price")
st.line_chart(pd.DataFrame({
    "Actual": true_prices,
    "Predicted": pred_prices
}, index=df.index[-len(pred_prices):]))
