import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

# Load .env locally only
if Path(".env").exists():
    load_dotenv()

# Alpaca clients
trading = TradingClient(
    api_key=os.environ["API_KEY"],
    secret_key=os.environ["API_KEY_SECRET"],
    paper=True,
)

data_client = StockHistoricalDataClient(
    api_key=os.environ["API_KEY"],
    secret_key=os.environ["API_KEY_SECRET"],
)

# -----------------------------
# Stock list (you should expand)
# -----------------------------
rus2000 = ["NVDA", "AAPL", "MSFT", "MA", "AMZN", "GOOGL", "META"]

# -----------------------------
# === Helper: Fetch indicators ===
# -----------------------------
def get_indicators(symbol):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=14)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=start,
        end=end,
    )

    bars = data_client.get_stock_bars(request).df
    if bars.empty:
        return None

    df = bars.xs(symbol)

    # Closing prices
    close = df["close"]

    # 1. Standard deviation volatility
    std_dev = close.pct_change().std() * 100  # %

    # 2. RSI calculation
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_value = float(rsi.iloc[-1])

    # 3. ADX calculation (simplified & standard)
    df["tr"] = np.maximum.reduce([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ])

    df["+dm"] = np.where((df["high"] - df["high"].shift()) > (df["low"].shift() - df["low"]), 
                         np.maximum(df["high"] - df["high"].shift(), 0), 
                         0)

    df["-dm"] = np.where((df["low"].shift() - df["low"]) > (df["high"] - df["high"].shift()), 
                         np.maximum(df["low"].shift() - df["low"], 0), 
                         0)

    df["+di"] = 100 * (df["+dm"].ewm(alpha=1/14).mean() / df["tr"].ewm(alpha=1/14).mean())
    df["-di"] = 100 * (df["-dm"].ewm(alpha=1/14).mean() / df["tr"].ewm(alpha=1/14).mean())
    df["dx"] = 100 * (abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"]))
    adx = df["dx"].rolling(14).mean().iloc[-1]

    return {
        "std": std_dev,
        "rsi": rsi_value,
        "adx": float(adx),
        "current_price": float(close.iloc[-1])
    }

# -----------------------------
# === Buy Logic ===
# -----------------------------
def buy_stock(symbol):
    print(f"BUYING {symbol}")

    order = MarketOrderRequest(
        symbol=symbol,
        notional=50,  # Buy $50 worth
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
        asset_class=AssetClass.US_EQUITY
    )
    trading.submit_order(order)


# -----------------------------
# === Sell Logic ===
# -----------------------------
def sell_stock(symbol, qty):
    print(f"SELLING {symbol}")

    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
        asset_class=AssetClass.US_EQUITY
    )
    trading.submit_order(order)


# =========================================================
# ===================== MAIN LOGIC ========================
# =========================================================

# A) Check buy conditions for all stocks in list
for symbol in rus2000:
    ind = get_indicators(symbol)
    if ind is None:
        continue

    print(symbol, ind)

    if (
        ind["std"] > 35           # high volatility
        and ind["adx"] < 25       # weak trend
        and ind["rsi"] < 30       # oversold
    ):
        buy_stock(symbol)

# B) Loop through currently held positions
positions = trading.get_all_positions()

for pos in positions:
    symbol = pos.symbol
    qty = pos.qty
    avg = float(pos.avg_entry_price)

    ind = get_indicators(symbol)
    if ind is None:
        continue

    price = ind["current_price"]

    # Take profit +10% or stop loss -5%
    if price < avg * 0.95 or price > avg * 1.10:
        sell_stock(symbol, qty)
