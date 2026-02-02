import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.trading.requests import MarketOrderRequest

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from dotenv import load_dotenv


# ───────────────────────────────────────────────
# LOAD .env (local only — GitHub Actions uses Secrets)
# ───────────────────────────────────────────────
if Path(".env").exists():
    load_dotenv()

# CORRECT Alpaca environment variables
API_KEY = os.environ["API_KEY"]
API_SECRET = os.environ["API_KEY_SECRET"]


# ───────────────────────────────────────────────
# ALPACA CLIENTS
# ───────────────────────────────────────────────
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

# SDK VERSION: feed MUST be set on the request, not here
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)


# ============================================================
# INDICATOR CALCULATIONS
# ============================================================

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_adx(df, period=14):
    df = df.copy()
    df["H-L"] = df["high"] - df["low"]
    df["H-PC"] = abs(df["high"] - df["close"].shift(1))
    df["L-PC"] = abs(df["low"] - df["close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)

    df["+DM"] = np.where(
        (df["high"] - df["high"].shift(1)) > (df["low"].shift(1) - df["low"]),
        df["high"] - df["high"].shift(1),
        0
    )
    df["-DM"] = np.where(
        (df["low"].shift(1) - df["low"]) > (df["high"] - df["high"].shift(1)),
        df["low"].shift(1) - df["low"],
        0
    )

    tr_smooth = df["TR"].rolling(period).sum()
    plus_dm_smooth = df["+DM"].rolling(period).sum()
    minus_dm_smooth = df["-DM"].rolling(period).sum()

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx


def get_indicators(symbol):
    """
    Fetches stock history using FREE IEX feed.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=14)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=start,
        end=end,
        feed="iex"   # CORRECT place for feed
    )

    try:
        bars = data_client.get_stock_bars(request).df
    except Exception as e:
        print(f"[ERROR] Couldn't fetch data for {symbol}: {e}")
        return None

    if bars.empty:
        print(f"[NO DATA] {symbol}")
        return None

    prices = bars["close"]

    rsi = calculate_rsi(prices).iloc[-1]
    adx = calculate_adx(bars).iloc[-1]
    volatility = prices.pct_change().std() * 100

    return {
        "rsi": float(rsi),
        "adx": float(adx),
        "volatility": float(volatility),
        "price": float(prices.iloc[-1])
    }


# ============================================================
# BST FOR VOLATILITY SORTING
# ============================================================

class Node:
    def __init__(self, volatility, symbol):
        self.volatility = volatility
        self.symbol = symbol
        self.left = None
        self.right = None


class VolatilityBST:
    def __init__(self):
        self.root = None

    def insert(self, vol, sym):
        def _insert(node, vol, sym):
            if node is None:
                return Node(vol, sym)
            if vol > node.volatility:  # highest vol gets priority
                node.left = _insert(node.left, vol, sym)
            else:
                node.right = _insert(node.right, vol, sym)
            return node

        self.root = _insert(self.root, vol, sym)

    def in_order(self):
        result = []
        def _traverse(node):
            if node is None:
                return
            _traverse(node.left)
            result.append(node.symbol)
            _traverse(node.right)
        _traverse(self.root)
        return result


# ============================================================
# TRADING HELPERS
# ============================================================

positions = {}   # symbol → buy price


def buy_stock(symbol, price):
    print(f"[BUY] {symbol} @ {price}")

    order = MarketOrderRequest(
        symbol=symbol,
        notional=50,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        asset_class=AssetClass.US_EQUITY
    )

    try:
        trading_client.submit_order(order)
        positions[symbol] = price
    except Exception as e:
        print(f"[BUY FAILED] {symbol}: {e}")


def sell_stock(symbol):
    print(f"[SELL] {symbol}")

    order = MarketOrderRequest(
        symbol=symbol,
        qty=1,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        asset_class=AssetClass.US_EQUITY
    )

    try:
        trading_client.submit_order(order)
        positions.pop(symbol, None)
    except Exception as e:
        print(f"[SELL FAILED] {symbol}: {e}")


# ============================================================
# MAIN BOT LOGIC
# ============================================================

RUSSELL_2000 = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]

bst = VolatilityBST()
stock_info = {}

print("Fetching indicators...")

# 1. Fetch indicators and build BST
for stock in RUSSELL_2000:
    ind = get_indicators(stock)
    if ind:
        print(f"{stock} → {ind}")
        stock_info[stock] = ind
        bst.insert(ind["volatility"], stock)

# 2. Buy logic (high → low volatility)
for stock in bst.in_order():
    ind = stock_info[stock]

    if ind["volatility"] > 35 and ind["adx"] < 25 and ind["rsi"] < 30:
        if stock not in positions:
            buy_stock(stock, ind["price"])

# 3. Sell logic
for stock in list(positions.keys()):
    ind = get_indicators(stock)
    if not ind:
        continue

    buy_price = positions[stock]
    current = ind["price"]

    if current <= buy_price * 0.95 or current >= buy_price * 1.10:
        sell_stock(stock)

print("Bot run complete.")
