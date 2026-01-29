from alpaca.trading.enums import TimeInForce, OrderSide, AssetClass
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env
# Load .env only if running locally
if Path(".env").exists():
    load_dotenv()


# Connect to Alpaca paper trading
trading_client = TradingClient(
    api_key=os.environ["API_KEY"],
    secret_key=os.environ["API_SECRET_KEY"],
    paper=True
)

# Create a market order to buy Bitcoin
order_request = MarketOrderRequest(
    symbol="BTC/USD",
    notional=50,                # Buy $50 worth of BTC
    side=OrderSide.BUY,
    time_in_force=TimeInForce.GTC,  # Must be GTC for crypto
    asset_class=AssetClass.CRYPTO
)

# Submit the order
order = trading_client.submit_order(order_request)

print("Bitcoin order submitted!")
print(order)
