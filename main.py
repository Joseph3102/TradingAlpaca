import os
from datetime import datetime, timezone
from pathlib import Path

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import TimeInForce, OrderSide, AssetClass
from alpaca.trading.requests import MarketOrderRequest
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Load .env ONLY for local development
# GitHub Actions uses secrets instead
# ─────────────────────────────────────────────
if Path(".env").exists():
    load_dotenv()

# ─────────────────────────────────────────────
# SAFETY GUARD 1:
# Only allow trades on scheduled (cron) runs
# ─────────────────────────────────────────────
event = os.getenv("GITHUB_EVENT_NAME")
if event != "schedule":
    print(f"Not a scheduled run (event={event}). Skipping trade.")
    exit(0)

# ─────────────────────────────────────────────
# SAFETY GUARD 2:
# Only trade at the top of the hour (UTC)
# ─────────────────────────────────────────────
now = datetime.now(timezone.utc)
if now.minute != 0:
    print("Not top of the hour. Skipping trade.")
    exit(0)

# ─────────────────────────────────────────────
# Connect to Alpaca (paper trading)
# ─────────────────────────────────────────────
trading_client = TradingClient(
    api_key=os.environ["APCA_API_KEY_ID"],
    secret_key=os.environ["APCA_API_SECRET_KEY"],
    paper=True
)

# ─────────────────────────────────────────────
# Optional SAFETY GUARD 3:
# Don't buy if BTC position already exists
# ─────────────────────────────────────────────
positions = trading_client.get_all_positions()
if any(p.symbol == "BTCUSD" for p in positions):
    print("BTC position already exists. Skipping trade.")
    exit(0)

# ─────────────────────────────────────────────
# Create BTC market order
# ─────────────────────────────────────────────
order_request = MarketOrderRequest(
    symbol="BTCUSD",
    notional=50,  # $50 worth of BTC
    side=OrderSide.BUY,
    time_in_force=TimeInForce.GTC,  # Required for crypto
    asset_class=AssetClass.CRYPTO
)

order = trading_client.submit_order(order_request)

print("✅ Bitcoin order submitted")
print(order)
