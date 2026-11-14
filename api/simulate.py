import json
import os
import sys
from http.server import BaseHTTPRequestHandler

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from trading_engine import (
    MarketContext,
    RiskConfig,
    ProTradingEngine,
    generate_sample_market_data,
    fetch_coindcx_candles,
)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))

        asset_class = body.get("asset_class", "CRYPTO").upper()
        symbol = body.get("symbol", "BTC_INR").upper()
        data_source = body.get("data_source", "sample")

        if asset_class == "CRYPTO" and data_source == "live":
            candles, source_label = fetch_coindcx_candles(symbol=symbol, interval="15m", limit=45)
        else:
            candles = generate_sample_market_data(symbol=symbol)
            source_label = f"BUILT_IN_SAMPLE ({symbol})"

        ctx = MarketContext(
            asset_class=asset_class,
            symbol=symbol,
            current_time_ist=body.get("time", "09:30"),
            lead_index_1_pct=float(body.get("lead1", 1.45)),
            lead_index_2_pct=float(body.get("lead2", 2.10)),
            gap_or_premium_pct=float(body.get("cue", 0.45)),
            volatility_value=float(body.get("vol", 3.2)),
            breadth_bullish_val=int(body.get("adv", 1420)),
            breadth_bearish_val=int(body.get("dec", 680)),
            htf_trend="BULLISH",
        )
        risk_cfg = RiskConfig(
            account_capital=float(body.get("cap", 500_000)),
            base_risk_pct=float(body.get("risk", 1.0)),
            min_rr_ratio=float(body.get("rr", 2.5)),
            sl_buffer_pct=float(body.get("sl_buf", 0.04)),
            exchange_fee_pct=float(body.get("fee_pct", 0.20 if asset_class == "CRYPTO" else 0.03)),
            tds_pct=float(body.get("tds_pct", 1.0 if asset_class == "CRYPTO" else 0.0)),
        )

        engine = ProTradingEngine(risk_cfg)
        result = engine.run_backtest(candles, ctx)
        result["candles"] = [c.__dict__ for c in candles]
        result["data_source"] = source_label

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode("utf-8"))