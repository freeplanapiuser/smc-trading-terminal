import csv
import json
import math
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple


# ==============================================================================
# 1. DATA MODELS
# ==============================================================================

@dataclass
class Candle:
    index: int
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


@dataclass
class MarketContext:
    asset_class: str = "CRYPTO"        # "CRYPTO" or "EQUITY"
    symbol: str = "BTC_INR"            # e.g., "NIFTY50", "BTC_INR", "ETH_INR", "SOL_INR", "POL_INR"
    current_time_ist: str = "09:30"    # Used for NSE session window in EQUITY mode
    lead_index_1_pct: float = 1.45     # NASDAQ % (Equity) or BTC 24h % (Crypto)
    lead_index_2_pct: float = 2.10     # DOWJ % (Equity) or ETH 24h % (Crypto)
    gap_or_premium_pct: float = 0.45   # GIFT Nifty Gap % (Equity) or USDT/INR Premium % (Crypto)
    volatility_value: float = 3.2      # India VIX (Equity, <15 stable) or 24h Vol % (Crypto, <4.5% stable)
    breadth_bullish_val: int = 1420    # NSE Advances (Equity) or 24h Buy Volume Ratio % (Crypto)
    breadth_bearish_val: int = 680     # NSE Declines (Equity) or 24h Sell Volume Ratio % (Crypto)
    htf_trend: str = "BULLISH"         # Multi-Timeframe (1H / 4H / Daily) bias


@dataclass
class RiskConfig:
    account_capital: float = 500_000.0   # INR 5,00,000
    base_risk_pct: float = 1.0           # 1.0% risk in stable volatility
    min_rr_ratio: float = 2.5            # Minimum 1:2.5 Risk:Reward
    sl_buffer_pct: float = 0.04          # 0.04% for NIFTY (~9 pts) or 0.20% for Crypto
    exchange_fee_pct: float = 0.20       # Per-side fee incl. GST (0.20% Crypto, 0.03% Equity)
    tds_pct: float = 1.0                 # 1.0% TDS on VDA sell value (0.0% for Equity)
    max_trades_per_session: int = 2


@dataclass
class SMCSetup:
    ob_index: int
    ob_low: float
    ob_high: float
    fvg_low: float
    fvg_high: float
    choch_level: float
    bsl_target: float
    confirmed: bool
    rejection_reason: Optional[str] = None


@dataclass
class JournalEntry:
    trade_id: str
    symbol: str
    timestamp: str
    direction: str
    status: str                  # WIN, LOSS, REJECTED
    entry_price: float
    stop_loss: float
    target_price: float
    exit_price: float
    quantity: float
    rr_planned: float
    gross_pnl_inr: float
    fees_and_tds_inr: float
    net_pnl_inr: float
    r_multiple: float
    vol_regime: str
    notes: str


# ==============================================================================
# 2. REGIME & PRE-TRADE GATE (EQUITY + 24/7 CRYPTO)
# ==============================================================================

class RegimeAnalyzer:
    @staticmethod
    def evaluate(ctx: MarketContext, risk_cfg: RiskConfig) -> Dict:
        is_crypto = ctx.asset_class.upper() == "CRYPTO"
        checks = []
        score = 0

        # Rule 1: Session Timing
        if is_crypto:
            market_open = True
            in_preopen = False
            checks.append({
                "rule": "24/7 Crypto Market Session",
                "passed": True,
                "detail": f"{ctx.symbol.replace('_', '/')} Continuous Spot Liquidity Active",
            })
        else:
            hh, mm = map(int, ctx.current_time_ist.split(":"))
            mins = hh * 60 + mm
            in_preopen = (9 * 60) <= mins < (9 * 60 + 15)
            market_open = (9 * 60 + 15) <= mins <= (15 * 60 + 30)
            checks.append({
                "rule": "NSE Session Window (9:00 Pre-Open / 9:15 CTS)",
                "passed": in_preopen or market_open,
                "detail": (
                    "Pre-Open Call Auction (Prepare Orders Only)"
                    if in_preopen else
                    ("Continuous Trading Session Active" if market_open else "Outside NSE Hours")
                ),
            })

        # Rule 2: Lead Benchmark Momentum
        leaders_bullish = (ctx.lead_index_1_pct > 0) and (ctx.lead_index_2_pct > 0)
        if leaders_bullish:
            score += 1
        checks.append({
            "rule": "Crypto Majors (BTC & ETH 24h)" if is_crypto else "US Markets (NASDAQ & DOWJ)",
            "passed": leaders_bullish,
            "detail": (
                f"BTC {ctx.lead_index_1_pct:+.2f}% | ETH {ctx.lead_index_2_pct:+.2f}%"
                if is_crypto else
                f"NAS {ctx.lead_index_1_pct:+.2f}% | DOWJ {ctx.lead_index_2_pct:+.2f}%"
            ),
        })

        # Rule 3: GIFT Nifty Gap / USDT-INR Liquidity Cue
        cue_bullish = ctx.gap_or_premium_pct >= 0.10
        if cue_bullish:
            score += 1
        checks.append({
            "rule": "INR Liquidity / USDT Premium Cue" if is_crypto else "GIFT Nifty Gap Cue",
            "passed": cue_bullish,
            "detail": f"{ctx.gap_or_premium_pct:+.2f}% ({'Bullish Flow' if cue_bullish else 'Neutral/Weak'})",
        })

        # Rule 4: Volatility Regime (India VIX < 15 vs Crypto 24h Vol < 4.5%)
        vol_threshold = 4.5 if is_crypto else 15.0
        is_stable_vol = ctx.volatility_value < vol_threshold
        effective_risk_pct = risk_cfg.base_risk_pct if is_stable_vol else (risk_cfg.base_risk_pct * 0.5)
        vol_label = f"<{vol_threshold}%" if is_crypto else "<15"
        vol_regime = f"STABLE ({vol_label})" if is_stable_vol else f"VOLATILE (>={vol_threshold})"
        if is_stable_vol:
            score += 1
        checks.append({
            "rule": f"{'24h Crypto Volatility' if is_crypto else 'India VIX'} Regime ({vol_label} Stable)",
            "passed": is_stable_vol,
            "detail": f"{ctx.volatility_value:.2f} -> {vol_regime} | Risk Cap: {effective_risk_pct:.2f}%",
        })

        # Rule 5: Breadth / Order-Flow Ratio
        ratio = ctx.breadth_bullish_val / max(ctx.breadth_bearish_val, 1)
        breadth_bullish = ctx.breadth_bullish_val > ctx.breadth_bearish_val
        if breadth_bullish:
            score += 1
        checks.append({
            "rule": "Taker Buy vs Sell Volume" if is_crypto else "NSE Breadth (Advances > Declines)",
            "passed": breadth_bullish,
            "detail": f"{ctx.breadth_bullish_val} vs {ctx.breadth_bearish_val} (Ratio: {ratio:.2f})",
        })

        # Rule 6: Multi-Timeframe Higher-TF Bias
        htf_ok = ctx.htf_trend.upper() == "BULLISH"
        if htf_ok:
            score += 1
        checks.append({
            "rule": "Multi-Timeframe Bias (1H / 4H / Daily)",
            "passed": htf_ok,
            "detail": f"HTF Trend: {ctx.htf_trend.upper()}",
        })

        allow_long_execution = market_open and htf_ok and (score >= 3)
        return {
            "score": score,
            "max_score": 5,
            "in_preopen": in_preopen,
            "market_open": market_open,
            "vol_regime": vol_regime,
            "effective_risk_pct": effective_risk_pct,
            "allow_long_execution": allow_long_execution,
            "checks": checks,
        }


# ==============================================================================
# 3. TECHNICAL INDICATORS & SMART MONEY CONCEPTS (SMC)
# ==============================================================================

class TechnicalSMCAnalyzer:
    @staticmethod
    def ema(values: List[float], period: int) -> List[float]:
        if not values:
            return []
        k = 2.0 / (period + 1)
        result = [values[0]]
        for price in values[1:]:
            result.append(price * k + result[-1] * (1.0 - k))
        return result

    @staticmethod
    def rsi(closes: List[float], period: int = 14) -> List[float]:
        if len(closes) < 2:
            return [50.0] * len(closes)
        rsis = [50.0]
        gains, losses = [], []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i - 1]
            gains.append(max(diff, 0.0))
            losses.append(max(-diff, 0.0))
            if i < period:
                avg_g = sum(gains) / i
                avg_l = sum(losses) / i
            else:
                avg_g = sum(gains[-period:]) / period
                avg_l = sum(losses[-period:]) / period
            if avg_l == 0:
                rsis.append(100.0)
            else:
                rs = avg_g / avg_l
                rsis.append(100.0 - (100.0 / (1.0 + rs)))
        return rsis

    @staticmethod
    def vwap(candles: List[Candle]) -> List[float]:
        cum_pv = 0.0
        cum_vol = 0.0
        out = []
        for c in candles:
            typical = (c.high + c.low + c.close) / 3.0
            cum_pv += typical * max(c.volume, 1e-6)
            cum_vol += max(c.volume, 1e-6)
            out.append(cum_pv / cum_vol)
        return out

    @staticmethod
    def detect_swings(candles: List[Candle], left: int = 2, right: int = 2) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        swing_highs, swing_lows = [], []
        for i in range(left, len(candles) - right):
            h, l = candles[i].high, candles[i].low
            if all(h >= candles[j].high for j in range(i - left, i + right + 1) if j != i):
                swing_highs.append((i, h))
            if all(l <= candles[j].low for j in range(i - left, i + right + 1) if j != i):
                swing_lows.append((i, l))
        return swing_highs, swing_lows

    @staticmethod
    def scan_order_blocks_and_fvg(
        candles: List[Candle],
        swing_highs: List[Tuple[int, float]],
    ) -> List[SMCSetup]:
        setups: List[SMCSetup] = []
        avg_vol = sum(c.volume for c in candles) / max(len(candles), 1)
        bsl_target = max(sh[1] for sh in swing_highs) if swing_highs else candles[-1].high

        for i in range(2, len(candles) - 2):
            c_ob = candles[i]
            c_imp1 = candles[i + 1]
            c_imp2 = candles[i + 2]

            if not (c_ob.is_bearish and c_imp1.is_bullish):
                continue

            ob_low = c_ob.low
            ob_high = max(c_ob.open, c_ob.close)
            has_fvg = c_imp2.low > c_ob.high
            fvg_low = c_ob.high if has_fvg else ob_high
            fvg_high = c_imp2.low if has_fvg else ob_high

            prior_sh = [sh[1] for sh in swing_highs if sh[0] < i and sh[1] < bsl_target]
            choch_level = prior_sh[-1] if prior_sh else c_ob.high

            broke_structure = max(c_imp1.close, c_imp2.close) > choch_level
            strong_volume = max(c_imp1.volume, c_imp2.volume) >= (avg_vol * 1.15)

            if not broke_structure:
                setups.append(SMCSetup(
                    ob_index=i, ob_low=ob_low, ob_high=ob_high,
                    fvg_low=fvg_low, fvg_high=fvg_high,
                    choch_level=choch_level, bsl_target=bsl_target,
                    confirmed=False,
                    rejection_reason="TRAP_OB: No ChoCH/BOS above prior Swing High"
                ))
            elif not has_fvg:
                setups.append(SMCSetup(
                    ob_index=i, ob_low=ob_low, ob_high=ob_high,
                    fvg_low=fvg_low, fvg_high=fvg_high,
                    choch_level=choch_level, bsl_target=bsl_target,
                    confirmed=False,
                    rejection_reason="WEAK_OB: Missing Fair Value Gap (FVG) displacement"
                ))
            elif not strong_volume:
                setups.append(SMCSetup(
                    ob_index=i, ob_low=ob_low, ob_high=ob_high,
                    fvg_low=fvg_low, fvg_high=fvg_high,
                    choch_level=choch_level, bsl_target=bsl_target,
                    confirmed=False,
                    rejection_reason="LOW_VOL: Insufficient volume on displacement"
                ))
            else:
                setups.append(SMCSetup(
                    ob_index=i, ob_low=ob_low, ob_high=ob_high,
                    fvg_low=fvg_low, fvg_high=fvg_high,
                    choch_level=choch_level, bsl_target=bsl_target,
                    confirmed=True,
                    rejection_reason=None
                ))
        return setups


# ==============================================================================
# 4. EXECUTION, FRACTIONAL SIZING, FEE/TDS ENGINE & BACKTESTER
# ==============================================================================

class ProTradingEngine:
    def __init__(self, risk_cfg: RiskConfig):
        self.risk_cfg = risk_cfg

    @staticmethod
    def _px_decimals(price: float) -> int:
        return 4 if price < 500 else 2

    def calculate_position_size(
        self, entry: float, stop_loss: float, effective_risk_pct: float, is_crypto: bool
    ) -> Tuple[float, float]:
        risk_per_unit = abs(entry - stop_loss)
        if risk_per_unit <= 0:
            return 0.0, 0.0
        risk_budget_inr = self.risk_cfg.account_capital * (effective_risk_pct / 100.0)
        raw_qty = risk_budget_inr / risk_per_unit
        qty = round(raw_qty, 6) if is_crypto else float(math.floor(raw_qty))
        return max(qty, 0.0), round(risk_budget_inr, 2)

    def compute_trade_pnl(self, entry: float, exit_px: float, qty: float, is_crypto: bool) -> Tuple[float, float, float]:
        gross_pnl = (exit_px - entry) * qty
        entry_val = entry * qty
        exit_val = exit_px * qty
        fees = (entry_val + exit_val) * (self.risk_cfg.exchange_fee_pct / 100.0)
        tds = exit_val * (self.risk_cfg.tds_pct / 100.0) if is_crypto else 0.0
        total_friction = round(fees + tds, 2)
        net_pnl = round(gross_pnl - total_friction, 2)
        return round(gross_pnl, 2), total_friction, net_pnl

    def run_backtest(self, candles: List[Candle], ctx: MarketContext) -> Dict:
        is_crypto = ctx.asset_class.upper() == "CRYPTO"
        px_dec = self._px_decimals(candles[0].close)
        regime = RegimeAnalyzer.evaluate(ctx, self.risk_cfg)

        closes = [c.close for c in candles]
        ema20 = TechnicalSMCAnalyzer.ema(closes, 20)
        rsi14 = TechnicalSMCAnalyzer.rsi(closes, 14)
        vwap_line = TechnicalSMCAnalyzer.vwap(candles)
        swing_highs, swing_lows = TechnicalSMCAnalyzer.detect_swings(candles)
        all_setups = TechnicalSMCAnalyzer.scan_order_blocks_and_fvg(candles, swing_highs)

        journal: List[JournalEntry] = []
        equity = self.risk_cfg.account_capital
        equity_curve = [{"index": candles[0].index, "time": candles[0].timestamp, "equity": equity}]

        for idx, s in enumerate(all_setups):
            if not s.confirmed:
                c = candles[s.ob_index]
                sl_buf = s.ob_low * (self.risk_cfg.sl_buffer_pct / 100.0)
                journal.append(JournalEntry(
                    trade_id=f"REJ-{idx + 1:02d}",
                    symbol=ctx.symbol,
                    timestamp=c.timestamp,
                    direction="LONG",
                    status="REJECTED",
                    entry_price=round(s.ob_high, px_dec),
                    stop_loss=round(s.ob_low - sl_buf, px_dec),
                    target_price=round(s.bsl_target, px_dec),
                    exit_price=0.0,
                    quantity=0.0,
                    rr_planned=0.0,
                    gross_pnl_inr=0.0,
                    fees_and_tds_inr=0.0,
                    net_pnl_inr=0.0,
                    r_multiple=0.0,
                    vol_regime=regime["vol_regime"],
                    notes=s.rejection_reason or "Rejected setup",
                ))

        confirmed_setups = [s for s in all_setups if s.confirmed]
        active_trade = None
        trades_taken = 0
        trade_markers = []

        for i, c in enumerate(candles):
            if active_trade is not None:
                if c.low <= active_trade["sl"]:
                    exit_px = active_trade["sl"]
                    gross_pnl, friction, net_pnl = self.compute_trade_pnl(
                        active_trade["entry"], exit_px, active_trade["qty"], is_crypto
                    )
                    equity += net_pnl
                    journal.append(JournalEntry(
                        trade_id=active_trade["id"],
                        symbol=ctx.symbol,
                        timestamp=active_trade["entry_time"],
                        direction="LONG",
                        status="LOSS",
                        entry_price=round(active_trade["entry"], px_dec),
                        stop_loss=round(active_trade["sl"], px_dec),
                        target_price=round(active_trade["tp"], px_dec),
                        exit_price=round(exit_px, px_dec),
                        quantity=active_trade["qty"],
                        rr_planned=active_trade["rr"],
                        gross_pnl_inr=gross_pnl,
                        fees_and_tds_inr=friction,
                        net_pnl_inr=net_pnl,
                        r_multiple=-1.0,
                        vol_regime=regime["vol_regime"],
                        notes=f"SL hit at {c.timestamp}",
                    ))
                    trade_markers.append({"index": i, "type": "EXIT_SL", "price": exit_px})
                    active_trade = None

                elif c.high >= active_trade["tp"]:
                    exit_px = active_trade["tp"]
                    gross_pnl, friction, net_pnl = self.compute_trade_pnl(
                        active_trade["entry"], exit_px, active_trade["qty"], is_crypto
                    )
                    equity += net_pnl
                    journal.append(JournalEntry(
                        trade_id=active_trade["id"],
                        symbol=ctx.symbol,
                        timestamp=active_trade["entry_time"],
                        direction="LONG",
                        status="WIN",
                        entry_price=round(active_trade["entry"], px_dec),
                        stop_loss=round(active_trade["sl"], px_dec),
                        target_price=round(active_trade["tp"], px_dec),
                        exit_price=round(exit_px, px_dec),
                        quantity=active_trade["qty"],
                        rr_planned=active_trade["rr"],
                        gross_pnl_inr=gross_pnl,
                        fees_and_tds_inr=friction,
                        net_pnl_inr=net_pnl,
                        r_multiple=active_trade["rr"],
                        vol_regime=regime["vol_regime"],
                        notes=f"BSL Target swept at {c.timestamp}",
                    ))
                    trade_markers.append({"index": i, "type": "EXIT_TP", "price": exit_px})
                    active_trade = None

            if (
                active_trade is None
                and regime["allow_long_execution"]
                and trades_taken < self.risk_cfg.max_trades_per_session
            ):
                for setup in confirmed_setups:
                    if i < setup.ob_index + 3:
                        continue
                    retested_zone = (c.low <= setup.fvg_high) and (c.low >= setup.ob_low)
                    bullish_rejection = c.is_bullish and (c.close >= ema20[i] or c.close >= vwap_line[i])
                    rsi_healthy = 40.0 <= rsi14[i] <= 72.0

                    if retested_zone and bullish_rejection and rsi_healthy:
                        entry_px = c.close
                        sl_buf = setup.ob_low * (self.risk_cfg.sl_buffer_pct / 100.0)
                        sl_px = setup.ob_low - sl_buf
                        tp_px = setup.bsl_target
                        risk_pts = entry_px - sl_px
                        reward_pts = tp_px - entry_px

                        if risk_pts <= 0:
                            continue
                        rr_ratio = round(reward_pts / risk_pts, 2)

                        if rr_ratio < self.risk_cfg.min_rr_ratio:
                            journal.append(JournalEntry(
                                trade_id=f"REJ-RR-{i}",
                                symbol=ctx.symbol,
                                timestamp=c.timestamp,
                                direction="LONG",
                                status="REJECTED",
                                entry_price=round(entry_px, px_dec),
                                stop_loss=round(sl_px, px_dec),
                                target_price=round(tp_px, px_dec),
                                exit_price=0.0,
                                quantity=0.0,
                                rr_planned=rr_ratio,
                                gross_pnl_inr=0.0,
                                fees_and_tds_inr=0.0,
                                net_pnl_inr=0.0,
                                r_multiple=0.0,
                                vol_regime=regime["vol_regime"],
                                notes=f"R:R {rr_ratio} below minimum {self.risk_cfg.min_rr_ratio}",
                            ))
                            continue

                        qty, risk_budget = self.calculate_position_size(
                            entry_px, sl_px, regime["effective_risk_pct"], is_crypto
                        )
                        if qty <= 0:
                            continue

                        trades_taken += 1
                        active_trade = {
                            "id": f"TRD-{trades_taken:02d}",
                            "entry_time": c.timestamp,
                            "entry": entry_px,
                            "sl": sl_px,
                            "tp": tp_px,
                            "qty": qty,
                            "rr": rr_ratio,
                        }
                        trade_markers.append({"index": i, "type": "ENTRY_LONG", "price": entry_px})
                        break

            equity_curve.append({"index": c.index, "time": c.timestamp, "equity": round(equity, 2)})

        executed = [j for j in journal if j.status in ("WIN", "LOSS")]
        wins = [j for j in executed if j.status == "WIN"]
        win_rate = (len(wins) / len(executed) * 100.0) if executed else 0.0
        net_pnl = round(equity - self.risk_cfg.account_capital, 2)
        total_friction = round(sum(j.fees_and_tds_inr for j in executed), 2)

        return {
            "symbol": ctx.symbol,
            "asset_class": ctx.asset_class,
            "regime": regime,
            "indicators": {
                "ema20": [round(x, px_dec) for x in ema20],
                "vwap": [round(x, px_dec) for x in vwap_line],
                "rsi14": [round(x, 2) for x in rsi14],
            },
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,
            "setups": [asdict(s) for s in all_setups],
            "markers": trade_markers,
            "journal": [asdict(j) for j in journal],
            "equity_curve": equity_curve,
            "metrics": {
                "initial_capital": self.risk_cfg.account_capital,
                "final_equity": round(equity, 2),
                "net_pnl_inr": net_pnl,
                "total_fees_tds_inr": total_friction,
                "return_pct": round((net_pnl / self.risk_cfg.account_capital) * 100.0, 2),
                "executed_trades": len(executed),
                "rejected_signals": len([j for j in journal if j.status == "REJECTED"]),
                "win_rate_pct": round(win_rate, 2),
            },
        }


# ==============================================================================
# 5. SAMPLE DATA & LIVE COINDCX PUBLIC CANDLE LOADER
# ==============================================================================

SYMBOL_BASE_PRICES = {
    "NIFTY50": 24470.0,
    "BTC_INR": 7850000.0,
    "ETH_INR": 265000.0,
    "SOL_INR": 14200.0,
    "POL_INR": 38.50,
}


def generate_sample_market_data(symbol: str = "BTC_INR") -> List[Candle]:
    """
    Generates 30 realistic 15-min/5-min candles scaled to the chosen asset:
    - NIFTY50 (Equity) or BTC_INR / ETH_INR / SOL_INR / POL_INR (Crypto)
    Contains: BSL Swing High, early unconfirmed Trap OB (rejected), Sweep to Swing Low,
    high-volume ChoCH + FVG displacement, OB+FVG retest entry, and BSL target sweep.
    """
    base = SYMBOL_BASE_PRICES.get(symbol.upper(), 7850000.0)
    scale = base / 24470.0
    dec = 4 if base < 500 else 2
    vol_mult = 0.001 if symbol.upper() == "BTC_INR" else (1.0 if symbol.upper() == "NIFTY50" else 12.0)

    template = [
        (0,  "09:15", 24470, 24505, 24460, 24498, 1800),
        (1,  "09:30", 24498, 24520, 24485, 24510, 2100),  # BSL Swing High
        (2,  "09:45", 24510, 24512, 24475, 24480, 1650),
        (3,  "10:00", 24480, 24485, 24445, 24450, 1550),
        (4,  "10:15", 24450, 24468, 24440, 24462, 1400),
        (5,  "10:30", 24462, 24465, 24420, 24425, 1600),
        (6,  "10:45", 24425, 24430, 24395, 24400, 1500),
        (7,  "11:00", 24400, 24404, 24372, 24376, 1450),  # Trap Order Block (No ChoCH/FVG)
        (8,  "11:15", 24376, 24392, 24374, 24388, 1200),
        (9,  "11:30", 24388, 24390, 24355, 24360, 1550),
        (10, "11:45", 24360, 24385, 24352, 24380, 1420),  # Internal Swing High (ChoCH level)
        (11, "12:00", 24380, 24382, 24345, 24348, 1600),
        (12, "12:15", 24348, 24352, 24328, 24332, 1700),
        (13, "12:30", 24332, 24344, 24322, 24338, 1380),
        (14, "12:45", 24338, 24342, 24318, 24325, 1520),
        (15, "13:00", 24335, 24340, 24310, 24315, 1950),  # Swing Low & Confirmed Bullish OB
        (16, "13:15", 24316, 24368, 24315, 24365, 3100),  # High-Vol Impulse 1
        (17, "13:30", 24366, 24415, 24360, 24408, 3400),  # High-Vol Impulse 2 (ChoCH + FVG)
        (18, "13:45", 24408, 24422, 24392, 24396, 1850),
        (19, "14:00", 24400, 24400, 24368, 24372, 1500),
        (20, "14:15", 24372, 24375, 24348, 24352, 1420),
        (21, "14:30", 24352, 24362, 24332, 24358, 2250),  # Retest into FVG + OB -> Entry
        (22, "14:45", 24358, 24385, 24354, 24382, 2150),
        (23, "15:00", 24382, 24410, 24378, 24405, 2300),
        (24, "15:15", 24405, 24432, 24398, 24428, 2400),
        (25, "15:30", 24428, 24455, 24420, 24450, 2550),
        (26, "15:45", 24450, 24478, 24442, 24472, 2600),
        (27, "16:00", 24472, 24495, 24465, 24490, 2700),
        (28, "16:15", 24490, 24532, 24485, 24525, 3600),  # BSL Target Swept -> Take Profit
        (29, "16:30", 24525, 24535, 24505, 24512, 2100),
    ]
    candles = []
    for idx, ts, o, h, l, c, v in template:
        candles.append(Candle(
            index=idx,
            timestamp=ts,
            open=round(o * scale, dec),
            high=round(h * scale, dec),
            low=round(l * scale, dec),
            close=round(c * scale, dec),
            volume=round(v * vol_mult, 4),
        ))
    return candles


def fetch_coindcx_candles(symbol: str = "BTC_INR", interval: str = "15m", limit: int = 45) -> Tuple[List[Candle], str]:
    """
    Fetches live public OHLCV candles from CoinDCX (e.g., pair=I-BTC_INR).
    Falls back cleanly to built-in sample data if offline or blocked.
    """
    pair = f"I-{symbol.upper()}"
    url = f"https://public.coindcx.com/market_data/candles/?pair={pair}&interval={interval}&limit={limit}"
    ist = timezone(timedelta(hours=5, minutes=30))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        if not isinstance(raw, list) or len(raw) < 10:
            raise ValueError("Insufficient live candle rows returned")
        raw.sort(key=lambda r: r["time"])
        candles = []
        for idx, r in enumerate(raw):
            ts = datetime.fromtimestamp(r["time"] / 1000.0, tz=ist).strftime("%d-%b %H:%M")
            candles.append(Candle(
                index=idx,
                timestamp=ts,
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["volume"]),
            ))
        return candles, f"LIVE_COINDCX ({pair} @ {interval})"
    except Exception as exc:
        return generate_sample_market_data(symbol), f"SAMPLE_FALLBACK ({exc})"


if __name__ == "__main__":
    candles = generate_sample_market_data("BTC_INR")
    ctx = MarketContext(asset_class="CRYPTO", symbol="BTC_INR")
    engine = ProTradingEngine(RiskConfig(sl_buffer_pct=0.04, exchange_fee_pct=0.20, tds_pct=1.0))
    report = engine.run_backtest(candles, ctx)
    print(json.dumps(report["metrics"], indent=2))
    print("\n--- CRYPTO TRADING JOURNAL ---")
    for row in report["journal"]:
        print(row)