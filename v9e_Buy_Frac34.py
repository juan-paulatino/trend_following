#!/usr/bin/env python3
"""
v9e_Buy_Frac34.py

A Bybit (linear) public WebSocket monitor that:
  - Builds 1m candles from the Bybit kline stream (confirm=True events).
  - Computes Heikin-Ashi (HA) candle color.
  - Computes 2 order-flow style metrics per closed candle:
      * agg_ratio: buy_volume / (buy_volume + sell_volume) from public trades inside the candle
      * avg_bbo_imba: average best-bid-vs-best-ask size imbalance during the candle from orderbook.1
  - Runs a simple state machine:
      * Setup detection: "Strict 3+1" (4 consecutive HA Green candles) or "Consolidation Watch"
      * Final entry confirmation gate: buy_frac from the AggTrades rollup must pass the entry-type threshold
        - Strict '3+1': uses the last 4 closed candles, buy_frac > 0.5499
        - Consolidation Watch: uses the last 7 closed candles, buy_frac > 0.5799
        - Logs an explicit BUY_FRAC gate verdict: ALLOW LONG or BLOCK LONG
      * Entry => "Ride the Wave" mode (suppresses new setups until exit)
      * Final entry-quality filters in v9e_Buy_Frac34:
          - Strict BUY_FRAC remains > 0.5499.
          - Consolidation Watch / CWatch BUY_FRAC is now > 0.5799.
          - Normal confirmation requires at least 40 aggressive trades in the rollup.
          - Low-sample exception: if total < 40, allow only when buy_frac > 0.75.
          - The old BUY_FRAC exhaustion band block (0.80 <= buy_frac < 0.90) is deprecated/removed.
      * Pre-emptive exit/hold layer while LONG:
          - Tracks max_close, the highest real Japanese close seen during the active HA Green streak.
          - A Watching state activates when a Green candle closes at or below the current max_close.
          - Exit hierarchy, evaluated before any red-grind fallback:
              Rule 1: EXIT immediately if current_close < max_close and agg < 0.12 OR bbo < 0.12.
              Rule 2: HOLD only if pullback >= 0.24%, median6 > 0.62, and all last-3 BBO values > 0.50.
              Rule 3: EXIT immediately if pullback >= 0.20% and median6 < 0.50.
              Rule 4: support-protected G-R-R RED-GRIND DECAY EXIT replaces the old 3-HA Red fallback.
                      It can exit on the 2nd HA Red candle only after evaluating the Green+Red+Red
                      support frame with the old median6/red_count/HOLD-ABSORPTION protection.
        NOTE: The old 3-HA Red fallback decision matrix is not restored as a 3-red exit; its support/absorption protection is reused inside Rule 4 on the G-R-R frame.

Usage:
  pip install pybit
  python v9e_Buy_Frac34.py --symbol POPCATUSDT

Optional:
  --testnet   to use testnet
  --loglevel  INFO (default), DEBUG, etc.
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

# ---- Strategy thresholds (ENTRY) ----
GREEN_ZONE_THRESHOLD = 0.55
RED_ZONE_THRESHOLD = 0.35
# Final / ultimate confirmation gate for LONG entries.
# A setup is allowed if buy_frac is above the entry-type threshold and either:
#   1) the rollup has at least BUY_FRAC_MIN_ROLLUP_TRADES aggressive trades; or
#   2) it is a low-sample exception with buy_frac > BUY_FRAC_LOW_SAMPLE_EXCEPTION_THRESHOLD.
# Frac34 keeps Strict at 0.5499 and tightens Consolidation Watch / CWatch to 0.5799.
# The old exhaustion-zone block (0.80 <= buy_frac < 0.90) is intentionally deprecated/removed.
BUY_FRAC_CONFIRMATION_THRESHOLD = 0.5499
BUY_FRAC_CWATCH_CONFIRMATION_THRESHOLD = 0.5799
BUY_FRAC_MIN_ROLLUP_TRADES = 40
BUY_FRAC_LOW_SAMPLE_EXCEPTION_THRESHOLD = 0.75

# ---- Pre-emptive exit/hold thresholds (evaluated before the red-grind decay fallback) ----
PREEMPTIVE_PANIC_THRESHOLD = 0.12                 # Rule 1: agg or bbo vanishes after a peak
PREEMPTIVE_HOLD_PULLBACK_THRESHOLD_PCT = 0.24     # Rule 2: peak-to-current close drop, in percent
PREEMPTIVE_HOLD_MEDIAN6_THRESHOLD = 0.62          # Rule 2: tighter absorption median threshold
PREEMPTIVE_HOLD_MIN_BBO = 0.50                    # Rule 2: all last-3 BBO values must stay above this
PREEMPTIVE_WEAR_TEAR_PULLBACK_THRESHOLD_PCT = 0.20
PREEMPTIVE_WEAR_TEAR_MEDIAN6_THRESHOLD = 0.50

# ---- Red-Grind Decay fallback thresholds (replaces the old 3-HA Red fallback) ----
DECAY_EXIT_RED_STREAK = 2
DECAY_MEDIAN6_WEAK_THRESHOLD = 0.62
DECAY_GIVEBACK_RATIO_THRESHOLD = 0.65

# Rule 4 support/absorption protection borrowed from the old 3-RED decision matrix,
# but applied to the Green + Red + Red frame that forms when the 2nd HA Red candle closes.
DECAY_SUPPORT_EXIT_THRESHOLD = 0.49999
DECAY_SUPPORT_BBO_GREEN = 0.80
DECAY_SUPPORT_NEUTRAL_LOW = 0.50
DECAY_SUPPORT_NEUTRAL_HIGH = 0.75

CONSOLIDATION_WATCH_PERIOD = 3   # candles
ALERT_COOLDOWN_PERIOD = 5        # candles (used only when NOT in-trade)

# ---- Buffer retention ----
TRADE_RETENTION_MS = 20 * 60_000      # keep ~20 minutes of trades
BBO_RETENTION_MS = 20 * 60_000        # keep ~20 minutes of BBO samples


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _utc_str_from_ms(ts_ms: int) -> str:
    # Match the style you showed in logs: 'YYYY-mm-dd HH:MM:SS,000'
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S,000")


@dataclass
class CandleSnapshot:
    utc: str
    close_price: float
    ha_open: float
    ha_close: float
    ha_color: str
    agg_ratio: float
    avg_bbo_imba: float

    # Trade counts for agg confidence
    buy_trades: int
    sell_trades: int
    total_trades: int

    # TickDirection (derived locally from SPOT publicTrade)
    tick_plus: int
    tick_zero_plus: int
    tick_minus: int
    tick_zero_minus: int
    tick_total: int
    tick_imbalance_ratio: float


class OrderFlowTracker:
    def __init__(self, symbol: str, testnet: bool = False) -> None:
        self.symbol = symbol.upper()
        self.testnet = testnet

        # Thread-safety for shared buffers
        self._lock = threading.Lock()

        # Buffers: trades + BBO imbalance samples (ts_ms, value)
        self._trade_buffer: Deque[Tuple[int, str, float]] = deque()
        self._bbo_samples: Deque[Tuple[int, float]] = deque()

        # TickDirection tracking (derived locally from SPOT publicTrade)
        # Bybit semantics:
        #  - PlusTick: current trade price > previous trade price
        #  - MinusTick: current trade price < previous trade price
        #  - ZeroPlusTick: current == previous, and the previous trade was higher than the trade before it
        #  - ZeroMinusTick: current == previous, and the previous trade was lower than the trade before it
        #
        # Locally we replicate this by remembering the last non-zero direction (PlusTick or MinusTick).
        # When price is unchanged, we emit ZeroPlusTick if the last non-zero move was PlusTick (or unknown),
        # otherwise ZeroMinusTick.
        self.ws_ticks = None  # separate WS connection for SPOT publicTrade
        self._tick_last_price: Optional[float] = None
        self._tick_last_non_zero: Optional[str] = None  # 'PlusTick' or 'MinusTick'
        self._tick_counts_by_minute: Dict[int, Dict[str, int]] = {}  # bucket_start_ms -> counts
        self._tick_bucket_retention_ms: int = 6 * 60 * 60 * 1000  # keep ~6h of buckets

        # Keep recent closed candles for rollups (Strict 3+1 and Exit NOW)
        self._recent_snaps: Deque[CandleSnapshot] = deque(maxlen=600)  # ~10h of 1m candles

        # Heikin Ashi state
        self._last_ha_open: Optional[float] = None
        self._last_ha_close: Optional[float] = None

        # Setup state machine
        self._ha_green_streak = 0
        self._consolidation_watch = 0
        self._cooldown = 0

        # Trade state
        self.in_trade_mode = False
        self.entry_price: Optional[float] = None
        self.entry_utc: Optional[str] = None
        self.entry_reason: Optional[str] = None

        # Pre-emptive exit/hold state (evaluated before the red-grind decay fallback)
        self.max_close: Optional[float] = None  # highest real close recorded during the active HA Green streak
        self._preemptive_watching = False
        self._preemptive_window: Deque[CandleSnapshot] = deque(maxlen=3)  # rolling 3-candle agg+bbo frame
        self._preemptive_last_decision: Optional[str] = None

        # Exit decision state (after 3 consecutive HA RED candles)
        self._trade_red_streak = 0
        self._trade_red_window: Deque[CandleSnapshot] = deque(maxlen=3)  # (agg_ratio, avg_bbo_imba)
        self._trade_last_decision: Optional[str] = None

        # Optional: maintain a deeper book (not used by strategy right now)
        self.live_bids: Dict[float, float] = {}
        self.live_asks: Dict[float, float] = {}

        self._stop_event = threading.Event()

        self.ws = None
        self.ws_depth = None

    # ----------------------------
    # Public API
    # ----------------------------
    def start(self) -> None:
        """Start websocket subscriptions and keep running until Ctrl+C."""
        try:
            from pybit.unified_trading import WebSocket
        except Exception as e:
            raise SystemExit(
                "Missing dependency 'pybit'. Install with: pip install pybit\n"
                f"Import error: {e}"
            )

        self.ws = WebSocket(testnet=self.testnet, channel_type="linear")

        # Dedicated SPOT publicTrade stream for tickDirection derivation (Option A)
        # Kept in a separate WS connection so the rest of the pipeline remains unchanged.
        try:
            self.ws_ticks = WebSocket(testnet=self.testnet, channel_type="spot")
        except Exception as e:
            self.ws_ticks = None
            logging.warning(f"TickDirection SPOT WS disabled (connect failed): {e}")

        # Subscribe to 1m klines, public trades, and top-of-book orderbook
        self._subscribe_kline_1m()
        self._subscribe_public_trades()
        try:
            self._subscribe_spot_trades_for_ticks()
        except Exception as e:
            logging.warning(f"TickDirection SPOT subscription failed; disabling ticks: {e}")
            self.ws_ticks = None
        self._subscribe_orderbook_1()

        # Optional: in a separate connection, keep a depth-target orderbook snapshot updated (v9a uses level 1)
        self.ws_depth = WebSocket(testnet=self.testnet, channel_type="linear")
        t = threading.Thread(target=self._subscribe_orderbook_50, name="orderbookDepth", daemon=True)
        t.start()

        logging.info("Started OrderFlowTracker for %s (testnet=%s).", self.symbol, self.testnet)

        try:
            while not self._stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            logging.info("Stopping...")
            self._stop_event.set()
        finally:
            # pybit WS has close() in newer versions; guard it.
            for w in (self.ws, self.ws_depth, self.ws_ticks):
                try:
                    if w:
                        w.exit()
                except Exception:
                    try:
                        if w:
                            w.close()
                    except Exception:
                        pass

    def stop(self) -> None:
        self._stop_event.set()

    # ----------------------------
    # Subscriptions (best-effort across pybit versions)
    # ----------------------------
    def _subscribe_kline_1m(self) -> None:
        assert self.ws is not None
        # Different pybit versions have slightly different signatures; try common ones.
        try:
            self.ws.kline_stream(callback=self._on_kline, symbol=self.symbol, interval=1)
            return
        except TypeError:
            pass
        try:
            self.ws.kline_stream(self._on_kline, self.symbol, "1")
            return
        except Exception as e:
            raise RuntimeError(f"Failed subscribing to kline stream: {e}")

    def _subscribe_public_trades(self) -> None:
        """Subscribe to Bybit public trades.

        pybit renamed this method across versions:
          - newer: trade_stream(symbol=..., callback=...)
          - some older forks/examples: public_trade_stream(...)
        We support both.
        """
        assert self.ws is not None

        # 1) Try public_trade_stream if present
        if hasattr(self.ws, "public_trade_stream"):
            try:
                self.ws.public_trade_stream(callback=self._on_trade, symbol=self.symbol)
                return
            except TypeError:
                pass
            try:
                self.ws.public_trade_stream(self._on_trade, self.symbol)
                return
            except Exception:
                # fall through to trade_stream
                pass

        # 2) Try trade_stream (official pybit unified_trading)
        if hasattr(self.ws, "trade_stream"):
            try:
                self.ws.trade_stream(symbol=self.symbol, callback=self._on_trade)
                return
            except TypeError:
                pass
            try:
                self.ws.trade_stream(self.symbol, self._on_trade)
                return
            except Exception as e:
                raise RuntimeError(f"Failed subscribing to trade stream: {e}")

        raise AttributeError(
            "WebSocket client has neither 'trade_stream' nor 'public_trade_stream'. "
            "Please upgrade/downgrade pybit to a compatible version."
        )


    def _subscribe_spot_trades_for_ticks(self) -> None:
        """Subscribe to SPOT public trades for tickDirection derivation only."""
        if self.ws_ticks is None:
            return

        ws = self.ws_ticks

        # Mirror compatibility logic used in _subscribe_public_trades()
        if hasattr(ws, "public_trade_stream"):
            try:
                ws.public_trade_stream(callback=self._on_spot_trade_tick, symbol=self.symbol)
                return
            except TypeError:
                pass
            try:
                ws.public_trade_stream(self._on_spot_trade_tick, self.symbol)
                return
            except Exception:
                pass

        if hasattr(ws, "trade_stream"):
            try:
                ws.trade_stream(symbol=self.symbol, callback=self._on_spot_trade_tick)
                return
            except TypeError:
                pass
            try:
                ws.trade_stream(self.symbol, self._on_spot_trade_tick)
                return
            except Exception as e:
                raise RuntimeError(f"Failed subscribing to SPOT trade stream for ticks: {e}")

        raise AttributeError(
            "SPOT WebSocket client has neither 'trade_stream' nor 'public_trade_stream'. "
            "Please upgrade/downgrade pybit to a compatible version."
        )
    def _subscribe_orderbook_1(self) -> None:
        assert self.ws is not None
        # Some versions use orderbook_stream(depth=1), others have separate methods.
        for attempt in (
            lambda: self.ws.orderbook_stream(callback=self._on_orderbook_1, symbol=self.symbol, depth=1),
            lambda: self.ws.orderbook_stream(self._on_orderbook_1, self.symbol, 1),
            lambda: self.ws.orderbook_stream(self._on_orderbook_1, self.symbol),
        ):
            try:
                attempt()
                return
            except TypeError:
                continue
            except Exception:
                continue
        logging.warning("Could not subscribe to orderbook.1 via pybit helpers. "
                        "If your pybit version differs, adjust _subscribe_orderbook_1().")

    def _subscribe_orderbook_50(self) -> None:
        """Keeps the secondary depth book updated (v9a configured to level 1). Strategy does not consume it yet."""
        assert self.ws_depth is not None
        for attempt in (
            lambda: self.ws_depth.orderbook_stream(callback=self._on_orderbook_50, symbol=self.symbol, depth=1),
            lambda: self.ws_depth.orderbook_stream(self._on_orderbook_50, self.symbol, 1),
            lambda: self.ws_depth.orderbook_stream(self._on_orderbook_50, self.symbol),
        ):
            try:
                attempt()
                logging.info("Subscribed to orderbook depth stream (target=1).")
                return
            except TypeError:
                continue
            except Exception:
                continue
        logging.warning("Could not subscribe to orderbook depth stream (target=1). Continuing without it.")

    # ----------------------------
    # WebSocket callbacks
    # ----------------------------
    def _on_trade(self, msg: Any) -> None:
        """
        Expected (typical) Bybit v5 trade message shape:
          {"topic":"publicTrade.SYMBOL","data":[{"T":..., "S":"Buy"/"Sell", "v":"..."}]}
        """
        try:
            data = msg.get("data") if isinstance(msg, dict) else None
            if not data:
                return
            if isinstance(data, dict):
                data = [data]
            now_ms = int(time.time() * 1000)
            with self._lock:
                for t in data:
                    ts_ms = int(t.get("T") or t.get("ts") or now_ms)
                    side = str(t.get("S") or t.get("side") or "").title()
                    if side not in ("Buy", "Sell"):
                        # some feeds use lowercase
                        side = str(t.get("S") or t.get("side") or "").capitalize()
                    size = _safe_float(t.get("v") or t.get("size") or t.get("qty") or 0.0, 0.0)
                    self._trade_buffer.append((ts_ms, side, size))
                self._purge_old(now_ms)
        except Exception:
            # avoid killing the WS thread on parse errors
            logging.debug("Trade parse error", exc_info=True)


    def _on_spot_trade_tick(self, msg: Any) -> None:
        """Process SPOT publicTrade messages and update per-candle tickDirection counts."""
        try:
            if not isinstance(msg, dict):
                return
            data = msg.get("data")
            if not data:
                return
            if isinstance(data, dict):
                data = [data]

            # Sort within message to reduce out-of-order effects
            try:
                data = sorted(data, key=lambda x: int(x.get("T") or 0))
            except Exception:
                pass

            now_ms = int(time.time() * 1000)

            with self._lock:
                for t in data:
                    ts_ms = int(t.get("T") or t.get("ts") or now_ms)
                    price = _safe_float(t.get("p") or t.get("price"), None)
                    if price is None:
                        continue
                    side = str(t.get("S") or "").lower()

                    # Derive tickDirection (replicates Bybit 'L' semantics)
                    # If we haven't observed a non-zero direction yet and price doesn't change,
                    # use aggressor side as a tie-breaker so we don't "skip" flat prints.
                    if self._tick_last_price is None:
                        tick = "ZeroPlusTick" if side == "buy" else "ZeroMinusTick" if side == "sell" else "ZeroPlusTick"
                    else:
                        if price > self._tick_last_price:
                            tick = "PlusTick"
                            self._tick_last_non_zero = "PlusTick"
                        elif price < self._tick_last_price:
                            tick = "MinusTick"
                            self._tick_last_non_zero = "MinusTick"
                        else:
                            if self._tick_last_non_zero == "MinusTick":
                                tick = "ZeroMinusTick"
                            elif self._tick_last_non_zero == "PlusTick":
                                tick = "ZeroPlusTick"
                            else:
                                tick = "ZeroPlusTick" if side == "buy" else "ZeroMinusTick" if side == "sell" else "ZeroPlusTick"

                    self._tick_last_price = price

                    bucket = (ts_ms // 60_000) * 60_000
                    cnt = self._tick_counts_by_minute.get(bucket)
                    if cnt is None:
                        cnt = {"plus": 0, "zero_plus": 0, "minus": 0, "zero_minus": 0, "total": 0}
                        self._tick_counts_by_minute[bucket] = cnt

                    if tick == "PlusTick":
                        cnt["plus"] += 1
                    elif tick == "ZeroPlusTick":
                        cnt["zero_plus"] += 1
                    elif tick == "MinusTick":
                        cnt["minus"] += 1
                    else:
                        cnt["zero_minus"] += 1
                    cnt["total"] += 1

                self._purge_old_tick_buckets(now_ms)
        except Exception:
            logging.debug("SPOT tickDirection trade parse error", exc_info=True)
    def _on_orderbook_1(self, msg: Any) -> None:
        """
        Expected (typical) Bybit v5 orderbook message shape:
          {"topic":"orderbook.1.SYMBOL","data":{"b":[[price, size],...],"a":[[price,size],...],"ts":...}}
        """
        try:
            if not isinstance(msg, dict):
                return
            data = msg.get("data")
            if not data or not isinstance(data, dict):
                return

            bids = data.get("b") or data.get("bids") or []
            asks = data.get("a") or data.get("asks") or []

            if not bids or not asks:
                return

            best_bid = bids[0]
            best_ask = asks[0]

            bid_sz = _safe_float(best_bid[1] if isinstance(best_bid, (list, tuple)) else best_bid.get("size"), 0.0)
            ask_sz = _safe_float(best_ask[1] if isinstance(best_ask, (list, tuple)) else best_ask.get("size"), 0.0)

            denom = bid_sz + ask_sz
            imbalance = (bid_sz / denom) if denom > 0 else 0.5

            ts_ms = int(data.get("ts") or msg.get("ts") or time.time() * 1000)
            with self._lock:
                self._bbo_samples.append((ts_ms, imbalance))
                self._purge_old(ts_ms)
        except Exception:
            logging.debug("Orderbook(1) parse error", exc_info=True)

    def _on_orderbook_50(self, msg: Any) -> None:
        """Maintain a live depth-book snapshot (v9a target=1, not used in decision logic yet)."""
        try:
            if not isinstance(msg, dict):
                return
            data = msg.get("data")
            if not data or not isinstance(data, dict):
                return

            msg_type = msg.get("type") or data.get("type") or "delta"

            bids = data.get("b") or data.get("bids") or []
            asks = data.get("a") or data.get("asks") or []

            with self._lock:
                if msg_type == "snapshot":
                    self.live_bids.clear()
                    self.live_asks.clear()

                for p, s in bids:
                    pf, sf = _safe_float(p), _safe_float(s)
                    if sf <= 0:
                        self.live_bids.pop(pf, None)
                    else:
                        self.live_bids[pf] = sf

                for p, s in asks:
                    pf, sf = _safe_float(p), _safe_float(s)
                    if sf <= 0:
                        self.live_asks.pop(pf, None)
                    else:
                        self.live_asks[pf] = sf
        except Exception:
            logging.debug("Orderbook(depth-target) parse error", exc_info=True)

    def _on_kline(self, msg: Any) -> None:
        """
        Expected (typical) Bybit v5 kline message shape:
          {"topic":"kline.1.SYMBOL","data":[{"start":..., "end":..., "open":"...", "high":"...", "low":"...", "close":"...",
                                           "confirm":true}]}
        """
        try:
            if not isinstance(msg, dict):
                return
            data = msg.get("data")
            if not data:
                return
            if isinstance(data, dict):
                data = [data]
            k = data[0]

            confirm = bool(k.get("confirm", False))
            if not confirm:
                return

            start_ms = int(k.get("start") or 0)
            end_ms = int(k.get("end") or 0)
            if end_ms <= 0:
                # fallback: use now
                end_ms = int(time.time() * 1000)
                start_ms = end_ms - 60_000

            o = _safe_float(k.get("open"), 0.0)
            h = _safe_float(k.get("high"), 0.0)
            l = _safe_float(k.get("low"), 0.0)
            c = _safe_float(k.get("close"), 0.0)

            snapshot = self._build_snapshot(
                candle_start_ms=start_ms,
                candle_end_ms=end_ms,
                o=o, h=h, l=l, c=c
            )

            # Log the candle snapshot (human-readable, fixed 5 decimals)
            logging.info(
                f"[{snapshot.utc}] close={snapshot.close_price:.5f} | "
                f"HA(O/C)={snapshot.ha_open:.5f}/{snapshot.ha_close:.5f} ({snapshot.ha_color}) | "
                f"agg={snapshot.agg_ratio:.5f} | bbo={snapshot.avg_bbo_imba:.5f}"
            )


            # Save closed candle for later tick rollups
            with self._lock:
                self._recent_snaps.append(snapshot)
            # Run trade logic
            if self.in_trade_mode:
                self._run_exit_logic(snapshot)
            else:
                self._run_setup_logic(snapshot)
        except Exception:
            logging.debug("Kline parse error", exc_info=True)

    # ----------------------------
    # Core computations
    # ----------------------------
    def _purge_old(self, now_ms: int) -> None:
        """Purge old items from buffers."""
        # Trades
        cutoff_trades = now_ms - TRADE_RETENTION_MS
        while self._trade_buffer and self._trade_buffer[0][0] < cutoff_trades:
            self._trade_buffer.popleft()

        # BBO samples
        cutoff_bbo = now_ms - BBO_RETENTION_MS
        while self._bbo_samples and self._bbo_samples[0][0] < cutoff_bbo:
            self._bbo_samples.popleft()


    def _purge_old_tick_buckets(self, now_ms: int) -> None:
        """Keep tickDirection minute buckets from growing without bound."""
        cutoff = now_ms - self._tick_bucket_retention_ms
        old_keys = [k for k in self._tick_counts_by_minute.keys() if k < cutoff]
        for k in old_keys:
            self._tick_counts_by_minute.pop(k, None)
    def _build_snapshot(self, candle_start_ms: int, candle_end_ms: int, o: float, h: float, l: float, c: float) -> CandleSnapshot:
        utc = _utc_str_from_ms(candle_end_ms)

        # Heikin Ashi
        ha_close = (o + h + l + c) / 4.0
        if self._last_ha_open is None or self._last_ha_close is None:
            ha_open = (o + c) / 2.0
        else:
            ha_open = (self._last_ha_open + self._last_ha_close) / 2.0

        self._last_ha_open = ha_open
        self._last_ha_close = ha_close

        ha_color = "Green" if ha_close > ha_open else "Red"

        with self._lock:
            # Aggression ratio from trades inside candle
            buy_vol = 0.0
            sell_vol = 0.0
            buy_count = 0
            sell_count = 0
            for ts_ms, side, size in self._trade_buffer:
                if candle_start_ms <= ts_ms < candle_end_ms:
                    if side == "Buy":
                        buy_vol += size
                        buy_count += 1
                    elif side == "Sell":
                        sell_vol += size
                        sell_count += 1

            total = buy_vol + sell_vol
            agg_ratio = (buy_vol / total) if total > 0 else 0.5

            # Average BBO imbalance from samples inside candle
            imbas: List[float] = [v for ts, v in self._bbo_samples if candle_start_ms <= ts < candle_end_ms]
            avg_bbo_imba = sum(imbas) / len(imbas) if imbas else 0.5

            # TickDirection counts for this candle (SPOT-derived, locally computed)
            b = ((candle_end_ms - 1) // 60_000) * 60_000  # align to trade buckets (minute of candle end)
            tc = self._tick_counts_by_minute.get(b) or {"plus": 0, "zero_plus": 0, "minus": 0, "zero_minus": 0, "total": 0}
            tick_plus = int(tc.get("plus", 0))
            tick_zero_plus = int(tc.get("zero_plus", 0))
            tick_minus = int(tc.get("minus", 0))
            tick_zero_minus = int(tc.get("zero_minus", 0))
            tick_total = int(tc.get("total", 0))
            tick_imb = (tick_plus + tick_zero_plus) - (tick_minus + tick_zero_minus)
            tick_imbalance_ratio = (tick_imb / tick_total) if tick_total > 0 else 0.0
            # If SPOT tickDirection stream is unavailable for this minute (tick_total==0) but we DO have trades
            # in the agg window, synthesize a non-null tick proxy from aggressor side counts.
            # This avoids misleading "all zeros" tick windows during brief SPOT WS gaps.
            if tick_total == 0 and (buy_count + sell_count) > 0:
                tick_plus = 0
                tick_minus = 0
                tick_zero_plus = int(buy_count)
                tick_zero_minus = int(sell_count)
                tick_total = int(buy_count + sell_count)
                tick_imb = (tick_plus + tick_zero_plus) - (tick_minus + tick_zero_minus)
                tick_imbalance_ratio = (tick_imb / tick_total) if tick_total > 0 else 0.0


        return CandleSnapshot(
            utc=utc,
            close_price=c,
            ha_open=ha_open,
            ha_close=ha_close,
            ha_color=ha_color,
            agg_ratio=agg_ratio,
            avg_bbo_imba=avg_bbo_imba,
            buy_trades=buy_count,
            sell_trades=sell_count,
            total_trades=(buy_count + sell_count),
            tick_plus=tick_plus,
            tick_zero_plus=tick_zero_plus,
            tick_minus=tick_minus,
            tick_zero_minus=tick_zero_minus,
            tick_total=tick_total,
            tick_imbalance_ratio=tick_imbalance_ratio,
        )


    def _rollup_ticks_last_n(self, n: int) -> Dict[str, Any]:
        """Roll up tickDirection counts across the last n closed candles."""
        with self._lock:
            snaps = list(self._recent_snaps)[-n:] if n > 0 else []

        plus = sum(s.tick_plus for s in snaps)
        zero_plus = sum(s.tick_zero_plus for s in snaps)
        minus = sum(s.tick_minus for s in snaps)
        zero_minus = sum(s.tick_zero_minus for s in snaps)
        total = sum(s.tick_total for s in snaps)

        imb = (plus + zero_plus) - (minus + zero_minus)
        ratio = (imb / total) if total > 0 else 0.0
        colors = ''.join('G' if s.ha_color == 'Green' else 'R' for s in snaps)

        return {
            "n": n,
            "colors": colors,
            "plus": plus,
            "zero_plus": zero_plus,
            "minus": minus,
            "zero_minus": zero_minus,
            "total": total,
            "tick_imbalance": imb,
            "tick_imbalance_ratio": ratio,
        }
    
    def _rollup_trade_counts_last_n(self, n: int) -> Dict[str, Any]:
        """Roll up trade side counts across the last n closed candles (from the same trade stream used for agg_ratio)."""
        with self._lock:
            snaps = list(self._recent_snaps)[-n:] if n > 0 else []

        buy = sum(getattr(s, "buy_trades", 0) for s in snaps)
        sell = sum(getattr(s, "sell_trades", 0) for s in snaps)
        total = buy + sell
        buy_frac = (buy / total) if total > 0 else 0.0
        sell_frac = (sell / total) if total > 0 else 0.0

        return {
            "n": n,
            "buy": int(buy),
            "sell": int(sell),
            "total": int(total),
            "buy_frac": float(buy_frac),
            "sell_frac": float(sell_frac),
        }

    def _rollup_trade_counts_red_window(self) -> Dict[str, Any]:
        """Roll up trade side counts across the current 3-red window used for exit decisions."""
        with self._lock:
            snaps = list(self._trade_red_window)

        buy = sum(getattr(s, "buy_trades", 0) for s in snaps)
        sell = sum(getattr(s, "sell_trades", 0) for s in snaps)
        total = buy + sell
        buy_frac = (buy / total) if total > 0 else 0.0
        sell_frac = (sell / total) if total > 0 else 0.0

        return {
            "n": len(snaps),
            "buy": int(buy),
            "sell": int(sell),
            "total": int(total),
            "buy_frac": float(buy_frac),
            "sell_frac": float(sell_frac),
        }
# ----------------------------
    # Trading state machine
    # ----------------------------
    def _check_entry_conditions(self, snap: CandleSnapshot) -> bool:
        # Must be a green HA candle
        if snap.ha_color != "Green":
            return False

        # No metric in red zone
        if snap.agg_ratio < RED_ZONE_THRESHOLD or snap.avg_bbo_imba < RED_ZONE_THRESHOLD:
            return False

        # Need at least one metric in green zone ("conviction")
        if snap.agg_ratio < GREEN_ZONE_THRESHOLD and snap.avg_bbo_imba < GREEN_ZONE_THRESHOLD:
            return False

        return True

    def _get_entry_rollups(self, entry_type: str) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        """Return the tick/trade rollups used by the last confirmation gate."""
        if entry_type == "Strict '3+1'":
            n = 4
        elif entry_type == "Consolidation Watch":
            n = 4 + CONSOLIDATION_WATCH_PERIOD
        else:
            raise ValueError(f"Unsupported entry_type for rollups: {entry_type}")

        label = f"last{n}"
        t = self._rollup_ticks_last_n(n)
        c = self._rollup_trade_counts_last_n(n)
        return t, c, label

    def _log_entry_rollups(self, label: str, t: Dict[str, Any], c: Dict[str, Any]) -> None:
        logging.info(
            "TickDirection rollup (%s): plus=%d zplus=%d zminus=%d minus=%d total=%d | imb=%d ratio=%.5f",
            label,
            int(t["plus"]),
            int(t["zero_plus"]),
            int(t["zero_minus"]),
            int(t["minus"]),
            int(t["total"]),
            int(t["tick_imbalance"]),
            float(t["tick_imbalance_ratio"]),
        )
        logging.info(
            "AggTrades rollup (%s): buy/total=%d/%d (buy_frac=%.3f) | sell=%d",
            label,
            int(c["buy"]),
            int(c["total"]),
            float(c["buy_frac"]),
            int(c["sell"]),
        )

    def _buy_frac_threshold_for_entry(self, entry_type: str) -> float:
        """Return the entry-type-specific BUY_FRAC threshold.

        Frac34 keeps Strict at 0.5499 and tightens Consolidation Watch
        / CWatch to 0.5799. Everything else in the gate remains unchanged.
        """
        if entry_type == "Consolidation Watch":
            return BUY_FRAC_CWATCH_CONFIRMATION_THRESHOLD
        return BUY_FRAC_CONFIRMATION_THRESHOLD

    def _buy_frac_gate_allows(self, entry_type: str, c: Dict[str, Any]) -> bool:
        """Final / ultimate confirmation gate for LONG entries.

        v9e_Buy_Frac34 uses an entry-type-specific minimum buy_frac gate,
        keeps the total>=40 participation filter, and keeps the low-sample
        exception:
          - Strict buy_frac must be > 0.5499;
          - Consolidation Watch buy_frac must be > 0.5799; and
          - either total >= 40, or total < 40 with buy_frac > 0.75.

        The previous exhaustion-zone block (0.80 <= buy_frac < 0.90) is
        intentionally deprecated and is not applied.
        """
        buy_frac = float(c.get("buy_frac", 0.0))
        total = int(c.get("total", 0))
        threshold = self._buy_frac_threshold_for_entry(entry_type)

        if buy_frac <= threshold:
            return False

        if total >= BUY_FRAC_MIN_ROLLUP_TRADES:
            return True

        # Low-sample exception: allow small rollups only when the buy_frac
        # is strong enough to compensate for the smaller sample.
        if buy_frac > BUY_FRAC_LOW_SAMPLE_EXCEPTION_THRESHOLD:
            return True

        return False

    def _log_buy_frac_gate(self, entry_type: str, label: str, c: Dict[str, Any], allowed: bool) -> None:
        buy_frac = float(c["buy_frac"])
        total = int(c["total"])
        threshold = self._buy_frac_threshold_for_entry(entry_type)
        gate_state = "ALLOW" if allowed else "BLOCK"

        if allowed and total >= BUY_FRAC_MIN_ROLLUP_TRADES:
            verdict = "supportive confirmation"
        elif allowed:
            verdict = "low-sample exception: buy_frac > %.2f despite total < %d" % (
                BUY_FRAC_LOW_SAMPLE_EXCEPTION_THRESHOLD,
                BUY_FRAC_MIN_ROLLUP_TRADES,
            )
        elif buy_frac <= threshold:
            verdict = "buy_frac failed final confirmation -> skip/block long"
        elif total < BUY_FRAC_MIN_ROLLUP_TRADES:
            verdict = "insufficient rollup trades and buy_frac <= %.2f -> skip/block long" % (
                BUY_FRAC_LOW_SAMPLE_EXCEPTION_THRESHOLD,
            )
        else:
            verdict = "entry-quality gate failed -> skip/block long"

        log_fn = logging.info if allowed else logging.warning
        log_fn(
            "BUY_FRAC confirmation gate (%s, %s): %s LONG | buy_frac=%.5f | "
            "rules: buy_frac > %.4f AND (total >= %d OR low-total buy_frac > %.2f) | "
            "buy/total=%d/%d | sell=%d | verdict=%s",
            entry_type,
            label,
            gate_state,
            buy_frac,
            threshold,
            BUY_FRAC_MIN_ROLLUP_TRADES,
            BUY_FRAC_LOW_SAMPLE_EXCEPTION_THRESHOLD,
            int(c["buy"]),
            total,
            int(c["sell"]),
            verdict,
        )

    def _attempt_entry(self, snap: CandleSnapshot, entry_type: str) -> bool:
        """Apply the final BUY_FRAC gate before allowing the LONG entry."""
        t, c, label = self._get_entry_rollups(entry_type)
        allowed = self._buy_frac_gate_allows(entry_type, c)

        if not allowed:
            if entry_type == "Strict '3+1'":
                logging.warning(f"*** TRADE BLOCKED: Strict '3+1' setup at {snap.utc} ***")
            else:
                logging.warning(f"*** TRADE BLOCKED: '{entry_type}' setup at {snap.utc} ***")
            self._log_entry_rollups(label, t, c)
            self._log_buy_frac_gate(entry_type, label, c, allowed=False)
            logging.info("LONG entry was blocked by the BUY_FRAC final confirmation gate. Staying flat.")
            return False

        self._trigger_entry(snap, entry_type=entry_type, t=t, c=c, label=label)
        return True

    def _run_setup_logic(self, snap: CandleSnapshot) -> None:
        # Cooldown only applies when not in-trade
        if self._cooldown > 0:
            self._cooldown -= 1
            return

        # Update HA streak + reset watch on Red
        if snap.ha_color == "Green":
            self._ha_green_streak += 1
        else:
            self._ha_green_streak = 0
            self._consolidation_watch = 0

        # If we're in consolidation watch, check entry each candle
        if self._consolidation_watch > 0:
            if self._check_entry_conditions(snap):
                if self._attempt_entry(snap, entry_type="Consolidation Watch"):
                    return
            self._consolidation_watch -= 1
            if self._consolidation_watch == 0:
                logging.info("Consolidation Watch ended without entry.")
            return

        # Strict "3+1" = 4 consecutive green HA candles
        if self._ha_green_streak == 4:
            if self._check_entry_conditions(snap):
                if not self._attempt_entry(snap, entry_type="Strict '3+1'"):
                    self._consolidation_watch = CONSOLIDATION_WATCH_PERIOD
                    logging.info(
                        "Strict '3+1' setup passed base checks but was blocked by BUY_FRAC gate. Entering Consolidation Watch for %s candles.",
                        CONSOLIDATION_WATCH_PERIOD,
                    )
            else:
                self._consolidation_watch = CONSOLIDATION_WATCH_PERIOD
                logging.info("Strict '3+1' failed. Entering Consolidation Watch for %s candles.", CONSOLIDATION_WATCH_PERIOD)

    def _trigger_entry(
        self,
        snap: CandleSnapshot,
        entry_type: str,
        t: Dict[str, Any],
        c: Dict[str, Any],
        label: str,
    ) -> None:
        # Log the setup warning (same style as your output)
        if entry_type == "Strict '3+1'":
            logging.warning(f"*** TRADE SETUP: Strict '3+1' Entry Triggered at {snap.utc} ***")
        else:
            logging.warning(f"*** TRADE SETUP: '{entry_type}' Entry Triggered at {snap.utc} ***")

        self._log_entry_rollups(label, t, c)
        self._log_buy_frac_gate(entry_type, label, c, allowed=True)

        # Enter trade mode
        self.in_trade_mode = True
        self.entry_price = snap.close_price
        self.entry_utc = snap.utc
        self.entry_reason = entry_type

        # Reset exit decision state
        self._trade_red_streak = 0
        self._trade_red_window.clear()
        self._trade_last_decision = None

        # Reset and seed pre-emptive state for the new LONG. The entry candle is
        # part of the active HA Green streak, so its real close is the initial max_close.
        self._reset_preemptive_state()
        self.max_close = snap.close_price
        self._preemptive_window.append(snap)
        logging.info(
            "Pre-emptive peak tracker seeded at entry | max_close=%.5f at %s",
            self.max_close,
            snap.utc,
        )

        # Reset setup state
        self._ha_green_streak = 0
        self._consolidation_watch = 0

        logging.info("Entering 'Ride the Wave' mode (LONG). Suppressing new trade setups until EXIT...")

    def _reset_preemptive_state(self) -> None:
        """Reset the peak/watch state used by the pre-emptive exit layer."""
        self.max_close = None
        self._preemptive_watching = False
        self._preemptive_window.clear()
        self._preemptive_last_decision = None

    def _reset_trade_state_after_exit(self) -> None:
        """Return the strategy to flat mode after any exit path."""
        self.in_trade_mode = False
        self.entry_price = None
        self.entry_utc = None
        self.entry_reason = None
        self._trade_red_streak = 0
        self._trade_red_window.clear()
        self._trade_last_decision = None
        self._reset_preemptive_state()

        # Optional: cooldown after an exit to avoid immediate re-entry on the same microstructure noise.
        # Set to 0 to mimic the original "immediate resume" behavior.
        self._cooldown = 0

    def _median6_from_snaps(self, snaps: List[CandleSnapshot]) -> Optional[float]:
        """Return median6 from 3 candles: 3 agg values + 3 bbo values."""
        if len(snaps) < 3:
            return None
        frame = [s.agg_ratio for s in snaps[-3:]] + [s.avg_bbo_imba for s in snaps[-3:]]
        sf = sorted(frame)
        return (sf[2] + sf[3]) / 2.0

    def _update_preemptive_peak_watch(self, snap: CandleSnapshot) -> None:
        """Track max_close during HA Green trend and activate Watching on first non-higher close.

        max_close is only advanced by HA Green candles. Once Watching is activated,
        it stays active until a new Green candle makes a higher close or the trade exits.
        """
        if snap.ha_color != "Green":
            return

        if self.max_close is None:
            self.max_close = snap.close_price
            self._preemptive_watching = False
            self._preemptive_last_decision = None
            logging.info(
                "Pre-emptive peak tracker initialized | max_close=%.5f at %s",
                self.max_close,
                snap.utc,
            )
            return

        if snap.close_price > self.max_close:
            old_max = self.max_close
            self.max_close = snap.close_price
            if self._preemptive_watching:
                logging.info(
                    "Pre-emptive Watching reset by new high | old_max=%.5f | new max_close=%.5f at %s",
                    old_max,
                    self.max_close,
                    snap.utc,
                )
            self._preemptive_watching = False
            self._preemptive_last_decision = None
            return

        if not self._preemptive_watching:
            self._preemptive_watching = True
            logging.info(
                "Pre-emptive Watching activated | current_close=%.5f <= max_close=%.5f at %s",
                snap.close_price,
                self.max_close,
                snap.utc,
            )

    def _run_preemptive_exit_hold_logic(self, snap: CandleSnapshot) -> bool:
        """Evaluate pre-emptive Rule 1/2/3 before the red-grind decay fallback.

        Returns True when the pre-emptive layer made a final decision for this
        candle (EXIT or HOLD) and the caller should not continue to Rule 4.
        Returns False when no pre-emptive condition matched.
        """
        if not self.in_trade_mode or self.entry_price is None or self.max_close is None:
            return False
        if not self._preemptive_watching:
            return False

        current_close = snap.close_price
        max_close = self.max_close
        if max_close <= 0:
            return False

        drop_pct = ((max_close - current_close) / max_close) * 100.0
        snaps3 = list(self._preemptive_window)
        median6 = self._median6_from_snaps(snaps3)
        aggs = [s.agg_ratio for s in snaps3[-3:]]
        bbos = [s.avg_bbo_imba for s in snaps3[-3:]]

        # Rule 1: Instant Panic Fuse (Agility Filter). Kept intentionally aggressive.
        if current_close < max_close and (snap.agg_ratio < PREEMPTIVE_PANIC_THRESHOLD or snap.avg_bbo_imba < PREEMPTIVE_PANIC_THRESHOLD):
            reasons = [
                "PRE-EMPTIVE RULE 1 PANIC FUSE",
                f"current_close={current_close:.5f} < max_close={max_close:.5f}",
                f"drop_pct={drop_pct:.3f}%",
                f"agg={snap.agg_ratio:.5f}",
                f"bbo={snap.avg_bbo_imba:.5f}",
                f"panic_thr<{PREEMPTIVE_PANIC_THRESHOLD:.2f}",
            ]
            self._exit_long(snap, reasons)
            return True

        # Rules 2/3 require a full rolling 3-candle median6 frame.
        if median6 is None:
            return False

        # Rule 2: Scientific Absorption Hold (Yo-Yo Filter), tightened.
        # It now requires a stronger median6 and all BBO values in the last-3 frame
        # to remain above 0.50. This avoids holding weak decays that merely have one
        # strong value carrying the median.
        if drop_pct >= PREEMPTIVE_HOLD_PULLBACK_THRESHOLD_PCT:
            strong_absorption = (
                median6 > PREEMPTIVE_HOLD_MEDIAN6_THRESHOLD
                and len(bbos) >= 3
                and min(bbos) > PREEMPTIVE_HOLD_MIN_BBO
            )
            if strong_absorption:
                decision = "PREEMPTIVE_RULE_2_HOLD_ABSORPTION"
                if decision != self._preemptive_last_decision:
                    logging.info(
                        "Pre-emptive Rule 2 HOLD | drop_pct=%.3f%% >= %.3f%% | median6=%.5f > %.2f | min_bbo=%.5f > %.2f | max_close=%.5f | current_close=%.5f | aggs=%s | bbos=%s",
                        drop_pct,
                        PREEMPTIVE_HOLD_PULLBACK_THRESHOLD_PCT,
                        median6,
                        PREEMPTIVE_HOLD_MEDIAN6_THRESHOLD,
                        min(bbos),
                        PREEMPTIVE_HOLD_MIN_BBO,
                        max_close,
                        current_close,
                        ",".join(f"{x:.5f}" for x in aggs),
                        ",".join(f"{x:.5f}" for x in bbos),
                    )
                    self._preemptive_last_decision = decision
                return True

        # Rule 3: Pre-emptive Wear-and-Tear Exit (Anti-Lag Filter), narrowed.
        # Earlier samples showed the old 0.24/0.55 rule was often late and broad.
        # This version exits only when the cushion is genuinely weak, but it can do
        # so slightly before the 0.24% threshold.
        if drop_pct >= PREEMPTIVE_WEAR_TEAR_PULLBACK_THRESHOLD_PCT and median6 < PREEMPTIVE_WEAR_TEAR_MEDIAN6_THRESHOLD:
            reasons = [
                "PRE-EMPTIVE RULE 3 WEAR-AND-TEAR EXIT",
                f"drop_pct={drop_pct:.3f}% >= {PREEMPTIVE_WEAR_TEAR_PULLBACK_THRESHOLD_PCT:.3f}%",
                f"median6={median6:.5f} < {PREEMPTIVE_WEAR_TEAR_MEDIAN6_THRESHOLD:.2f}",
                f"max_close={max_close:.5f}",
                f"current_close={current_close:.5f}",
                f"aggs={','.join(f'{x:.5f}' for x in aggs)}",
                f"bbos={','.join(f'{x:.5f}' for x in bbos)}",
            ]
            self._exit_long(snap, reasons)
            return True

        return False

    def _run_red_grind_decay_exit(self, snap: CandleSnapshot) -> bool:
        """Rule 4: support-protected Green+Red+Red slow-grind decay exit.

        v9e_Buy_Frac34 keeps the faster 2-HA-Red trigger, but restores the
        old 3-candle support/absorption protection by evaluating the 3-candle
        frame formed by the candle immediately before the red streak plus the
        two HA Red candles: Green + Red + Red.

        The exit only fires on the second red candle when:
          - the trade is back to breakeven/loss, or it has given back too much
            of its peak profit with weak cushion; and
          - the Green+Red+Red frame does NOT match HOLD/ABSORPTION; and
          - the old 3-red weakness confirmation is present:
                red_count >= 4 out of 6, OR median6 <= 0.49999.
        """
        if not self.in_trade_mode or self.entry_price is None or self.max_close is None:
            return False
        if self._trade_red_streak != DECAY_EXIT_RED_STREAK:
            return False
        if snap.close_price > self.max_close:
            return False

        # If the trade slipped under max_close through red candles without the Green-only
        # watch activator firing first, force Watching here so Rule 4 is state-consistent.
        if not self._preemptive_watching:
            self._preemptive_watching = True
            logging.info(
                "Pre-emptive Watching activated by red-grind decay | current_close=%.5f <= max_close=%.5f at %s",
                snap.close_price,
                self.max_close,
                snap.utc,
            )

        support_snaps = list(self._preemptive_window)[-3:]
        if len(support_snaps) < 3:
            return False

        colors = "".join("G" if s.ha_color == "Green" else "R" for s in support_snaps)
        if colors != "GRR":
            decision = "RULE_4_RED_GRIND_HOLD_NO_GRR_FRAME"
            if decision != self._trade_last_decision:
                logging.info(
                    "Rule 4 RED-GRIND HOLD | expected G-R-R support frame but got %s | red_streak=%d | max_close=%.5f | current_close=%.5f",
                    colors,
                    self._trade_red_streak,
                    self.max_close,
                    snap.close_price,
                )
                self._trade_last_decision = decision
            return False

        median6 = self._median6_from_snaps(support_snaps)
        aggs = [s.agg_ratio for s in support_snaps]
        bbos = [s.avg_bbo_imba for s in support_snaps]
        frame = aggs + bbos

        red_count = sum(1 for v in frame if v <= DECAY_SUPPORT_EXIT_THRESHOLD)
        support_exit_now = median6 is not None and (
            red_count >= 4 or median6 <= DECAY_SUPPORT_EXIT_THRESHOLD
        )

        # Old HOLD/ABSORPTION protection, now applied to the G-R-R frame:
        # any BBO >= 0.80, every agg in [0.50, 0.75], and every BBO either
        # >= 0.80 or in [0.50, 0.75].
        bbo_green_any = any(b >= DECAY_SUPPORT_BBO_GREEN for b in bbos)
        hold_absorption = False
        if bbo_green_any:
            ok = True
            for s in support_snaps:
                if not (DECAY_SUPPORT_NEUTRAL_LOW <= s.agg_ratio <= DECAY_SUPPORT_NEUTRAL_HIGH):
                    ok = False
                    break
                if s.avg_bbo_imba < DECAY_SUPPORT_BBO_GREEN and not (
                    DECAY_SUPPORT_NEUTRAL_LOW <= s.avg_bbo_imba <= DECAY_SUPPORT_NEUTRAL_HIGH
                ):
                    ok = False
                    break
            hold_absorption = ok

        peak_profit = self.max_close - self.entry_price
        current_pnl = snap.close_price - self.entry_price
        giveback = self.max_close - snap.close_price
        giveback_ratio = (giveback / peak_profit) if peak_profit > 0 else 1.0

        back_to_breakeven_or_worse = current_pnl <= 0
        gave_back_too_much = peak_profit > 0 and giveback_ratio >= DECAY_GIVEBACK_RATIO_THRESHOLD
        weak_cushion = median6 is not None and median6 < DECAY_MEDIAN6_WEAK_THRESHOLD
        base_decay_condition = back_to_breakeven_or_worse or (weak_cushion and gave_back_too_much)

        t3 = self._rollup_ticks_last_n(3)
        tc3 = self._rollup_trade_counts_last_n(3)

        if hold_absorption:
            decision = "RULE_4_GRR_HOLD_ABSORPTION"
            if decision != self._trade_last_decision:
                logging.info(
                    "Rule 4 G-R-R HOLD/ABSORPTION | bbo>=%.2f seen; others in [%.2f, %.2f] | red_count=%d/6 | median6=%.5f | current_pnl=%.5f | peak_profit=%.5f | giveback_ratio=%.3f | aggs=%s | bbos=%s | tick3 plus=%d zplus=%d zminus=%d minus=%d total=%d | imb=%d ratio=%.5f | AggTrades3 buy/total=%d/%d (buy_frac=%.3f) | sell=%d",
                    DECAY_SUPPORT_BBO_GREEN,
                    DECAY_SUPPORT_NEUTRAL_LOW,
                    DECAY_SUPPORT_NEUTRAL_HIGH,
                    red_count,
                    median6 if median6 is not None else float("nan"),
                    current_pnl,
                    peak_profit,
                    giveback_ratio,
                    ",".join(f"{x:.5f}" for x in aggs),
                    ",".join(f"{x:.5f}" for x in bbos),
                    int(t3["plus"]),
                    int(t3["zero_plus"]),
                    int(t3["zero_minus"]),
                    int(t3["minus"]),
                    int(t3["total"]),
                    int(t3["tick_imbalance"]),
                    float(t3["tick_imbalance_ratio"]),
                    int(tc3["buy"]),
                    int(tc3["total"]),
                    float(tc3["buy_frac"]),
                    int(tc3["sell"]),
                )
                self._trade_last_decision = decision
            return False

        if not support_exit_now:
            decision = "RULE_4_GRR_HOLD_SUPPORT"
            if decision != self._trade_last_decision:
                logging.info(
                    "Rule 4 G-R-R HOLD | red_count=%d/6 | median6=%s | support_exit_now=False | current_pnl=%.5f | peak_profit=%.5f | giveback_ratio=%.3f | weak_cushion=%s | base_decay_condition=%s | aggs=%s | bbos=%s | tick3 plus=%d zplus=%d zminus=%d minus=%d total=%d | imb=%d ratio=%.5f | AggTrades3 buy/total=%d/%d (buy_frac=%.3f) | sell=%d",
                    red_count,
                    "None" if median6 is None else f"{median6:.5f}",
                    current_pnl,
                    peak_profit,
                    giveback_ratio,
                    weak_cushion,
                    base_decay_condition,
                    ",".join(f"{x:.5f}" for x in aggs),
                    ",".join(f"{x:.5f}" for x in bbos),
                    int(t3["plus"]),
                    int(t3["zero_plus"]),
                    int(t3["zero_minus"]),
                    int(t3["minus"]),
                    int(t3["total"]),
                    int(t3["tick_imbalance"]),
                    float(t3["tick_imbalance_ratio"]),
                    int(tc3["buy"]),
                    int(tc3["total"]),
                    float(tc3["buy_frac"]),
                    int(tc3["sell"]),
                )
                self._trade_last_decision = decision
            return False

        if not base_decay_condition:
            decision = "RULE_4_GRR_HOLD_NO_DECAY_CONDITION"
            if decision != self._trade_last_decision:
                logging.info(
                    "Rule 4 G-R-R HOLD | support weakness confirmed but decay condition not met | red_count=%d/6 | median6=%s | current_pnl=%.5f | peak_profit=%.5f | giveback_ratio=%.3f | weak_cushion=%s | aggs=%s | bbos=%s",
                    red_count,
                    "None" if median6 is None else f"{median6:.5f}",
                    current_pnl,
                    peak_profit,
                    giveback_ratio,
                    weak_cushion,
                    ",".join(f"{x:.5f}" for x in aggs),
                    ",".join(f"{x:.5f}" for x in bbos),
                )
                self._trade_last_decision = decision
            return False

        reasons = [
            "PRE-EMPTIVE RULE 4 G-R-R RED-GRIND DECAY EXIT",
            f"frame={colors}",
            f"red_streak={self._trade_red_streak} >= {DECAY_EXIT_RED_STREAK}",
            f"current_pnl={current_pnl:.5f}",
            f"peak_profit={peak_profit:.5f}",
            f"giveback={giveback:.5f}",
            f"giveback_ratio={giveback_ratio:.3f}",
            f"max_close={self.max_close:.5f}",
            f"current_close={snap.close_price:.5f}",
            f"red_count={red_count}/6",
            f"median6={'None' if median6 is None else f'{median6:.5f}'}",
            f"support_thr<={DECAY_SUPPORT_EXIT_THRESHOLD:.5f}",
            f"weak_cushion={weak_cushion} (thr<{DECAY_MEDIAN6_WEAK_THRESHOLD:.2f})",
            f"hold_absorption=False",
            f"support_exit_now=True",
            f"aggs={','.join(f'{x:.5f}' for x in aggs)}",
            f"bbos={','.join(f'{x:.5f}' for x in bbos)}",
        ]
        if back_to_breakeven_or_worse:
            reasons.append("back_to_breakeven_or_worse=True")
        if gave_back_too_much:
            reasons.append(f"gave_back_too_much=True (ratio>={DECAY_GIVEBACK_RATIO_THRESHOLD:.2f})")
        reasons.append(
            f"tick3 plus={t3['plus']} zplus={t3['zero_plus']} zminus={t3['zero_minus']} minus={t3['minus']} "
            f"total={t3['total']} imb={t3['tick_imbalance']} ratio={float(t3['tick_imbalance_ratio']):.5f}"
        )
        reasons.append(
            f"trades3 sell/total={tc3['sell']}/{tc3['total']} (sell_frac={tc3['sell_frac']:.3f})"
        )
        self._exit_long(snap, reasons)
        return True

    def _exit_long(self, snap: CandleSnapshot, reasons: List[str]) -> None:
        """Log and apply a LONG exit. Used by all exit paths."""
        if self.entry_price is None:
            return
        exit_price = snap.close_price
        pnl = exit_price - self.entry_price

        logging.warning(
            f"*** EXIT LONG at {snap.utc} | exit_price={exit_price:.5f} | entry_price={self.entry_price:.5f} "
            f"| pnl={pnl:.5f} | reasons: {'; '.join(reasons)} ***"
        )

        self._reset_trade_state_after_exit()

    def _run_exit_logic(self, snap: CandleSnapshot) -> None:
        """Exit logic for an active LONG.

        Priority order in v9e_Buy_Frac34:
          1) Rule 1 Panic Fuse.
          2) Tightened Rule 2 Absorption Hold.
          3) Narrowed Rule 3 Wear-and-Tear Exit.
          4) Rule 4 support-protected G-R-R Red-Grind Decay Exit after 2 HA Red candles.

        The old 3-HA Red fallback decision matrix is not restored as a 3-red exit; its support/absorption protection is reused inside Rule 4 on the G-R-R frame.
        """
        if not self.in_trade_mode or self.entry_price is None:
            return

        # Maintain a rolling 3-candle frame for pre-emptive median6 and Rule 4.
        self._preemptive_window.append(snap)

        # Track max_close during HA Green trend and activate Watching on a non-higher close.
        self._update_preemptive_peak_watch(snap)

        # Rules 1/2/3 have priority. A True return means EXIT or HOLD already decided.
        if self._run_preemptive_exit_hold_logic(snap):
            return

        # Rule 4 uses the second HA Red candle as slow-grind confirmation and evaluates
        # the Green+Red+Red support frame, not the old 3-red fallback delay.
        if snap.ha_color == "Red":
            self._trade_red_streak += 1
            self._trade_red_window.append(snap)
        else:
            self._trade_red_streak = 0
            self._trade_red_window.clear()
            self._trade_last_decision = None
            return

        self._run_red_grind_decay_exit(snap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="POPCATUSDT", help="Bybit linear symbol, e.g. BTCUSDT")
    ap.add_argument("--testnet", action="store_true", help="Use Bybit testnet")
    ap.add_argument("--loglevel", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    tracker = OrderFlowTracker(symbol=args.symbol, testnet=args.testnet)
    tracker.start()


if __name__ == "__main__":
    main()
