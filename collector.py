#!/usr/bin/env python3
"""
Live Bybit collector. Records RAW websocket messages and classifies candles.

    pip install pybit
    python3 collector.py --symbol XYZUSDT --tick-size 0.00001 --category spot

Why it records RAW messages rather than computed features: the feature set is
still changing. Saved features are frozen at the moment they were written and
cannot be recomputed when a definition improves. Saved raw messages can be
replayed through any future version of the code. Storage is cheap; a lost tape
is not.

    python3 collector.py --replay tape/XYZUSDT-20260807.jsonl

Replay needs no network and no pybit.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from microstructure.bybit import BybitAssembler
from microstructure.phases import Phase, PhaseMachine

INTERESTING = {
    Phase.ABSORPTION,
    Phase.EXHAUSTION,
    Phase.ARRIVAL,
    Phase.VACUUM,
    Phase.DISTRIBUTION,
    Phase.INVALIDATED,
}


def fmt_minute(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%H:%M")


class Runner:
    def __init__(self, tick_size: float, tape: Path | None, verbose: bool):
        self.assembler = BybitAssembler(tick_size=tick_size)
        self.machine = PhaseMachine()
        self.tape = tape.open("a") if tape else None
        self.verbose = verbose
        self.lock = threading.Lock()  # pybit dispatches on socket threads
        self.n_candles = 0
        self.n_gaps = 0
        self._tick_warned = False

    def handle(self, msg: dict, record: bool = True) -> None:
        with self.lock:
            if self.tape and record:
                self.tape.write(json.dumps(msg, separators=(",", ":")) + "\n")
            for em in self.assembler.on_message(msg):
                self._on_candle(em)
            self._check_tick_size()

    def _check_tick_size(self) -> None:
        """Warn once if --tick-size disagrees with the observed tape."""
        if self._tick_warned or self.n_candles < 5:
            return
        warning = self.assembler.tick_size_warning()
        if warning:
            print(f"\n*** TICK SIZE WARNING: {warning}\n"
                  f"    Restart with --tick-size "
                  f"{self.assembler.observed_tick_size():g} for correct "
                  f"impact metrics. The recorded tape is unaffected and can be "
                  f"replayed with the right value.\n")
        self._tick_warned = True

    def _on_candle(self, em) -> None:
        self.n_candles += 1
        c = em.candle

        if not em.usable:
            self.n_gaps += 1
            print(f"[{fmt_minute(em.minute_start_ms)}] GAP -- feed unhealthy "
                  f"({c.bbo_samples} book samples). Candle NOT fed to the model.")
            return

        result = self.machine.update(c)

        if result.phase in INTERESTING or self.verbose:
            agg = f"{c.agg:.3f}" if c.agg is not None else " None"
            bbo = f"{c.bbo_avg:.3f}" if c.bbo_avg is not None else " None"
            atr = (f"{c.absorption_tick_ratio:.0%}"
                   if c.absorption_tick_ratio is not None else "  n/a")
            flag = "" if result.confident else "  (low confidence)"
            # Absolute depth is logged alongside the ratio because the ratio is
            # blind to which side moved, and because depth_tot (bid+ask) is the
            # only feature that has reached significance on POPCAT. Without
            # these two numbers a console log cannot test that hypothesis --
            # it needs the raw tape, which is a much larger file to move around.
            bsz = f"{c.bid_sz_avg:,.0f}" if c.bid_sz_avg is not None else "None"
            asz = f"{c.ask_sz_avg:,.0f}" if c.ask_sz_avg is not None else "None"
            print(f"[{fmt_minute(em.minute_start_ms)}] {result.phase.value:<12} "
                  f"close={c.close:<12.8g} agg={agg} (n={c.agg_n:>10,.0f}) "
                  f"bbo={bbo} pinned={atr} bid_sz={bsz:>10} ask_sz={asz:>10}{flag}")
            for r in result.reasons:
                print(f"      - {r}")

        if em.dropped_block_trades or em.dropped_duplicate_books:
            print(f"      [filtered {em.dropped_block_trades} block trades, "
                  f"{em.dropped_duplicate_books} duplicate book snapshots]")

    def finish(self) -> None:
        with self.lock:
            for em in self.assembler.flush():
                self._on_candle(em)
            if self.tape:
                self.tape.close()
        print(f"\n{self.n_candles} candles, {self.n_gaps} unusable "
              f"({self.n_gaps / max(self.n_candles, 1):.1%})")
        if self.n_candles < 240:
            print("NOTE: percentile gates need ~240 candles (4h) before they "
                  "report anything. Keep collecting.")


def run_live(args) -> None:
    try:
        from pybit.unified_trading import WebSocket
    except ImportError:
        sys.exit("pybit is required for live collection:  pip install pybit")

    tape = None
    if not args.no_tape:
        Path("tape").mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        tape = Path("tape") / f"{args.symbol}-{stamp}.jsonl"
        print(f"recording raw messages -> {tape}")

    runner = Runner(args.tick_size, tape, args.verbose)

    ws = WebSocket(testnet=args.testnet, channel_type=args.category)
    ws.trade_stream(symbol=args.symbol, callback=runner.handle)
    # Level 1 for linear/inverse/spot is snapshot-only; no delta bookkeeping.
    ws.orderbook_stream(depth=1, symbol=args.symbol, callback=runner.handle)

    print(f"subscribed to {args.symbol} ({args.category}). Ctrl-C to stop.\n")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()

    print("\nstopping...")
    runner.finish()


def run_replay(args) -> None:
    path = Path(args.replay)
    if not path.exists():
        sys.exit(f"no such tape: {path}")

    runner = Runner(args.tick_size, None, args.verbose)
    bad = 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                runner.handle(json.loads(line), record=False)
            except json.JSONDecodeError:
                bad += 1
    runner.finish()
    if bad:
        print(f"{bad} unparseable lines skipped")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--tick-size", type=float, required=True,
                   help="price increment, e.g. 0.00001 for a sub-penny pair")
    p.add_argument("--category", default="spot",
                   choices=["spot", "linear", "inverse"],
                   help="NOTE: tickDirection (L) is only sent for perps and "
                        "futures. On spot it is derived locally instead.")
    p.add_argument("--testnet", action="store_true")
    p.add_argument("--replay", metavar="TAPE.jsonl",
                   help="replay a recorded tape offline; no network or pybit needed")
    p.add_argument("--no-tape", action="store_true", help="do not record")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="print every candle, not just classified ones")
    args = p.parse_args()

    if args.replay:
        run_replay(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
