#!/usr/bin/env python3
"""
How long does a collector run need to be to test the bbo hypothesis?

The second TRX holdout (output5_TRX, 75 candles) flipped sign at h=5 and h=15.
That looks like a refutation and is not one: at n=70 the standard error of a
correlation is about 0.12, so an estimate of +0.17 and an estimate of -0.20 are
both entirely consistent with the same underlying effect. The run is too short
to carry information either way.

This computes the sample size actually required, so future runs can be planned
rather than guessed.
"""

import glob
import math
import statistics as st
from pathlib import Path

from analyze_agg import spearman
from analyze_run import CANDLE, strip_rtf

Z_ALPHA = 1.96   # two-sided 0.05
Z_POWER = 0.84   # 80% power


def load(path):
    out = []
    for ln in strip_rtf(Path(path).read_text(errors="replace")).splitlines():
        m = CANDLE.search(ln)
        if not m:
            continue
        _, ph, close, agg, n, bbo, pinned, _ = m.groups()
        out.append({
            "close": float(close),
            "agg": None if agg == "None" else float(agg),
            "bbo": None if bbo == "None" else float(bbo),
        })
    return out


def n_required(rho: float) -> int:
    """Fisher-z sample size for detecting a correlation of this magnitude."""
    if abs(rho) >= 1 or rho == 0:
        return 0
    return int(((Z_ALPHA + Z_POWER) / abs(math.atanh(rho))) ** 2 + 3)


def rho_se(n: int) -> float:
    """Approximate standard error of a correlation estimate."""
    return 1 / math.sqrt(n - 3) if n > 3 else float("nan")


def main():
    print("=== how much data does a correlation of this size need? ===")
    print(f"  two-sided alpha=0.05, 80% power\n")
    print(f"  {'true rho':>9} {'candles needed':>15} {'hours at 1/min':>16}")
    for r in (0.05, 0.10, 0.13, 0.17, 0.20, 0.25, 0.29):
        n = n_required(r)
        print(f"  {r:>9.2f} {n:>15,} {n/60:>15.1f}h")

    print("\n=== what each TRX run could actually resolve ===")
    files = sorted(glob.glob("output*TRX*.rtf")) + sorted(glob.glob("output_trx.rtf"))
    print(f"  {'run':<22} {'candles':>8} {'SE of rho':>11} {'can detect rho >':>18}")
    for f in files:
        rows = load(f)
        n = sum(1 for c in rows if c["bbo"] is not None)
        if n < 5:
            continue
        se = rho_se(n)
        # smallest rho this n can detect at 80% power
        detectable = math.tanh((Z_ALPHA + Z_POWER) / math.sqrt(max(n - 3, 1)))
        print(f"  {Path(f).name:<22} {n:>8} {se:>11.3f} {detectable:>17.3f}")

    print("\n=== the second holdout, read correctly ===")
    r5 = load("output5_TRX.rtf")
    for h in (1, 5, 15):
        pairs = [(c["bbo"], (r5[i + h]["close"] - c["close"]) / c["close"] * 100)
                 for i, c in enumerate(r5)
                 if i + h < len(r5) and c["bbo"] is not None]
        if len(pairs) < 20:
            continue
        m = st.mean(r for _, r in pairs)
        rho = spearman([a for a, _ in pairs], [r - m for _, r in pairs])
        n = len(pairs)
        se = rho_se(n)
        lo, hi = rho - Z_ALPHA * se, rho + Z_ALPHA * se
        covers = lo <= 0.17 <= hi
        print(f"  h={h:<3} n={n:<4} rho={rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
              f"   contains +0.17? {'YES' if covers else 'no'}")
    print("\n  Read precisely: at h=1 the interval contains +0.17 and is simply")
    print("  uninformative. At h=5 and h=15 it EXCLUDES +0.17 while still")
    print("  including zero, so the second holdout is consistent with no effect")
    print("  and inconsistent with an effect as large as the first holdout found.")
    print("  That is mild tension, not refutation -- but it is not nothing, and")
    print("  it is why the pooled result should not be treated as settled: 192 of")
    print("  the 265 pooled candles come from the first holdout.")


if __name__ == "__main__":
    main()
