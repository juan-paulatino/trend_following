# trend_following
opus 5

## microstructure

Order-flow features for detecting **absorption** (aggressive sellers pressing
into a bid that holds) and distinguishing it from **exhaustion** (sellers
simply running out) and **vacuum drift** (price floating up on a thin ask).

Built around one organising idea: the orderbook and the tape are different
classes of evidence.

|                  | Orderbook depth | Executed ticks |
| ---------------- | --------------- | -------------- |
| Timing           | *ex-ante*       | *ex-post*      |
| Revocable        | yes, cancel is free | no, a fill is settled |
| Spoofable        | yes             | **no**         |
| Role             | anticipates     | **confirms**   |

A resting bid is a *promise*. A `ZeroMinusTick` — a trade that executes without
moving price — is that promise being *materialised*. The link is deductive, not
statistical: for a sell aggressor to leave price unchanged, the bid must have
held more size than the order that hit it, or refreshed instantly. So the tape
verifies what the book merely claims.

### Layout

```
microstructure/
  features.py   per-candle measurement
  phases.py     state machine across candles
tests/
  test_features.py   49 assertions
```

Run: `python3 tests/test_features.py`

### Data model

Three layers, split by whether the information is recoverable later.

**Primitives** — stored. `buy_vol`, `sell_vol`, trade counts, OHLC, `pv_sum`,
book sums.

**Path data** — must be captured live; impossible to reconstruct once the
candle closes. Cumulative-delta extremes, volume-weighted time centroids,
4-state tick counts, per-quartile buckets.

**Derived** — computed properties, never stored as independent model features.

### Traps this encodes

**`delta / volume` is not a feature.** It equals `2 * agg - 1` exactly. The
trade-flow content of a candle has only two degrees of freedom — SCALE
(`volume`) and BALANCE (`agg`). Storing `delta`, `volume`, `agg` and
`delta/volume` as four columns gives perfectly collinear inputs.

**`agg = 0.5` must not double as "no data".** 0.5 is a legal observable value:
`800 buy / 800 sell` is one of the most informative states in the dataset and
is indistinguishable from an empty candle under a sentinel default. `agg`
returns `None` when there is no volume, is shrunk toward 0.5 by a finite prior,
and always ships with `agg_n`. The information is two-dimensional; no scalar
carries both location and confidence.

**Min/max are contaminated by sample count.** Extreme order statistics drift
outward as trade count rises, so `agg_max - agg_min` tracks tick count rather
than market structure. Replaced with fixed 15s quartile buckets, and sequencing
uses volume-weighted time centroids (`flow_lag`) rather than path argmin/argmax.

**A ratio conceals which leg moved.** `agg` rising means buyers arrived
(numerator) *or* sellers left (denominator) — different events, same reading.
`sell_decay` and `buy_growth` separate them. `bbo` rising is the same problem:
bid grew (real wall) or ask evaporated (vacuum). Absolute sizes separate those.

**An untested wall is not evidence.** A 900k displayed bid that only ever
absorbed 50-unit sells proves nothing — nobody challenged it. `wall_tested`
reports the largest absorbed sell as a fraction of displayed depth.

**Raw percentages are not interpretable.** A −0.22% decline against heavy
selling looks like absorption, but on an instrument whose typical 4-minute move
is 0.05% it is a *large* move. `Episode.price_effect_z` normalises against the
instrument's own realised volatility.

### Selected features

| Feature | Meaning |
| --- | --- |
| `absorption_tick_ratio` | share of down-pressure ticks that left price pinned — replenishment at tick resolution, no book feed needed |
| `sell_efficiency` | fraction of aggressive sells that achieved a downtick; low = pressing with nothing to show |
| `inferred_bid_depth` | `(lower, upper)` bound on real bid depth from fills alone; the lower bound is hard and unspoofable |
| `hidden_liquidity` | price pinned while displayed bid is *small* — iceberg. Strongest state available, since there is nothing displayed to spoof with |
| `absorption_per_tick` | net supply swallowed per tick surrendered |
| `delta_recovery` | net buying after the trough, in volume units |
| `flow_lag` | `t_buy_centroid - t_sell_centroid`; positive = capitulation preceded arrival, so demand is unspent |
| `impact_asymmetry` | `impact_up / impact_down`; book asymmetry proxy from trades alone |

### Phases

`ABSORPTION -> EXHAUSTION -> ARRIVAL -> MARKUP`, plus `VACUUM`,
`DISTRIBUTION`, `INVALIDATED`, `STALE`.

`ABSORPTION` has two admissible entry paths: a *displayed* wall (`bbo`
bid-heavy) or an *executed* wall (`hidden_liquidity` — tick-confirmed with
nothing visible). The second is the stronger of the two and a `bbo`-only gate
rejects it outright.

Invalidation is explicit. A bid that is large but lets every sell tick price
down is not support, whatever the ratio says.

A zero-trade candle is a **feature, not a hole** — it is maximal seller
exhaustion, and `bbo` remains fully valid because the book is sampled
independently of whether anything trades.

## Status

**Calibrated against synthetic data only.** Every threshold in
`phases.Thresholds`, the `agg_k` prior strength, and the volatility baseline
need real history before any of this is tradeable. The percentile gates return
`None` below 20 observations rather than guessing.

Known limitation: `bid_replenishment` is derived from sampled book snapshots,
so a wall hit and refilled between two samples looks untouched. Proper
measurement needs the book *delta* stream. `absorption_tick_ratio` does not
have this blind spot and should be preferred.

Not yet wired to a live feed. `build_candle` consumes raw trades (with Bybit's
`tickDirection`) plus book samples; a logger emitting only collapsed per-candle
scalars cannot drive it.
