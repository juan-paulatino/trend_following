"""
Order-flow microstructure features for absorption / exhaustion detection.

The package separates two classes of evidence:

  features.py  -- per-candle measurement. Stores primitives and path data,
                  derives everything else. Distinguishes what the ORDERBOOK
                  promises (revocable, spoofable) from what the TAPE
                  materialises (settled, unspoofable).

  phases.py    -- a state machine over those features, tracking absorption ->
                  exhaustion -> arrival across candles, with explicit
                  invalidation conditions.
"""

from .features import (
    BookSample,
    Bucket,
    Candle,
    TickDirection,
    Trade,
    build_candle,
)
from .phases import Classified, Episode, Phase, PhaseMachine, Thresholds

__all__ = [
    "BookSample",
    "Bucket",
    "Candle",
    "TickDirection",
    "Trade",
    "build_candle",
    "Classified",
    "Episode",
    "Phase",
    "PhaseMachine",
    "Thresholds",
]
