"""Model architectures.

two_stream_film  -- single-frame RGB + spectral indices, FiLM-style gating.
                    Predicts [gsw, Tleaf]. This is the thesis architecture.

temporal_bigru   -- 5-frame clips, two-stream, BiGRU over time. Predicts Tleaf
                    only. This is what the RELEASED CHECKPOINTS implement.

The two differ in backbone (V2-B3 vs v1-B3), fusion (FiLM vs concat) and
outputs. See docs/architecture.md before assuming they are interchangeable.
"""

from . import temporal_bigru, two_stream_film  # noqa: F401

__all__ = ["two_stream_film", "temporal_bigru"]
