"""Representative golden grid for offline coefficient-L∞ SIS width regression."""

from __future__ import annotations

# Module ranks exercised in golden refresh and replay.
GOLDEN_RANKS = [1, 5, 20]

# (family, ring dimension d, coefficient-L∞ collision bucket B).
GOLDEN_WORK_ITEMS: list[tuple[str, int, int]] = [
    # q32
    ("q32", 32, 16_383),
    ("q32", 32, 65_535),
    ("q32", 32, 262_143),
    ("q32", 64, 16_383),
    ("q32", 64, 65_535),
    ("q32", 64, 1_048_575),
    ("q32", 128, 16_383),
    ("q32", 128, 65_535),
    ("q32", 256, 16_383),
    ("q32", 256, 65_535),
    # q64
    ("q64", 32, 16_383),
    ("q64", 32, 65_535),
    ("q64", 64, 16_383),
    ("q64", 64, 65_535),
    ("q64", 128, 16_383),
    ("q64", 256, 16_383),
    # q128
    ("q128", 32, 16_383),
    ("q128", 32, 65_535),
    ("q128", 64, 16_383),
    ("q128", 128, 16_383),
    ("q128", 256, 16_383),
]
