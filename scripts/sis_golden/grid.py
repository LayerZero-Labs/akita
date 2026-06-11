"""Representative golden grid for offline SIS width regression."""

from __future__ import annotations

# Module ranks exercised in golden refresh and replay.
GOLDEN_RANKS = [1, 5, 20]

# (family, ring dimension d, collision_l2_sq bucket). Includes a known degenerate
# knee at (q32, d=32, collision=16384).
GOLDEN_WORK_ITEMS: list[tuple[str, int, int]] = [
    # q32
    ("q32", 32, 16_384),
    ("q32", 32, 65_536),
    ("q32", 32, 262_144),
    ("q32", 64, 16_384),
    ("q32", 64, 65_536),
    ("q32", 64, 1_048_576),
    ("q32", 128, 16_384),
    ("q32", 128, 65_536),
    ("q32", 256, 16_384),
    ("q32", 256, 65_536),
    # q64
    ("q64", 32, 16_384),
    ("q64", 32, 65_536),
    ("q64", 64, 16_384),
    ("q64", 64, 65_536),
    ("q64", 128, 16_384),
    ("q64", 256, 16_384),
    # q128
    ("q128", 32, 16_384),
    ("q128", 32, 65_536),
    ("q128", 64, 16_384),
    ("q128", 128, 16_384),
    ("q128", 256, 16_384),
]
