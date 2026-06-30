"""Representative golden grid for SIS infinity-norm estimator parity."""

from __future__ import annotations

TARGET_BITS = 138.0

INFINITY_RANKS = [1, 5, 20]

# Keep this subset broad enough to exercise tiny-probability amplification and
# Akita-style coefficient envelopes without making every refresh a full table
# generation job.
INFINITY_COEFF_LINF_BOUNDS = [2, 15, 255, 4095]

INFINITY_WIDTH_FACTORS = [2, 8]

INFINITY_FAMILIES = ["q32", "q64", "q128"]
INFINITY_RING_DIMS = [32, 64, 128, 256]


def infinity_work_items() -> list[dict[str, int | str]]:
    """Return fixed-width infinity cells for all families, dimensions, and ranks."""
    rows: list[dict[str, int | str]] = []
    for family in INFINITY_FAMILIES:
        for d in INFINITY_RING_DIMS:
            for rank in INFINITY_RANKS:
                for width_factor in INFINITY_WIDTH_FACTORS:
                    width = rank * width_factor
                    for coeff_linf_bound in INFINITY_COEFF_LINF_BOUNDS:
                        rows.append(
                            {
                                "family": family,
                                "d": d,
                                "rank": rank,
                                "width": width,
                                "coeff_linf_bound": coeff_linf_bound,
                            }
                        )
    return rows
