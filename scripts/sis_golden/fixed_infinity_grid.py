"""Small fixed-beta, fixed-zeta infinity golden grid for Slice 3 parity."""

from __future__ import annotations

TARGET_BITS = 138.0

# Explicit beta/zeta pairs kept small so refresh stays fast and Rust parity
# tests stay focused on the fixed-cost path rather than optimizer behavior.
FIXED_INFINITY_CELLS: list[dict[str, int | str]] = [
    {
        "family": "q32",
        "d": 32,
        "rank": 1,
        "width": 2,
        "coeff_linf_bound": 2,
        "beta": 63,
        "zeta": 0,
    },
    {
        "family": "q32",
        "d": 32,
        "rank": 1,
        "width": 2,
        "coeff_linf_bound": 15,
        "beta": 63,
        "zeta": 0,
    },
    {
        "family": "q32",
        "d": 32,
        "rank": 1,
        "width": 2,
        "coeff_linf_bound": 15,
        "beta": 40,
        "zeta": 0,
    },
    {
        "family": "q32",
        "d": 32,
        "rank": 1,
        "width": 2,
        "coeff_linf_bound": 15,
        "beta": 40,
        "zeta": 5,
    },
    {
        "family": "q32",
        "d": 32,
        "rank": 5,
        "width": 10,
        "coeff_linf_bound": 15,
        "beta": 50,
        "zeta": 0,
    },
    {
        "family": "q32",
        "d": 64,
        "rank": 1,
        "width": 2,
        "coeff_linf_bound": 15,
        "beta": 63,
        "zeta": 0,
    },
    {
        "family": "q64",
        "d": 32,
        "rank": 1,
        "width": 2,
        "coeff_linf_bound": 15,
        "beta": 63,
        "zeta": 0,
    },
    {
        "family": "q128",
        "d": 32,
        "rank": 1,
        "width": 2,
        "coeff_linf_bound": 15,
        "beta": 63,
        "zeta": 0,
    },
]


def fixed_infinity_work_items() -> list[dict[str, int | str]]:
    return list(FIXED_INFINITY_CELLS)
