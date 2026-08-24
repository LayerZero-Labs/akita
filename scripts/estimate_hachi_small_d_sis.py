#!/usr/bin/env python3
"""
Estimate SIS security for Hachi at ring dimensions D=32 and D=16 (one-hot).

This script mirrors estimate_hachi_d64_k256_onehot_sis.py but targets the
smaller ring dimensions used in the proof-size reduction study. It sweeps
(n_a, n_b, n_d) combinations to find the minimum ranks that achieve 128-bit
security across all three commitment roles (A, B, D).

Field: q = 2^128 - 275  (k=2 ring split for all D, resolves LS18 invertibility)

Two configurations are studied:
  D=32:  n_a=2, challenge Uniform(weight=32, coeffs in [-8..8] excl. 0), l1_mass=256
  D=16:  n_a=4, challenge Uniform(weight=16, coeffs in [-128..128] excl. 0), l1_mass=2048

Both use onehot mode with delta_commit=1 and LOG_BASIS=5 (the planner's
chosen lb at the recursive levels where small-D would be used).

The SIS estimation follows the same methodology as the D=64 script:
  - Flatten Module-SIS to SIS: n = rank * D, m = width_ring * D
  - L2 bound: B_l2 = sqrt(m) * collision_inf
  - Estimator: SIS.lattice(...) with BDGL16 + lgsa

Run:
    sage -python scripts/estimate_hachi_small_d_sis.py

Or with explicit estimator path:
    LATTICE_ESTIMATOR_PATH="../lattice-estimator" \
      sage -python scripts/estimate_hachi_small_d_sis.py
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path


Q = (1 << 128) - 275
Q_LABEL = "2^128 - 275"

MAX_ABS_CHALLENGE_COEFF_D32 = 8
MAX_ABS_CHALLENGE_COEFF_D16 = 128


def compute_num_digits(log_bound: int, log_basis: int) -> int:
    if log_basis <= 0 or log_basis >= 128:
        raise ValueError("invalid log_basis")
    if log_bound == 0:
        return 1
    levels = (log_bound + log_basis - 1) // log_basis
    total_bits = levels * log_basis
    if total_bits <= log_bound:
        b = 1 << log_basis
        half_b_minus_1 = b // 2 - 1
        b_pow = b ** levels
        max_positive = half_b_minus_1 * ((b_pow - 1) // (b - 1))
        required = (1 << (log_bound - 1)) - 1
        if max_positive < required:
            levels += 1
    return max(levels, 1)


def compute_num_digits_fold_code(r_vars: int, challenge_mass: int, log_basis: int) -> int:
    shift = r_vars + log_basis - 1
    if shift >= 127 or challenge_mass == 0:
        return compute_num_digits(128, log_basis)
    beta = challenge_mass * (1 << shift)
    if beta == 0:
        return 1
    return compute_num_digits(beta.bit_length(), log_basis)


def compute_num_digits_fold_tight(r_vars: int, max_abs_coeff: int, log_basis: int) -> int:
    beta = (1 << r_vars) * max_abs_coeff
    return compute_num_digits(beta.bit_length(), log_basis)


@dataclass(frozen=True)
class SmallDConfig:
    label: str
    d: int
    n_a: int
    challenge_mass: int
    max_abs_challenge_coeff: int
    log_basis: int
    delta_commit: int
    delta_open: int

    @property
    def alpha(self) -> int:
        return self.d.bit_length() - 1


D32_CONFIG = SmallDConfig(
    label="D=32",
    d=32,
    n_a=2,
    challenge_mass=256,
    max_abs_challenge_coeff=MAX_ABS_CHALLENGE_COEFF_D32,
    log_basis=5,
    delta_commit=1,
    delta_open=compute_num_digits(128, 5),
)

D16_CONFIG = SmallDConfig(
    label="D=16 (n_a=4)",
    d=16,
    n_a=4,
    challenge_mass=2048,
    max_abs_challenge_coeff=MAX_ABS_CHALLENGE_COEFF_D16,
    log_basis=5,
    delta_commit=1,
    delta_open=compute_num_digits(128, 5),
)

D16_NA3_CONFIG = SmallDConfig(
    label="D=16 (n_a=3)",
    d=16,
    n_a=3,
    challenge_mass=2048,
    max_abs_challenge_coeff=MAX_ABS_CHALLENGE_COEFF_D16,
    log_basis=5,
    delta_commit=1,
    delta_open=compute_num_digits(128, 5),
)


@dataclass(frozen=True)
class Layout:
    nv: int
    m_vars: int
    r_vars: int
    delta_fold_tight: int
    delta_fold_code: int
    cfg: SmallDConfig

    @property
    def num_blocks(self) -> int:
        return 1 << self.r_vars

    @property
    def block_len(self) -> int:
        return 1 << self.m_vars

    @property
    def inner_width(self) -> int:
        return self.block_len * self.cfg.delta_commit

    def outer_width(self, n_a: int) -> int:
        return n_a * self.cfg.delta_open * self.num_blocks

    @property
    def d_matrix_width(self) -> int:
        return self.cfg.delta_open * self.num_blocks


def best_layout(nv: int, cfg: SmallDConfig) -> Layout:
    reduced_vars = nv - cfg.alpha
    if reduced_vars <= 1:
        raise ValueError(f"nv={nv} too small for D={cfg.d}")

    n_a = cfg.n_a
    best = None
    for r_vars in range(1, reduced_vars):
        m_vars = reduced_vars - r_vars
        delta_fold_tight = compute_num_digits_fold_tight(
            r_vars, cfg.max_abs_challenge_coeff, cfg.log_basis
        )
        cost = (
            (cfg.delta_open + n_a * cfg.delta_commit) * (1 << r_vars)
            + cfg.delta_commit * delta_fold_tight * (1 << m_vars)
        )
        candidate = (cost, m_vars, r_vars, delta_fold_tight)
        if best is None or candidate < best:
            best = candidate

    assert best is not None
    _, m_vars, r_vars, delta_fold_tight = best
    delta_fold_code = compute_num_digits_fold_code(
        r_vars, cfg.challenge_mass, cfg.log_basis
    )
    return Layout(
        nv=nv,
        m_vars=m_vars,
        r_vars=r_vars,
        delta_fold_tight=delta_fold_tight,
        delta_fold_code=delta_fold_code,
        cfg=cfg,
    )


def recursive_level_layout(cfg: SmallDConfig, current_w_len: int) -> Layout:
    """Compute layout for a recursive level given the witness length from
    the previous level."""
    num_ring = current_w_len // cfg.d
    ring_pow2 = 1
    while ring_pow2 < num_ring:
        ring_pow2 <<= 1
    reduced_vars = ring_pow2.bit_length() - 1

    n_a = cfg.n_a
    lb = cfg.log_basis
    best = None
    for r_vars in range(1, reduced_vars):
        m_vars = reduced_vars - r_vars
        delta_fold_tight = compute_num_digits_fold_tight(
            r_vars, cfg.max_abs_challenge_coeff, lb
        )
        cost = (
            (cfg.delta_open + n_a * cfg.delta_commit) * (1 << r_vars)
            + cfg.delta_commit * delta_fold_tight * (1 << m_vars)
        )
        candidate = (cost, m_vars, r_vars, delta_fold_tight)
        if best is None or candidate < best:
            best = candidate

    assert best is not None
    _, m_vars, r_vars, delta_fold_tight = best
    delta_fold_code = compute_num_digits_fold_code(
        r_vars, cfg.challenge_mass, lb
    )

    nv = reduced_vars + cfg.alpha
    return Layout(
        nv=nv,
        m_vars=m_vars,
        r_vars=r_vars,
        delta_fold_tight=delta_fold_tight,
        delta_fold_code=delta_fold_code,
        cfg=cfg,
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def locate_estimator_repo() -> Path:
    env_path = os.environ.get("LATTICE_ESTIMATOR_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    root = repo_root()
    candidates.extend([
        root / "lattice-estimator",
        root / "third_party" / "lattice-estimator",
        root.parent / "lattice-estimator",
    ])
    for c in candidates:
        if (c / "estimator" / "__init__.py").exists():
            return c.resolve()
    raise SystemExit(
        "Could not locate lattice-estimator. "
        "Set LATTICE_ESTIMATOR_PATH or pass --estimator-path."
    )


def load_estimator(estimator_repo: Path):
    sys.path.insert(0, str(estimator_repo))
    from estimator import SIS
    from estimator.reduction import RC
    from sage.all import log
    return SIS, RC, log


def estimate_sec_bits(SIS, RC, log, rank: int, width_ring_elems: int,
                      collision_inf: int, d: int) -> float:
    n = rank * d
    m = width_ring_elems * d
    length_bound = (m ** 0.5) * collision_inf
    out = SIS.lattice(
        SIS.Parameters(n=n, q=Q, m=m, length_bound=length_bound, norm=2, tag="repro"),
        red_cost_model=RC.BDGL16,
        red_shape_model="lgsa",
        log_level=0,
    )
    return float(log(out["rop"], 2))


def r_decomp_levels_for_bound(log_basis: int) -> int:
    field_bits = 128
    half_field_bound = (Q) // 2
    levels = compute_num_digits(field_bits, log_basis)
    if levels == 0:
        levels = 1
    total_bits = levels * log_basis
    if total_bits <= field_bits:
        b = 1 << log_basis
        half_b_minus_1 = b // 2 - 1
        b_minus_1 = b - 1
        b_pow = b ** levels
        max_positive = half_b_minus_1 * ((b_pow - 1) // b_minus_1)
        if max_positive < half_field_bound:
            levels += 1
    return levels


def w_ring_count(layout: Layout, n_a: int, n_b: int, n_d: int) -> int:
    """Compute the witness ring element count for a level."""
    lb = layout.cfg.log_basis
    w_hat = layout.num_blocks * layout.cfg.delta_open
    t_hat = layout.num_blocks * n_a * layout.cfg.delta_open
    z_pre = layout.inner_width * layout.delta_fold_code
    m_row = n_d + n_b + 2 + n_a
    r_ct = m_row * r_decomp_levels_for_bound(lb)
    return w_hat + t_hat + z_pre + r_ct


def print_header(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def fmt(bits: float) -> str:
    return f"{bits:.2f}"


def sweep_for_config(SIS, RC, log, cfg: SmallDConfig):
    """Sweep n_b/n_d combinations and report security for each."""

    print_header(f"SIS Security Sweep: {cfg.label}")
    print(f"  q                  = {Q_LABEL}")
    print(f"  D                  = {cfg.d}")
    print(f"  n_a (fixed)        = {cfg.n_a}")
    print(f"  log_basis          = {cfg.log_basis}")
    print(f"  delta_commit       = {cfg.delta_commit}")
    print(f"  delta_open         = {cfg.delta_open}")
    print(f"  challenge_mass     = {cfg.challenge_mass}")
    print(f"  max_abs_coeff      = {cfg.max_abs_challenge_coeff}")
    print(f"  estimator model    = BDGL16 + lgsa")

    # Compute the layouts for the recursive levels reachable from the
    # baseline D=64 onehot nv=32 schedule. The first recursive level with
    # the small D receives the witness from the root D=64 level.
    # From the validated model: L0 output with lb=3 -> next_w = 29,898,176
    # L1 output with lb=4 -> next_w = 1,747,264
    # L2 output with lb=4 -> next_w = 411,968
    # L3 output with lb=4 -> next_w = 211,264
    # L4 output with lb=5 -> next_w = 110,720
    # L5 output with lb=5 -> next_w = 86,144
    #
    # We use L1's output (1,747,264) as the representative input for
    # the first small-D recursive level, which is the worst case (widest
    # witness) and thus the lowest security.
    #
    # But really, the SIS instance geometry depends on the planner's choice
    # of (m, r) for that level, which depends on cfg. So we compute the
    # recursive layout using the actual witness length.

    representative_w_lens = [
        ("after L0 (root D=64)", 29_898_176),
        ("after L1 (D=64)", 1_747_264),
        ("after L2 (D=64)", 411_968),
        ("after L4 (D=64)", 110_720),
    ]

    for w_label, w_len in representative_w_lens:
        layout = recursive_level_layout(cfg, w_len)
        print(f"\n  --- Recursive level layout from w_len={w_len:,} ({w_label}) ---")
        print(f"  reduced_vars={layout.nv - cfg.alpha}, m={layout.m_vars}, r={layout.r_vars}")
        print(f"  delta_fold_tight={layout.delta_fold_tight}, delta_fold_code={layout.delta_fold_code}")
        print(f"  inner_width={layout.inner_width}, d_matrix_width={layout.d_matrix_width}")

        print(f"\n  {'n_b':>4} {'n_d':>4} "
              f"{'outer_w':>10} {'d_mat_w':>10} "
              f"{'A_bits':>8} {'B_bits':>8} {'D_bits':>8} {'min':>8} {'>=128':>6}")

        for n_b in range(1, 5):
            for n_d in range(1, 5):
                inner_w = layout.inner_width
                outer_w = layout.outer_width(cfg.n_a)
                d_mat_w = layout.d_matrix_width

                a_bits = estimate_sec_bits(SIS, RC, log, cfg.n_a, inner_w, 2, cfg.d)
                b_bits = estimate_sec_bits(SIS, RC, log, n_b, outer_w, 7, cfg.d)
                d_bits = estimate_sec_bits(SIS, RC, log, n_d, d_mat_w, 7, cfg.d)
                overall = min(a_bits, b_bits, d_bits)
                ok = "YES" if overall >= 128.0 else "no"

                print(f"  {n_b:>4} {n_d:>4} "
                      f"{outer_w:>10,} {d_mat_w:>10,} "
                      f"{fmt(a_bits):>8} {fmt(b_bits):>8} {fmt(d_bits):>8} "
                      f"{fmt(overall):>8} {ok:>6}")

    # Also sweep nv to find the max supported nv at various n_b/n_d
    print_header(f"Max nv sweep for {cfg.label} (root-level geometry)")
    print("  Sweeps the root-level layout at each nv to find where B/D drops below 128 bits.")
    print()

    for n_b, n_d in [(1, 1), (2, 2), (cfg.n_a, cfg.n_a)]:
        print(f"\n  n_b={n_b}, n_d={n_d}:")
        print(f"  {'nv':>4} {'m':>4} {'r':>4} {'d_fold':>6} "
              f"{'A':>8} {'B':>8} {'D':>8} {'min':>8}")
        for nv in range(cfg.alpha + 2, 45):
            try:
                layout = best_layout(nv, cfg)
            except ValueError:
                continue
            inner_w = layout.inner_width
            outer_w = layout.outer_width(cfg.n_a)
            d_mat_w = layout.d_matrix_width

            a_bits = estimate_sec_bits(SIS, RC, log, cfg.n_a, inner_w, 2, cfg.d)
            b_bits = estimate_sec_bits(SIS, RC, log, n_b, outer_w, 7, cfg.d)
            d_bits = estimate_sec_bits(SIS, RC, log, n_d, d_mat_w, 7, cfg.d)
            overall = min(a_bits, b_bits, d_bits)
            marker = " <<<" if overall < 128.0 else ""
            print(f"  {nv:>4} {layout.m_vars:>4} {layout.r_vars:>4} "
                  f"{layout.delta_fold_tight:>6} "
                  f"{fmt(a_bits):>8} {fmt(b_bits):>8} {fmt(d_bits):>8} "
                  f"{fmt(overall):>8}{marker}")


def main() -> None:
    estimator_repo = locate_estimator_repo()
    print(f"Lattice estimator: {estimator_repo}")
    SIS, RC, log = load_estimator(estimator_repo)

    sweep_for_config(SIS, RC, log, D32_CONFIG)
    sweep_for_config(SIS, RC, log, D16_CONFIG)
    sweep_for_config(SIS, RC, log, D16_NA3_CONFIG)

    print_header("Summary")
    print("Check the tables above for the minimum n_b/n_d that gives >= 128 bits")
    print("on ALL roles (A, B, D) for the recursive-level witness geometries.")


if __name__ == "__main__":
    main()
