#!/usr/bin/env python3
"""
Estimate SIS security for a proposed Hachi one-hot parameter family.

This script studies the specific family:

- base field modulus `q = 2^128 - 275`
- ring degree `D = 64`
- one-hot chunk size `K = 256`
- mixed exact sparse challenges over `{-2, -1, 1, 2}`

It prints two standalone reports:

1. a single "main configuration" estimate for a chosen parameter point
   (defaults: `nv=44`, `N_A=1`, `N_B=2`, `N_D=2`)
2. a sweep over `nv` with `N_B = N_D = 1` to locate the largest `nv`
   that still clears 128-bit SIS security under the same model

Modeling choices used here:

- The script uses the Euclidean `SIS.lattice(...)` path from
  `lattice-estimator`.
- The reduction cost / shape model is pinned to `BDGL16 + lgsa` so the output
  matches the Hachi-side analysis that was used during parameter exploration.
- For the folded one-hot witness `z_pre`, this script uses the tighter
  onehot-aware bound

      ||z_pre||_inf <= 2^r * max_abs_challenge_coeff = 2^r * 2

  rather than the older dense proxy `2^r * challenge_mass * 2^(log_basis-1)`.

Why this script exists:

- to make the D=64 / K=256 one-hot SIS analysis reproducible without relying
  on a separate markdown note
- to provide a single command that prints the key parameter point and the
  rank-1 cutoff sweep

Run from the `hachi/` repo root with either:

    sage -python scripts/estimate_hachi_d64_k256_onehot_sis.py

or, if the estimator is not in the default sibling location:

    LATTICE_ESTIMATOR_PATH="../lattice-estimator" \
      sage -python scripts/estimate_hachi_d64_k256_onehot_sis.py
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


# Fixed family parameters for the experiment.
Q = (1 << 128) - 275
D = 64
K = 256
LOG_BASIS = 3
DELTA_COMMIT = 1
DELTA_OPEN = 43
MAX_ABS_CHALLENGE_COEFF = 2
DEFAULT_CHALLENGE_MASS = 50


@dataclass(frozen=True)
class Layout:
    nv: int
    m_vars: int
    r_vars: int
    delta_fold_tight: int
    delta_fold_code: int

    @property
    def num_blocks(self) -> int:
        return 1 << self.r_vars

    @property
    def block_len(self) -> int:
        return 1 << self.m_vars

    @property
    def inner_width(self) -> int:
        return self.block_len * DELTA_COMMIT

    def outer_width(self, n_a: int) -> int:
        return n_a * DELTA_OPEN * self.num_blocks

    @property
    def d_matrix_width(self) -> int:
        return DELTA_OPEN * self.num_blocks


@dataclass(frozen=True)
class LayerEstimate:
    name: str
    sec_bits: float
    rank: int
    width_ring_elems: int
    collision_inf: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate SIS security for the Hachi D=64, K=256 one-hot family "
            "and print both a main configuration report and a rank-1 nv sweep."
        )
    )
    parser.add_argument(
        "--estimator-path",
        help=(
            "Path to the lattice-estimator repo. Defaults to "
            "LATTICE_ESTIMATOR_PATH or a sibling ../lattice-estimator checkout."
        ),
    )
    parser.add_argument(
        "--candidate-nv",
        type=int,
        default=44,
        help="Number of multilinear variables for the main configuration report.",
    )
    parser.add_argument(
        "--candidate-na",
        type=int,
        default=1,
        help="Module rank N_A for the main configuration report.",
    )
    parser.add_argument(
        "--candidate-nb",
        type=int,
        default=2,
        help="Module rank N_B for the main configuration report.",
    )
    parser.add_argument(
        "--candidate-nd",
        type=int,
        default=2,
        help="Module rank N_D for the main configuration report.",
    )
    parser.add_argument(
        "--sweep-min-nv",
        type=int,
        default=28,
        help="Smallest nv included in the rank-1 cutoff sweep.",
    )
    parser.add_argument(
        "--sweep-max-nv",
        type=int,
        default=44,
        help="Largest nv included in the rank-1 cutoff sweep.",
    )
    parser.add_argument(
        "--challenge-mass",
        type=int,
        default=DEFAULT_CHALLENGE_MASS,
        help=(
            "L1 mass of the mixed exact D=64 family. Use 50 for (28,11) "
            "or 51 for (31,10)."
        ),
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def locate_estimator_repo(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())

    env_path = os.environ.get("LATTICE_ESTIMATOR_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    root = repo_root()
    candidates.extend(
        [
            root / "lattice-estimator",
            root / "third_party" / "lattice-estimator",
            root / "vendor" / "lattice-estimator",
            root.parent / "lattice-estimator",
        ]
    )

    for candidate in candidates:
        if (candidate / "estimator" / "__init__.py").exists():
            return candidate.resolve()

    raise SystemExit(
        "Could not locate lattice-estimator. "
        "Set LATTICE_ESTIMATOR_PATH or pass --estimator-path."
    )


def load_estimator(estimator_repo: Path):
    sys.path.insert(0, str(estimator_repo))
    from estimator import SIS  # type: ignore
    from estimator.reduction import RC  # type: ignore
    from sage.all import log  # type: ignore

    return SIS, RC, log


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
        b_pow = b**levels
        max_positive = half_b_minus_1 * ((b_pow - 1) // (b - 1))
        required = (1 << (log_bound - 1)) - 1
        if max_positive < required:
            levels += 1

    return max(levels, 1)


def compute_num_digits_fold_code(r_vars: int, challenge_mass: int) -> int:
    # Generic dense proxy used by the current code path.
    beta = challenge_mass * (1 << (r_vars + LOG_BASIS - 1))
    return compute_num_digits(beta.bit_length(), LOG_BASIS)


def compute_num_digits_fold_tight(r_vars: int) -> int:
    # Onehot-aware bound: a monomial times a sparse challenge only sees the
    # maximum absolute challenge coefficient, not the full challenge L1 mass.
    beta = (1 << r_vars) * MAX_ABS_CHALLENGE_COEFF
    return compute_num_digits(beta.bit_length(), LOG_BASIS)


def best_layout_onehot(nv: int, n_a: int, challenge_mass: int) -> Layout:
    alpha = D.bit_length() - 1
    reduced_vars = nv - alpha
    if reduced_vars <= 1:
        raise ValueError(f"nv={nv} is too small for D={D}")

    best: tuple[int, int, int, int] | None = None
    for r_vars in range(1, reduced_vars):
        m_vars = reduced_vars - r_vars
        delta_fold_tight = compute_num_digits_fold_tight(r_vars)
        cost = (
            (DELTA_OPEN + n_a * DELTA_COMMIT) * (1 << r_vars)
            + DELTA_COMMIT * delta_fold_tight * (1 << m_vars)
        )
        candidate = (cost, m_vars, r_vars, delta_fold_tight)
        if best is None or candidate < best:
            best = candidate

    assert best is not None
    _, m_vars, r_vars, delta_fold_tight = best
    delta_fold_code = compute_num_digits_fold_code(r_vars, challenge_mass)
    return Layout(
        nv=nv,
        m_vars=m_vars,
        r_vars=r_vars,
        delta_fold_tight=delta_fold_tight,
        delta_fold_code=delta_fold_code,
    )


def estimate_sec_bits(SIS, RC, log, rank: int, width_ring_elems: int, collision_inf: int) -> float:
    n = rank * D
    m = width_ring_elems * D
    length_bound = (m**0.5) * collision_inf
    out = SIS.lattice(
        SIS.Parameters(n=n, q=Q, m=m, length_bound=length_bound, norm=2, tag="repro"),
        red_cost_model=RC.BDGL16,
        red_shape_model="lgsa",
        log_level=0,
    )
    return float(log(out["rop"], 2))


def main_configuration_estimates(
    SIS, RC, log, layout: Layout, n_a: int, n_b: int, n_d: int
) -> list[LayerEstimate]:
    inner_width = layout.inner_width
    outer_width = layout.outer_width(n_a)
    d_matrix_width = layout.d_matrix_width

    a_bits = estimate_sec_bits(SIS, RC, log, n_a, inner_width, 2)
    b_bits = estimate_sec_bits(SIS, RC, log, n_b, outer_width, 7)
    d_bits = estimate_sec_bits(SIS, RC, log, n_d, d_matrix_width, 7)

    m_collision_inf = 2 * ((1 << layout.r_vars) * MAX_ABS_CHALLENGE_COEFF)
    m_rank = n_a + n_b + n_d + 2
    m_code_width = d_matrix_width + outer_width + inner_width * layout.delta_fold_code
    m_tight_width = d_matrix_width + outer_width + inner_width * layout.delta_fold_tight
    m_code_bits = estimate_sec_bits(SIS, RC, log, m_rank, m_code_width, m_collision_inf)
    m_tight_bits = estimate_sec_bits(SIS, RC, log, m_rank, m_tight_width, m_collision_inf)

    return [
        LayerEstimate("A_fullwidth", a_bits, n_a, inner_width, 2),
        LayerEstimate("B", b_bits, n_b, outer_width, 7),
        LayerEstimate("D", d_bits, n_d, d_matrix_width, 7),
        LayerEstimate("M_code", m_code_bits, m_rank, m_code_width, m_collision_inf),
        LayerEstimate("M_tight", m_tight_bits, m_rank, m_tight_width, m_collision_inf),
    ]


def sweep_rank1_cutoff(SIS, RC, log, min_nv: int, max_nv: int, n_a: int, challenge_mass: int):
    rows = []
    for nv in range(min_nv, max_nv + 1):
        layout = best_layout_onehot(nv, n_a=n_a, challenge_mass=challenge_mass)
        outer_width = layout.outer_width(n_a)
        d_matrix_width = layout.d_matrix_width
        m_rank = n_a + 1 + 1 + 2
        m_collision_inf = 2 * ((1 << layout.r_vars) * MAX_ABS_CHALLENGE_COEFF)
        m_tight_width = d_matrix_width + outer_width + layout.inner_width * layout.delta_fold_tight

        a_bits = estimate_sec_bits(SIS, RC, log, n_a, layout.inner_width, 2)
        b_bits = estimate_sec_bits(SIS, RC, log, 1, outer_width, 7)
        d_bits = estimate_sec_bits(SIS, RC, log, 1, d_matrix_width, 7)
        m_bits = estimate_sec_bits(SIS, RC, log, m_rank, m_tight_width, m_collision_inf)
        overall = min(a_bits, b_bits, d_bits, m_bits)
        rows.append(
            {
                "nv": nv,
                "m_vars": layout.m_vars,
                "r_vars": layout.r_vars,
                "delta_fold": layout.delta_fold_tight,
                "A_bits": a_bits,
                "B_bits": b_bits,
                "D_bits": d_bits,
                "BD_floor_bits": min(b_bits, d_bits),
                "M_bits": m_bits,
                "overall_bits": overall,
            }
        )

    at_least_128 = [row["nv"] for row in rows if row["overall_bits"] >= 128.0]
    cutoff = max(at_least_128) if at_least_128 else None
    return rows, cutoff


def fmt(bits: float) -> str:
    return f"{bits:.2f}"


def print_header(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def print_intro(estimator_repo: Path, challenge_mass: int) -> None:
    print_header("Hachi D=64, K=256 one-hot SIS estimator")
    print(f"repo_root            = {repo_root()}")
    print(f"estimator_repo       = {estimator_repo}")
    print(f"field modulus        = 2^128 - 275")
    print(f"ring degree D        = {D}")
    print(f"one-hot chunk size K = {K}")
    print(f"challenge family     = D=64 mixed exact shell with L1 mass {challenge_mass}")
    print(f"folded z bound       = onehot-aware: ||z_pre||_inf <= 2^r * {MAX_ABS_CHALLENGE_COEFF}")
    print(f"estimator model      = BDGL16 + lgsa")


def print_main_configuration(
    layout: Layout,
    estimates: list[LayerEstimate],
    n_a: int,
    n_b: int,
    n_d: int,
    challenge_mass: int,
) -> None:
    print_header("Main configuration estimate")
    print("This section estimates a single parameter point for the family above.")
    print()
    print(f"main nv              = {layout.nv}")
    print(f"N_A, N_B, N_D        = {n_a}, {n_b}, {n_d}")
    print(f"challenge mass       = {challenge_mass}")
    print(f"layout               = (m_vars={layout.m_vars}, r_vars={layout.r_vars})")
    print(f"delta_fold_tight     = {layout.delta_fold_tight}")
    print(f"delta_fold_code      = {layout.delta_fold_code}")
    print()
    print(f"{'layer':<12} {'sec_bits':>10} {'rank':>8} {'width_ring':>14} {'collision_inf':>14}")
    for estimate in estimates:
        print(
            f"{estimate.name:<12} {fmt(estimate.sec_bits):>10} "
            f"{estimate.rank:>8} {estimate.width_ring_elems:>14} {estimate.collision_inf:>14}"
        )
    overall = min(estimate.sec_bits for estimate in estimates)
    overall_layer = min(estimates, key=lambda estimate: estimate.sec_bits).name
    print()
    print("Layer legend:")
    print("- A_fullwidth: conservative full-support proxy for the inner A layer")
    print("- B / D: outer commitment layers with digit-collision bound 7")
    print("- M_code: folded witness width using the generic code-style delta_fold proxy")
    print("- M_tight: folded witness width using the tighter onehot-aware delta_fold")
    print()
    print(f"overall floor        = {fmt(overall)} bits ({overall_layer})")


def print_sweep(rows: list[dict], cutoff: int | None) -> None:
    print_header("Rank-1 cutoff sweep")
    print("This sweep fixes N_B = N_D = 1 and searches for the largest nv with overall >= 128 bits.")
    print()
    print(
        f"{'nv':>4} {'m_vars':>7} {'r_vars':>7} {'d_fold':>7} "
        f"{'A':>8} {'B/D':>8} {'M':>8} {'overall':>8}"
    )
    for row in rows:
        print(
            f"{row['nv']:>4} {row['m_vars']:>7} {row['r_vars']:>7} {row['delta_fold']:>7} "
            f"{fmt(row['A_bits']):>8} {fmt(row['BD_floor_bits']):>8} "
            f"{fmt(row['M_bits']):>8} {fmt(row['overall_bits']):>8}"
        )
    print()
    if cutoff is None:
        print("largest nv with overall >= 128 bits: none in sweep")
    else:
        print(f"largest nv with overall >= 128 bits: {cutoff}")


def main() -> None:
    args = parse_args()
    estimator_repo = locate_estimator_repo(args.estimator_path)
    SIS, RC, log = load_estimator(estimator_repo)

    main_layout = best_layout_onehot(
        args.candidate_nv,
        n_a=args.candidate_na,
        challenge_mass=args.challenge_mass,
    )
    main_estimates = main_configuration_estimates(
        SIS,
        RC,
        log,
        main_layout,
        n_a=args.candidate_na,
        n_b=args.candidate_nb,
        n_d=args.candidate_nd,
    )
    rows, cutoff = sweep_rank1_cutoff(
        SIS,
        RC,
        log,
        min_nv=args.sweep_min_nv,
        max_nv=args.sweep_max_nv,
        n_a=args.candidate_na,
        challenge_mass=args.challenge_mass,
    )

    print_intro(estimator_repo, args.challenge_mass)
    print_main_configuration(
        main_layout,
        main_estimates,
        n_a=args.candidate_na,
        n_b=args.candidate_nb,
        n_d=args.candidate_nd,
        challenge_mass=args.challenge_mass,
    )
    print_sweep(rows, cutoff)


if __name__ == "__main__":
    main()
