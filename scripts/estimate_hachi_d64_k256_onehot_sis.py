#!/usr/bin/env python3
"""
Estimate SIS security for a prospective Hachi D=64 one-hot family.

This script is meant to be readable on its own. It fixes one concrete
prospective `D = 64` parameter regime, explains the challenge family in plain
language, and prints the extracted SIS instances that determine the security
floor.

The setup studied here is:

- field modulus `q = 2^128 - 5823`
- ring degree `D = 64`, i.e. ring `F_q[X] / (X^64 + 1)`
- one-hot chunk size `K = 256`
- sparse challenge coefficients in `{-2, -1, 1, 2}`

The default challenge family is the rigorous split `k=32` family
`C_{21,<=6}`. Intuitively:

- split the `64` ring coefficients into the even and odd positions
- in each half, choose exactly `21` active slots out of `32`
- each active slot gets sign `+/-`
- at most `6` of those active slots may use magnitude `2`
- the two halves are then interleaved back into one ring element

That is what the name `C_{21,<=6}` means. Its conservative challenge mass is
`54 = 2 * (21 + 6)`. The same-budget larger variant `C_{22,<=6}` has mass
`56`.

For comparison, the script also supports two older direct full-ring shells:

- `(28,11)`: exactly `28` coefficients in `+/-1` and `11` in `+/-2`
- `(31,10)`: exactly `31` coefficients in `+/-1` and `10` in `+/-2`

Those legacy shells are useful comparison points, but they are not proven.

The script prints two reports:

1. one main estimate for a chosen parameter point
2. one sweep over `nv` to find the largest point that still clears 128 bits

Glossary for the printed quantities:

- `nv`: total multilinear variables before ring packing
- `alpha = log2(D)`: variables absorbed by the ring structure; here `alpha = 6`
- `m_vars`, `r_vars`: inner-block and outer/fold variables, with
  `nv - alpha = m_vars + r_vars`
- `num_blocks = 2^r_vars`, `block_len = 2^m_vars`
- `LOG_BASIS = 3`: base-8 digit decomposition
- `N_A`, `N_B`, `N_D`: module ranks of the three main commitment layers
- `inner_width`, `outer_width`, `d_matrix_width`: extracted SIS widths in ring
  elements
- `width_ring`: SIS width measured in ring elements; the estimator sees
  `m = width_ring * D` field coordinates
- `collision_inf`: `l_inf` collision bound passed to the estimator
- `A_fullwidth`, `B`, `D`, `M_code`, `M_tight`: the extracted SIS instances
- `overall floor`: the minimum security estimate across those instances

Modeling choices:

- the script calls the Euclidean `SIS.lattice(...)` path from
  `lattice-estimator`
- the reduction model is pinned to `BDGL16 + lgsa`
- for the folded one-hot witness `z_pre`, it uses the tighter onehot-aware
  bound `||z_pre||_inf <= 2^r_vars * 2`
  instead of the older dense proxy
  `2^r_vars * challenge_mass * 2^(LOG_BASIS - 1)`

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
from textwrap import dedent


# Fixed family parameters for the experiment.
Q = (1 << 128) - 5823
Q_LABEL = "2^128 - 5823"
D = 64
K = 256
LOG_BASIS = 3
DELTA_COMMIT = 1
DELTA_OPEN = 43
MAX_ABS_CHALLENGE_COEFF = 2
DEFAULT_CHALLENGE_MASS = 54

ALPHA = D.bit_length() - 1


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Combine preserved formatting with automatic default display."""


HELP_EPILOG = dedent(
    """\
    Terminology:
      nv           total multilinear variables before ring packing
      alpha        log2(D), i.e. packing variables absorbed by the ring
      m_vars       inner-block variables; block_len = 2^m_vars
      r_vars       outer/fold variables; num_blocks = 2^r_vars
      N_A/N_B/N_D  module ranks of the A, B, and D commitment layers
      challenge_mass
                   conservative L1 bound for the sparse challenge family
      delta_*      base-8 digit counts used in the extracted bounds
      A/B/D/M      SIS instances reported by the script
    """
)


TERMINOLOGY_LINES = [
    ("q", "field modulus used in the SIS instance"),
    ("D", "ring degree; ring is F_q[X] / (X^D + 1)"),
    ("K", "one-hot chunk size before ring packing"),
    ("alpha", "log2(D), the number of variables absorbed by ring packing"),
    ("nv", "total multilinear variables before subtracting alpha"),
    ("m_vars", "inner-block variables; block_len = 2^m_vars"),
    ("r_vars", "outer variables; num_blocks = 2^r_vars"),
    ("LOG_BASIS", "digit decomposition base exponent, so base = 2^LOG_BASIS = 8"),
    ("challenge_mass", "conservative L1 bound for the sparse challenge family"),
    ("delta_commit/open/fold", "numbers of base-8 digits used in the extracted bounds"),
    ("inner/outer/D widths", "ring-element counts of the A, B, and D SIS instances"),
    ("width_ring", "SIS width measured in ring elements; field-coordinate width is width_ring * D"),
    ("collision_inf", "l_inf collision bound passed to the SIS estimator"),
    ("A/B/D/M", "the extracted SIS instances whose minimum gives the overall floor"),
]


RIGOROUS_SPLIT_BY_MASS = {
    54: (21, 6),
    56: (22, 6),
}

LEGACY_RAW_SHELL_BY_MASS = {
    50: (28, 11),
    51: (31, 10),
}


@dataclass(frozen=True)
class Layout:
    """Split `reduced_vars = nv - alpha` into inner-block and outer variables."""

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
    """Security estimate for one extracted SIS instance."""

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
        ),
        formatter_class=HelpFormatter,
        epilog=HELP_EPILOG,
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
        help="Total multilinear variables nv for the main configuration report.",
    )
    parser.add_argument(
        "--candidate-na",
        type=int,
        default=1,
        help="Module rank N_A of the inner A commitment layer.",
    )
    parser.add_argument(
        "--candidate-nb",
        type=int,
        default=2,
        help="Module rank N_B of the outer B commitment layer.",
    )
    parser.add_argument(
        "--candidate-nd",
        type=int,
        default=2,
        help="Module rank N_D of the outer D commitment layer.",
    )
    parser.add_argument(
        "--sweep-min-nv",
        type=int,
        default=28,
        help="Smallest total multilinear variable count nv in the sweep.",
    )
    parser.add_argument(
        "--sweep-max-nv",
        type=int,
        default=44,
        help="Largest total multilinear variable count nv in the sweep.",
    )
    parser.add_argument(
        "--challenge-mass",
        type=int,
        default=DEFAULT_CHALLENGE_MASS,
        help=(
            "Conservative L1 mass of the D=64 {+/-1, +/-2} family. "
            "Use 54 for rigorous split C_{21,<=6} (default), 56 for "
            "rigorous split C_{22,<=6}, or 50/51 for the older raw "
            "direct full-ring shells (28,11)/(31,10)."
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
    """Return the number of signed base-2^log_basis digits needed for a bound."""

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
    """Fold-digit count from the generic dense proxy used by the code path."""

    beta = challenge_mass * (1 << (r_vars + LOG_BASIS - 1))
    return compute_num_digits(beta.bit_length(), LOG_BASIS)


def compute_num_digits_fold_tight(r_vars: int) -> int:
    """Fold-digit count from the tighter onehot-aware bound on z_pre."""

    # A monomial times a sparse challenge only sees the maximum absolute
    # challenge coefficient, not the full challenge L1 mass.
    beta = (1 << r_vars) * MAX_ABS_CHALLENGE_COEFF
    return compute_num_digits(beta.bit_length(), LOG_BASIS)


def describe_challenge_family(challenge_mass: int) -> str:
    if challenge_mass == 54:
        return "rigorous split k=32 family C_{21,<=6}"
    if challenge_mass == 56:
        return "rigorous split k=32 family C_{22,<=6}"
    if challenge_mass == 50:
        return "legacy raw direct full-ring shell (28,11) [not yet proven]"
    if challenge_mass == 51:
        return "legacy raw direct full-ring shell (31,10) [not yet proven]"
    return f"custom D=64 {{+/-1, +/-2}} family with conservative L1 mass {challenge_mass}"


def challenge_family_definition_lines(challenge_mass: int) -> list[str]:
    """Return a self-contained description of the selected challenge family."""

    if challenge_mass in RIGOROUS_SPLIT_BY_MASS:
        w, m = RIGOROUS_SPLIT_BY_MASS[challenge_mass]
        return [
            "This is the rigorous split family.",
            "Think of a challenge as an even half plus an odd half.",
            f"In each half: choose exactly {w} active slots out of 32, give them signs +/- ,",
            f"and allow magnitude 2 on at most {m} of those active slots.",
            "Then interleave the two halves back into one ring element as a0(X^2) + X a1(X^2).",
            f"Selected parameters: w = {w}, m = {m}.",
            f"Exact support = 2w = {2 * w}.",
            f"Conservative challenge_mass = 2(w + m) = {2 * (w + m)}.",
        ]

    if challenge_mass in LEGACY_RAW_SHELL_BY_MASS:
        n1, n2 = LEGACY_RAW_SHELL_BY_MASS[challenge_mass]
        return [
            "This is the older direct full-ring comparison shell.",
            f"Choose exactly {n1} positions with coefficients +/-1 and exactly {n2} positions with coefficients +/-2",
            "across all 64 slots of the ring element.",
            f"Selected parameters: n1 = {n1}, n2 = {n2}.",
            f"Exact support = n1 + n2 = {n1 + n2}.",
            f"Conservative challenge_mass = n1 + 2*n2 = {n1 + 2 * n2}.",
            "This shell is kept only for comparison; it is not proven.",
        ]

    return [
        "Custom conservative model with no named exact family attached.",
        "The script will treat challenge_mass only as an L1 proxy in the folded bound.",
        f"Selected challenge_mass = {challenge_mass}.",
    ]


def best_layout_onehot(nv: int, n_a: int, challenge_mass: int) -> Layout:
    """Choose the cheapest (m_vars, r_vars) split for a given nv and N_A."""

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
    """Estimate security in bits for one extracted SIS instance."""

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
    """Estimate all reported SIS layers for one chosen parameter point."""

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
    """Sweep nv with N_B = N_D = 1 and report the 128-bit cutoff."""

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


def print_terminology() -> None:
    """Print the glossary used by the report."""

    print_header("Terminology")
    for term, definition in TERMINOLOGY_LINES:
        print(f"- {term}: {definition}")


def print_challenge_family_definition(challenge_mass: int) -> None:
    """Print the exact family definition associated with the chosen mass."""

    print_header("Challenge Family Definition")
    for line in challenge_family_definition_lines(challenge_mass):
        print(line)


def print_intro(estimator_repo: Path, challenge_mass: int) -> None:
    print_header("Hachi D=64, K=256 one-hot SIS estimator")
    print(f"repo_root            = {repo_root()}")
    print(f"estimator_repo       = {estimator_repo}")
    print(f"field modulus        = {Q_LABEL}")
    print(f"ring degree D        = {D}")
    print(f"one-hot chunk size K = {K}")
    print(
        f"one-hot sparsity     = 1-of-{K} "
        f"(equiv. 1-sparse over {K} slots, density = {100.0 / K:.2f}%)"
    )
    print(f"packing alpha        = log2(D) = {ALPHA}")
    print(f"digit basis          = 2^{LOG_BASIS} = {1 << LOG_BASIS}")
    print(f"challenge family     = {describe_challenge_family(challenge_mass)}")
    print(f"challenge mass (L1)  = {challenge_mass}")
    print(f"folded z bound       = onehot-aware: ||z_pre||_inf <= 2^r_vars * {MAX_ABS_CHALLENGE_COEFF}")
    print(f"estimator model      = BDGL16 + lgsa")


def print_main_configuration(
    layout: Layout,
    estimates: list[LayerEstimate],
    n_a: int,
    n_b: int,
    n_d: int,
    challenge_mass: int,
) -> None:
    inner_width = layout.inner_width
    outer_width = layout.outer_width(n_a)
    d_matrix_width = layout.d_matrix_width
    m_code_width = d_matrix_width + outer_width + inner_width * layout.delta_fold_code
    m_tight_width = d_matrix_width + outer_width + inner_width * layout.delta_fold_tight
    m_collision_inf = 2 * ((1 << layout.r_vars) * MAX_ABS_CHALLENGE_COEFF)

    print_header("Main configuration estimate")
    print("This section estimates a single parameter point for the family above.")
    print()
    print(f"main nv              = {layout.nv}")
    print(f"N_A, N_B, N_D        = {n_a}, {n_b}, {n_d}")
    print(f"challenge family     = {describe_challenge_family(challenge_mass)}")
    print(f"challenge mass (L1)  = {challenge_mass}")
    print(f"reduced vars         = nv - alpha = {layout.nv - ALPHA}")
    print(f"layout               = (m_vars={layout.m_vars}, r_vars={layout.r_vars})")
    print(f"num_blocks           = 2^r_vars = {layout.num_blocks}")
    print(f"block_len            = 2^m_vars = {layout.block_len}")
    print(f"delta_fold_tight     = {layout.delta_fold_tight}")
    print(f"delta_fold_code      = {layout.delta_fold_code}")
    print(f"inner_width          = {inner_width}")
    print(f"outer_width          = {outer_width}")
    print(f"d_matrix_width       = {d_matrix_width}")
    print(f"M_code width         = {m_code_width}")
    print(f"M_tight width        = {m_tight_width}")
    print(f"collision_inf(M)     = {m_collision_inf}")
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
    print(
        "Columns: nv = total variables, m_vars/r_vars = layout split, "
        "d_fold = delta_fold_tight, A/B/D/M = security bits by layer."
    )
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
    print_challenge_family_definition(args.challenge_mass)
    print_terminology()
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
