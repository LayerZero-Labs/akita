#!/usr/bin/env python3
"""
Hachi Universal Proof Size Planner

Computes optimal proof sizes for Hachi polynomial commitment scheme proofs
across multiple ring dimension configurations (D=64, D=32, D=16) with
128-bit SIS security enforced via lattice-estimator-verified thresholds.

Five complementary techniques:
  1. Ring dimension reduction (D=64→32→16)
  2. Eq-compressed sumcheck (1 fewer element/round for any sumcheck with eq)
  3. Fully 4-ary GKR tree for Stage 1
  4. Column-major block layout (tight z_pre)
  5. Serialization header stripping (remove redundant Vec length prefixes)

Supports arbitrary polynomial types (onehot, dense, etc.) and any nv.

Usage:
    python3 scripts/hachi_proof_size_planner.py              # All results
    python3 scripts/hachi_proof_size_planner.py --validate    # Baseline validation only
    python3 scripts/hachi_proof_size_planner.py --breakdown   # Detailed level breakdowns
    python3 scripts/hachi_proof_size_planner.py --compare     # Tight z_pre comparison

Validated against the Rust planner (cargo test dump_planned_schedules).
See docs/proof-size-reduction-study.md for full analysis.
"""

import math
import sys
from dataclasses import dataclass


# =============================================================================
#  Constants
# =============================================================================

FIELD_BITS = 128
ELEM_BYTES = FIELD_BITS // 8  # 16

HALF_FIELD_BOUND_P5823 = (2**128 - 5823) // 2
HALF_FIELD_BOUND_P275 = (2**128 - 275) // 2


# =============================================================================
#  Core computation: digit decomposition
# =============================================================================

def compute_num_digits(log_bound: int, log_basis: int) -> int:
    """Number of base-2^log_basis digits to represent a value with log_bound bits."""
    if log_bound == 0 or log_basis == 0:
        return 1
    levels = math.ceil(log_bound / log_basis)
    total_bits = levels * log_basis
    if total_bits <= log_bound:
        b = 1 << log_basis
        b_pow = b ** levels
        max_positive = (b // 2 - 1) * ((b_pow - 1) // (b - 1))
        required = (1 << (log_bound - 1)) - 1 if log_bound <= 128 else (2**128 - 1) // 2
        if max_positive < required:
            levels += 1
    return max(levels, 1)


def compute_num_digits_fold(r_vars: int, challenge_l1_mass: int, log_basis: int) -> int:
    """Number of digits for the folded witness decomposition."""
    shift = r_vars + log_basis - 1
    if shift >= 127 or challenge_l1_mass == 0:
        return compute_num_digits(128, log_basis)
    beta = challenge_l1_mass * (1 << shift)
    if beta == 0:
        return 1
    return compute_num_digits(beta.bit_length(), log_basis)


def r_decomp_levels(log_basis: int, half_field_bound: int) -> int:
    """Number of r-decomposition levels for quotient rows."""
    levels = compute_num_digits(FIELD_BITS, log_basis)
    if levels == 0:
        levels = 1
    total_bits = levels * log_basis
    if total_bits <= FIELD_BITS:
        b = 1 << log_basis
        b_pow = b ** levels
        max_positive = (b // 2 - 1) * ((b_pow - 1) // (b - 1))
        if max_positive < half_field_bound:
            levels += 1
    return levels


# =============================================================================
#  Core computation: proof element sizes
# =============================================================================

def ring_vec_bytes(ring_len: int, ring_dim: int) -> int:
    return ring_len * ring_dim * ELEM_BYTES


def compressed_unipoly_bytes(degree: int) -> int:
    return degree * ELEM_BYTES


def sumcheck_bytes(rounds: int, degree: int) -> int:
    return rounds * compressed_unipoly_bytes(degree)


def packed_digits_bytes(num_elems: int, bits_per_elem: int) -> int:
    return math.ceil(num_elems * bits_per_elem / 8)


def stage1_bytes_optimized(n_rounds: int, lb: int) -> int:
    """Stage 1 cost with eq-compression + fully 4-ary GKR tree.

    Merges pairs of binary levels into degree-4 stages wherever possible.
    If lb-1 is odd, the leftover binary level goes at the top (root)
    to minimise inter-stage claims (fewer nodes at the top).

    lb=2 (d=2): 1 stage, degree 2
    lb=3 (d=4): 1 stage, degree 4
    lb≥4:       ⌊(lb-1)/2⌋ degree-4 stages
                + (lb-1)%2  degree-2 stage at root (if odd)
                + inter-stage claims
    """
    d = (1 << lb) >> 1
    if lb <= 3:
        return n_rounds * d * ELEM_BYTES
    num_levels = lb - 1
    num_4ary = num_levels // 2
    has_binary_top = num_levels % 2

    deg4_cost = n_rounds * 4 * ELEM_BYTES
    deg2_cost = n_rounds * 2 * ELEM_BYTES
    stage_cost = num_4ary * deg4_cost + has_binary_top * deg2_cost

    total_stages = num_4ary + has_binary_top
    if total_stages <= 1:
        inter_claims = 0
    elif has_binary_top:
        claims = 2
        nodes = 2
        for _ in range(num_4ary - 1):
            claims += 4 * nodes
            nodes *= 4
        inter_claims = claims * ELEM_BYTES
    else:
        claims = 0
        nodes = 1
        for _ in range(num_4ary - 1):
            claims += 4 * nodes
            nodes *= 4
        inter_claims = claims * ELEM_BYTES
    return stage_cost + inter_claims


def sumcheck_rounds(D: int, next_w_len: int) -> int:
    """Total sumcheck rounds (num_u + num_l)."""
    num_l = D.bit_length() - 1
    num_ring = next_w_len // D
    nrp = 1
    while nrp < num_ring:
        nrp <<= 1
    num_u = nrp.bit_length() - 1
    return num_u + num_l


# =============================================================================
#  Ring configuration
# =============================================================================

@dataclass
class RingConfig:
    """Ring configuration with SIS security thresholds.

    SIS thresholds are enforced via lattice-estimator-derived width caps
    keyed by `(D, collision_inf, rank)`. The same width table applies to
    the A/B/D roles; only the extracted width and collision bound differ.
    The A-role uses a challenge-aware proxy: its raw digit collision is
    scaled by the maximum absolute coefficient in the stage-1 challenge
    family and rounded up to the next supported SIS bucket.
    """
    D: int
    n_a: int
    challenge_l1_mass: int
    max_abs_challenge_coeff: int
    label: str

# Max secure SIS width (in ring elements) at 128-bit security for
# `(D, collision_inf, rank)`, verified with lattice-estimator
# (BDGL16 + lgsa, q = 2^128 - 275).
#
# collision_inf values come from:
#   - root onehot A-role: 2
#   - balanced base-2^lb digits: 2^lb - 1, so lb=2..7 -> 3,7,15,31,63,127
SIS_MAX_WIDTH_BY_D_AND_COLLISION = {
    16: {
        2: (158, 10450, 260593, 200000),
        3: (158, 10450, 260593, 200000),
        7: (31, 1919, 47864, 200000),
        15: (21, 418, 10423, 155015),
        31: (18, 97, 2440, 36294),
        63: (15, 38, 590, 8787),
        127: (14, 30, 145, 2162),
        255: (12, 26, 50, 536),
        511: (11, 23, 40, 133),
        1023: (10, 21, 34, 55),
        2047: (9, 19, 31, 46),
        4095: (9, 18, 28, 41),
        8191: (8, 17, 26, 37),
        16383: (7, 15, 24, 33),
    },
    32: {
        2: (11757, 4359823, 5000000, 5000000),
        3: (5225, 1937699, 5000000, 5000000),
        7: (959, 355903, 5000000, 5000000),
        15: (209, 77507, 7357796, 5000000),
        31: (48, 18147, 1722689, 5000000),
        63: (19, 4393, 417108, 5000000),
        127: (15, 1081, 102641, 4824061),
        255: (13, 268, 25459, 1196574),
        511: (11, 66, 6339, 297974),
        1023: (10, 27, 1581, 74347),
        2047: (9, 23, 395, 18568),
    },
    64: {
        2: (2179911, 20000000, 20000000, 20000000),
        3: (968849, 20000000, 20000000, 20000000),
        7: (177951, 20000000, 20000000, 20000000),
        15: (38753, 20000000, 20000000, 20000000),
        31: (9073, 20000000, 20000000, 20000000),
        63: (2196, 9801875, 20000000, 20000000),
        127: (540, 2412030, 20000000, 20000000),
        255: (134, 598287, 20000000, 20000000),
        511: (33, 148987, 20000000, 20000000),
    },
}


def min_rank_for_secure_width(D: int, collision_inf: int, width: int) -> int | None:
    """Smallest Module-SIS rank whose width cap covers `width`."""
    by_collision = SIS_MAX_WIDTH_BY_D_AND_COLLISION.get(D, {})
    widths = by_collision.get(collision_inf)
    if widths is None:
        raise ValueError(f"missing SIS table for D={D}, collision_inf={collision_inf}")
    for i, max_w in enumerate(widths):
        if width <= max_w:
            return i + 1
    return None


def ceil_supported_collision(D: int, collision_inf: int) -> int | None:
    """Round a requested collision bound up to the next available SIS bucket."""
    buckets = sorted(SIS_MAX_WIDTH_BY_D_AND_COLLISION.get(D, {}))
    for bucket in buckets:
        if collision_inf <= bucket:
            return bucket
    return None


def a_role_collision(cfg: RingConfig, level: int, log_cb: int, lb: int) -> int | None:
    """Challenge-aware A-role collision proxy used by the proof-size planner."""
    raw_collision = 2 if level == 0 and log_cb == 1 else (1 << lb) - 1
    requested = raw_collision * cfg.max_abs_challenge_coeff
    return ceil_supported_collision(cfg.D, requested)


ALL_RING_CONFIGS = [
    RingConfig(D=64, n_a=1, challenge_l1_mass=54, max_abs_challenge_coeff=2, label="D64-na1"),
    RingConfig(D=64, n_a=2, challenge_l1_mass=54, max_abs_challenge_coeff=2, label="D64-na2"),
    RingConfig(D=32, n_a=1, challenge_l1_mass=256, max_abs_challenge_coeff=8, label="D32-na1"),
    RingConfig(D=32, n_a=2, challenge_l1_mass=256, max_abs_challenge_coeff=8, label="D32-na2"),
    RingConfig(D=32, n_a=3, challenge_l1_mass=256, max_abs_challenge_coeff=8, label="D32-na3"),
    RingConfig(D=16, n_a=1, challenge_l1_mass=2048, max_abs_challenge_coeff=128, label="D16-na1"),
    RingConfig(D=16, n_a=2, challenge_l1_mass=2048, max_abs_challenge_coeff=128, label="D16-na2"),
    RingConfig(D=16, n_a=3, challenge_l1_mass=2048, max_abs_challenge_coeff=128, label="D16-na3"),
    RingConfig(D=16, n_a=4, challenge_l1_mass=2048, max_abs_challenge_coeff=128, label="D16-na4"),
]

MIN_LB = 2
MAX_LB = 7


# =============================================================================
#  Witness computation
# =============================================================================

@dataclass
class LevelComputation:
    """Result of computing one folding level's witness structure."""
    m_vars: int
    r_vars: int
    delta_commit: int
    delta_open: int
    delta_fold: int
    w_ring_elems: int
    next_w_len: int
    rounds: int


def optimal_m_r_split(
    n_a: int, challenge_l1_mass: int,
    log_commit_bound: int, log_basis: int,
    reduced_vars: int, num_ring: int = 0,
) -> tuple[int, int]:
    """Find (m, r) split minimizing next-level witness size.

    When num_ring > 0, uses tight z_pre: ceil(num_ring / 2^r) instead of 2^m.
    """
    delta_open = compute_num_digits(128 if log_commit_bound < 128 else log_commit_bound, log_basis)
    delta_commit = compute_num_digits(log_commit_bound, log_basis)
    c1 = delta_open + n_a * delta_commit

    best_r, best_cost = reduced_vars // 2, float('inf')
    for r in range(1, reduced_vars):
        m = reduced_vars - r
        delta_fold = compute_num_digits_fold(r, challenge_l1_mass, log_basis)
        m_eff = -(-num_ring // (1 << r)) if num_ring > 0 else 1 << m
        cost = c1 * (1 << r) + delta_commit * delta_fold * m_eff
        if cost < best_cost:
            best_cost = cost
            best_r = r
    return (reduced_vars - best_r, best_r)


def compute_level_witness(
    cfg: RingConfig, level: int, current_w_len: int,
    max_num_vars: int, log_basis: int,
    half_field_bound: int, nb: int, nd: int,
    log_commit_bound: int = 0,
    tight_zpre: bool = False,
) -> LevelComputation:
    """Compute witness structure for one folding level.

    At level 0, uses log_commit_bound for the opening decomposition.
    At level > 0, uses the caller-supplied log_commit_bound (prev_lb)
    so that delta_commit correctly accounts for re-decomposition when
    the gadget base changes between levels.
    """
    D = cfg.D
    alpha = D.bit_length() - 1

    if level == 0:
        reduced_vars = max_num_vars - alpha
        log_cb = log_commit_bound
        num_ring_actual = 1 << reduced_vars
    else:
        num_ring = current_w_len // D
        ring_pow2 = 1
        while ring_pow2 < num_ring:
            ring_pow2 <<= 1
        reduced_vars = ring_pow2.bit_length() - 1
        log_cb = log_commit_bound if log_commit_bound > 0 else log_basis
        num_ring_actual = num_ring

    nr_arg = num_ring_actual if tight_zpre else 0
    m_vars, r_vars = optimal_m_r_split(
        cfg.n_a, cfg.challenge_l1_mass,
        log_cb, log_basis, reduced_vars, num_ring=nr_arg,
    )

    open_bound = 128 if log_cb < 128 else log_cb
    delta_open = compute_num_digits(open_bound, log_basis)
    delta_commit = compute_num_digits(log_cb, log_basis)
    delta_fold = compute_num_digits_fold(r_vars, cfg.challenge_l1_mass, log_basis)

    num_blocks = 1 << r_vars
    if tight_zpre:
        m_actual = -(-num_ring_actual // num_blocks)
    else:
        m_actual = 1 << m_vars
    inner_width = m_actual * delta_commit

    w_hat = num_blocks * delta_open
    t_hat = num_blocks * cfg.n_a * delta_open
    z_pre = inner_width * delta_fold
    m_row = nd + nb + 2 + cfg.n_a
    r_ct = m_row * r_decomp_levels(log_basis, half_field_bound)
    w_ring_elems = w_hat + t_hat + z_pre + r_ct
    next_w_len = w_ring_elems * D
    rounds = sumcheck_rounds(D, next_w_len)

    return LevelComputation(
        m_vars=m_vars, r_vars=r_vars,
        delta_commit=delta_commit, delta_open=delta_open,
        delta_fold=delta_fold,
        w_ring_elems=w_ring_elems, next_w_len=next_w_len,
        rounds=rounds,
    )


# =============================================================================
#  Universal planner
# =============================================================================

def run_universal_planner(
    log_commit_bound: int,
    max_num_vars: int,
    ring_configs: list[RingConfig] | None = None,
    half_field_bound: int = HALF_FIELD_BOUND_P275,
    opt_sumcheck: bool = True,
    monotone_d: bool = True,
    tight_zpre: bool = False,
    verbose: bool = False,
) -> dict:
    """Security-aware universal planner via dynamic programming.

    Searches over all ring configs (D, n_a) and lb values at each level.
    Supports any polynomial type via log_commit_bound (1=onehot, 128=full)
    and any number of variables.

    Each recursive call to best_from(w_len, cur_D, prev_lb) returns costs
    INCLUDING the entry commitment from the previous level, so the root
    level only adds its own prefix + best_from().

    Args:
        log_commit_bound: bits per coefficient (1=onehot, 128=full field)
        max_num_vars: number of variables (polynomial has 2^max_num_vars coefficients)
        ring_configs: list of RingConfig to search over
        half_field_bound: (q-1)/2 for the working prime
        opt_sumcheck: use eq-compression + tree@4 for Stage 1
        monotone_d: restrict D to only decrease across levels
        tight_zpre: use column-major block layout (z_pre = ceil(num_ring/2^r))
        verbose: print detailed schedule

    Returns dict with keys: total, num_levels, level_bytes, tail_bytes,
    final_w_len, final_lb, levels_list.
    """
    if ring_configs is None:
        ring_configs = ALL_RING_CONFIGS
    unique_ds = sorted({c.D for c in ring_configs}, reverse=True)
    cfgs_by_d = {d: [c for c in ring_configs if c.D == d] for d in unique_ds}

    memo: dict = {}

    def _level_prefix(cfg: RingConfig, lb: int, rounds: int, nd: int) -> int:
        if opt_sumcheck:
            s1 = stage1_bytes_optimized(rounds, lb)
        else:
            s1 = sumcheck_bytes(rounds, (1 << lb) // 2 + 1)
        return (
            ring_vec_bytes(1, cfg.D)      # y
            + ring_vec_bytes(nd, cfg.D)   # v
            + s1 + ELEM_BYTES             # stage1 + s_claim
            + sumcheck_bytes(rounds, 3)   # stage2
            + ELEM_BYTES                  # eval
        )

    def _try_level(cfg: RingConfig, level: int, w_len: int, lb: int, log_cb: int):
        """Try a folding level. Returns None or (prefix, lc, nb, nd)."""
        nb, nd = 1, 1
        lc = compute_level_witness(
            cfg, level, w_len, max_num_vars, lb,
            half_field_bound, nb, nd,
            log_commit_bound=log_cb, tight_zpre=tight_zpre,
        )
        if lc.next_w_len >= w_len:
            return None

        num_ring = w_len // cfg.D if level > 0 else (1 << (max_num_vars - (cfg.D.bit_length() - 1)))
        if tight_zpre:
            inner_width = -(-num_ring // (1 << lc.r_vars)) * lc.delta_commit
        else:
            inner_width = (1 << lc.m_vars) * lc.delta_commit
        a_collision = a_role_collision(cfg, level, log_cb, lb)
        if a_collision is None:
            return None
        na_needed = min_rank_for_secure_width(cfg.D, a_collision, inner_width)
        if na_needed is None or na_needed > cfg.n_a:
            return None

        bd_collision = (1 << lb) - 1
        outer = cfg.n_a * lc.delta_open * (1 << lc.r_vars)
        d_mat = lc.delta_open * (1 << lc.r_vars)
        nb = min_rank_for_secure_width(cfg.D, bd_collision, outer)
        nd = min_rank_for_secure_width(cfg.D, bd_collision, d_mat)
        if nb is None or nd is None:
            return None

        lc = compute_level_witness(
            cfg, level, w_len, max_num_vars, lb,
            half_field_bound, nb, nd,
            log_commit_bound=log_cb, tight_zpre=tight_zpre,
        )
        if lc.next_w_len >= w_len:
            return None
        return (_level_prefix(cfg, lb, lc.rounds, nd), lc, nb, nd)

    def _tail_entry_nb(w_len: int, d: int, tail_lb: int) -> int | None:
        ring_elems = (w_len + d - 1) // d
        return min_rank_for_secure_width(d, (1 << tail_lb) - 1, ring_elems)

    def best_from(w_len: int, cur_D: int, prev_lb: int, depth: int = 8):
        """Optimal cost from this recursive level onward.

        Returns (total_cost, levels_list, tail_lb).
        total_cost INCLUDES entry commitment into this level.
        """
        key = (w_len, cur_D, prev_lb)
        if key in memo:
            return memo[key]

        best = (float('inf'), [], prev_lb)
        tnb = _tail_entry_nb(w_len, cur_D, prev_lb)
        if tnb is not None:
            t = ring_vec_bytes(tnb, cur_D) + packed_digits_bytes(w_len, prev_lb)
            best = (t, [], prev_lb)

        if depth <= 0:
            memo[key] = best
            return best

        for cfg in cfgs_by_d.get(cur_D, []):
            for lb in range(MIN_LB, MAX_LB + 1):
                result = _try_level(cfg, 1, w_len, lb, prev_lb)
                if result is None:
                    continue
                prefix, lc, nb_self, nd_self = result
                entry_commit = ring_vec_bytes(nb_self, cur_D)

                for next_D in unique_ds:
                    if monotone_d and next_D > cur_D:
                        continue
                    suffix = best_from(lc.next_w_len, next_D, lb, depth - 1)
                    total = entry_commit + prefix + suffix[0]
                    if total < best[0]:
                        best = (
                            total,
                            [(lb, entry_commit + prefix, lc, cfg.label,
                              cfg.n_a, nb_self, nd_self, cfg.D)]
                            + suffix[1],
                            suffix[2],
                        )

        memo[key] = best
        return best

    # --- Root level (level 0) ---
    root_w_len = 1 << max_num_vars
    overall_best = None

    for root_cfg in ring_configs:
        for root_lb in range(MIN_LB, MAX_LB + 1):
            result = _try_level(root_cfg, 0, root_w_len, root_lb, log_commit_bound)
            if result is None:
                continue
            root_prefix, root_lc, root_nb, root_nd = result

            for next_D in unique_ds:
                if monotone_d and next_D > root_cfg.D:
                    continue
                suffix = best_from(root_lc.next_w_len, next_D, root_lb)
                total = root_prefix + suffix[0] + 4  # +4 for wrapper
                if overall_best is None or total < overall_best[0]:
                    root_entry = (root_lb, root_prefix, root_lc, root_cfg.label,
                                  root_cfg.n_a, root_nb, root_nd, root_cfg.D)
                    overall_best = (total, [root_entry] + suffix[1], suffix[2])

    if overall_best is None:
        return {"total": float('inf'), "num_levels": 0, "level_bytes": 0,
                "tail_bytes": 0, "final_w_len": 0, "final_lb": 0,
                "levels_list": []}

    total, levels_list, tail_lb = overall_best
    last_lc = levels_list[-1][2]
    term_w = last_lc.next_w_len
    term_tail = packed_digits_bytes(term_w, tail_lb)
    level_bytes_sum = sum(e[1] for e in levels_list)

    if verbose:
        print(f"  levels ({len(levels_list)}):")
        for entry in levels_list:
            lb, lv_bytes, lc, lbl, na, nb, nd, D = entry
            print(f"    lb={lb} m={lc.m_vars} r={lc.r_vars} "
                  f"D={D} na={na} nb={nb} nd={nd} "
                  f"δo={lc.delta_open} δf={lc.delta_fold} δc={lc.delta_commit} "
                  f"w_ring={lc.w_ring_elems:,} next_w={lc.next_w_len:,} "
                  f"level={lv_bytes:,}B [{lbl}]")
        print(f"  terminal: w_len={term_w:,} lb={tail_lb} tail={term_tail:,}B")
        print(f"  TOTAL: {total:,} B  ({total/1024:.1f} KB)")

    return {
        "total": total,
        "num_levels": len(levels_list),
        "level_bytes": level_bytes_sum,
        "tail_bytes": term_tail,
        "final_w_len": term_w,
        "final_lb": tail_lb,
        "levels_list": levels_list,
    }


# =============================================================================
#  Baseline planner (for Rust validation only)
# =============================================================================

def _baseline_ring_vec_bytes(ring_len: int, ring_dim: int) -> int:
    return 8 + ring_len * ring_dim * ELEM_BYTES


def _baseline_sumcheck_bytes(rounds: int, degree: int) -> int:
    return 8 + rounds * (8 + degree * ELEM_BYTES)


def _baseline_packed_digits_bytes(num_elems: int, bits_per_elem: int) -> int:
    return 8 + 1 + math.ceil(num_elems * bits_per_elem / 8)


def _run_baseline_planner(
    D: int, n_a: int, n_b: int, n_d: int,
    challenge_l1_mass: int,
    log_commit_bound: int,
    max_num_vars: int,
    min_lb: int = 2, max_lb: int = 5,
) -> dict:
    """Simplified planner matching the Rust best_recursive_suffix logic.

    Used only for validating against the Rust planner output. Does not
    support ring dimension reduction, sumcheck optimization, tight z_pre,
    or header stripping.
    """
    half_q = HALF_FIELD_BOUND_P5823
    alpha = D.bit_length() - 1

    def _compute_level(level: int, current_w_len: int, lb: int):
        if level == 0:
            reduced = max_num_vars - alpha
            log_cb = log_commit_bound
        else:
            num_ring = current_w_len // D
            rp2 = 1
            while rp2 < num_ring:
                rp2 <<= 1
            reduced = rp2.bit_length() - 1
            log_cb = lb

        m, r = optimal_m_r_split(n_a, challenge_l1_mass, log_cb, lb, reduced)
        op = 128 if log_cb < 128 else log_cb
        d_open = compute_num_digits(op, lb)
        d_commit = compute_num_digits(log_cb, lb)
        d_fold = compute_num_digits_fold(r, challenge_l1_mass, lb)
        bl = 1 << m
        iw = bl * d_commit
        w_hat = (1 << r) * d_open
        t_hat = (1 << r) * n_a * d_open
        z_pre = iw * d_fold
        r_ct = (n_d + n_b + 2 + n_a) * r_decomp_levels(lb, half_q)
        w_ring = w_hat + t_hat + z_pre + r_ct
        nw = w_ring * D
        rnds = sumcheck_rounds(D, nw)
        return m, r, d_open, d_commit, d_fold, w_ring, nw, rnds

    def _level_bytes(lb: int, rounds: int) -> int:
        s1_deg = (1 << lb) // 2 + 1
        return (
            _baseline_ring_vec_bytes(1, D) + _baseline_ring_vec_bytes(n_d, D)
            + _baseline_sumcheck_bytes(rounds, s1_deg) + ELEM_BYTES
            + _baseline_sumcheck_bytes(rounds, 3)
            + _baseline_ring_vec_bytes(n_b, D) + ELEM_BYTES
        )

    memo: dict = {}

    def best_suffix(level: int, w_len: int, lb: int):
        key = (level, w_len, lb)
        if key in memo:
            return memo[key]
        tail = _baseline_packed_digits_bytes(w_len, lb)
        best = (tail, [])
        _, _, _, _, _, _, nw, rnds = _compute_level(level, w_len, lb)
        if nw < w_len:
            for nlb in range(max(lb, min_lb), max_lb + 1):
                lbytes = _level_bytes(lb, rnds)
                sb, sl = best_suffix(level + 1, nw, nlb)
                cand = lbytes + sb
                if cand < best[0]:
                    best = (cand, [(lb, lbytes, nw, rnds)] + sl)
        memo[key] = best
        return best

    root_w = 1 << max_num_vars
    overall = None
    for rlb in range(min_lb, max_lb + 1):
        _, _, _, _, _, _, nw, rnds = _compute_level(0, root_w, rlb)
        if nw >= root_w:
            continue
        for nlb in range(max(rlb, min_lb), max_lb + 1):
            rb = _level_bytes(rlb, rnds)
            sb, sl = best_suffix(1, nw, nlb)
            total = rb + sb
            if overall is None or total < overall[0]:
                overall = (total, [(rlb, rb, nw, rnds)] + sl)

    if overall is None:
        return {"total": float('inf')}

    total = overall[0] + 4
    levels = overall[1]
    last_lb = levels[-1][0]
    term_w = levels[-1][2]
    return {
        "total": total,
        "tail_bytes": packed_digits_bytes(term_w, last_lb),
        "final_w_len": term_w,
        "final_lb": last_lb,
        "num_levels": len(levels),
    }


# =============================================================================
#  Output formatting
# =============================================================================

def _d_schedule(levels_list: list) -> str:
    return "→".join(str(e[7]) for e in levels_list)


def _print_headline_table(results: list[tuple]):
    """Print the headline results table."""
    print(f"\n  {'Poly type':<15} {'nv':>4} {'Baseline':>10} {'Optimized':>10} {'Reduction':>10}")
    print(f"  {'-'*15} {'-'*4} {'-'*10} {'-'*10} {'-'*10}")
    for name, nv, baseline, optimized in results:
        pct = (1 - optimized / baseline) * 100
        print(f"  {name:<15} {nv:>4} {baseline:>10,} {optimized:>10,} {f'−{pct:.1f}%':>10}")


def _print_detailed_breakdown(res: dict, title: str = ""):
    """Print per-level detailed breakdown."""
    if title:
        print(f"\n  {title}")
    levels = res['levels_list']
    print(f"  levels ({len(levels)}):")
    for i, entry in enumerate(levels):
        lb, lv_bytes, lc, lbl, na, nb, nd, D = entry
        print(f"    L{i}: D={D} lb={lb} m={lc.m_vars} r={lc.r_vars} [{lbl}]")
        print(f"        na={na} nb={nb} nd={nd}  "
              f"δo={lc.delta_open} δf={lc.delta_fold} δc={lc.delta_commit}  "
              f"w_ring={lc.w_ring_elems:,}  next_w={lc.next_w_len:,}  "
              f"level={lv_bytes:,}B")
    print(f"  terminal: w_len={res['final_w_len']:,}  lb={res['final_lb']}  "
          f"tail={res['tail_bytes']:,}B")
    print(f"  TOTAL: {res['total']:,} B  ({res['total']/1024:.1f} KB)")


def _print_waste_analysis(res: dict):
    """Print per-level zero-padding waste analysis for tight z_pre."""
    levels = res['levels_list']
    print(f"\n    Per-level waste analysis:")
    print(f"    {'Level':>5} {'num_ring':>10} {'m_actual':>10} {'2^m':>6} {'waste%':>7}")
    for i, entry in enumerate(levels):
        if i == 0:
            continue
        lb, _, lc, _, _, _, _, D = entry
        prev_lc = levels[i-1][2]
        nr = prev_lc.next_w_len // D
        m_act = -(-nr // (1 << lc.r_vars))
        blen = 1 << lc.m_vars
        waste = (1 - m_act / blen) * 100
        print(f"    L{i:>4} {nr:>10,} {m_act:>10,} {blen:>6,} {waste:>6.1f}%")


# =============================================================================
#  Entry points
# =============================================================================

RUST_BASELINES = {
    ("onehot", 64, 1, 32): 99805,
    ("full128", 128, 128, 25): 166613,
    ("full128", 128, 128, 32): 173197,
}


def cmd_validate():
    """Validate the proof size model against Rust planner output."""
    print("=" * 70)
    print("  Baseline Validation (vs Rust planner)")
    print("=" * 70)

    all_ok = True
    for (name, D, lcb, nv), expected in sorted(RUST_BASELINES.items()):
        l1 = 54 if D == 64 else 31
        r = _run_baseline_planner(D, 1, 1, 1, l1, lcb, nv)
        ok = r["total"] == expected
        mark = "✓" if ok else "✗"
        if not ok:
            all_ok = False
        print(f"  {mark}  {name} nv={nv}: python={r['total']:,}  rust={expected:,}")

    if all_ok:
        print("\n  All baselines match.")
    else:
        print("\n  ⚠ MISMATCH — model diverges from Rust planner!")
    return all_ok


def cmd_results():
    """Print the headline optimized results."""
    print("=" * 70)
    print("  Hachi Universal Planner — Optimized Results")
    print("  (eq-comp + tree@4 + tight z_pre + header stripping, 128-bit SIS)")
    print("=" * 70)

    configs = [
        ("onehot", 1, [20, 25, 30, 32, 38, 44]),
        ("full",  128, [20, 25, 30, 32]),
    ]

    headline = []
    for poly_name, lcb, nvs in configs:
        print(f"\n  {poly_name.upper()} (log_commit_bound={lcb})")
        print(f"  {'nv':>4} {'total':>10} {'D schedule':<25} {'tail':>10}")
        print(f"  {'-'*4} {'-'*10} {'-'*25} {'-'*10}")

        for nv in nvs:
            r = run_universal_planner(lcb, nv, tight_zpre=True, verbose=False)
            ds = _d_schedule(r['levels_list'])
            print(f"  {nv:>4} {r['total']:>10,} {ds:<25} {r['tail_bytes']:>10,}")

            baseline = _get_baseline(lcb, nv)
            if baseline:
                headline.append((poly_name, nv, baseline, r['total']))

    if headline:
        print(f"\n\n{'─' * 70}")
        print("  Headline: optimized vs baseline")
        _print_headline_table(headline)


def _get_baseline(lcb: int, nv: int) -> int | None:
    """Get the D=64/D=128 baseline for comparison."""
    if lcb == 1:
        r = _run_baseline_planner(64, 1, 1, 1, 54, 1, nv)
    elif lcb >= 128:
        r = _run_baseline_planner(128, 1, 1, 1, 31, lcb, nv)
    else:
        return None
    return r['total'] if r['total'] < float('inf') else None


def cmd_breakdown():
    """Print detailed per-level breakdowns for representative configs."""
    print("=" * 70)
    print("  Detailed Level Breakdowns")
    print("=" * 70)

    cases = [
        ("onehot", 1, 32),
        ("onehot", 1, 44),
        ("full", 128, 32),
        ("full", 128, 25),
    ]

    for name, lcb, nv in cases:
        baseline = _get_baseline(lcb, nv)
        r = run_universal_planner(lcb, nv, tight_zpre=True, verbose=False)
        title = f"{name} nv={nv}"
        if baseline:
            pct = (1 - r['total'] / baseline) * 100
            title += f"  (baseline: {baseline:,} B → −{pct:.1f}%)"
        _print_detailed_breakdown(r, title)
        _print_waste_analysis(r)
        print()


def cmd_compare():
    """Compare standard vs tight z_pre optimization."""
    print("=" * 70)
    print("  Standard vs Tight z_pre (Column-Major Blocks)")
    print("=" * 70)

    configs = [
        ("onehot", 1, [20, 25, 30, 32, 38, 44]),
        ("full",  128, [20, 25, 30, 32]),
    ]

    for poly_name, lcb, nvs in configs:
        print(f"\n  {poly_name.upper()} (log_commit_bound={lcb})")
        print(f"  {'nv':>4} {'standard':>10} {'tight':>10} {'saved':>8} {'%':>7}")
        print(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*8} {'-'*7}")

        for nv in nvs:
            std = run_universal_planner(lcb, nv, tight_zpre=False, verbose=False)
            tgt = run_universal_planner(lcb, nv, tight_zpre=True, verbose=False)
            saved = std['total'] - tgt['total']
            pct = saved / std['total'] * 100 if std['total'] > 0 else 0
            print(f"  {nv:>4} {std['total']:>10,} {tgt['total']:>10,} "
                  f"{saved:>8,} {pct:>6.1f}%")

        detail_nv = 32 if 32 in nvs else nvs[-1]
        print(f"\n  ▶ Detailed comparison for {poly_name} nv={detail_nv}:")

        print(f"\n    Standard (2^m blocks):")
        std = run_universal_planner(lcb, detail_nv, tight_zpre=False, verbose=True)

        print(f"\n    Tight z_pre (ceil(num_ring/2^r) blocks):")
        tgt = run_universal_planner(lcb, detail_nv, tight_zpre=True, verbose=True)
        _print_waste_analysis(tgt)


def main():
    if "--validate" in sys.argv:
        cmd_validate()
    elif "--breakdown" in sys.argv:
        cmd_breakdown()
    elif "--compare" in sys.argv:
        cmd_compare()
    else:
        ok = cmd_validate()
        print()
        cmd_results()
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
