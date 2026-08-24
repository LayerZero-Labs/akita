#!/usr/bin/env python3
"""Hachi proof-size planner for 32-bit prime field (q = 2^32 - 99).

Security-aware DP planner that derives (n_a, n_b, n_d) from MSIS security
constraints at 128-bit security (BDGL16+lgsa), sweeping D ∈ {64, 128, 256}.

Key differences from the 128-bit planner:
  - field_bits = 32, base field elements = 4 bytes
  - Sumcheck runs over degree-4 extension field F_{q^4} for Fiat-Shamir
    security, so sumcheck messages / claims / evals = 16 bytes each
  - log_open_bound = max(log_commit_bound, 32)  (not 128)
  - SIS width table for q = 2^32 - 99
  - Ring configs with D ∈ {64, 128, 256}

Two element sizes coexist in the proof:
  - BASE_ELEM_BYTES = 4:  committed ring vectors, packed digits
  - EXT_ELEM_BYTES  = 16: sumcheck messages, claims, evaluations, y_ring
"""

import math
import sys
from dataclasses import dataclass, field
from typing import Optional

# ── Field parameters ──────────────────────────────────────────────────────

FIELD_BITS = 32
EXT_DEGREE = 4                          # degree of extension for sumcheck
BASE_ELEM_BYTES = 4                     # base field element (F_q)
EXT_ELEM_BYTES = BASE_ELEM_BYTES * EXT_DEGREE  # extension field (F_{q^4}) = 16 bytes

# ── SIS width table: q = 2^32 - 99, 128-bit security (BDGL16+lgsa) ──────
#
# sis_max_widths[(D, collision_inf)] = [rank1, rank2, rank3, rank4]
# where rank_i is the maximum number of ring-element columns secure at rank i.

SIS_MAX_WIDTHS = {
    # D=64
    (64,   2): [89, 5_878, 146_583, 2_179_911],
    (64,   3): [39, 2_612, 65_148, 968_849],
    (64,   7): [7, 479, 11_966, 177_951],
    (64,  15): [5, 104, 2_605, 38_753],
    (64,  31): [4, 24, 610, 9_073],
    (64,  63): [3, 9, 147, 2_196],
    (64, 127): [3, 7, 36, 540],
    (64, 255): [3, 6, 12, 134],
    # D=128
    (128,   2): [2_939, 1_089_955, 5_000_000, 5_000_000],
    (128,   3): [1_306, 484_424, 3_000_000, 3_000_000],
    (128,   7): [239, 88_975, 3_000_000, 3_000_000],
    (128,  15): [52, 19_376, 1_839_449, 3_000_000],
    (128,  31): [12, 4_536, 430_672, 3_000_000],
    (128,  63): [4, 1_098, 104_277, 3_000_000],
    (128, 127): [3, 270, 25_660, 3_000_000],
    # D=256
    (256,   2): [500_000, 500_000, 500_000, 500_000],
    (256,   3): [242_212, 500_000, 500_000, 500_000],
    (256,   7): [44_487, 500_000, 500_000, 500_000],
    (256,  15): [9_688, 500_000, 500_000, 500_000],
    (256,  31): [2_268, 500_000, 500_000, 500_000],
    (256,  63): [549, 500_000, 500_000, 500_000],
    (256, 127): [135, 500_000, 500_000, 500_000],
}

MAX_RANK = 4


def min_rank_for_secure_width(d: int, collision_inf: int, width: int) -> Optional[int]:
    """Smallest MSIS rank achieving 128-bit security for the given SIS width."""
    key = (d, collision_inf)
    if key not in SIS_MAX_WIDTHS:
        return None
    widths = SIS_MAX_WIDTHS[key]
    for rank_minus_1, max_w in enumerate(widths):
        if width <= max_w:
            return rank_minus_1 + 1
    return None


def ceil_supported_collision(d: int, collision_inf: int) -> Optional[int]:
    buckets = sorted(key[1] for key in SIS_MAX_WIDTHS if key[0] == d)
    for bucket in buckets:
        if collision_inf <= bucket:
            return bucket
    return None


# ── Digit math (from Rust planner) ───────────────────────────────────────

def balanced_digit_max(log_basis: int, num_digits: int) -> int:
    b = 1 << log_basis
    max_digit = b // 2 - 1
    b_pow = b ** num_digits
    return max_digit * (b_pow - 1) // (b - 1)


def compute_num_digits(log_bound: int, log_basis: int) -> int:
    assert 0 < log_basis < 128
    if log_bound == 0:
        return 1
    n = -(-log_bound // log_basis)  # ceil division
    total_bits = n * log_basis
    if total_bits <= log_bound:
        required = (1 << (log_bound - 1)) - 1 if log_bound <= 128 else (1 << 127) - 1
        if balanced_digit_max(log_basis, n) < required:
            n += 1
    return max(n, 1)


def compute_num_digits_fold(r_vars: int, l1_mass: int, log_basis: int) -> int:
    shift = r_vars + log_basis - 1
    if shift >= 127 or l1_mass == 0:
        return compute_num_digits(128, log_basis)
    beta = l1_mass * (1 << shift)
    log_beta = beta.bit_length()
    return compute_num_digits(log_beta, log_basis)


def optimal_m_r_split(n_a, l1_mass, log_cb, log_basis, reduced_vars, num_ring=0):
    if reduced_vars <= 2 or reduced_vars >= 53:
        r = reduced_vars // 2
        return (reduced_vars - r, r)

    open_bound = max(log_cb, FIELD_BITS)  # w_i ∈ R_q → coefficients in Z_q
    d_open = compute_num_digits(open_bound, log_basis)
    d_commit = compute_num_digits(log_cb, log_basis)
    per_block = d_open + n_a * d_open

    best_cost, best_r = float('inf'), reduced_vars // 2
    for r in range(1, reduced_vars):
        nb = 1 << r
        m_eff = -(-num_ring // nb) if num_ring > 0 else (1 << (reduced_vars - r))
        d_fold = compute_num_digits_fold(r, l1_mass, log_basis)
        cost = per_block * nb + d_commit * d_fold * m_eff
        if cost < best_cost:
            best_cost, best_r = cost, r
    return (reduced_vars - best_r, best_r)


# ── Proof size helpers ────────────────────────────────────────────────────
#
# Proof components use two different element sizes:
#   - Committed ring vectors (entry commitment, v, next_w_commitment):
#     coefficients are in F_q → BASE_ELEM_BYTES per coefficient.
#   - Sumcheck messages, claims, evaluations, y_ring:
#     over extension field F_{q^4} → EXT_ELEM_BYTES per element.
#
# y_ring: result of folding with extension-field challenges, lives in
#   R_{q^4} = F_{q^4}[X]/(X^D+1). Cost: D * EXT_ELEM_BYTES.

def ring_vec_bytes_base(ring_len, ring_dim):
    """Ring vector with base-field coefficients (commitments)."""
    return ring_len * ring_dim * BASE_ELEM_BYTES


def ring_vec_bytes_ext(ring_len, ring_dim):
    """Ring vector with extension-field coefficients (evaluations)."""
    return ring_len * ring_dim * EXT_ELEM_BYTES


def sumcheck_bytes(rounds, degree):
    """Sumcheck proof: rounds * degree extension-field elements."""
    return rounds * degree * EXT_ELEM_BYTES


def packed_digits_bytes(num_elems, bits_per_elem):
    """Packed balanced digits (base field)."""
    return -(-(num_elems * bits_per_elem) // 8)


def stage1_bytes_optimized(n_rounds, lb):
    """Stage 1 sumcheck with eq-compression + 4-ary GKR tree (extension field)."""
    if lb <= 3:
        d = (1 << lb) >> 1
        return n_rounds * d * EXT_ELEM_BYTES
    num_levels = lb - 1
    num_4ary = num_levels // 2
    has_binary_top = num_levels % 2
    deg4_cost = n_rounds * 4 * EXT_ELEM_BYTES
    deg2_cost = n_rounds * 2 * EXT_ELEM_BYTES
    stage_cost = num_4ary * deg4_cost + has_binary_top * deg2_cost
    total_stages = num_4ary + has_binary_top
    if total_stages <= 1:
        inter_claims = 0
    elif has_binary_top:
        claims, nodes = 2, 2
        for _ in range(max(num_4ary - 1, 0)):
            claims += 4 * nodes
            nodes *= 4
        inter_claims = claims * EXT_ELEM_BYTES
    else:
        claims, nodes = 0, 1
        for _ in range(max(num_4ary - 1, 0)):
            claims += 4 * nodes
            nodes *= 4
        inter_claims = claims * EXT_ELEM_BYTES
    return stage_cost + inter_claims


def sumcheck_rounds(level_d, next_w_len):
    num_l = (level_d & -level_d).bit_length() - 1
    num_ring = next_w_len // level_d
    p = 1
    while p < num_ring:
        p <<= 1
    num_u = p.bit_length() - 1 if p > 0 else 0
    return num_u + num_l


# ── Ring configurations ───────────────────────────────────────────────────

@dataclass
class RingConfig:
    d: int
    n_a: int
    l1_mass: int
    max_abs_challenge_coeff: int
    label: str


# Challenge L1 masses:
#   D=64:  same as 128-bit Hachi (54 for ternary weight-54 challenge)
#   D=128: conservative estimate (54, same as D=64)
#   D=256: scaled down (27)
# These are configurable and should be validated against the actual
# challenge distribution for each ring.
ALL_RING_CONFIGS = [
    # D=64: rank 1-2 (rank 1 is very constrained at 32-bit)
    RingConfig(64,  1, 54, 2, "D64-na1"),
    RingConfig(64,  2, 54, 2, "D64-na2"),
    # D=128: rank 1-2 (l1_mass lower than D=64: sparser challenges suffice)
    RingConfig(128, 1, 27, 1, "D128-na1"),
    RingConfig(128, 2, 27, 1, "D128-na2"),
    # D=256: rank 1-2
    RingConfig(256, 1, 19, 1, "D256-na1"),
    RingConfig(256, 2, 19, 1, "D256-na2"),
]

MIN_LB = 2
MAX_LB = 7


# ── Level computation ─────────────────────────────────────────────────────

@dataclass
class LevelComputation:
    m_vars: int
    r_vars: int
    delta_commit: int
    delta_open: int
    delta_fold: int
    w_ring_elems: int
    next_w_len: int
    rounds: int


def compute_level_witness(cfg: RingConfig, m_vars, r_vars, log_basis, log_cb,
                          nb, nd, num_ring_actual, tight_zpre=True):
    d = cfg.d
    # Committed witness coefficients are in Z_q (32-bit), so:
    #   delta_open = ceil(log_q / lb) — decomposition of w_i ∈ R_q
    #   delta_commit = ceil(log_cb / lb) — decomposition of original poly coeffs
    # The quotient r ∈ Z_q^{<d-1}[X] also has Z_q coefficients.
    # See Hachi paper Section 4.2 Eq(14) and Section 4.4 proof size analysis.
    open_bound = max(log_cb, FIELD_BITS)  # w_i ∈ R_q → coeffs in Z_q
    delta_open = compute_num_digits(open_bound, log_basis)
    delta_commit = compute_num_digits(log_cb, log_basis)
    delta_fold = compute_num_digits_fold(r_vars, cfg.l1_mass, log_basis)

    num_blocks = 1 << r_vars
    m_actual = -(-num_ring_actual // num_blocks) if tight_zpre else (1 << m_vars)
    inner_width = m_actual * delta_commit

    w_hat = num_blocks * delta_open
    t_hat = num_blocks * cfg.n_a * delta_open
    z_pre = inner_width * delta_fold
    m_row = nd + nb + 2 + cfg.n_a
    # r_decomp: quotient r ∈ Z_q^{<d-1}[X] → coefficients in Z_q (32-bit)
    r_ct = m_row * compute_num_digits(FIELD_BITS, log_basis)
    w_ring_elems = w_hat + t_hat + z_pre + r_ct
    next_w_len = w_ring_elems * d
    rounds = sumcheck_rounds(d, next_w_len)

    return LevelComputation(
        m_vars=m_vars, r_vars=r_vars,
        delta_commit=delta_commit, delta_open=delta_open, delta_fold=delta_fold,
        w_ring_elems=w_ring_elems, next_w_len=next_w_len, rounds=rounds,
    )


def a_role_collision(cfg: RingConfig, level: int, log_cb: int, lb: int) -> Optional[int]:
    raw_collision = 2 if (level == 0 and log_cb == 1) else ((1 << lb) - 1)
    requested = raw_collision * cfg.max_abs_challenge_coeff
    return ceil_supported_collision(cfg.d, requested)


# ── Planner output types ─────────────────────────────────────────────────

@dataclass
class PlannedLevel:
    d: int
    lb: int
    m_vars: int
    r_vars: int
    na: int
    nb: int
    nd: int
    delta_open: int
    delta_fold: int
    delta_commit: int
    w_ring: int
    next_w_len: int
    level_bytes: int
    label: str


@dataclass
class Schedule:
    levels: list
    tail_bytes: int
    total_bytes: int
    final_w_len: int
    final_lb: int


# ── DP Planner ────────────────────────────────────────────────────────────

class Planner:
    def __init__(self, log_commit_bound, max_num_vars, ring_configs=None,
                 tight_zpre=True, monotone_d=True, opt_sumcheck=True):
        self.log_cb = log_commit_bound
        self.nv = max_num_vars
        self.cfgs = ring_configs or ALL_RING_CONFIGS
        self.tight_zpre = tight_zpre
        self.monotone_d = monotone_d
        self.opt_sumcheck = opt_sumcheck
        self.unique_ds = sorted(set(c.d for c in self.cfgs), reverse=True)
        self.memo = {}

    def cfgs_for_d(self, d):
        return [c for c in self.cfgs if c.d == d]

    def level_prefix_bytes(self, cfg, lb, rounds, nd):
        """Proof bytes for one level (excluding entry commitment).

        From Hachi paper Section 4.4 (proof size), one iteration sends:
          1. Ring switch (Section 3.2): k-1 partial evaluations y_i in F_{q^k},
             each costing k Z_q-elements. Total: (k-1)*k base field elements.
             In our implementation: 1 R_q element y_ring (d Z_q-elements).
          2. v = D * w_hat: n_D ring elements over R_q (committed, base field)
          3. Sumcheck round polynomials over F_{q^k} (extension field):
             - Stage 1 (range proof F_{0,τ₀}): degree b+1, extension elements
             - Stage 2 (linear relation F_{α,τ₁}): degree 3, extension elements
          4. s_claim: extension field element
          5. next_w_eval: extension field element (evaluation of next-level poly)
        """
        if self.opt_sumcheck:
            s1 = stage1_bytes_optimized(rounds, lb)
        else:
            deg = ((1 << lb) // 2) + 1
            s1 = sumcheck_bytes(rounds, deg)

        return (ring_vec_bytes_base(1, cfg.d)        # y_ring: d Z_q-elements
                + ring_vec_bytes_base(nd, cfg.d)      # v: n_D * d Z_q-elements
                + s1                                   # stage1 sumcheck (extension)
                + EXT_ELEM_BYTES                       # s_claim (extension)
                + sumcheck_bytes(rounds, 3)            # stage2 sumcheck (extension)
                + EXT_ELEM_BYTES)                      # next_w_eval (extension)

    def try_level_mr(self, cfg, level, w_len, lb, log_cb, m_vars, r_vars):
        """Try a specific (cfg, lb, m, r) at a given level. Returns (prefix_bytes, lc, nb, nd) or None."""
        d = cfg.d
        alpha = (d & -d).bit_length() - 1

        if level == 0:
            num_ring = 1 << (self.nv - alpha)
        else:
            num_ring = w_len // d

        # Initial witness computation with nb=nd=1
        lc = compute_level_witness(cfg, m_vars, r_vars, lb, log_cb, 1, 1,
                                   num_ring, self.tight_zpre)
        if lc.next_w_len >= w_len:
            return None

        # Derive security-constrained ranks
        inner_width = (-(-num_ring // (1 << r_vars)) * lc.delta_commit
                       if self.tight_zpre else (1 << m_vars) * lc.delta_commit)

        a_collision = a_role_collision(cfg, level, log_cb, lb)
        if a_collision is None:
            return None
        na_needed = min_rank_for_secure_width(d, a_collision, inner_width)
        if na_needed is None or na_needed > cfg.n_a:
            return None

        bd_collision = (1 << lb) - 1
        outer = cfg.n_a * lc.delta_open * (1 << r_vars)
        d_mat = lc.delta_open * (1 << r_vars)
        nb = min_rank_for_secure_width(d, bd_collision, outer)
        nd = min_rank_for_secure_width(d, bd_collision, d_mat)
        if nb is None or nd is None:
            return None

        # Recompute with actual nb, nd
        lc = compute_level_witness(cfg, m_vars, r_vars, lb, log_cb, nb, nd,
                                   num_ring, self.tight_zpre)
        if lc.next_w_len >= w_len:
            return None

        prefix = self.level_prefix_bytes(cfg, lb, lc.rounds, nd)
        return (prefix, lc, nb, nd)

    def try_level(self, cfg, level, w_len, lb, log_cb):
        d = cfg.d
        alpha = (d & -d).bit_length() - 1

        if level == 0:
            rv = self.nv - alpha
            num_ring = 1 << rv
        else:
            nr = w_len // d
            p = 1
            while p < nr:
                p <<= 1
            rv = p.bit_length() - 1 if p > 0 else 0
            num_ring = nr

        nr_arg = num_ring if self.tight_zpre else 0
        m, r = optimal_m_r_split(cfg.n_a, cfg.l1_mass, log_cb, lb, rv, nr_arg)
        return self.try_level_mr(cfg, level, w_len, lb, log_cb, m, r)

    def tail_cost(self, w_len, d, tail_lb):
        """Tail = entry commitment (base ring) + packed digits."""
        ring_elems = -(-w_len // d)
        nb = min_rank_for_secure_width(d, (1 << tail_lb) - 1, ring_elems)
        if nb is None:
            return None
        return ring_vec_bytes_base(nb, d) + packed_digits_bytes(w_len, tail_lb)

    def best_from(self, w_len, cur_d, prev_lb):
        key = (w_len, cur_d, prev_lb)
        if key in self.memo:
            return self.memo[key]

        # Base case: terminate here
        tc = self.tail_cost(w_len, cur_d, prev_lb)
        best = (tc if tc is not None else float('inf'), [], prev_lb)

        # Try each config at this D
        for cfg in self.cfgs_for_d(cur_d):
            for lb in range(MIN_LB, MAX_LB + 1):
                result = self.try_level(cfg, 1, w_len, lb, prev_lb)
                if result is None:
                    continue
                prefix, lc, nb_self, nd_self = result
                entry_commit = ring_vec_bytes_base(nb_self, cur_d)

                for next_d in self.unique_ds:
                    if self.monotone_d and next_d > cur_d:
                        continue
                    suffix_cost, suffix_levels, suffix_lb = self.best_from(
                        lc.next_w_len, next_d, lb)
                    if suffix_cost == float('inf'):
                        continue
                    total = entry_commit + prefix + suffix_cost
                    if total < best[0]:
                        lvl = PlannedLevel(
                            d=cfg.d, lb=lb, m_vars=lc.m_vars, r_vars=lc.r_vars,
                            na=cfg.n_a, nb=nb_self, nd=nd_self,
                            delta_open=lc.delta_open, delta_fold=lc.delta_fold,
                            delta_commit=lc.delta_commit, w_ring=lc.w_ring_elems,
                            next_w_len=lc.next_w_len,
                            level_bytes=entry_commit + prefix, label=cfg.label,
                        )
                        best = (total, [lvl] + suffix_levels, suffix_lb)

        self.memo[key] = best
        return best

    def plan(self) -> Schedule:
        root_w = 1 << self.nv
        overall_best = None  # (total, levels, tail_lb)

        for cfg in self.cfgs:
            d = cfg.d
            alpha = (d & -d).bit_length() - 1
            rv = self.nv - alpha
            if rv <= 0:
                continue
            num_ring = 1 << rv

            for root_lb in range(MIN_LB, MAX_LB + 1):
                # Enumerate (m, r) splits at root
                nr_arg = num_ring if self.tight_zpre else 0
                opt_m, opt_r = optimal_m_r_split(
                    cfg.n_a, cfg.l1_mass, self.log_cb, root_lb, rv, nr_arg)

                for root_r in range(1, rv):
                    root_m = rv - root_r
                    # Prune: skip splits far from local optimum
                    if abs(root_r - opt_r) > 4:
                        continue

                    result = self.try_level_mr(
                        cfg, 0, root_w, root_lb, self.log_cb, root_m, root_r)
                    if result is None:
                        continue
                    prefix, lc, root_nb, root_nd = result
                    entry_commit = ring_vec_bytes_base(root_nb, d)

                    for next_d in self.unique_ds:
                        if self.monotone_d and next_d > d:
                            continue
                        suffix_cost, suffix_levels, suffix_lb = self.best_from(
                            lc.next_w_len, next_d, root_lb)
                        if suffix_cost == float('inf'):
                            continue
                        total = entry_commit + prefix + suffix_cost
                        if overall_best is None or total < overall_best[0]:
                            lvl = PlannedLevel(
                                d=d, lb=root_lb, m_vars=lc.m_vars, r_vars=lc.r_vars,
                                na=cfg.n_a, nb=root_nb, nd=root_nd,
                                delta_open=lc.delta_open, delta_fold=lc.delta_fold,
                                delta_commit=lc.delta_commit, w_ring=lc.w_ring_elems,
                                next_w_len=lc.next_w_len,
                                level_bytes=entry_commit + prefix, label=cfg.label,
                            )
                            overall_best = (total, [lvl] + suffix_levels, suffix_lb)

        if overall_best is None:
            return Schedule([], 0, 0, 0, 0)

        total, levels, tail_lb = overall_best
        final_w = levels[-1].next_w_len if levels else 0
        tail = packed_digits_bytes(final_w, tail_lb)
        return Schedule(levels=levels, tail_bytes=tail, total_bytes=total,
                        final_w_len=final_w, final_lb=tail_lb)


# ── Output ────────────────────────────────────────────────────────────────

def d_schedule(sched):
    return "->".join(str(l.d) for l in sched.levels)


def print_detailed(sched):
    for i, l in enumerate(sched.levels):
        print(f"    L{i}: D={l.d} lb={l.lb} m={l.m_vars} r={l.r_vars} [{l.label}]")
        print(f"        na={l.na} nb={l.nb} nd={l.nd}  "
              f"do={l.delta_open} df={l.delta_fold} dc={l.delta_commit}  "
              f"w_ring={l.w_ring}  next_w={l.next_w_len}  level={l.level_bytes}B")
    print(f"    TERMINAL: w={sched.final_w_len}  lb={sched.final_lb}  "
          f"tail={sched.tail_bytes}B")
    print(f"    TOTAL: {sched.total_bytes} B  ({sched.total_bytes/1024:.1f} KB)")


def main():
    print("=" * 80)
    print("Hachi Proof-Size Planner — 32-bit Prime (q = 2^32 - 99)")
    print(f"  field_bits={FIELD_BITS}, base_elem={BASE_ELEM_BYTES}B, ext_elem={EXT_ELEM_BYTES}B")
    print(f"  D ∈ {sorted(set(c.d for c in ALL_RING_CONFIGS))}")
    print(f"  SIS: 128-bit security (BDGL16+lgsa)")
    print("=" * 80)

    configs = [
        ("onehot", 1),
        ("dense-32bit", 32),
    ]
    NVS = [20, 25, 30, 32]

    # Summary table
    print(f"\n{'='*80}")
    print(f"  {'Poly':>12} {'nv':>4} {'total':>10} {'D schedule':<30} {'tail':>10}")
    print(f"  {'-'*12} {'-'*4} {'-'*10} {'-'*30} {'-'*10}")

    all_results = {}
    for name, lcb in configs:
        for nv in NVS:
            p = Planner(log_commit_bound=lcb, max_num_vars=nv)
            sched = p.plan()
            all_results[(name, nv)] = sched
            ds = d_schedule(sched) if sched.levels else "N/A"
            print(f"  {name:>12} {nv:>4} {sched.total_bytes:>10} {ds:<30} {sched.tail_bytes:>10}")

    # Detailed breakdowns
    print(f"\n{'='*80}")
    print("DETAILED BREAKDOWNS")
    print(f"{'='*80}")
    for name, lcb in configs:
        for nv in NVS:
            sched = all_results[(name, nv)]
            if not sched.levels:
                continue
            print(f"\n  {name} nv={nv}  ({len(sched.levels)} levels, "
                  f"{sched.total_bytes:,} B = {sched.total_bytes/1024:.1f} KB)")
            print_detailed(sched)

    # Comparison with 128-bit planner estimates
    print(f"\n{'='*80}")
    print("COMPARISON: 32-bit vs 128-bit field proof sizes")
    print("  (128-bit numbers from Rust universal planner)")
    print(f"{'='*80}")
    # Known 128-bit optimized results (from Rust planner output)
    ref_128 = {
        ("onehot", 20): 64224, ("onehot", 25): 70736,
        ("onehot", 30): 74800, ("onehot", 32): 75632,
    }
    print(f"  {'Poly':>12} {'nv':>4} {'32-bit':>10} {'128-bit':>10} {'ratio':>8}")
    print(f"  {'-'*12} {'-'*4} {'-'*10} {'-'*10} {'-'*8}")
    for name, lcb in configs:
        for nv in NVS:
            s32 = all_results[(name, nv)].total_bytes
            ref_key = (name, nv)
            if ref_key in ref_128:
                s128 = ref_128[ref_key]
                ratio = f"{s32/s128:.2f}x"
            else:
                s128 = "N/A"
                ratio = ""
            print(f"  {name:>12} {nv:>4} {s32:>10} {str(s128):>10} {ratio:>8}")


if __name__ == "__main__":
    main()
