#!/usr/bin/env python3
"""Hachi proof-size planner for 16-bit prime field (q = 2^16 - 99).

Exploratory DP planner that derives (n_a, n_b, n_d) from MSIS security
constraints, using the same proof-size model as the 32-bit and 64-bit
planners but with a 16-bit base field:

  - field_bits = 16, base field elements = 2 bytes
  - Sumcheck runs over degree-8 extension field F_{q^8}; extension
    elements are still 16 bytes, matching the 32/64/128-bit studies
  - log_open_bound = max(log_commit_bound, 16)
  - 2-splitting only (LS18-compatible q ≡ 5 mod 8)
  - Ring configs D ∈ {128, 256, 512}, shifted up from the 32-bit sweep

Security note:
  Degree-8 over q = 65437 gives about 127.98 bits of field soundness, so
  this script should be read as a "128-bit-ish" exploratory planner rather
  than a final strict 128-bit claim.

Two element sizes coexist in the proof:
  - BASE_ELEM_BYTES = 2: committed ring vectors, packed digits
  - EXT_ELEM_BYTES  = 16: sumcheck messages, claims, evaluations, y_ring
"""

import math
from dataclasses import dataclass
from typing import Optional

# ── Field parameters ──────────────────────────────────────────────────────

FIELD_BITS = 16
EXT_DEGREE = 8
BASE_ELEM_BYTES = 2
EXT_ELEM_BYTES = BASE_ELEM_BYTES * EXT_DEGREE
Q = (1 << FIELD_BITS) - 99
SUMCHECK_BITS = EXT_DEGREE * math.log2(Q)

# ── SIS width table: q = 2^16 - 99, 128-bit MSIS target (BDGL16+lgsa) ───
#
# These widths were freshly estimated with the lattice-estimator using the
# same methodology as the 32-bit large-D sweep, with the extra guard
# l2 < (q - 1) / 2 to avoid the trivial "short vector already wraps mod q"
# regime that becomes important at 16 bits.
#
# sis_max_widths[(D, collision_inf)] = [rank1, rank2, rank3, rank4]

SIS_MAX_WIDTHS = {
    # D=128
    (128,   2): [44, 2_936, 73_204, 1_088_459],
    (128,   3): [19, 1_305, 32_535, 483_759],
    (128,   7): [3, 239, 5_975, 88_853],
    (128,  15): [2, 52, 1_301, 19_350],
    (128,  31): [2, 12, 304, 4_530],
    (128,  63): [1, 4, 73, 1_096],
    (128, 127): [1, 3, 18, 269],
    (128, 255): [1, 3, 6, 66],
    # D=256
    (256,   2): [1_468, 544_229, 1_045_378, 1_045_378],
    (256,   3): [652, 241_879, 464_612, 464_612],
    (256,   7): [119, 44_426, 85_337, 85_337],
    (256,  15): [26, 9_675, 18_584, 18_584],
    (256,  31): [6, 2_265, 4_351, 4_351],
    (256,  63): [2, 548, 1_053, 1_053],
    (256, 127): [1, 134, 259, 259],
    (256, 255): [1, 33, 64, 64],
    # D=512
    (512,   2): [272_114, 522_689, 522_689, 522_689],
    (512,   3): [120_939, 232_306, 232_306, 232_306],
    (512,   7): [22_213, 42_668, 42_668, 42_668],
    (512,  15): [4_837, 9_292, 9_292, 9_292],
    (512,  31): [1_132, 2_175, 2_175, 2_175],
    (512,  63): [274, 526, 526, 526],
    (512, 127): [67, 129, 129, 129],
    (512, 255): [16, 32, 32, 32],
}

MAX_RANK = 4


def min_rank_for_secure_width(d: int, collision_inf: int, width: int) -> Optional[int]:
    """Smallest MSIS rank achieving the target security for the given width."""
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
    n = -(-log_bound // log_basis)
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

    open_bound = max(log_cb, FIELD_BITS)
    d_open = compute_num_digits(open_bound, log_basis)
    d_commit = compute_num_digits(log_cb, log_basis)
    per_block = d_open + n_a * d_open

    best_cost, best_r = float("inf"), reduced_vars // 2
    for r in range(1, reduced_vars):
        nb = 1 << r
        m_eff = -(-num_ring // nb) if num_ring > 0 else (1 << (reduced_vars - r))
        d_fold = compute_num_digits_fold(r, l1_mass, log_basis)
        cost = per_block * nb + d_commit * d_fold * m_eff
        if cost < best_cost:
            best_cost, best_r = cost, r
    return (reduced_vars - best_r, best_r)


# ── Proof size helpers ────────────────────────────────────────────────────

def ring_vec_bytes_base(ring_len, ring_dim):
    return ring_len * ring_dim * BASE_ELEM_BYTES


def ring_vec_bytes_ext(ring_len, ring_dim):
    return ring_len * ring_dim * EXT_ELEM_BYTES


def sumcheck_bytes(rounds, degree):
    return rounds * degree * EXT_ELEM_BYTES


def packed_digits_bytes(num_elems, bits_per_elem):
    return -(-(num_elems * bits_per_elem) // 8)


def stage1_bytes_optimized(n_rounds, lb):
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


ALL_RING_CONFIGS = [
    RingConfig(128, 1, 54, 2, "D128-na1"),
    RingConfig(128, 2, 54, 2, "D128-na2"),
    RingConfig(256, 1, 27, 1, "D256-na1"),
    RingConfig(256, 2, 27, 1, "D256-na2"),
    RingConfig(512, 1, 19, 1, "D512-na1"),
    RingConfig(512, 2, 19, 1, "D512-na2"),
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
    open_bound = max(log_cb, FIELD_BITS)
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


# ── Planner output types ──────────────────────────────────────────────────

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


# ── DP planner ────────────────────────────────────────────────────────────

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
        if self.opt_sumcheck:
            s1 = stage1_bytes_optimized(rounds, lb)
        else:
            deg = ((1 << lb) // 2) + 1
            s1 = sumcheck_bytes(rounds, deg)

        return (ring_vec_bytes_base(1, cfg.d)
                + ring_vec_bytes_base(nd, cfg.d)
                + s1
                + EXT_ELEM_BYTES
                + sumcheck_bytes(rounds, 3)
                + EXT_ELEM_BYTES)

    def try_level_mr(self, cfg, level, w_len, lb, log_cb, m_vars, r_vars):
        d = cfg.d
        alpha = (d & -d).bit_length() - 1

        if level == 0:
            num_ring = 1 << (self.nv - alpha)
        else:
            num_ring = w_len // d

        lc = compute_level_witness(cfg, m_vars, r_vars, lb, log_cb, 1, 1,
                                   num_ring, self.tight_zpre)
        if lc.next_w_len >= w_len:
            return None

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
        ring_elems = -(-w_len // d)
        nb = min_rank_for_secure_width(d, (1 << tail_lb) - 1, ring_elems)
        if nb is None:
            return None
        return ring_vec_bytes_base(nb, d) + packed_digits_bytes(w_len, tail_lb)

    def best_from(self, w_len, cur_d, prev_lb):
        key = (w_len, cur_d, prev_lb)
        if key in self.memo:
            return self.memo[key]

        tc = self.tail_cost(w_len, cur_d, prev_lb)
        best = (tc if tc is not None else float("inf"), [], prev_lb)

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
                    if suffix_cost == float("inf"):
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
        overall_best = None

        for cfg in self.cfgs:
            d = cfg.d
            alpha = (d & -d).bit_length() - 1
            rv = self.nv - alpha
            if rv <= 0:
                continue
            num_ring = 1 << rv

            for root_lb in range(MIN_LB, MAX_LB + 1):
                nr_arg = num_ring if self.tight_zpre else 0
                _, opt_r = optimal_m_r_split(
                    cfg.n_a, cfg.l1_mass, self.log_cb, root_lb, rv, nr_arg)

                for root_r in range(1, rv):
                    root_m = rv - root_r
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
                        if suffix_cost == float("inf"):
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


# ── Output helpers ────────────────────────────────────────────────────────

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
    print(f"Hachi Proof-Size Planner — 16-bit Prime (q = 2^16 - 99 = {Q})")
    print(f"  field_bits={FIELD_BITS}, base_elem={BASE_ELEM_BYTES}B, ext_elem={EXT_ELEM_BYTES}B")
    print(f"  ext_degree={EXT_DEGREE} (~{SUMCHECK_BITS:.2f}-bit-ish sumcheck field)")
    print(f"  D ∈ {sorted(set(c.d for c in ALL_RING_CONFIGS))}")
    print("  SIS: exploratory 128-bit target (BDGL16+lgsa), with l2 < (q-1)/2 cutoff")
    print("=" * 80)

    configs = [
        ("onehot", 1, [20, 25, 30, 32]),
        ("dense-16bit", 16, [20, 25, 30]),
    ]

    print(f"\n{'='*80}")
    print(f"  {'Poly':>12} {'nv':>4} {'total':>10} {'D schedule':<30} {'tail':>10}")
    print(f"  {'-'*12} {'-'*4} {'-'*10} {'-'*30} {'-'*10}")

    all_results = {}
    for name, lcb, nvs in configs:
        for nv in nvs:
            p = Planner(log_commit_bound=lcb, max_num_vars=nv)
            sched = p.plan()
            all_results[(name, nv)] = sched
            ds = d_schedule(sched) if sched.levels else "N/A"
            print(f"  {name:>12} {nv:>4} {sched.total_bytes:>10} {ds:<30} {sched.tail_bytes:>10}")

    print(f"\n{'='*80}")
    print("DETAILED BREAKDOWNS")
    print(f"{'='*80}")
    for name, _, nvs in configs:
        for nv in nvs:
            sched = all_results[(name, nv)]
            if not sched.levels:
                continue
            print(f"\n  {name} nv={nv}  ({len(sched.levels)} levels, "
                  f"{sched.total_bytes:,} B = {sched.total_bytes/1024:.1f} KB)")
            print_detailed(sched)

    print(f"\n{'='*80}")
    print("COMPARISON: 16-bit vs corrected 32-bit field proof sizes")
    print(f"{'='*80}")
    ref_32 = {
        ("onehot", 20): 38_960,
        ("onehot", 25): 45_056,
        ("onehot", 30): 48_208,
        ("onehot", 32): 49_568,
        ("dense-16bit", 20): 45_360,
        ("dense-16bit", 25): 48_016,
        ("dense-16bit", 30): 50_192,
    }
    print(f"  {'Poly':>12} {'nv':>4} {'16-bit':>10} {'32-bit':>10} {'ratio':>8}")
    print(f"  {'-'*12} {'-'*4} {'-'*10} {'-'*10} {'-'*8}")
    for name, _, nvs in configs:
        for nv in nvs:
            s16 = all_results[(name, nv)].total_bytes
            ref_key = (name, nv)
            if ref_key in ref_32:
                s32 = ref_32[ref_key]
                ratio = f"{s16/s32:.2f}x"
            else:
                s32 = "N/A"
                ratio = ""
            print(f"  {name:>12} {nv:>4} {s16:>10} {str(s32):>10} {ratio:>8}")


if __name__ == "__main__":
    main()
