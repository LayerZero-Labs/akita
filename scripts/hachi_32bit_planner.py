#!/usr/bin/env python3
"""Hachi proof-size planner for 32-bit and 128-bit prime fields.

Simulates the recursive folding planner from schedule.rs to produce
proof-size estimates at both field sizes.

Output:
  - Per-level breakdown showing witness shrinkage and proof-byte budget
  - Comparison table across nv in {20, 25, 30, 32}
"""

import math

# ---------------------------------------------------------------------------
# Helpers matching Rust u128 saturating arithmetic
# ---------------------------------------------------------------------------

U128_MAX = (1 << 128) - 1


def sat_mul(a: int, b: int) -> int:
    return min(a * b, U128_MAX)


def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def trailing_zeros(n: int) -> int:
    if n == 0:
        return 0
    return (n & -n).bit_length() - 1


# ---------------------------------------------------------------------------
# Core functions (faithful translations from config.rs / schedule.rs)
# ---------------------------------------------------------------------------


def compute_num_digits(log_bound: int, log_basis: int) -> int:
    """config.rs: gadget decomposition depth delta."""
    assert 0 < log_basis < 128
    if log_bound == 0:
        return 1
    levels = -(-log_bound // log_basis)
    total_bits = levels * log_basis
    if total_bits <= log_bound:
        b = 1 << log_basis
        b_pow = 1
        for _ in range(levels):
            b_pow = sat_mul(b_pow, b)
        max_pos = sat_mul(
            b // 2 - 1, (max(b_pow, 1) - 1) // (b - 1)
        )
        if log_bound > 128:
            required = U128_MAX // 2
        else:
            required = (1 << (log_bound - 1)) - 1
        if max_pos < required:
            levels += 1
    return max(levels, 1)


def compute_num_digits_fold(r_vars: int, l1_mass: int, log_basis: int) -> int:
    """config.rs: folded-witness decomposition depth tau."""
    shift = r_vars + log_basis - 1
    if shift >= 127 or l1_mass == 0:
        return compute_num_digits(128, log_basis)
    beta = l1_mass * (1 << shift)
    if beta == 0:
        return 1
    log_beta = beta.bit_length()
    return compute_num_digits(log_beta, log_basis)


def recursive_r_decomp(field_bits: int, half_field_bound: int,
                        log_basis: int) -> int:
    """schedule.rs: decomposition levels for the r vectors."""
    levels = compute_num_digits(field_bits, log_basis)
    if levels == 0:
        levels = 1
    total_bits = levels * log_basis
    if total_bits <= field_bits:
        b = 1 << log_basis
        b_pow = 1
        for _ in range(levels):
            b_pow = sat_mul(b_pow, b)
        max_pos = sat_mul(
            b // 2 - 1, (max(b_pow, 1) - 1) // (b - 1)
        )
        if max_pos < half_field_bound:
            levels += 1
    return levels


def packed_digits_bytes(num_elems: int, log_basis: int) -> int:
    """schedule.rs: tail size in bytes."""
    return -(-(num_elems * log_basis) // 8)


def sumcheck_rounds(d: int, next_w_len: int) -> int:
    """schedule.rs: total sumcheck round count."""
    num_l = trailing_zeros(d)
    num_ring_elems = next_w_len // d
    padded = next_pow2(num_ring_elems)
    num_u = padded.bit_length() - 1 if padded > 0 else 0
    return num_u + num_l


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class FieldConfig:
    def __init__(self, name, D, log_commit_bound, log_open_bound,
                 n_a, n_b, n_d, l1_mass, half_field_bound, field_bits,
                 min_lb=2, max_lb=5):
        self.name = name
        self.D = D
        self.log_commit_bound = log_commit_bound
        self.log_open_bound = log_open_bound
        self.n_a = n_a
        self.n_b = n_b
        self.n_d = n_d
        self.l1_mass = l1_mass
        self.half_field_bound = half_field_bound
        self.field_bits = field_bits
        self.field_bytes = -(-field_bits // 8)
        self.min_lb = min_lb
        self.max_lb = max_lb

    def m_row_count(self):
        return self.n_d + self.n_b + 2 + self.n_a


# ---------------------------------------------------------------------------
# Planner functions
# ---------------------------------------------------------------------------


def optimal_m_r_split(cfg, lcb, lob, lb, S):
    """config.rs: find (m, r) minimizing witness expansion."""
    if S <= 2 or S >= 53:
        r = S // 2
        return (S - r, r)
    d_open = compute_num_digits(lob, lb)
    d_commit = compute_num_digits(lcb, lb)
    c1 = d_open + cfg.n_a * d_open
    best_r, best_cost = S // 2, 1 << 63
    for r in range(1, S):
        m = S - r
        df = compute_num_digits_fold(r, cfg.l1_mass, lb)
        cost = c1 * (1 << r) + d_commit * df * (1 << m)
        if cost < best_cost:
            best_cost = cost
            best_r = r
    return (S - best_r, best_r)


def build_layout(cfg, m, r, lcb, lob, lb):
    """Construct layout dict from split parameters."""
    return {
        'm': m,
        'r': r,
        'nb': 1 << r,
        'bl': 1 << m,
        'iw': (1 << m) * compute_num_digits(lcb, lb),
        'ndc': compute_num_digits(lcb, lb),
        'ndo': compute_num_digits(lob, lb),
        'ndf': compute_num_digits_fold(r, cfg.l1_mass, lb),
        'lb': lb,
    }


def w_re_count(cfg, ly):
    """schedule.rs: next-witness ring-element count and breakdown."""
    w_hat = ly['nb'] * ly['ndo']
    t_hat = ly['nb'] * cfg.n_a * ly['ndo']
    z_pre = ly['iw'] * ly['ndf']
    rd = recursive_r_decomp(cfg.field_bits, cfg.half_field_bound, ly['lb'])
    r_count = cfg.m_row_count() * rd
    return w_hat + t_hat + z_pre + r_count, w_hat, t_hat, z_pre, r_count, rd


def lv_proof_bytes(cfg, ly, nw):
    """schedule.rs: proof bytes for one folding level."""
    fb = cfg.field_bytes
    rounds = sumcheck_rounds(cfg.D, nw)
    b = 1 << ly['lb']
    s1d = b // 2 + 1
    return (cfg.D * fb                    # y_ring
            + cfg.n_d * cfg.D * fb        # v
            + rounds * s1d * fb           # stage1 sumcheck
            + fb                          # s_claim
            + rounds * 3 * fb             # stage2 sumcheck
            + cfg.n_b * cfg.D * fb        # next_w_commitment
            + fb)                         # next_w_eval


def lv_proof_breakdown(cfg, ly, nw):
    """Detailed byte breakdown for one level."""
    fb = cfg.field_bytes
    rounds = sumcheck_rounds(cfg.D, nw)
    b = 1 << ly['lb']
    s1d = b // 2 + 1
    y = cfg.D * fb
    v = cfg.n_d * cfg.D * fb
    s1 = rounds * s1d * fb
    sc = fb
    s2 = rounds * 3 * fb
    nc = cfg.n_b * cfg.D * fb
    ne = fb
    return dict(y=y, v=v, s1=s1, sc=sc, s2=s2, nc=nc, ne=ne,
                rounds=rounds, s1d=s1d, total=y+v+s1+sc+s2+nc+ne)


# ---------------------------------------------------------------------------
# DP planner (matches planned_schedule + best_recursive_suffix)
# ---------------------------------------------------------------------------


def plan_schedule(cfg, nv):
    alpha = trailing_zeros(cfg.D)
    root_w = 1 << nv
    memo = {}

    def suffix(level, w, lb):
        key = (level, w, lb)
        if key in memo:
            return memo[key]
        tail = packed_digits_bytes(w, lb)
        best = ([], tail, w, lb)

        lcb, lob = lb, cfg.log_open_bound
        re = w // cfg.D
        if re == 0:
            memo[key] = best
            return best
        p = next_pow2(re)
        S = p.bit_length() - 1 if p > 1 else 0
        if S <= 0:
            memo[key] = best
            return best

        m, r = optimal_m_r_split(cfg, lcb, lob, lb, S)
        ly = build_layout(cfg, m, r, lcb, lob, lb)
        tot, *_ = w_re_count(cfg, ly)
        nw = tot * cfg.D
        if nw >= w:
            memo[key] = best
            return best

        lb_bytes = lv_proof_bytes(cfg, ly, nw)
        for nlb in range(max(lb, cfg.min_lb), cfg.max_lb + 1):
            sl, sb, sw, slb = suffix(level + 1, nw, nlb)
            cand = lb_bytes + sb
            if cand < best[1]:
                ent = dict(lv=level, w=w, lb=lb, m=m, r=r, ly=ly,
                           nw=nw, bytes=lb_bytes)
                best = ([ent] + sl, cand, sw, slb)

        memo[key] = best
        return best

    overall = None
    for rlb in range(cfg.min_lb, cfg.max_lb + 1):
        lcb, lob = cfg.log_commit_bound, cfg.log_open_bound
        S = nv - alpha
        if S <= 0:
            continue
        m, r = optimal_m_r_split(cfg, lcb, lob, rlb, S)
        ly = build_layout(cfg, m, r, lcb, lob, rlb)
        tot, *_ = w_re_count(cfg, ly)
        nw = tot * cfg.D
        lb_bytes = lv_proof_bytes(cfg, ly, nw)

        for nlb in range(max(rlb, cfg.min_lb), cfg.max_lb + 1):
            if nw < root_w:
                sl, sb, sw, slb = suffix(1, nw, nlb)
                ct = lb_bytes + sb
                slevels = sl
            else:
                t = packed_digits_bytes(nw, nlb)
                ct = lb_bytes + t
                sw, slb = nw, nlb
                slevels = []
            if overall is None or ct < overall[1]:
                ent = dict(lv=0, w=root_w, lb=rlb, m=m, r=r, ly=ly,
                           nw=nw, bytes=lb_bytes)
                overall = ([ent] + slevels, ct, sw, slb)

    if overall is None:
        raise ValueError(f"No valid schedule for nv={nv}")
    levels, total, term_w, term_lb = overall
    return dict(levels=levels, total=total, term_w=term_w, term_lb=term_lb)


# ---------------------------------------------------------------------------
# Concrete configs
# ---------------------------------------------------------------------------

FP128_MOD = (1 << 128) - 5823
FP128_HALF = FP128_MOD // 2

CFG_128 = FieldConfig(
    name='128-bit OneHot', D=64,
    log_commit_bound=1, log_open_bound=128,
    n_a=1, n_b=1, n_d=1, l1_mass=54,
    half_field_bound=FP128_HALF, field_bits=128,
)

# MSIS security constraints at q ~ 2^32 (from lattice estimator):
#   beta=1000:   min n=128 → D=64,n_a=2 or D=128,n_a=1
#   beta=10000:  min n=256 → D=64,n_a=4 or D=128,n_a=2 or D=256,n_a=1
#   beta=100000: min n=384 → D=64,n_a=6 or D=128,n_a=3 or D=256,n_a=2
#   Realistic tail beta~233K: D=64,n_a=7 or D=128,n_a=4 or D=256,n_a=2
#
# We set n_b=n_d=n_a for consistency (as in Lantern's kappa_MSIS).

# --- Configs with correct MSIS security ---

CFG_32_D64 = FieldConfig(
    name='32b D=64 na=4', D=64,
    log_commit_bound=32, log_open_bound=32,
    n_a=4, n_b=4, n_d=4, l1_mass=54,
    half_field_bound=1 << 31, field_bits=32,
)

CFG_32_D128 = FieldConfig(
    name='32b D=128 na=2', D=128,
    log_commit_bound=32, log_open_bound=32,
    n_a=2, n_b=2, n_d=2, l1_mass=31,
    half_field_bound=1 << 31, field_bits=32,
)

CFG_32_D256 = FieldConfig(
    name='32b D=256 na=1', D=256,
    log_commit_bound=32, log_open_bound=32,
    n_a=1, n_b=1, n_d=1, l1_mass=31,
    half_field_bound=1 << 31, field_bits=32,
)

# Also test the aggressive D=128,na=4 for realistic beta~233K
CFG_32_D128_na4 = FieldConfig(
    name='32b D=128 na=4', D=128,
    log_commit_bound=32, log_open_bound=32,
    n_a=4, n_b=4, n_d=4, l1_mass=31,
    half_field_bound=1 << 31, field_bits=32,
)

# And D=256,na=2 for realistic beta
CFG_32_D256_na2 = FieldConfig(
    name='32b D=256 na=2', D=256,
    log_commit_bound=32, log_open_bound=32,
    n_a=2, n_b=2, n_d=2, l1_mass=31,
    half_field_bound=1 << 31, field_bits=32,
)

NV_LIST = [20, 25, 30, 32]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_decomp_info(cfg):
    print(f"\n  Config: {cfg.name}")
    print(f"    D={cfg.D}, field_bits={cfg.field_bits}, "
          f"field_bytes={cfg.field_bytes}")
    print(f"    log_commit_bound={cfg.log_commit_bound}, "
          f"log_open_bound={cfg.log_open_bound}")
    print(f"    n_a={cfg.n_a}, n_b={cfg.n_b}, n_d={cfg.n_d}, "
          f"l1_mass={cfg.l1_mass}")
    hfb_bits = math.log2(cfg.half_field_bound) if cfg.half_field_bound > 0 else 0
    print(f"    half_field_bound ~ 2^{hfb_bits:.1f}")
    print(f"    {'lb':>4} | {'d_open':>6} | {'d_c_root':>8} | "
          f"{'d_c_rec':>7} | {'r_decomp':>8}")
    print(f"    {'----':>4}-+-{'------':>6}-+-{'--------':>8}-+-"
          f"{'-------':>7}-+-{'--------':>8}")
    for lb in range(cfg.min_lb, cfg.max_lb + 1):
        ndo = compute_num_digits(cfg.log_open_bound, lb)
        ndc_root = compute_num_digits(cfg.log_commit_bound, lb)
        ndc_rec = compute_num_digits(lb, lb)
        rd = recursive_r_decomp(cfg.field_bits, cfg.half_field_bound, lb)
        print(f"    {lb:>4} | {ndo:>6} | {ndc_root:>8} | "
              f"{ndc_rec:>7} | {rd:>8}")


def print_schedule(cfg, nv, result):
    n_lv = len(result['levels'])
    print(f"\n{'='*95}")
    print(f"nv={nv}  config={cfg.name}  "
          f"({n_lv} levels, {result['total']:,} bytes = "
          f"{result['total']/1024:.1f} KB)")
    print(f"{'='*95}")

    for entry in result['levels']:
        ly = entry['ly']
        ratio = entry['nw'] / entry['w']
        tot, wh, th, zp, rc, rd = w_re_count(cfg, ly)
        print(f"  L{entry['lv']}: w={entry['w']:<12,}  lb={ly['lb']}  "
              f"m={entry['m']:2}  r={entry['r']:2}  "
              f"dc={ly['ndc']:2}  do={ly['ndo']:2}  df={ly['ndf']:2}")
        print(f"      wh={wh:7}  th={th:7}  zp={zp:7}  "
              f"rc={rc:3}(m_row={cfg.m_row_count()},rd={rd})  "
              f"total_RE={tot:7}")
        print(f"      -> next_w={entry['nw']:<12,}  "
              f"shrink={ratio:.4f}  lv_bytes={entry['bytes']:,}")
        bd = lv_proof_breakdown(cfg, ly, entry['nw'])
        print(f"         [y={bd['y']} v={bd['v']} "
              f"s1={bd['s1']}({bd['rounds']}r*{bd['s1d']}d) "
              f"sc={bd['sc']} s2={bd['s2']} "
              f"nc={bd['nc']} ne={bd['ne']}]")

    tail = packed_digits_bytes(result['term_w'], result['term_lb'])
    total_lv = sum(e['bytes'] for e in result['levels'])
    print(f"\n  TERMINAL: w_len={result['term_w']:,}  lb={result['term_lb']}  "
          f"tail={tail:,} bytes ({tail/1024:.1f} KB)")
    print(f"  TOTAL: levels={total_lv:,} + tail={tail:,} = "
          f"{result['total']:,} bytes ({result['total']/1024:.1f} KB)")


def main():
    print("Hachi Proof-Size Planner: 32-bit vs 128-bit Field Comparison")
    print("=" * 70)

    for cfg in [CFG_32_D64, CFG_32_D128, CFG_32_D256, CFG_128]:
        print_decomp_info(cfg)

    ALL_CFGS = [CFG_32_D64, CFG_32_D128, CFG_32_D256,
                 CFG_32_D128_na4, CFG_32_D256_na2, CFG_128]

    results = {}
    for nv in NV_LIST:
        for cfg in ALL_CFGS:
            try:
                result = plan_schedule(cfg, nv)
                print_schedule(cfg, nv, result)
                results[(nv, cfg.name)] = result
            except Exception as e:
                print(f"\nERROR for nv={nv}, {cfg.name}: {e}")

    # Summary comparison table
    print(f"\n\n{'='*110}")
    print("COMPARISON TABLE (with correct MSIS security)")
    print(f"{'='*110}")
    hdr = (f"{'nv':>4} | {'Config':>20} | {'D':>4} | {'na':>3} | {'Lvls':>4} | "
           f"{'Term w_len':>12} | {'Tail KB':>8} | {'Lvl KB':>8} | {'Total KB':>8}")
    print(hdr)
    sep = "-" * len(hdr)
    print(sep)

    for nv in NV_LIST:
        for cfg in ALL_CFGS:
            key = (nv, cfg.name)
            if key not in results:
                continue
            r = results[key]
            tail = packed_digits_bytes(r['term_w'], r['term_lb'])
            total_lv = sum(e['bytes'] for e in r['levels'])
            print(f"{nv:>4} | {cfg.name:>20} | {cfg.D:>4} | {cfg.n_a:>3} | "
                  f"{len(r['levels']):>4} | "
                  f"{r['term_w']:>12,} | "
                  f"{tail/1024:>8.1f} | {total_lv/1024:>8.1f} | "
                  f"{r['total']/1024:>8.1f}")
        print(sep)


if __name__ == '__main__':
    main()
