#!/usr/bin/env python3
"""
Anatomy of the onehot tail: why can't it shrink further?

Dissects the terminal witness into its 4 sub-components and sweeps
every possible (m, r) split to show why the planner's choice is optimal
and where the floor comes from.
"""

import math

FIELD_BITS = 128
D = 64  # onehot uses D=64
ELEM_BYTES = 16
N_A = 1
N_B = 1
N_D = 1
HALF_FIELD_BOUND = (2**128 - 5823) // 2
CHALLENGE_L1_MASS = 54  # D=64 SplitRing


def compute_num_digits(log_bound, log_basis):
    if log_bound == 0 or log_basis == 0:
        return 1
    levels = math.ceil(log_bound / log_basis)
    total_bits = levels * log_basis
    if total_bits <= log_bound:
        b = 1 << log_basis
        half_b_minus_1 = b // 2 - 1
        b_minus_1 = b - 1
        b_pow = b ** levels
        max_positive = half_b_minus_1 * ((b_pow - 1) // b_minus_1)
        required = (1 << (log_bound - 1)) - 1 if log_bound <= 128 else (2**128 - 1) // 2
        if max_positive < required:
            levels += 1
    return max(levels, 1)


def compute_num_digits_fold(r_vars, log_basis):
    shift = r_vars + log_basis - 1
    if shift >= 127:
        return compute_num_digits(128, log_basis)
    beta = CHALLENGE_L1_MASS * (1 << shift)
    if beta == 0:
        return 1
    log_beta = beta.bit_length()
    return compute_num_digits(log_beta, log_basis)


def r_decomp_levels_for_bound(log_basis):
    levels = compute_num_digits(FIELD_BITS, log_basis)
    total_bits = levels * log_basis
    if total_bits <= FIELD_BITS:
        b = 1 << log_basis
        half_b_minus_1 = b // 2 - 1
        b_minus_1 = b - 1
        b_pow = b ** levels
        max_positive = half_b_minus_1 * ((b_pow - 1) // b_minus_1)
        if max_positive < HALF_FIELD_BOUND:
            levels += 1
    return levels


def m_row_count():
    return N_D + N_B + 2 + N_A  # = 5


def packed_digits_bytes(num_elems, bits_per_elem):
    return 8 + 1 + math.ceil(num_elems * bits_per_elem / 8)


def witness_anatomy(m_vars, r_vars, log_basis, label=""):
    num_blocks = 1 << r_vars
    block_len = 1 << m_vars

    delta_commit = compute_num_digits(log_basis, log_basis)  # recursive: commit_bound = lb
    delta_open = compute_num_digits(128, log_basis)
    delta_fold = compute_num_digits_fold(r_vars, log_basis)
    r_levels = r_decomp_levels_for_bound(log_basis)
    inner_width = block_len * delta_commit

    w_hat = num_blocks * delta_open
    t_hat = num_blocks * N_A * delta_open
    z_pre = inner_width * delta_fold
    r_ct = m_row_count() * r_levels

    w_ring = w_hat + t_hat + z_pre + r_ct
    next_w_len = w_ring * D
    tail_bytes = packed_digits_bytes(next_w_len, log_basis)
    tail_bits = next_w_len * log_basis

    return {
        "m": m_vars, "r": r_vars, "lb": log_basis,
        "num_blocks": num_blocks, "block_len": block_len,
        "delta_commit": delta_commit, "delta_open": delta_open,
        "delta_fold": delta_fold, "r_levels": r_levels,
        "inner_width": inner_width,
        "w_hat": w_hat, "t_hat": t_hat, "z_pre": z_pre, "r_ct": r_ct,
        "w_ring": w_ring, "next_w_len": next_w_len,
        "tail_bytes": tail_bytes, "tail_bits": tail_bits,
    }


def bar(value, max_val, width=40):
    filled = int(value / max_val * width) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)


def print_anatomy(a, label=""):
    if label:
        print(f"\n  {label}")
    print(f"    m={a['m']}, r={a['r']}, lb={a['lb']}")
    print(f"    blocks=2^{a['r']}={a['num_blocks']}, "
          f"block_len=2^{a['m']}={a['block_len']}")
    print(f"    δ_commit={a['delta_commit']}, δ_open={a['delta_open']}, "
          f"δ_fold={a['delta_fold']}, r_levels={a['r_levels']}")
    print()

    total = a['w_ring']
    terms = [
        ("ŵ  = 2^r × δ_open", a['w_hat'],
         f"{a['num_blocks']} × {a['delta_open']}"),
        ("t̂  = 2^r × n_a × δ_open", a['t_hat'],
         f"{a['num_blocks']} × {N_A} × {a['delta_open']}"),
        ("z' = 2^m × δ_c × δ_fold", a['z_pre'],
         f"{a['block_len']} × {a['delta_commit']} × {a['delta_fold']}"),
        ("r  = 5 × r_levels", a['r_ct'],
         f"5 × {a['r_levels']}"),
    ]
    print(f"    Witness ring elements = {total:,}")
    for name, val, formula in terms:
        pct = val / total * 100
        print(f"      {name:<30} = {formula:<20} = {val:>8,}  "
              f"({pct:>5.1f}%)  {bar(val, total, 30)}")

    print(f"\n    next_w_len = {total:,} × {D} = {a['next_w_len']:,} coefficients")
    print(f"    tail = {a['next_w_len']:,} × {a['lb']} bits = "
          f"{a['tail_bits']:,} bits = {a['tail_bytes']:,} bytes")


def main():
    print("=" * 72)
    print("  ANATOMY OF THE ONEHOT TAIL")
    print("  Why can't the terminal witness shrink further?")
    print("=" * 72)

    # ── Part 1: What the current terminal level looks like ──
    print("\n\n" + "━" * 72)
    print("  PART 1: CURRENT TERMINAL LEVEL (onehot nv=32)")
    print("━" * 72)
    print("""
  The recursive folding in onehot nv=32 bottoms out at:
    86,144 coefficients → 86,144 / 64 = 1,346 ring elements

  The planner chose m=7, r=4, lb=5 for this last level (L5).
  After folding, the OUTPUT of L5 becomes the tail witness.
  Let's see what that output looks like:
""")
    a = witness_anatomy(7, 4, 5)
    print_anatomy(a, "L5 (final level) → produces the tail")

    print(f"\n  This tail of {a['tail_bytes']:,} bytes is 54% of the 99,805 B total proof.")

    # ── Part 2: What if we tried different m/r splits? ──
    print("\n\n" + "━" * 72)
    print("  PART 2: SWEEPING ALL (m, r) SPLITS AT THE TERMINAL LEVEL")
    print("━" * 72)
    print("""
  The terminal level has ~1,346 ring elements → needs ~11 variables
  (2^11 = 2048 ≥ 1346). With D=64, α=6, so reduced_vars = 11.
  We can split 11 into any (m, r) with m+r = 11.
  For each split, what tail would we get?
""")

    for lb in [3, 4, 5, 8, 13, 19]:
        print(f"\n  log_basis = {lb}:")
        print(f"  {'m':>3} {'r':>3}  {'ŵ':>8} {'t̂':>8} {'z_pre':>8} "
              f"{'r_ct':>6}  {'w_ring':>8}  {'coeff':>10}  {'tail_B':>8}  "
              f"{'tail_bits':>10}")
        print(f"  {'-'*3} {'-'*3}  {'-'*8} {'-'*8} {'-'*8} "
              f"{'-'*6}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")

        reduced_vars = 11
        best_tail = float('inf')
        best_m = 0
        for r in range(1, reduced_vars):
            m = reduced_vars - r
            a = witness_anatomy(m, r, lb)
            marker = ""
            if a['tail_bytes'] < best_tail:
                best_tail = a['tail_bytes']
                best_m = m
                marker = " ◄"
            print(f"  {m:>3} {r:>3}  {a['w_hat']:>8,} {a['t_hat']:>8,} "
                  f"{a['z_pre']:>8,} {a['r_ct']:>6,}  {a['w_ring']:>8,}  "
                  f"{a['next_w_len']:>10,}  {a['tail_bytes']:>8,}  "
                  f"{a['tail_bits']:>10,}{marker}")

    # ── Part 3: The z_pre trap explained ──
    print("\n\n" + "━" * 72)
    print("  PART 3: THE z_pre TRAP — WHY THE FLOOR EXISTS")
    print("━" * 72)
    print("""
  The witness has 4 sub-components. Three scale with 2^r, one with 2^m:

    ŵ     = 2^r × δ_open          ← scales with r (number of blocks)
    t̂     = 2^r × δ_open          ← scales with r
    z_pre = 2^m × δ_commit × δ_fold  ← scales with m (block size)!!
    r_ct  = 5 × r_levels          ← constant (tiny)

  Since m + r = reduced_vars (fixed), increasing r decreases m.
  ŵ and t̂ grow exponentially in r, z_pre grows exponentially in m.
  The optimum is where these opposing forces balance.

  But here's the trap: z_pre involves δ_fold, which ALSO depends on r!
    δ_fold = ceil(log2(ω × 2^(r + lb - 1)) / lb)
  As r increases, δ_fold increases, making z_pre grow even when m shrinks.

  The result is a U-shaped curve: small r → z_pre dominates,
  large r → ŵ + t̂ dominate. The minimum is the floor.
""")

    # Show the U-shape for lb=5
    print("  U-SHAPE for lb=5, reduced_vars=11:")
    print()
    reduced_vars = 11
    lb = 5
    results = []
    for r in range(1, reduced_vars):
        m = reduced_vars - r
        a = witness_anatomy(m, r, lb)
        results.append(a)

    max_ring = max(a['w_ring'] for a in results)
    min_ring = min(a['w_ring'] for a in results)

    for a in results:
        m, r = a['m'], a['r']
        w_ring = a['w_ring']
        wt = a['w_hat'] + a['t_hat']
        zp = a['z_pre']
        is_min = "◄ MIN" if w_ring == min_ring else ""
        scale = 50
        wt_bar = int(wt / max_ring * scale)
        zp_bar = int(zp / max_ring * scale)
        rc_bar = max(1, int(a['r_ct'] / max_ring * scale))
        print(f"  m={m:>2} r={r:>2}  "
              f"{'▓' * wt_bar}{'░' * zp_bar}{'·' * rc_bar} "
              f" {w_ring:>8,} ring elems  {is_min}")

    print()
    print(f"  Legend: ▓ = ŵ + t̂ (grows with r)   ░ = z_pre (grows with m)   · = r_ct")

    # ── Part 4: Comparing across lb values ──
    print("\n\n" + "━" * 72)
    print("  PART 4: OPTIMAL TAIL AT EACH log_basis")
    print("━" * 72)
    print("""
  For each lb, we find the best (m,r) split and show the minimum tail.
  The tail_bytes = w_ring × D × lb / 8.  Even if w_ring shrinks with
  larger lb (fewer δ_open digits), the wider bits per element fight back.
""")

    print(f"  {'lb':>4}  {'best_m':>6} {'best_r':>6}  {'δ_o':>4} {'δ_f':>4}  "
          f"{'w_ring':>8}  {'coeffs':>10}  {'bits':>12}  {'tail_B':>8}  "
          f"{'vs_lb5':>8}")

    lb5_best = None
    for lb in range(2, 33):
        reduced_vars = 11
        best = None
        for r in range(1, reduced_vars):
            m = reduced_vars - r
            a = witness_anatomy(m, r, lb)
            if best is None or a['tail_bytes'] < best['tail_bytes']:
                best = a
        if lb == 5:
            lb5_best = best
        delta_str = ""
        if lb5_best:
            diff = best['tail_bytes'] - lb5_best['tail_bytes']
            delta_str = f"{diff:>+8,}"
        print(f"  {lb:>4}  {best['m']:>6} {best['r']:>6}  "
              f"{best['delta_open']:>4} {best['delta_fold']:>4}  "
              f"{best['w_ring']:>8,}  {best['next_w_len']:>10,}  "
              f"{best['tail_bits']:>12,}  {best['tail_bytes']:>8,}  "
              f"{delta_str}")

    # ── Part 5: The bit-product ceiling ──
    print("\n\n" + "━" * 72)
    print("  PART 5: WHY CAN'T JL BREAK THROUGH THE FLOOR?")
    print("━" * 72)
    print("""
  With JL, we remove stage 1 (the range check), so lb can be anything.
  But the tail is:
    tail_bits = w_ring × D × lb

  Even though w_ring shrinks with larger lb (fewer digits), the product
  w_ring × lb has a FLOOR because:

    w_ring ≈ 2 × 2^r × δ_open + 2^m × δ_fold
           ≈ 2 × 2^r × ceil(128/lb) + 2^m × ceil(log_beta/lb)

  The first term's contribution to tail_bits is:
    2 × 2^r × ceil(128/lb) × D × lb ≈ 2 × 2^r × 128 × D  (constant!)

  That is, the ŵ + t̂ contribution to tail BITS is roughly constant
  regardless of lb — because ceil(128/lb) × lb ≈ 128.

  The z_pre contribution to tail bits:
    2^m × δ_commit × δ_fold × D × lb

  For recursive levels, δ_commit = ceil(lb/lb) = 1, so:
    2^m × δ_fold × D × lb ≈ 2^m × log_beta × D  (also roughly constant)

  So the tail BIT COUNT has a floor around:
    2 × 2^r × 128 × D + 2^m × log_beta × D

  This is ~independent of lb!  Larger lb just moves bits between
  "fewer elements" and "wider elements" without reducing the product.
""")

    print("  Tail BIT count at optimal (m,r) for each lb:")
    print(f"  {'lb':>4}  {'w_ring':>8}  {'tail_bits':>12}  {'ratio_vs_lb3':>12}")
    for lb in range(2, 33):
        reduced_vars = 11
        best = None
        for r in range(1, reduced_vars):
            m = reduced_vars - r
            a = witness_anatomy(m, r, lb)
            if best is None or a['tail_bytes'] < best['tail_bytes']:
                best = a
        lb3 = witness_anatomy(7, 4, 3)
        ratio = best['tail_bits'] / lb3['tail_bits']
        print(f"  {lb:>4}  {best['w_ring']:>8,}  {best['tail_bits']:>12,}  {ratio:>12.2f}×")


if __name__ == "__main__":
    main()
