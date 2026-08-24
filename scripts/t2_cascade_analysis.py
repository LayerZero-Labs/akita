#!/usr/bin/env python3
"""
T2 Witness Cascade Analysis

Computes the downstream witness blowup when Technique 2 (claim-reduction
sumcheck) is applied at one or more Hachi folding levels.

Uses hardcoded baseline schedules from the planner (already validated)
to avoid re-running the slow DP search.
"""

import math
from dataclasses import dataclass

from hachi_proof_size_planner import (
    HALF_FIELD_BOUND_P275,
    SIS_MAX_WIDTH_BY_D_AND_COLLISION,
    compute_num_digits,
    compute_num_digits_fold,
    min_rank_for_secure_width,
    r_decomp_levels,
)

HALF_FIELD_BOUND = HALF_FIELD_BOUND_P275


@dataclass
class LevelParams:
    D: int
    lb: int
    m: int
    r: int
    na: int
    nb: int
    nd: int
    l1_mass: int
    next_w_len: int


BASELINE_SCHEDULES = {
    32: [
        LevelParams(D=32, lb=2, m=16, r=11, na=3, nb=2, nd=2, l1_mass=256, next_w_len=0),
        LevelParams(D=16, lb=2, m=14, r=8, na=2, nb=3, nd=3, l1_mass=2048, next_w_len=0),
        LevelParams(D=16, lb=2, m=12, r=6, na=2, nb=2, nd=2, l1_mass=2048, next_w_len=0),
        LevelParams(D=16, lb=3, m=10, r=6, na=2, nb=3, nd=3, l1_mass=2048, next_w_len=0),
        LevelParams(D=16, lb=4, m=9, r=5, na=2, nb=3, nd=3, l1_mass=2048, next_w_len=0),
    ],
    38: [
        LevelParams(D=32, lb=2, m=20, r=13, na=2, nb=2, nd=2, l1_mass=256, next_w_len=0),
        LevelParams(D=16, lb=2, m=15, r=10, na=2, nb=3, nd=3, l1_mass=2048, next_w_len=0),
        LevelParams(D=16, lb=2, m=12, r=7, na=2, nb=2, nd=2, l1_mass=2048, next_w_len=0),
        LevelParams(D=16, lb=3, m=10, r=6, na=2, nb=3, nd=3, l1_mass=2048, next_w_len=0),
        LevelParams(D=16, lb=4, m=9, r=5, na=2, nb=3, nd=3, l1_mass=2048, next_w_len=0),
    ],
    44: [
        LevelParams(D=64, lb=2, m=21, r=17, na=1, nb=2, nd=2, l1_mass=54, next_w_len=0),
        LevelParams(D=32, lb=2, m=16, r=11, na=2, nb=2, nd=2, l1_mass=256, next_w_len=0),
        LevelParams(D=16, lb=2, m=14, r=8, na=2, nb=3, nd=3, l1_mass=2048, next_w_len=0),
        LevelParams(D=16, lb=2, m=12, r=6, na=2, nb=2, nd=2, l1_mass=2048, next_w_len=0),
        LevelParams(D=16, lb=3, m=10, r=6, na=2, nb=3, nd=3, l1_mass=2048, next_w_len=0),
    ],
}


def compute_next_w(
    D: int, lb: int, m: int, r: int, na: int, nb: int, nd: int,
    l1_mass: int, num_ring: int, delta_commit: int,
) -> dict:
    """Compute next witness size for one polynomial commitment."""
    delta_open = compute_num_digits(128, lb)
    delta_fold = compute_num_digits_fold(r, l1_mass, lb)
    delta_128 = r_decomp_levels(lb, HALF_FIELD_BOUND)

    num_blocks = 1 << r
    m_eff = -(-num_ring // num_blocks)

    w_hat = num_blocks * delta_open
    t_hat = num_blocks * na * delta_open
    z_pre = m_eff * delta_commit * delta_fold
    m_row = nd + nb + 2 + na
    r_ct = m_row * delta_128
    total = w_hat + t_hat + z_pre + r_ct

    return {
        "w_hat": w_hat, "t_hat": t_hat, "z_pre": z_pre, "r_ct": r_ct,
        "total_ring": total, "next_w_len": total * D,
        "m_eff": m_eff, "delta_open": delta_open, "delta_commit": delta_commit,
        "delta_fold": delta_fold, "delta_128": delta_128,
        "num_blocks": num_blocks, "m_row": m_row,
    }


def compute_baseline_next_w(p: LevelParams, nv: int, level: int, prev_next_w: int) -> dict:
    """Compute next_w for a baseline level."""
    if level == 0:
        alpha = p.D.bit_length() - 1
        num_ring = 1 << (nv - alpha)
    else:
        num_ring = prev_next_w // p.D

    delta_commit = 1 if level > 0 else 1  # onehot at root, recursive later
    return compute_next_w(p.D, p.lb, p.m, p.r, p.na, p.nb, p.nd,
                          p.l1_mass, num_ring, delta_commit)


def compute_l0_matrix(p: LevelParams, nv: int) -> dict:
    """Compute L0 shared matrix size."""
    alpha = p.D.bit_length() - 1
    num_ring = 1 << (nv - alpha)
    num_blocks = 1 << p.r
    m_eff = -(-num_ring // num_blocks)

    delta_open = compute_num_digits(128, p.lb)
    delta_commit = 1  # onehot

    a_cols = m_eff * delta_commit
    b_cols = p.na * delta_open * num_blocks
    d_cols = delta_open * num_blocks
    max_rows = max(p.na, p.nb, p.nd)
    max_cols = max(a_cols, b_cols, d_cols)

    return {
        "max_rows": max_rows, "max_cols": max_cols,
        "a_cols": a_cols, "b_cols": b_cols, "d_cols": d_cols,
        "ring_elems": max_rows * max_cols,
        "field_elems": max_rows * max_cols * p.D,
        "m_eff": m_eff,
    }


def find_optimal_r_for_S(
    D: int, lb: int, na: int, l1_mass: int,
    num_ring_S: int, delta_commit_S: int,
) -> tuple[int, int]:
    """Find optimal r for the S polynomial, respecting SIS constraints."""
    delta_open = compute_num_digits(128, lb)
    collision_inf = (1 << lb) - 1

    widths_by_rank = SIS_MAX_WIDTH_BY_D_AND_COLLISION.get(D, {}).get(collision_inf)
    if widths_by_rank is None:
        return 1, 1

    max_rank_width = max(widths_by_rank)
    outer_per_block = na * delta_open
    if outer_per_block == 0:
        return 1, 1
    max_blocks_sis = max_rank_width // outer_per_block
    if max_blocks_sis == 0:
        return 1, 1
    max_r_sis = max_blocks_sis.bit_length() - 1

    ring_pow2 = 1
    while ring_pow2 < num_ring_S:
        ring_pow2 <<= 1
    reduced_vars = ring_pow2.bit_length() - 1

    c1 = delta_open + na * delta_open
    best_r = 1
    best_cost = float('inf')

    for r in range(1, min(reduced_vars, max_r_sis + 1)):
        df = compute_num_digits_fold(r, l1_mass, lb)
        m_eff = -(-num_ring_S // (1 << r))
        cost = c1 * (1 << r) + delta_commit_S * df * m_eff
        if cost < best_cost:
            best_cost = cost
            best_r = r

    nb_S = min_rank_for_secure_width(D, collision_inf, na * delta_open * (1 << best_r))
    if nb_S is None:
        nb_S = 4
    return best_r, nb_S


def analyze_t2_at_l0(nv: int):
    """Full T2@L0 cascade analysis."""
    schedule = BASELINE_SCHEDULES[nv]
    L0 = schedule[0]
    L1 = schedule[1]

    print(f"\n{'='*72}")
    print(f"  T2 CASCADE ANALYSIS: onehot nv={nv}")
    print(f"{'='*72}")

    print(f"\n  Baseline schedule:")
    prev_w = 1 << nv
    for i, p in enumerate(schedule):
        res = compute_baseline_next_w(p, nv, i, prev_w)
        p.next_w_len = res["next_w_len"]
        print(f"    L{i}: D={p.D} lb={p.lb} m={p.m} r={p.r} "
              f"na={p.na} nb={p.nb} nd={p.nd} "
              f"next_w={res['next_w_len']:,} ({res['next_w_len']/1e6:.2f}M)")
        prev_w = res["next_w_len"]

    mat = compute_l0_matrix(L0, nv)
    print(f"\n  L0 shared matrix:")
    print(f"    max_rows={mat['max_rows']}, max_cols={mat['max_cols']:,}")
    print(f"    a_cols={mat['a_cols']:,}, b_cols={mat['b_cols']:,}, d_cols={mat['d_cols']:,}")
    print(f"    S0 = {mat['ring_elems']:,} ring elems = {mat['field_elems']:,} field elems "
          f"({mat['field_elems']/1e6:.1f}M)")

    next_w_from_L0 = schedule[0].next_w_len
    S0_field = mat["field_elems"]

    delta_commit_S = compute_num_digits(128, L1.lb)
    delta_commit_w = compute_num_digits(L1.lb, L1.lb)
    delta_open_L1 = compute_num_digits(128, L1.lb)

    print(f"\n  L1 params: D={L1.D}, lb={L1.lb}, na={L1.na}, l1_mass={L1.l1_mass}")
    print(f"  delta_commit_w={delta_commit_w}, delta_commit_S={delta_commit_S}, delta_open={delta_open_L1}")

    num_ring_w = next_w_from_L0 // L1.D
    num_ring_S = S0_field // L1.D

    print(f"\n  --- T2@L0 split design at L1 ---")
    print(f"  w: {num_ring_w:,} ring elems ({next_w_from_L0/1e6:.1f}M field)")
    print(f"  S: {num_ring_S:,} ring elems ({S0_field/1e6:.1f}M field)")

    w_res = compute_next_w(L1.D, L1.lb, L1.m, L1.r, L1.na, L1.nb, L1.nd,
                           L1.l1_mass, num_ring_w, delta_commit_w)

    S_r, S_nb = find_optimal_r_for_S(L1.D, L1.lb, L1.na, L1.l1_mass,
                                      num_ring_S, delta_commit_S)
    S_m_eff = -(-num_ring_S // (1 << S_r))
    S_delta_fold = compute_num_digits_fold(S_r, L1.l1_mass, L1.lb)

    S_w_hat = (1 << S_r) * delta_open_L1
    S_t_hat = (1 << S_r) * L1.na * delta_open_L1
    S_z_pre = S_m_eff * delta_commit_S * S_delta_fold

    print(f"\n  w polynomial at L1:")
    print(f"    (m,r)=({L1.m},{L1.r}), m_eff={w_res['m_eff']:,}, δf={w_res['delta_fold']}")
    print(f"    w_hat={w_res['w_hat']:,}, t_hat={w_res['t_hat']:,}, z_pre={w_res['z_pre']:,}")

    print(f"\n  S polynomial at L1:")
    print(f"    r_S={S_r} (SIS-constrained), m_eff_S={S_m_eff:,}, δf={S_delta_fold}, nb_S={S_nb}")
    print(f"    w_hat_S={S_w_hat:,}, t_hat_S={S_t_hat:,}, z_pre_S={S_z_pre:,}")

    collision_inf = (1 << L1.lb) - 1
    d_width = w_res["w_hat"] + S_w_hat
    nd_combined = min_rank_for_secure_width(L1.D, collision_inf, d_width) or 4

    m_row_combined = nd_combined + w_res["m_row"] - L1.nd + S_nb
    # m_row = nd_new + nb_w + nb_S + 2 + na
    m_row_combined = nd_combined + L1.nb + S_nb + 2 + L1.na
    delta_128 = r_decomp_levels(L1.lb, HALF_FIELD_BOUND)
    r_ct = m_row_combined * delta_128

    w_contrib = w_res["w_hat"] + w_res["t_hat"] + w_res["z_pre"]
    S_contrib = S_w_hat + S_t_hat + S_z_pre
    total_ring = w_contrib + S_contrib + r_ct
    t2_next_w = total_ring * L1.D

    baseline_next_w = w_res["next_w_len"]

    print(f"\n  Combined next witness (L2 input):")
    print(f"    nd={nd_combined}, nb_w={L1.nb}, nb_S={S_nb}, m_row={m_row_combined}")
    print(f"    r_ct = {r_ct:,}")
    print(f"    w contribution: {w_contrib:,} ({w_contrib/total_ring*100:.1f}%)")
    print(f"    S contribution: {S_contrib:,} ({S_contrib/total_ring*100:.1f}%)")
    print(f"      z_pre_S:      {S_z_pre:,} ({S_z_pre/total_ring*100:.1f}%)")
    print(f"    r_ct:           {r_ct:,}")
    print(f"    TOTAL:          {total_ring:,} ring elems")
    print(f"    next_w_len:     {t2_next_w:,} ({t2_next_w/1e6:.1f}M field elems)")

    print(f"\n  Comparison:")
    print(f"    Baseline L2 input: {baseline_next_w:,} ({baseline_next_w/1e6:.2f}M)")
    print(f"    T2@L0 L2 input:    {t2_next_w:,} ({t2_next_w/1e6:.1f}M)")
    ratio = t2_next_w / baseline_next_w if baseline_next_w > 0 else float('inf')
    print(f"    Ratio:             {ratio:.1f}x")

    if len(schedule) > 2:
        L2 = schedule[2]
        L2_alpha = L2.D.bit_length() - 1
        L2_ring_cap = 1 << (L2.m + L2.r)
        L2_field_cap = L2_ring_cap * L2.D
        print(f"\n  L2 overflow check:")
        print(f"    L2 baseline: D={L2.D}, m={L2.m}, r={L2.r}")
        print(f"    L2 capacity: 2^{L2.m+L2.r} x {L2.D} = {L2_field_cap:,} ({L2_field_cap/1e6:.1f}M)")
        if t2_next_w > L2_field_cap:
            print(f"    OVERFLOW: {t2_next_w/L2_field_cap:.1f}x over capacity")
        else:
            print(f"    Fits ({t2_next_w/L2_field_cap*100:.0f}% utilized)")

    a_cols_S = S_m_eff * delta_commit_S
    b_cols_S = L1.na * delta_open_L1 * (1 << S_r)
    b_cols_w = L1.na * delta_open_L1 * (1 << L1.r)
    l1_max_rows = max(L1.na, L1.nb, S_nb, nd_combined)
    l1_max_cols = max(w_res["m_eff"] * delta_commit_w, a_cols_S, b_cols_w, b_cols_S, d_width)
    S1_ring = l1_max_rows * l1_max_cols
    S1_field = S1_ring * L1.D

    print(f"\n  L1 shared matrix (for T2@L1):")
    print(f"    max_rows={l1_max_rows}, max_cols={l1_max_cols:,}")
    print(f"    a_cols_S={a_cols_S:,}, b_cols_w={b_cols_w:,}, b_cols_S={b_cols_S:,}")
    print(f"    S1 = {S1_ring:,} ring = {S1_field:,} field ({S1_field/1e6:.1f}M)")

    return {
        "nv": nv,
        "S0_field": mat["field_elems"],
        "S0_max_rows": mat["max_rows"],
        "S0_max_cols": mat["max_cols"],
        "baseline_L2": baseline_next_w,
        "t2_L2": t2_next_w,
        "ratio": ratio,
        "z_pre_S": S_z_pre,
        "z_pre_S_pct": S_z_pre / total_ring * 100,
        "S_r": S_r,
        "S_nb": S_nb,
        "S_m_eff": S_m_eff,
        "delta_commit_S": delta_commit_S,
        "m_row": m_row_combined,
        "w_contrib": w_contrib,
        "S_contrib": S_contrib,
        "S1_field": S1_field,
    }


def analyze_t2_at_l0_and_l1(nv: int):
    """T2@L0+L1 cascade: trace S_0 through L1, and S_1 through L2, into L3."""
    schedule = BASELINE_SCHEDULES[nv]
    if len(schedule) < 3:
        print(f"  nv={nv}: not enough levels for T2@L0+L1 analysis")
        return None

    t2_l0 = analyze_t2_at_l0(nv)

    L1 = schedule[1]
    L2 = schedule[2]

    S1_field = t2_l0["S1_field"]
    t2_L2_input = t2_l0["t2_L2"]

    delta_commit_S1 = compute_num_digits(128, L2.lb)
    delta_open_L2 = compute_num_digits(128, L2.lb)

    num_ring_w_L2 = t2_L2_input // L2.D
    num_ring_S1_L2 = S1_field // L2.D

    print(f"\n  --- T2@L0+L1: S_1 at L2 ---")
    print(f"  L2 w input (from T2@L0 L1): {t2_L2_input:,} ({t2_L2_input/1e6:.1f}M field)")
    print(f"  S_1 from L1: {S1_field:,} ({S1_field/1e6:.1f}M field)")

    l1_mass_L2 = L2.l1_mass

    w_res_L2 = compute_next_w(L2.D, L2.lb, L2.m, L2.r, L2.na, L2.nb, L2.nd,
                               l1_mass_L2, num_ring_w_L2, 1)

    S1_r, S1_nb = find_optimal_r_for_S(L2.D, L2.lb, L2.na, l1_mass_L2,
                                         num_ring_S1_L2, delta_commit_S1)
    S1_m_eff = -(-num_ring_S1_L2 // (1 << S1_r))
    S1_delta_fold = compute_num_digits_fold(S1_r, l1_mass_L2, L2.lb)

    S1_w_hat = (1 << S1_r) * delta_open_L2
    S1_t_hat = (1 << S1_r) * L2.na * delta_open_L2
    S1_z_pre = S1_m_eff * delta_commit_S1 * S1_delta_fold

    collision_inf = (1 << L2.lb) - 1
    d_width = w_res_L2["w_hat"] + S1_w_hat
    nd_L2 = min_rank_for_secure_width(L2.D, collision_inf, d_width) or 4

    m_row_L2 = nd_L2 + L2.nb + S1_nb + 2 + L2.na
    delta_128 = r_decomp_levels(L2.lb, HALF_FIELD_BOUND)
    r_ct_L2 = m_row_L2 * delta_128

    w_contrib_L2 = w_res_L2["w_hat"] + w_res_L2["t_hat"] + w_res_L2["z_pre"]
    S1_contrib_L2 = S1_w_hat + S1_t_hat + S1_z_pre
    total_ring_L2 = w_contrib_L2 + S1_contrib_L2 + r_ct_L2
    t2_L3_input = total_ring_L2 * L2.D

    baseline_L2_result = compute_baseline_next_w(L2, nv, 2, schedule[1].next_w_len)
    baseline_L3 = baseline_L2_result["next_w_len"]

    print(f"\n  w polynomial at L2 (inflated from T2@L0):")
    print(f"    num_ring_w={num_ring_w_L2:,}, (m,r)=({L2.m},{L2.r})")
    print(f"    w_hat={w_res_L2['w_hat']:,}, t_hat={w_res_L2['t_hat']:,}, z_pre={w_res_L2['z_pre']:,}")

    print(f"\n  S_1 polynomial at L2:")
    print(f"    num_ring_S1={num_ring_S1_L2:,}, r_S1={S1_r}, m_eff_S1={S1_m_eff:,}")
    print(f"    w_hat_S1={S1_w_hat:,}, t_hat_S1={S1_t_hat:,}, z_pre_S1={S1_z_pre:,}")
    print(f"    nb_S1={S1_nb}")

    print(f"\n  Combined L3 input:")
    print(f"    w contribution: {w_contrib_L2:,}")
    print(f"    S1 contribution: {S1_contrib_L2:,}")
    print(f"    r_ct: {r_ct_L2:,}")
    print(f"    TOTAL: {total_ring_L2:,} ring elems")
    print(f"    L3 input: {t2_L3_input:,} ({t2_L3_input/1e6:.1f}M field)")

    print(f"\n  Cascade comparison:")
    print(f"    Baseline L3 input:       {baseline_L3:,} ({baseline_L3/1e6:.2f}M)")
    print(f"    T2@L0+L1 L3 input:       {t2_L3_input:,} ({t2_L3_input/1e6:.1f}M)")
    ratio = t2_L3_input / baseline_L3 if baseline_L3 > 0 else float('inf')
    print(f"    Compound ratio (vs baseline): {ratio:.1f}x")

    if len(schedule) > 3:
        L3 = schedule[3]
        L3_cap = (1 << (L3.m + L3.r)) * L3.D
        print(f"\n  L3 overflow check:")
        print(f"    L3 baseline: D={L3.D}, m={L3.m}, r={L3.r}")
        print(f"    L3 capacity: {L3_cap:,} ({L3_cap/1e6:.1f}M)")
        if t2_L3_input > L3_cap:
            print(f"    OVERFLOW: {t2_L3_input/L3_cap:.1f}x over capacity")
        else:
            print(f"    Fits ({t2_L3_input/L3_cap*100:.0f}% utilized)")

    return {
        "nv": nv,
        "baseline_L3": baseline_L3,
        "t2_L3": t2_L3_input,
        "ratio": ratio,
        "S1_z_pre": S1_z_pre,
    }


def print_summary(results: list[dict]):
    print(f"\n{'='*72}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*72}")

    def fmt(v):
        if v >= 1e9:
            return f"{v/1e9:.1f}B"
        if v >= 1e6:
            return f"{v/1e6:.1f}M"
        if v >= 1e3:
            return f"{v/1e3:.0f}K"
        return str(int(v))

    r = {res["nv"]: res for res in results}
    nvs = sorted(r.keys())

    header = f"  {'':>20}" + "".join(f" {'nv='+str(nv):>14}" for nv in nvs)
    print(header)
    print(f"  {'-'*20}" + " ".join(f"{'-'*14}" for _ in nvs))

    rows = [
        ("S0 (field elems)", lambda nv: fmt(r[nv]["S0_field"])),
        ("δ_commit_S", lambda nv: str(r[nv]["delta_commit_S"])),
        ("Baseline L2 input", lambda nv: fmt(r[nv]["baseline_L2"])),
        ("T2@L0 L2 input", lambda nv: fmt(r[nv]["t2_L2"])),
        ("Blowup ratio", lambda nv: f"{r[nv]['ratio']:.1f}x"),
        ("z_pre_S fraction", lambda nv: f"{r[nv]['z_pre_S_pct']:.0f}%"),
        ("S optimal r", lambda nv: str(r[nv]["S_r"])),
        ("S m_eff", lambda nv: f"{r[nv]['S_m_eff']:,}"),
        ("Split m_row", lambda nv: str(r[nv]["m_row"])),
        ("S1 (for T2@L1)", lambda nv: fmt(r[nv]["S1_field"])),
    ]

    for label, fn in rows:
        print(f"  {label:>20}" + "".join(f" {fn(nv):>14}" for nv in nvs))


def analyze_k_chunk(nv: int, k: int = 4, fuse_d: bool = False):
    """Analyze k-chunk commitment variant.

    k chunks from halving both A and B widths: each chunk has
    m_per_chunk = m-1, r_per_chunk = r-1, giving N/4 per chunk.
    Fold challenges shared across all k * 2^(r-1) = 2^(r+1) blocks.

    fuse_d: if True, use a single D-commitment spanning all chunks'
    blocks (width = delta_open * 2^(r+1)). Otherwise 4 separate D's.
    """
    assert k == 4, "Only k=4 (halving A and B) supported for now"

    schedule = BASELINE_SCHEDULES[nv]
    L0 = schedule[0]
    L1 = schedule[1]

    prev_w = 1 << nv
    for i, p in enumerate(schedule):
        res = compute_baseline_next_w(p, nv, i, prev_w)
        p.next_w_len = res["next_w_len"]
        prev_w = res["next_w_len"]

    alpha = L0.D.bit_length() - 1
    num_ring = 1 << (nv - alpha)
    delta_open = compute_num_digits(128, L0.lb)
    delta_128 = r_decomp_levels(L0.lb, HALF_FIELD_BOUND)

    m_per_chunk = L0.m - 1
    r_per_chunk = L0.r - 1
    r_fold = L0.r + 1
    m_eff_tight = -(-num_ring // (1 << r_fold))

    collision_A = 2
    collision_BD = (1 << L0.lb) - 1

    a_inner = m_eff_tight * 1
    na_chunk = L0.na
    for try_na in range(1, L0.na + 1):
        na_needed_try = min_rank_for_secure_width(L0.D, collision_A, a_inner)
        if na_needed_try is not None and na_needed_try <= try_na:
            na_chunk = try_na
            break

    b_outer = na_chunk * delta_open * (1 << r_per_chunk)
    nb_chunk = min_rank_for_secure_width(L0.D, collision_BD, b_outer)
    if nb_chunk is None:
        nb_chunk = 4

    if fuse_d:
        d_width_sis = delta_open * (1 << r_fold)
    else:
        d_width_sis = delta_open * (1 << r_per_chunk)
    nd_chunk = min_rank_for_secure_width(L0.D, collision_BD, d_width_sis)
    if nd_chunk is None:
        nd_chunk = 4

    l1_mass_chunk = L0.l1_mass

    delta_fold_chunk = compute_num_digits_fold(r_fold, l1_mass_chunk, L0.lb)

    w_hat = delta_open * (1 << r_fold)
    t_hat = na_chunk * delta_open * (1 << r_fold)
    z_pre = m_eff_tight * 1 * delta_fold_chunk

    if fuse_d:
        m_row_chunk = nd_chunk + k * nb_chunk + 2 + na_chunk
    else:
        m_row_chunk = k * nd_chunk + k * nb_chunk + 2 + na_chunk

    r_ct = m_row_chunk * delta_128
    total_ring = w_hat + t_hat + z_pre + r_ct
    next_w_chunk = total_ring * L0.D

    baseline_res = compute_baseline_next_w(L0, nv, 0, 1 << nv)
    baseline_next_w = baseline_res["next_w_len"]

    a_cols_chunk = m_eff_tight * 1
    b_cols_chunk = na_chunk * delta_open * (1 << r_per_chunk)
    d_cols_per_chunk = delta_open * (1 << r_per_chunk)
    d_cols_fused = delta_open * (1 << r_fold)

    if fuse_d:
        max_rows_for_S = max(na_chunk, nb_chunk, nd_chunk)
        b_role_cols = b_cols_chunk
        d_role_cols = d_cols_fused
        max_cols_for_S = max(a_cols_chunk, b_role_cols, d_role_cols)
    else:
        max_rows_for_S = max(na_chunk, nb_chunk, nd_chunk)
        max_cols_for_S = max(a_cols_chunk, b_cols_chunk, d_cols_per_chunk)

    S_ring = max_rows_for_S * max_cols_for_S
    S_field = S_ring * L0.D

    baseline_mat = compute_l0_matrix(L0, nv)

    label = f"{k}-chunk" + (" D-fused" if fuse_d else " D-separate")

    print(f"\n{'='*72}")
    print(f"  {label.upper()}: onehot nv={nv}")
    print(f"{'='*72}")

    print(f"\n  Parameters:")
    print(f"    m_chunk={m_per_chunk}, r_chunk={r_per_chunk}, "
          f"r_fold={r_fold}")
    print(f"    na={na_chunk}, nb={nb_chunk}, nd={nd_chunk}")
    print(f"    m_eff={m_eff_tight:,}, δ_fold={delta_fold_chunk}")
    if fuse_d:
        print(f"    m_row={m_row_chunk} (= {nd_chunk} + {k}×{nb_chunk} + 2 + {na_chunk})  "
              f"[D fused]")
    else:
        print(f"    m_row={m_row_chunk} (= {k}×{nd_chunk} + {k}×{nb_chunk} + 2 + {na_chunk})  "
              f"[D separate]")

    print(f"\n  SIS widths:")
    print(f"    A inner: {a_inner:,}")
    print(f"    B outer (per chunk): {b_outer:,}")
    if fuse_d:
        print(f"    D width (fused): {d_cols_fused:,}")
    else:
        print(f"    D width (per chunk): {d_cols_per_chunk:,}")

    print(f"\n  Next witness (L1 input):")
    print(f"    w_hat={w_hat:,}, t_hat={t_hat:,}, "
          f"z_pre={z_pre:,}, r_ct={r_ct:,}")
    print(f"    total_ring={total_ring:,}, next_w={next_w_chunk:,} "
          f"({next_w_chunk/1e6:.1f}M)")
    print(f"    vs baseline: {baseline_next_w:,} ({baseline_next_w/1e6:.1f}M) "
          f"→ {next_w_chunk/baseline_next_w:.3f}×")

    print(f"\n  Shared matrix S (for T2):")
    print(f"    max_rows={max_rows_for_S}, max_cols={max_cols_for_S:,}")
    if fuse_d:
        print(f"    (D-fused dominates: d_cols={d_cols_fused:,} vs "
              f"b_cols={b_cols_chunk:,})")
    print(f"    S = {S_ring:,} ring = {S_field:,} field ({S_field/1e6:.1f}M)")
    print(f"    vs baseline: {baseline_mat['field_elems']:,} ({baseline_mat['field_elems']/1e6:.1f}M) "
          f"→ {baseline_mat['field_elems']/S_field:.1f}× reduction")

    if fuse_d:
        proof_extra = (k - 1) * nb_chunk * L0.D * 16
    else:
        proof_extra = (k - 1) * (nb_chunk * L0.D * 16 + nd_chunk * L0.D * 16)
    print(f"\n  Proof size extra: {proof_extra:,} B ({proof_extra/1024:.1f} KB)")

    delta_commit_S = compute_num_digits(128, L1.lb)
    num_ring_w_L1 = next_w_chunk // L1.D
    num_ring_S_L1 = S_field // L1.D

    w_res_L1 = compute_next_w(L1.D, L1.lb, L1.m, L1.r, L1.na, L1.nb, L1.nd,
                               L1.l1_mass, num_ring_w_L1, 1)
    S_r, S_nb = find_optimal_r_for_S(L1.D, L1.lb, L1.na, L1.l1_mass,
                                      num_ring_S_L1, delta_commit_S)
    delta_open_L1 = compute_num_digits(128, L1.lb)
    S_m_eff = -(-num_ring_S_L1 // (1 << S_r))
    S_delta_fold = compute_num_digits_fold(S_r, L1.l1_mass, L1.lb)
    S_z_pre = S_m_eff * delta_commit_S * S_delta_fold

    S_w_hat = (1 << S_r) * delta_open_L1
    S_t_hat = (1 << S_r) * L1.na * delta_open_L1
    collision_inf = (1 << L1.lb) - 1
    d_width_L1 = w_res_L1["w_hat"] + S_w_hat
    nd_L1 = min_rank_for_secure_width(L1.D, collision_inf, d_width_L1) or 4
    m_row_L1 = nd_L1 + L1.nb + S_nb + 2 + L1.na
    delta_128_L1 = r_decomp_levels(L1.lb, HALF_FIELD_BOUND)
    r_ct_L1 = m_row_L1 * delta_128_L1

    w_contrib_L1 = w_res_L1["w_hat"] + w_res_L1["t_hat"] + w_res_L1["z_pre"]
    S_contrib_L1 = S_w_hat + S_t_hat + S_z_pre
    total_ring_L1 = w_contrib_L1 + S_contrib_L1 + r_ct_L1
    t2_L2 = total_ring_L1 * L1.D

    print(f"\n  T2@L0 cascade:")
    print(f"    S entering L1: {S_field/1e6:.1f}M → z_pre_S={S_z_pre:,}")
    print(f"    T2 L2 input: {t2_L2:,} ({t2_L2/1e6:.1f}M)")

    return {
        "nv": nv, "fuse_d": fuse_d,
        "na": na_chunk, "nb": nb_chunk, "nd": nd_chunk,
        "m_row": m_row_chunk,
        "baseline_next_w": baseline_next_w,
        "chunk_next_w": next_w_chunk,
        "next_w_ratio": next_w_chunk / baseline_next_w,
        "S_field": S_field,
        "S_baseline_field": baseline_mat["field_elems"],
        "S_reduction": baseline_mat["field_elems"] / S_field,
        "proof_extra": proof_extra,
        "t2_L2": t2_L2,
        "d_cols": d_cols_fused if fuse_d else d_cols_per_chunk,
        "b_cols": b_cols_chunk,
        "max_cols": max_cols_for_S,
    }


# =============================================================================
#  Tiered commitment analysis
# =============================================================================

STORAGE_BYTES_PER_FIELD = 32  # plan convention (includes alignment overhead)


def _storage_gb(ring_elems: int, D: int) -> float:
    """Raw (FlatMatrix) storage in GB."""
    return ring_elems * D * STORAGE_BYTES_PER_FIELD / (1024**3)


def analyze_tiered_l0(nv: int, f: int, use_t1: bool = False) -> dict:
    """Analyze tiered commitment at L0 with shrink factor f.

    f=1 gives the (non-tiered) baseline.
    use_t1: if True and D>=64, adjust l1_mass for tensor challenges (4x).
    """
    schedule = BASELINE_SCHEDULES[nv]
    L0, L1 = schedule[0], schedule[1]

    log_f = int(math.log2(f)) if f > 1 else 0
    alpha = L0.D.bit_length() - 1
    num_ring = 1 << (nv - alpha)
    delta_open = compute_num_digits(128, L0.lb)
    delta_commit = 1  # onehot
    delta_128 = r_decomp_levels(L0.lb, HALF_FIELD_BOUND)

    l1_mass = L0.l1_mass * 4 if (use_t1 and L0.D >= 64) else L0.l1_mass

    k = f * f
    r_chunk = L0.r - log_f
    r_fold = L0.r + log_f
    m_eff_chunk = -(-num_ring // (1 << r_fold))
    delta_fold = compute_num_digits_fold(r_fold, l1_mass, L0.lb)

    collision_BD = (1 << L0.lb) - 1
    collision_A = 2  # onehot

    if f > 1:
        d_cols_chunk = delta_open * (1 << r_chunk)
        a_cols_chunk = m_eff_chunk * delta_commit

        na = min_rank_for_secure_width(L0.D, collision_A, a_cols_chunk)
        if na is None:
            na = L0.na

        b_cols_chunk = na * delta_open * (1 << r_chunk)
        nd_chunk = min_rank_for_secure_width(L0.D, collision_BD, d_cols_chunk) or 4
        nb_chunk = min_rank_for_secure_width(L0.D, collision_BD, b_cols_chunk) or 4
    else:
        d_cols_chunk = delta_open * (1 << L0.r)
        a_cols_chunk = m_eff_chunk * delta_commit
        b_cols_chunk = L0.na * delta_open * (1 << L0.r)
        na, nd_chunk, nb_chunk = L0.na, L0.nd, L0.nb

    w_hat = delta_open * (1 << r_fold)
    t_hat = na * delta_open * (1 << r_fold)
    z_pre = m_eff_chunk * delta_commit * delta_fold

    if f > 1:
        v_digits = k * nd_chunk * delta_open
        u_digits = k * nb_chunk * delta_open

        N_meta = k * (nd_chunk + nb_chunk)
        delta_commit_meta = compute_num_digits(128, L0.lb)

        rp2 = 1
        while rp2 < N_meta:
            rp2 <<= 1
        reduced_meta = max(rp2.bit_length() - 1, 2)

        best_r_meta, best_cost = 1, float('inf')
        c1_meta = delta_open + 1 * delta_commit_meta
        for rm in range(1, reduced_meta):
            dfm = compute_num_digits_fold(rm, l1_mass, L0.lb)
            mem = -(-N_meta // (1 << rm))
            cost = c1_meta * (1 << rm) + delta_commit_meta * dfm * mem
            if cost < best_cost:
                best_cost = cost
                best_r_meta = rm

        r_meta = best_r_meta
        m_eff_meta = -(-N_meta // (1 << r_meta))

        na_meta = min_rank_for_secure_width(
            L0.D, collision_BD, m_eff_meta * delta_commit_meta) or 1
        nb_meta = min_rank_for_secure_width(
            L0.D, collision_BD, na_meta * delta_open * (1 << r_meta)) or 1
        nd_meta = min_rank_for_secure_width(
            L0.D, collision_BD, delta_open * (1 << r_meta)) or 1

        delta_fold_meta = compute_num_digits_fold(r_meta, l1_mass, L0.lb)
        w_hat_meta = delta_open * (1 << r_meta)
        t_hat_meta = na_meta * delta_open * (1 << r_meta)
        z_pre_meta = m_eff_meta * delta_commit_meta * delta_fold_meta

        m_row = (k * (nd_chunk + nb_chunk) + 2 + na
                 + nd_meta + nb_meta + 2 + na_meta)
    else:
        v_digits = u_digits = 0
        N_meta = 0
        na_meta = nb_meta = nd_meta = 0
        w_hat_meta = t_hat_meta = z_pre_meta = 0
        r_meta = 0
        m_eff_meta = 0
        m_row = nd_chunk + nb_chunk + 2 + na

    r_ct = m_row * delta_128
    total_ring = (w_hat + t_hat + z_pre + v_digits + u_digits
                  + w_hat_meta + t_hat_meta + z_pre_meta + r_ct)
    next_w_len = total_ring * L0.D

    # S matrix (shared across all chunks)
    max_rows_S = max(na, nb_chunk, nd_chunk)
    max_cols_S = max(a_cols_chunk, b_cols_chunk, d_cols_chunk)
    S_ring = max_rows_S * max_cols_S
    raw_gb = _storage_gb(S_ring, L0.D)
    ntt_gb = raw_gb * 2

    # Baseline (same T1 setting) for fair growth comparison
    delta_fold_base = compute_num_digits_fold(L0.r, l1_mass, L0.lb)
    m_eff_base = -(-num_ring // (1 << L0.r))
    w_hat_base = delta_open * (1 << L0.r)
    t_hat_base = L0.na * delta_open * (1 << L0.r)
    z_pre_base = m_eff_base * delta_commit * delta_fold_base
    m_row_base = L0.nd + L0.nb + 2 + L0.na
    r_ct_base = m_row_base * delta_128
    baseline_ring = w_hat_base + t_hat_base + z_pre_base + r_ct_base

    witness_growth = total_ring / baseline_ring

    # Baseline S for reduction comparison
    mat_base = compute_l0_matrix(L0, nv)
    S_ring_base = mat_base["max_rows"] * mat_base["max_cols"]
    S_reduction = S_ring_base / S_ring if S_ring > 0 else float('inf')

    delta_commit_S_L1 = compute_num_digits(128, L1.lb)
    t2_ratio = (S_ring * delta_commit_S_L1) / total_ring

    L1_ring_cap = 1 << (L1.m + L1.r)
    L1_ring_needed = next_w_len // L1.D
    if L1_ring_needed > L1_ring_cap:
        extra_bits = math.ceil(math.log2(L1_ring_needed / L1_ring_cap))
    else:
        extra_bits = 0

    return {
        "nv": nv, "f": f, "k": k,
        "r_chunk": r_chunk, "r_fold": r_fold,
        "m_eff_chunk": m_eff_chunk,
        "na": na, "nb_chunk": nb_chunk, "nd_chunk": nd_chunk,
        "delta_fold": delta_fold, "l1_mass": l1_mass,
        "w_hat": w_hat, "t_hat": t_hat, "z_pre": z_pre,
        "v_digits": v_digits, "u_digits": u_digits,
        "w_hat_meta": w_hat_meta, "t_hat_meta": t_hat_meta,
        "z_pre_meta": z_pre_meta, "N_meta": N_meta,
        "na_meta": na_meta, "nb_meta": nb_meta, "nd_meta": nd_meta,
        "r_ct": r_ct, "m_row": m_row,
        "total_ring": total_ring, "next_w_len": next_w_len,
        "witness_growth": witness_growth,
        "baseline_ring": baseline_ring,
        "S_ring": S_ring, "S_ring_base": S_ring_base,
        "S_reduction": S_reduction,
        "max_rows_S": max_rows_S, "max_cols_S": max_cols_S,
        "raw_gb": raw_gb, "ntt_gb": ntt_gb,
        "t2_ratio": t2_ratio,
        "L1_extra_bits": extra_bits,
    }


def _l1_cascade(nv: int, l0: dict) -> dict:
    """Compute L1 T2 cascade given L0 tiered results.

    Traces S_chunk from L0 through L1 to determine the L1 shared matrix
    (S_L1) and the T2 ratio at L2.
    """
    schedule = BASELINE_SCHEDULES[nv]
    L0_p, L1 = schedule[0], schedule[1]

    S_field_L0 = l0["S_ring"] * L0_p.D
    num_ring_w_L1 = l0["next_w_len"] // L1.D
    num_ring_S_L1 = S_field_L0 // L1.D

    delta_commit_S = compute_num_digits(128, L1.lb)
    delta_commit_w = 1
    delta_open_L1 = compute_num_digits(128, L1.lb)

    w_res = compute_next_w(L1.D, L1.lb, L1.m, L1.r, L1.na, L1.nb, L1.nd,
                           L1.l1_mass, num_ring_w_L1, delta_commit_w)

    S_r, S_nb = find_optimal_r_for_S(L1.D, L1.lb, L1.na, L1.l1_mass,
                                      num_ring_S_L1, delta_commit_S)
    S_m_eff = -(-num_ring_S_L1 // (1 << S_r))
    S_delta_fold = compute_num_digits_fold(S_r, L1.l1_mass, L1.lb)

    S_w_hat = (1 << S_r) * delta_open_L1
    S_t_hat = (1 << S_r) * L1.na * delta_open_L1
    S_z_pre = S_m_eff * delta_commit_S * S_delta_fold

    collision_inf = (1 << L1.lb) - 1
    d_width = w_res["w_hat"] + S_w_hat
    nd_combined = min_rank_for_secure_width(L1.D, collision_inf, d_width) or 4
    m_row_L1 = nd_combined + L1.nb + S_nb + 2 + L1.na
    delta_128 = r_decomp_levels(L1.lb, HALF_FIELD_BOUND)
    r_ct = m_row_L1 * delta_128

    w_contrib = w_res["w_hat"] + w_res["t_hat"] + w_res["z_pre"]
    S_contrib = S_w_hat + S_t_hat + S_z_pre
    total_ring_L1 = w_contrib + S_contrib + r_ct

    a_cols_S = S_m_eff * delta_commit_S
    b_cols_w = L1.na * delta_open_L1 * (1 << L1.r)
    b_cols_S = L1.na * delta_open_L1 * (1 << S_r)
    l1_max_rows = max(L1.na, L1.nb, S_nb, nd_combined)
    l1_max_cols = max(
        w_res["m_eff"] * delta_commit_w, a_cols_S,
        b_cols_w, b_cols_S, d_width)
    S1_ring = l1_max_rows * l1_max_cols

    L2 = schedule[2] if len(schedule) > 2 else schedule[-1]
    delta_commit_S1 = compute_num_digits(128, L2.lb)
    t2_ratio_L2 = (S1_ring * delta_commit_S1) / total_ring_L1

    l1_raw = _storage_gb(S1_ring, L1.D)

    return {
        "S1_ring": S1_ring,
        "total_ring_L1": total_ring_L1,
        "t2_ratio_L2": t2_ratio_L2,
        "l1_raw": l1_raw,
    }


def print_tiered_sweep(nv: int, use_t1: bool = False) -> list[dict]:
    """Print the per-f sweep table for one nv setting."""
    t1_label = "T1+T2" if use_t1 else "T2 only"
    print(f"\n{'='*95}")
    print(f"  TIERED COMMITMENT SWEEP: onehot nv={nv} ({t1_label})")
    print(f"{'='*95}")

    L0 = BASELINE_SCHEDULES[nv][0]
    f_values = [1, 2, 4, 8, 16, 32, 64]
    results = []
    for f in f_values:
        log_f = int(math.log2(f)) if f > 1 else 0
        if L0.r - log_f < 1 or L0.m - log_f < 1:
            break
        results.append(analyze_tiered_l0(nv, f, use_t1))

    b = results[0]
    print(f"\n  Baseline: D={L0.D}, lb={L0.lb}, m={L0.m}, r={L0.r}")
    print(f"  na={b['na']}, nb={b['nb_chunk']}, nd={b['nd_chunk']}")
    if use_t1:
        print(f"  l1_mass={b['l1_mass']} (T1-adjusted from {L0.l1_mass})")
    print(f"  baseline_ring={b['baseline_ring']:,}")

    print(f"\n  Witness breakdown:")
    print(f"  {'f':>3} {'k':>5} {'r_ch':>4} {'na':>3} {'nb':>3} {'nd':>3} "
          f"{'w_hat':>10} {'t_hat':>10} {'z_pre':>10} "
          f"{'v+u_dig':>10} {'meta':>8} {'r_ct':>8} {'total(M)':>10}")
    print(f"  {'─'*3} {'─'*5} {'─'*4} {'─'*3} {'─'*3} {'─'*3} "
          f"{'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*10}")
    for r in results:
        meta = r['w_hat_meta'] + r['t_hat_meta'] + r['z_pre_meta']
        vu = r['v_digits'] + r['u_digits']
        print(f"  {r['f']:>3} {r['k']:>5} {r['r_chunk']:>4} "
              f"{r['na']:>3} {r['nb_chunk']:>3} {r['nd_chunk']:>3} "
              f"{r['w_hat']:>10,} {r['t_hat']:>10,} {r['z_pre']:>10,} "
              f"{vu:>10,} {meta:>8,} {r['r_ct']:>8,} "
              f"{r['total_ring']/1e6:>10.1f}")

    print(f"\n  Storage and T2 summary:")
    print(f"  {'f':>3} {'k':>5} {'S_red':>6} "
          f"{'S raw(GB)':>10} {'S ntt(GB)':>10} "
          f"{'Witness(M)':>11} {'Growth':>7} {'T2 ratio':>9} {'L1 xtra':>8}")
    print(f"  {'─'*3} {'─'*5} {'─'*6} {'─'*10} {'─'*10} "
          f"{'─'*11} {'─'*7} {'─'*9} {'─'*8}")
    for r in results:
        l1_str = "—" if r['L1_extra_bits'] == 0 else f"+{r['L1_extra_bits']}b"
        red_str = f"{r['S_reduction']:.0f}x" if r['f'] > 1 else "—"
        print(f"  {r['f']:>3} {r['k']:>5} {red_str:>6} "
              f"{r['raw_gb']:>10.2f} {r['ntt_gb']:>10.2f} "
              f"{r['total_ring']/1e6:>11.1f} {r['witness_growth']:>6.2f}x "
              f"{r['t2_ratio']:>8.1f}x {l1_str:>8}")

    return results


def print_tiered_scenarios(nv: int, use_t1: bool = False):
    """Print the combined T1+T2@L0+L1 scenario table."""
    scenarios = [
        (4, 1, "f=4, f_L1=1"),
        (8, 1, "f=8, f_L1=1"),
        (8, 4, "f=8, f_L1=4"),
        (16, 4, "f=16, f_L1=4"),
        (4, 2, "f=4, f_L1=2"),
    ]

    t1_label = "T1+T2" if use_t1 else "T2"
    print(f"\n{'='*105}")
    print(f"  COMBINED {t1_label} SCENARIOS @ L0+L1: nv={nv}")
    print(f"  (T2 applied at both L0 and L1; tiering optional at each)")
    print(f"{'='*105}")

    print(f"\n  {'Scenario':<18} {'L0 raw':>8} {'L0 NTT':>8} {'Tot raw':>8} "
          f"{'L0 wit':>7} {'L1 T2':>7} {'L2 T2':>7}  {'Viable?'}")
    print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*8} "
          f"{'─'*7} {'─'*7} {'─'*7}  {'─'*30}")

    for f, f_L1, label in scenarios:
        l0 = analyze_tiered_l0(nv, f, use_t1)
        cascade = _l1_cascade(nv, l0)

        l0_raw = l0["raw_gb"]
        l0_ntt = l0["ntt_gb"]

        if f_L1 > 1:
            t2_L2 = cascade["t2_ratio_L2"] / (f_L1 * f_L1)
            l1_raw = cascade["l1_raw"] / f_L1
        else:
            t2_L2 = cascade["t2_ratio_L2"]
            l1_raw = cascade["l1_raw"]

        total_raw = l0_raw + l1_raw

        if l0["t2_ratio"] <= 1.5 and t2_L2 <= 1.5:
            viable = "Both OK"
        elif l0["t2_ratio"] <= 1.5:
            viable = f"L1 OK, L2 overflows"
        elif l0["t2_ratio"] <= 4:
            if t2_L2 <= 1.5:
                viable = "L1 marginal"
            else:
                viable = "L1 marginal, L2 overflows"
        else:
            viable = f"L1 overflows"

        print(f"  {label:<18} {l0_raw:>7.1f}G {l0_ntt:>7.1f}G "
              f"{total_raw:>7.1f}G "
              f"{l0['witness_growth']:>6.1f}x {l0['t2_ratio']:>6.1f}x "
              f"{t2_L2:>6.1f}x  {viable}")


def run_tiered_analysis():
    """Entry point for the tiered commitment analysis."""
    print(f"\n\n{'#'*95}")
    print(f"  TIERED COMMITMENT ANALYSIS")
    print(f"  Trades f x witness growth for f x shared-matrix shrinkage.")
    print(f"  Enables T2 by eliminating the cascade; reduces storage for large nv.")
    print(f"{'#'*95}")

    all_results = {}
    for nv in [32, 38, 44]:
        use_t1 = (nv >= 40)
        results = print_tiered_sweep(nv, use_t1)
        all_results[nv] = results

    print_tiered_scenarios(44, use_t1=True)

    print(f"\n{'='*95}")
    print(f"  CROSS-NV SUMMARY (sweet-spot recommendations)")
    print(f"{'='*95}")

    recs = [
        (32, "T2@L0 only (no T1 at D=32)", 8, False),
        (38, "T2@L0 only (no T1 at D=32)", 8, False),
        (44, "T1+T2@L0, f=8 (sweet spot)", 8, True),
    ]
    print(f"\n  {'Setting':<12} {'Scenario':<30} {'S raw(GB)':>10} {'Growth':>7} "
          f"{'T2 ratio':>9}")
    print(f"  {'─'*12} {'─'*30} {'─'*10} {'─'*7} {'─'*9}")
    for nv, desc, f, t1 in recs:
        r = analyze_tiered_l0(nv, f, t1)
        print(f"  nv={nv:<8} {desc:<30} {r['raw_gb']:>10.2f} "
              f"{r['witness_growth']:>6.2f}x {r['t2_ratio']:>8.1f}x")


def main():
    results = []
    for nv in [32, 38, 44]:
        res = analyze_t2_at_l0(nv)
        results.append(res)

    print_summary(results)

    print(f"\n\n{'#'*72}")
    print(f"  T2@L0+L1 CASCADE (compound effect)")
    print(f"{'#'*72}")

    cascade_results = []
    for nv in [32, 38, 44]:
        res = analyze_t2_at_l0_and_l1(nv)
        if res:
            cascade_results.append(res)

    if cascade_results:
        print(f"\n{'='*72}")
        print(f"  T2@L0+L1 COMPOUND SUMMARY")
        print(f"{'='*72}")
        for r in cascade_results:
            def fmt(v):
                if v >= 1e9:
                    return f"{v/1e9:.1f}B"
                if v >= 1e6:
                    return f"{v/1e6:.1f}M"
                return f"{v:,}"
            print(f"  nv={r['nv']}: baseline L3={fmt(r['baseline_L3'])}, "
                  f"T2@L0+L1 L3={fmt(r['t2_L3'])}, "
                  f"ratio={r['ratio']:.1f}x")

    print(f"\n\n{'#'*72}")
    print(f"  4-CHUNK: D-SEPARATE vs D-FUSED COMPARISON")
    print(f"{'#'*72}")

    sep_results = {}
    fused_results = {}
    for nv in [32, 38, 44]:
        sep_results[nv] = analyze_k_chunk(nv, k=4, fuse_d=False)
        fused_results[nv] = analyze_k_chunk(nv, k=4, fuse_d=True)

    def fmt(v):
        if v >= 1e9:
            return f"{v/1e9:.1f}B"
        if v >= 1e6:
            return f"{v/1e6:.1f}M"
        if v >= 1e3:
            return f"{v/1e3:.0f}K"
        return str(int(v))

    print(f"\n{'='*72}")
    print(f"  D-SEPARATE vs D-FUSED SUMMARY")
    print(f"{'='*72}")

    print(f"\n  {'':>22} {'nv=32':>12} {'nv=38':>12} {'nv=44':>12}")
    print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*12}")

    for label, fn_s, fn_f in [
        ("m_row (sep / fused)",
         lambda r: str(r["m_row"]), lambda r: str(r["m_row"])),
        ("next_w (sep)", lambda r: fmt(r["chunk_next_w"]), None),
        ("next_w (fused)", None, lambda r: fmt(r["chunk_next_w"])),
        ("next_w ratio vs base",
         lambda r: f"{r['next_w_ratio']:.3f}×",
         lambda r: f"{r['next_w_ratio']:.3f}×"),
        ("S (sep)", lambda r: fmt(r["S_field"]), None),
        ("S (fused)", None, lambda r: fmt(r["S_field"])),
        ("S reduction vs base",
         lambda r: f"{r['S_reduction']:.1f}×",
         lambda r: f"{r['S_reduction']:.1f}×"),
        ("d_cols", lambda r: f"{r['d_cols']:,}", lambda r: f"{r['d_cols']:,}"),
        ("b_cols", lambda r: f"{r['b_cols']:,}", lambda r: f"{r['b_cols']:,}"),
        ("max_cols", lambda r: f"{r['max_cols']:,}", lambda r: f"{r['max_cols']:,}"),
        ("Proof extra",
         lambda r: f"{r['proof_extra']:,}B",
         lambda r: f"{r['proof_extra']:,}B"),
        ("T2 L2 (sep)", lambda r: fmt(r["t2_L2"]), None),
        ("T2 L2 (fused)", None, lambda r: fmt(r["t2_L2"])),
    ]:
        fn = fn_s or fn_f
        data = sep_results if fn_s else fused_results
        print(f"  {label:>22}", end="")
        for nv in [32, 38, 44]:
            print(f" {fn(data[nv]):>12}", end="")
        print()


    run_tiered_analysis()


if __name__ == "__main__":
    main()
