#!/usr/bin/env python3
"""Rough planner-backed models for Hachi modulus-switch bridges.

This script compares three switch-side opening models:

1. ``replay_structured``:
   replay the old structured opening side with old ``D`` rows, old public
   ``y_ring`` row, old challenge-fold row, and a trace row.
2. ``direct_y_ring``:
   transport the old opening as ``w -> y_ring -> y`` with explicit secret
   ``y_ring`` digits plus outer-evaluation rows and one trace row.
3. ``direct_dense``:
   transport the old scalar opening claim directly as one dense row
   ``lambda(r) · w = y``.

Each model is recursively proved with the ordinary smaller-field planner.

The main output is the fused bridge overhead:

    full_bridge_cost - native_lo_suffix_cost

which is the extra proof size added by the bridge on top of the smaller-field
suffix, under the optimistic assumption that the outgoing commitment is fully
reused by the first native lower-field level.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
from dataclasses import dataclass


ROOT = pathlib.Path(__file__).resolve().parent
PLANNER_PATH = ROOT / "hachi_proof_planner.py"

spec = importlib.util.spec_from_file_location("hachi_proof_planner", PLANNER_PATH)
planner_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = planner_mod
spec.loader.exec_module(planner_mod)


def bitlen(n: int) -> int:
    return max(1, int(n).bit_length())


def min_digits(q_bits: int, base: int) -> int:
    log_base = int(math.log2(base))
    return (q_bits + log_base - 1) // log_base


@dataclass(frozen=True)
class BoundaryCase:
    name: str
    hi_profile_name: str
    lo_profile_name: str
    bridge_profile_name: str
    q_hi_bits: int
    q_lo: int
    witness_len: int
    prev_bound: int
    eta: int
    d_hi: int
    n_a_hi: int
    n_b_hi: int
    n_d_hi: int
    num_blocks_hi: int
    delta_open_hi: int


PROFILE_MAP = {
    "128": planner_mod.PROFILE_128,
    "64": planner_mod.PROFILE_64,
    "32": planner_mod.PROFILE_32,
    "16": planner_mod.PROFILE_16,
    "64b": planner_mod.EXPERIMENTAL_BOOL_PROFILES[64],
    "32b": planner_mod.EXPERIMENTAL_BOOL_PROFILES[32],
    "k7pack": planner_mod.PROFILE_K7_PACK,
}


PLANNER_CACHE = {
    name: planner_mod.Planner(profile, log_commit_bound=1, max_num_vars=32)
    for name, profile in PROFILE_MAP.items()
}

BEST_SUFFIX_CACHE: dict[tuple[str, int, int], tuple[int, int, list, int]] = {}


def best_suffix(profile_name: str, w_len: int, prev_bound: int):
    key = (profile_name, w_len, prev_bound)
    cached = BEST_SUFFIX_CACHE.get(key)
    if cached is not None:
        return cached

    planner = PLANNER_CACHE[profile_name]
    best = None
    for d in planner.unique_ds:
        cost, levels, tail_lb = planner._best_from(w_len, d, prev_bound)
        if cost == float("inf"):
            continue
        if best is None or cost < best[0]:
            best = (cost, d, levels, tail_lb)
    BEST_SUFFIX_CACHE[key] = best
    return best


def t_hat_len(level) -> int:
    return level.d * level.na * (1 << level.r_vars) * level.delta_open


def best_tail_breakdown(profile_name: str, w_len: int, tail_bits: int):
    planner = PLANNER_CACHE[profile_name]
    tail_cinf = 1 if tail_bits == 1 else ((1 << tail_bits) - 1)
    best = None
    for d in planner.unique_ds:
        ring_elems = -(-w_len // d)
        nb = planner_mod.min_rank_for_secure_width(planner.p, d, tail_cinf, ring_elems)
        if nb is None:
            continue
        commit_bytes = planner_mod.ring_vec_bytes_base(nb, d, planner.p)
        packed_bytes = planner_mod.packed_digits_bytes(w_len, tail_bits)
        total = commit_bytes + packed_bytes
        cand = {
            "d": d,
            "nb": nb,
            "ring_elems": ring_elems,
            "commit_bytes": commit_bytes,
            "packed_bytes": packed_bytes,
            "total": total,
        }
        if best is None or cand["total"] < best["total"]:
            best = cand
    return best


def suffix_breakdown(profile_name: str, w_len: int, prev_bound: int):
    total, start_d, levels, tail_lb = best_suffix(profile_name, w_len, prev_bound)
    final_w = levels[-1].next_w_len if levels else w_len
    tail = best_tail_breakdown(profile_name, final_w, tail_lb)
    levels_total = sum(level.level_bytes for level in levels)
    assert tail is not None
    assert levels_total + tail["total"] == total
    return {
        "total": total,
        "start_d": start_d,
        "levels": levels,
        "levels_total": levels_total,
        "tail_lb": tail_lb,
        "final_w": final_w,
        "tail": tail,
    }


def case_core_setup(case: BoundaryCase):
    hi_suffix_cost, _, _, _ = best_suffix(case.hi_profile_name, case.witness_len, case.prev_bound)
    lo_suffix_cost, _, lo_levels, _ = best_suffix(case.lo_profile_name, case.witness_len, case.prev_bound)
    lo_first = lo_levels[0]
    t_lo = t_hat_len(lo_first)

    core_rows_slots = case.d_hi * (case.n_a_hi + case.n_b_hi)
    t_hi = case.d_hi * case.n_a_hi * case.num_blocks_hi * case.delta_open_hi
    support = case.n_a_hi * case.num_blocks_hi * case.delta_open_hi
    s_old = support * case.eta
    live_x_cols = -(-case.witness_len // case.d_hi)
    return {
        "hi_suffix_cost": hi_suffix_cost,
        "lo_suffix_cost": lo_suffix_cost,
        "lo_first": lo_first,
        "t_lo": t_lo,
        "core_rows_slots": core_rows_slots,
        "t_hi": t_hi,
        "support": support,
        "s_old": s_old,
        "live_x_cols": live_x_cols,
    }


def search_case(case: BoundaryCase):
    setup = case_core_setup(case)
    hi_suffix_cost = setup["hi_suffix_cost"]
    lo_suffix_cost = setup["lo_suffix_cost"]
    t_lo = setup["t_lo"]
    core_rows_slots = setup["core_rows_slots"]
    open_rows_slots = case.d_hi * (case.n_d_hi + 2)
    trace_rows = 1
    t_hi = setup["t_hi"]
    support = setup["support"]
    s_old = setup["s_old"]
    old_elem_bytes = (case.q_hi_bits + 7) // 8
    old_explicit_open_bytes = (case.n_d_hi + 1) * case.d_hi * old_elem_bytes

    best = None
    for log_b in range(1, 31):
        b = 1 << log_b
        if b >= case.q_lo:
            break
        m_q = min_digits(case.q_hi_bits, b)
        m_k = max(1, math.ceil(math.log2(s_old + 1) / log_b))
        mu = min(m_q, m_k)
        c_bound = (b // 2) * (s_old + 1) + (mu * (b**2)) // 4
        if 2 * c_bound >= case.q_lo:
            continue
        f_bound = math.ceil(c_bound / (b - 1))

        core_aux = core_rows_slots * (m_q + 2 * m_k - 1)
        open_aux = open_rows_slots * (m_q + 2 * m_k - 1)
        trace_aux = trace_rows * (m_q + 2 * m_k - 1)
        total_witness = case.witness_len + t_hi + core_aux + open_aux + trace_aux + t_lo
        bridge_prev_bound = max(case.prev_bound, bitlen(b // 2), bitlen(f_bound))
        bridge_full_cost, _, _, _ = best_suffix(
            case.bridge_profile_name, total_witness, bridge_prev_bound
        )
        bridge_full_cost += old_explicit_open_bytes
        fused_overhead = bridge_full_cost - lo_suffix_cost

        cand = {
            "base": b,
            "m_q": m_q,
            "m_k": m_k,
            "f_bound": f_bound,
            "core_rows_slots": core_rows_slots,
            "open_rows_slots": open_rows_slots,
            "trace_rows": trace_rows,
            "t_hi": t_hi,
            "t_lo": t_lo,
            "support": support,
            "s_old": s_old,
            "core_aux": core_aux,
            "open_aux": open_aux,
            "trace_aux": trace_aux,
            "y_ring_digits": 0,
            "old_explicit_open_bytes": old_explicit_open_bytes,
            "bridge_witness": total_witness,
            "bridge_prev_bound": bridge_prev_bound,
            "hi_suffix_cost": hi_suffix_cost,
            "lo_suffix_cost": lo_suffix_cost,
            "bridge_full_cost": bridge_full_cost,
            "fused_overhead": fused_overhead,
            "bridge_budget": hi_suffix_cost - lo_suffix_cost,
            "net_after_bridge": hi_suffix_cost - (lo_suffix_cost + fused_overhead),
            "model": "replay_structured",
        }
        if best is None or (
            cand["fused_overhead"],
            cand["bridge_full_cost"],
            cand["bridge_witness"],
        ) < (
            best["fused_overhead"],
            best["bridge_full_cost"],
            best["bridge_witness"],
        ):
            best = cand
    return best


def search_case_direct_y_ring(case: BoundaryCase):
    setup = case_core_setup(case)
    hi_suffix_cost = setup["hi_suffix_cost"]
    lo_suffix_cost = setup["lo_suffix_cost"]
    t_lo = setup["t_lo"]
    core_rows_slots = setup["core_rows_slots"]
    t_hi = setup["t_hi"]
    s_old = setup["s_old"]
    live_x_cols = setup["live_x_cols"]

    best = None
    for log_b in range(1, 31):
        b = 1 << log_b
        if b >= case.q_lo:
            break
        m_q = min_digits(case.q_hi_bits, b)

        m_k_core = max(1, math.ceil(math.log2(s_old + 1) / log_b))
        mu_core = min(m_q, m_k_core)
        c_core = (b // 2) * (s_old + 1) + (mu_core * (b**2)) // 4

        s_open = live_x_cols * case.eta
        m_k_open = max(1, math.ceil(math.log2(s_open + 1) / log_b))
        mu_open = min(m_q, m_k_open)
        c_open = (b // 2) * (s_open + 1) + (mu_open * (b**2)) // 4

        c_trace = (b // 2) * (case.d_hi + 1) + (min(m_q, 1) * (b**2)) // 4
        worst_c = max(c_core, c_open, c_trace)
        if 2 * worst_c >= case.q_lo:
            continue
        f_bound = math.ceil(worst_c / (b - 1))

        core_aux = core_rows_slots * (m_q + 2 * m_k_core - 1)
        y_ring_digits = case.d_hi * m_q
        open_rows = case.d_hi + 1
        open_aux = open_rows * (m_q + 2 * m_k_open - 1)
        total_witness = case.witness_len + t_hi + core_aux + y_ring_digits + open_aux + t_lo
        bridge_prev_bound = max(case.prev_bound, bitlen(b // 2), bitlen(f_bound))
        bridge_full_cost, _, _, _ = best_suffix(
            case.bridge_profile_name, total_witness, bridge_prev_bound
        )
        fused_overhead = bridge_full_cost - lo_suffix_cost

        cand = {
            "base": b,
            "m_q": m_q,
            "m_k_core": m_k_core,
            "m_k_open": m_k_open,
            "f_bound": f_bound,
            "core_rows_slots": core_rows_slots,
            "open_rows_slots": open_rows,
            "trace_rows": 1,
            "t_hi": t_hi,
            "t_lo": t_lo,
            "support": setup["support"],
            "s_old": s_old,
            "s_open": s_open,
            "core_aux": core_aux,
            "y_ring_digits": y_ring_digits,
            "open_aux": open_aux,
            "trace_aux": 0,
            "old_explicit_open_bytes": 0,
            "bridge_witness": total_witness,
            "bridge_prev_bound": bridge_prev_bound,
            "hi_suffix_cost": hi_suffix_cost,
            "lo_suffix_cost": lo_suffix_cost,
            "bridge_full_cost": bridge_full_cost,
            "fused_overhead": fused_overhead,
            "bridge_budget": hi_suffix_cost - lo_suffix_cost,
            "net_after_bridge": hi_suffix_cost - (lo_suffix_cost + fused_overhead),
            "model": "direct_y_ring",
        }
        if best is None or (
            cand["fused_overhead"],
            cand["bridge_full_cost"],
            cand["bridge_witness"],
        ) < (
            best["fused_overhead"],
            best["bridge_full_cost"],
            best["bridge_witness"],
        ):
            best = cand
    return best


def search_case_direct_dense(case: BoundaryCase):
    setup = case_core_setup(case)
    hi_suffix_cost = setup["hi_suffix_cost"]
    lo_suffix_cost = setup["lo_suffix_cost"]
    t_lo = setup["t_lo"]
    core_rows_slots = setup["core_rows_slots"]
    t_hi = setup["t_hi"]
    s_old = setup["s_old"]

    best = None
    for log_b in range(1, 31):
        b = 1 << log_b
        if b >= case.q_lo:
            break
        m_q = min_digits(case.q_hi_bits, b)

        m_k_core = max(1, math.ceil(math.log2(s_old + 1) / log_b))
        mu_core = min(m_q, m_k_core)
        c_core = (b // 2) * (s_old + 1) + (mu_core * (b**2)) // 4

        s_open = case.witness_len * case.eta
        m_k_open = max(1, math.ceil(math.log2(s_open + 1) / log_b))
        mu_open = min(m_q, m_k_open)
        c_open = (b // 2) * (s_open + 1) + (mu_open * (b**2)) // 4

        worst_c = max(c_core, c_open)
        if 2 * worst_c >= case.q_lo:
            continue
        f_bound = math.ceil(worst_c / (b - 1))

        core_aux = core_rows_slots * (m_q + 2 * m_k_core - 1)
        open_aux = m_q + 2 * m_k_open - 1
        total_witness = case.witness_len + t_hi + core_aux + open_aux + t_lo
        bridge_prev_bound = max(case.prev_bound, bitlen(b // 2), bitlen(f_bound))
        bridge_full_cost, _, _, _ = best_suffix(
            case.bridge_profile_name, total_witness, bridge_prev_bound
        )
        fused_overhead = bridge_full_cost - lo_suffix_cost

        cand = {
            "base": b,
            "m_q": m_q,
            "m_k_core": m_k_core,
            "m_k_open": m_k_open,
            "f_bound": f_bound,
            "core_rows_slots": core_rows_slots,
            "open_rows_slots": 1,
            "trace_rows": 0,
            "t_hi": t_hi,
            "t_lo": t_lo,
            "support": setup["support"],
            "s_old": s_old,
            "s_open": s_open,
            "core_aux": core_aux,
            "y_ring_digits": 0,
            "open_aux": open_aux,
            "trace_aux": 0,
            "old_explicit_open_bytes": 0,
            "bridge_witness": total_witness,
            "bridge_prev_bound": bridge_prev_bound,
            "hi_suffix_cost": hi_suffix_cost,
            "lo_suffix_cost": lo_suffix_cost,
            "bridge_full_cost": bridge_full_cost,
            "fused_overhead": fused_overhead,
            "bridge_budget": hi_suffix_cost - lo_suffix_cost,
            "net_after_bridge": hi_suffix_cost - (lo_suffix_cost + fused_overhead),
            "model": "direct_dense",
        }
        if best is None or (
            cand["fused_overhead"],
            cand["bridge_full_cost"],
            cand["bridge_witness"],
        ) < (
            best["fused_overhead"],
            best["bridge_full_cost"],
            best["bridge_witness"],
        ):
            best = cand
    return best


VARIANT_SEARCHERS = [
    ("replay_structured", search_case),
    ("direct_y_ring", search_case_direct_y_ring),
    ("direct_dense", search_case_direct_dense),
]


def print_128_to_32_detail() -> None:
    case = next(c for c in CASES if c.name == "128->32 late onehot")
    best = search_case(case)
    direct_y_ring = search_case_direct_y_ring(case)
    direct_dense = search_case_direct_dense(case)
    lo = suffix_breakdown(case.lo_profile_name, case.witness_len, case.prev_bound)
    bridge = suffix_breakdown(
        case.bridge_profile_name,
        best["bridge_witness"],
        best["bridge_prev_bound"],
    )
    row_a = case.d_hi * case.n_a_hi
    row_b = case.d_hi * case.n_b_hi
    row_d = case.d_hi * case.n_d_hi
    row_public = case.d_hi
    row_fold = case.d_hi
    k_old_core = best["core_rows_slots"] * best["m_k"]
    f_old_core = best["core_rows_slots"] * (best["m_q"] + best["m_k"] - 1)
    k_old_open = (best["open_rows_slots"] + best["trace_rows"]) * best["m_k"]
    f_old_open = (best["open_rows_slots"] + best["trace_rows"]) * (best["m_q"] + best["m_k"] - 1)
    assert (
        case.witness_len
        + best["t_hi"]
        + best["core_aux"]
        + best["open_aux"]
        + best["trace_aux"]
        + best["t_lo"]
        == best["bridge_witness"]
    )

    lo_first = lo["levels"][0]
    bridge_first = bridge["levels"][0]
    tail_reduction = lo["tail"]["total"] - bridge["tail"]["total"]
    extra_level_bytes = bridge["levels_total"] - lo["levels_total"]
    structural_overhead = best["fused_overhead"] - extra_level_bytes + tail_reduction

    print("Detailed 128->32 late onehot anatomy:")
    print(
        "  variants          = "
        f"replay {best['bridge_full_cost']} B, "
        f"direct_y_ring {direct_y_ring['bridge_full_cost']} B, "
        f"direct_dense {direct_dense['bridge_full_cost']} B"
    )
    print(
        "  old boundary      = "
        f"D_hi={case.d_hi}, n_a={case.n_a_hi}, n_b={case.n_b_hi}, n_d={case.n_d_hi}, "
        f"blocks={case.num_blocks_hi}, "
        f"delta_open={case.delta_open_hi}, N={case.witness_len}, eta={case.eta}"
    )
    print(f"  old row slots     = A:{row_a}, B:{row_b}, D:{row_d}, y:{row_public}, fold:{row_fold}, trace:1")
    print(
        "  radix             = "
        f"B={best['base']}, m_q={best['m_q']}, m_k={best['m_k']}, "
        f"carry_bound={best['f_bound']}"
    )
    print(
        "  witness pieces    = "
        f"w {case.witness_len}, t_hat_hi {best['t_hi']}, "
        f"k_core {k_old_core}, f_core {f_old_core}, "
        f"k_open {k_old_open}, f_open {f_old_open}, "
        f"t_hat_lo {best['t_lo']}"
    )
    print(
        "  explicit old objs = "
        f"v_hi+y_ring_hi {best['old_explicit_open_bytes']} B"
    )
    print(
        "  native 32 suffix  = "
        f"levels {[level.level_bytes for level in lo['levels']]}, "
        f"tail {lo['tail']['total']} (= commit {lo['tail']['commit_bytes']} "
        f"+ packed {lo['tail']['packed_bytes']})"
    )
    print(
        "  bridge full       = "
        f"levels {[level.level_bytes for level in bridge['levels']]}, "
        f"tail {bridge['tail']['total']} (= commit {bridge['tail']['commit_bytes']} "
        f"+ packed {bridge['tail']['packed_bytes']})"
    )
    print(
        "  first level delta = "
        f"native D{lo_first.d}/lb{lo_first.lb}/m{lo_first.m_vars}/r{lo_first.r_vars}"
        f"/na{lo_first.na}/nb{lo_first.nb}/nd{lo_first.nd} -> {lo_first.level_bytes} B; "
        f"bridge D{bridge_first.d}/lb{bridge_first.lb}/m{bridge_first.m_vars}"
        f"/r{bridge_first.r_vars}/na{bridge_first.na}/nb{bridge_first.nb}"
        f"/nd{bridge_first.nd} -> {bridge_first.level_bytes} B"
    )
    print(
        "  overhead          = "
        f"extra levels {extra_level_bytes} + explicit old objs "
        f"{best['old_explicit_open_bytes']} - tail reduction {tail_reduction} "
        f"+ residual structural {structural_overhead} = fused overhead {best['fused_overhead']}"
    )
    print()


CASES = [
    BoundaryCase(
        name="64->32 early onehot",
        hi_profile_name="64b",
        lo_profile_name="32b",
        bridge_profile_name="32",
        q_hi_bits=64,
        q_lo=(1 << 32) - 99,
        witness_len=1_704_320,
        prev_bound=1,
        eta=1,
        d_hi=64,
        n_a_hi=1,
        n_b_hi=1,
        n_d_hi=1,
        num_blocks_hi=128,
        delta_open_hi=64,
    ),
    BoundaryCase(
        name="64->32 late onehot",
        hi_profile_name="64b",
        lo_profile_name="32",
        bridge_profile_name="32",
        q_hi_bits=64,
        q_lo=(1 << 32) - 99,
        witness_len=171_456,
        prev_bound=2,
        eta=2,
        d_hi=64,
        n_a_hi=1,
        n_b_hi=1,
        n_d_hi=1,
        num_blocks_hi=16,
        delta_open_hi=33,
    ),
    BoundaryCase(
        name="128->64 late onehot",
        hi_profile_name="128",
        lo_profile_name="64",
        bridge_profile_name="64",
        q_hi_bits=128,
        q_lo=(1 << 64) - 59,
        witness_len=199_584,
        prev_bound=4,
        eta=8,
        d_hi=32,
        n_a_hi=2,
        n_b_hi=2,
        n_d_hi=2,
        num_blocks_hi=32,
        delta_open_hi=33,
    ),
    BoundaryCase(
        name="128->32 late onehot",
        hi_profile_name="128",
        lo_profile_name="32",
        bridge_profile_name="32",
        q_hi_bits=128,
        q_lo=(1 << 32) - 99,
        witness_len=199_584,
        prev_bound=4,
        eta=8,
        d_hi=32,
        n_a_hi=2,
        n_b_hi=2,
        n_d_hi=2,
        num_blocks_hi=32,
        delta_open_hi=33,
    ),
    BoundaryCase(
        name="128->k7pack late onehot",
        hi_profile_name="128",
        lo_profile_name="k7pack",
        bridge_profile_name="k7pack",
        q_hi_bits=128,
        q_lo=319_541,
        witness_len=199_584,
        prev_bound=4,
        eta=8,
        d_hi=32,
        n_a_hi=2,
        n_b_hi=2,
        n_d_hi=2,
        num_blocks_hi=32,
        delta_open_hi=33,
    ),
]


def main() -> None:
    print("Modulus-switch bridge model")
    print("Assumption: fused commitment-boundary bridge")
    print("Variants: replay_structured, direct_y_ring, direct_dense")
    print()
    for case in CASES:
        print(f"{case.name}:")
        for variant_name, searcher in VARIANT_SEARCHERS:
            best = searcher(case)
            if best is None:
                print(f"  {variant_name}: no feasible radix")
                continue
            m_k_label = (
                f"{best['m_k']}"
                if "m_k" in best
                else f"core {best['m_k_core']} / open {best['m_k_open']}"
            )
            print(f"  {variant_name}:")
            print(f"    radix B          = {best['base']}")
            print(f"    m_q / m_k        = {best['m_q']} / {m_k_label}")
            print(f"    old support S    = {best['s_old']}")
            print(
                "    row slots        = "
                f"core {best['core_rows_slots']}, open {best['open_rows_slots']}, trace {best['trace_rows']}"
            )
            print(f"    t_hi / t_lo      = {best['t_hi']} / {best['t_lo']}")
            print(f"    y_ring digits    = {best['y_ring_digits']}")
            print(f"    carry bound      = {best['f_bound']}")
            print(f"    explicit old obj = {best['old_explicit_open_bytes']} B")
            print(f"    bridge witness   = {best['bridge_witness']}")
            print(f"    prev_bound bits  = {best['bridge_prev_bound']}")
            print(f"    stay suffix      = {best['hi_suffix_cost']} B")
            print(f"    lo suffix        = {best['lo_suffix_cost']} B")
            print(f"    bridge budget    = {best['bridge_budget']} B")
            print(f"    bridge full      = {best['bridge_full_cost']} B")
            print(f"    fused overhead   = {best['fused_overhead']} B")
            print(f"    net after bridge = {best['net_after_bridge']} B")
        print()
    print_128_to_32_detail()


if __name__ == "__main__":
    main()
