#!/usr/bin/env python3
"""Search proof-byte-optimal Hachi schedules under the current serializer model.

This script mirrors the byte formulas used by the Rust proof objects and the
current D=64 recursive witness recurrence:

- `PackedDigits`: `8 + 1 + ceil(num_elems * bits_per_elem / 8)`
- `FlatRingVec`: `4 + 8 + ring_len * D * field_bytes`
- `HachiLevelProof`: `y_ring + v + body_tag + body`
- recursive witness length:
  `w = w_hat + t_hat + z_pre + r`

It is meant for planner / design-note work, not for in-protocol decision
making. The search space is intentionally the same family explored in the
recent notes:

- `D = 64`
- `N_A = N_B = N_D = 1`
- root `nv` supplied on the command line
- per-level `log_basis` may increase, but never decrease, because recursive
  witnesses consist of balanced digits from the previous level
- root family is one of:
  - `onehot`: `log_commit_bound = 1` with the tighter first-level onehot
    folded bound
  - `log`: `log_commit_bound = log_basis`
  - `full`: `log_commit_bound = 128`

The `b = 4` / `log_basis = 2` case can be evaluated either with the historical
two-stage model or with the current combined single-stage serializer path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal


FIELD_BYTES = 16
Q_BITS = 128
Q = (1 << 128) - 275
HALF_Q = Q // 2
D = 64
ALPHA = D.bit_length() - 1
CHALLENGE_MASS = 54
MAX_ABS_CHALLENGE_COEFF = 2

N_A = 1
N_B = 1
N_D = 1
M_ROW_COUNT = N_A + N_B + N_D + 2

ROOT_FAMILIES = ("onehot", "log", "full")
B4_MODELS = ("combined", "two-stage", "both")


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def compute_num_digits(log_bound: int, log_basis: int) -> int:
    """Mirror `compute_num_digits()` from `src/protocol/commitment/config.rs`."""

    if log_basis <= 0 or log_basis >= 128:
        raise ValueError(f"invalid log_basis={log_basis}")
    if log_bound == 0:
        return 1

    levels = ceil_div(log_bound, log_basis)
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


def compute_num_digits_full_field(field_bits: int, log_basis: int) -> int:
    """Asymmetric centering: ceil(field_bits / log_basis) with no +1."""
    if log_basis <= 0 or log_basis >= 128:
        raise ValueError(f"invalid log_basis={log_basis}")
    if field_bits == 0:
        return 1
    return max(ceil_div(field_bits, log_basis), 1)


def num_digits_for_bound(log_bound: int, log_basis: int) -> int:
    """Select asymmetric or symmetric centering based on bound width."""
    if log_bound >= 128:
        return compute_num_digits_full_field(log_bound, log_basis)
    return compute_num_digits(log_bound, log_basis)


def compute_num_digits_fold_dense(r_vars: int, log_basis: int) -> int:
    """Mirror `compute_num_digits_fold()` for the generic dense bound."""

    shift = r_vars + log_basis - 1
    if shift >= 127 or CHALLENGE_MASS == 0:
        return compute_num_digits(Q_BITS, log_basis)
    beta = CHALLENGE_MASS * (1 << shift)
    if beta == 0:
        return 1
    return compute_num_digits(beta.bit_length(), log_basis)


def compute_num_digits_fold_onehot_root(r_vars: int, log_basis: int) -> int:
    """Tighter first-level folded bound for the sparse onehot family."""

    beta = (1 << r_vars) * MAX_ABS_CHALLENGE_COEFF
    return compute_num_digits(beta.bit_length(), log_basis)


def r_decomp_levels(log_basis: int) -> int:
    """Mirror `r_decomp_levels()` from `src/protocol/ring_switch.rs`.

    Always full-field, so uses asymmetric centering (no +1 correction).
    """
    bits = (Q - 1).bit_length()
    return max(compute_num_digits_full_field(bits, log_basis), 1)


def flat_ring_vec_bytes(ring_len: int) -> int:
    coeff_count = ring_len * D
    return 4 + 8 + coeff_count * FIELD_BYTES


def packed_digits_bytes(num_elems: int, bits_per_elem: int) -> int:
    return 8 + 1 + ceil_div(num_elems * bits_per_elem, 8)


def compressed_unipoly_bytes(degree: int) -> int:
    return 8 + degree * FIELD_BYTES


def sumcheck_bytes(rounds: int, degree: int) -> int:
    return 8 + rounds * compressed_unipoly_bytes(degree)


@dataclass(frozen=True)
class LevelTransition:
    level: int
    log_basis: int
    m_vars: int
    r_vars: int
    num_blocks: int
    block_len: int
    num_digits_commit: int
    num_digits_open: int
    num_digits_fold: int
    rounds: int
    next_w_len: int
    level_total: int
    y_bytes: int
    v_bytes: int
    normcheck_kind: Literal["combined", "two-stage"]
    normcheck_a_bytes: int
    normcheck_b_bytes: int
    next_commit_bytes: int
    next_eval_bytes: int

    def total_if_stop(self, suffix_bytes: int) -> int:
        return self.level_total + suffix_bytes


@dataclass(frozen=True)
class SearchResult:
    family: str
    b4_model: str
    nv: int
    no_wrapper_bytes: int
    exact_proof_bytes: int
    final_tail_bytes: int
    final_tail_basis: int
    final_w_len: int
    levels: tuple[LevelTransition, ...]


def rounds_for_next_w(next_w_ring_elems: int) -> int:
    return (max(next_w_ring_elems, 1) - 1).bit_length() + ALPHA


def reduced_vars_for_w_len(w_len: int) -> int:
    num_ring_elems = w_len // D
    total = 1 << (max(num_ring_elems, 1) - 1).bit_length()
    return total.bit_length() - 1


def root_delta_commit_bits(family: str, log_basis: int) -> int:
    if family == "onehot":
        return compute_num_digits(1, log_basis)
    if family == "log":
        return compute_num_digits(log_basis, log_basis)
    if family == "full":
        return num_digits_for_bound(Q_BITS, log_basis)
    raise ValueError(f"unknown family={family}")


def root_delta_fold(family: str, r_vars: int, log_basis: int) -> int:
    if family == "onehot":
        return compute_num_digits_fold_onehot_root(r_vars, log_basis)
    if family in {"log", "full"}:
        return compute_num_digits_fold_dense(r_vars, log_basis)
    raise ValueError(f"unknown family={family}")


def level_bytes(
    *,
    log_basis: int,
    rounds: int,
    b4_model: str,
) -> tuple[int, str, int, int]:
    y_bytes = flat_ring_vec_bytes(1)
    v_bytes = flat_ring_vec_bytes(N_D)
    next_commit_bytes = flat_ring_vec_bytes(N_B)
    next_eval_bytes = FIELD_BYTES

    if log_basis == 2 and b4_model == "combined":
        combined_bytes = sumcheck_bytes(rounds, 5)
        total = y_bytes + v_bytes + 1 + combined_bytes + next_commit_bytes + next_eval_bytes
        return total, "combined", combined_bytes, 0

    if log_basis == 2:
        stage1_degree = 3
    else:
        stage1_degree = (1 << (log_basis - 1)) + 1
    stage1_bytes = sumcheck_bytes(rounds, stage1_degree)
    stage2_bytes = sumcheck_bytes(rounds, 3)
    total = (
        y_bytes
        + v_bytes
        + 1
        + stage1_bytes
        + FIELD_BYTES
        + stage2_bytes
        + next_commit_bytes
        + next_eval_bytes
    )
    return total, "two-stage", stage1_bytes + FIELD_BYTES, stage2_bytes


def make_transition(
    *,
    level: int,
    log_basis: int,
    m_vars: int,
    r_vars: int,
    num_digits_commit: int,
    num_digits_open: int,
    num_digits_fold: int,
    next_w_ring_elems: int,
    b4_model: str,
) -> LevelTransition:
    rounds = rounds_for_next_w(next_w_ring_elems)
    level_total, normcheck_kind, normcheck_a_bytes, normcheck_b_bytes = level_bytes(
        log_basis=log_basis,
        rounds=rounds,
        b4_model=b4_model,
    )
    return LevelTransition(
        level=level,
        log_basis=log_basis,
        m_vars=m_vars,
        r_vars=r_vars,
        num_blocks=1 << r_vars,
        block_len=1 << m_vars,
        num_digits_commit=num_digits_commit,
        num_digits_open=num_digits_open,
        num_digits_fold=num_digits_fold,
        rounds=rounds,
        next_w_len=next_w_ring_elems * D,
        level_total=level_total,
        y_bytes=flat_ring_vec_bytes(1),
        v_bytes=flat_ring_vec_bytes(N_D),
        normcheck_kind=normcheck_kind,
        normcheck_a_bytes=normcheck_a_bytes,
        normcheck_b_bytes=normcheck_b_bytes,
        next_commit_bytes=flat_ring_vec_bytes(N_B),
        next_eval_bytes=FIELD_BYTES,
    )


def recursive_transitions(
    *,
    current_w_len: int,
    min_log_basis: int,
    max_log_basis: int,
    b4_model: str,
) -> list[LevelTransition]:
    reduced_vars = reduced_vars_for_w_len(current_w_len)
    if reduced_vars <= 1:
        return []

    transitions: list[LevelTransition] = []
    for log_basis in range(min_log_basis, max_log_basis + 1):
        num_digits_commit = 1
        num_digits_open = num_digits_for_bound(Q_BITS, log_basis)
        r_levels = r_decomp_levels(log_basis)
        for r_vars in range(1, reduced_vars):
            m_vars = reduced_vars - r_vars
            num_digits_fold = compute_num_digits_fold_dense(r_vars, log_basis)
            num_blocks = 1 << r_vars
            block_len = 1 << m_vars
            next_w_ring_elems = (
                num_blocks * num_digits_open
                + num_blocks * N_A * num_digits_open
                + block_len * num_digits_commit * num_digits_fold
                + M_ROW_COUNT * r_levels
            )
            next_w_len = next_w_ring_elems * D
            if next_w_len >= current_w_len:
                continue
            transitions.append(
                make_transition(
                    level=-1,
                    log_basis=log_basis,
                    m_vars=m_vars,
                    r_vars=r_vars,
                    num_digits_commit=num_digits_commit,
                    num_digits_open=num_digits_open,
                    num_digits_fold=num_digits_fold,
                    next_w_ring_elems=next_w_ring_elems,
                    b4_model=b4_model,
                )
            )
    return transitions


def root_transitions(
    *,
    family: str,
    nv: int,
    max_log_basis: int,
    b4_model: str,
) -> list[LevelTransition]:
    reduced_vars = nv - ALPHA
    if reduced_vars <= 1:
        raise ValueError(f"nv={nv} is too small for D={D}")

    transitions: list[LevelTransition] = []
    for log_basis in range(2, max_log_basis + 1):
        num_digits_commit = root_delta_commit_bits(family, log_basis)
        num_digits_open = num_digits_for_bound(Q_BITS, log_basis)
        r_levels = r_decomp_levels(log_basis)
        for r_vars in range(1, reduced_vars):
            m_vars = reduced_vars - r_vars
            num_digits_fold = root_delta_fold(family, r_vars, log_basis)
            num_blocks = 1 << r_vars
            block_len = 1 << m_vars
            next_w_ring_elems = (
                num_blocks * num_digits_open
                + num_blocks * N_A * num_digits_open
                + block_len * num_digits_commit * num_digits_fold
                + M_ROW_COUNT * r_levels
            )
            next_w_len = next_w_ring_elems * D
            if next_w_len >= (1 << nv):
                continue
            transitions.append(
                make_transition(
                    level=0,
                    log_basis=log_basis,
                    m_vars=m_vars,
                    r_vars=r_vars,
                    num_digits_commit=num_digits_commit,
                    num_digits_open=num_digits_open,
                    num_digits_fold=num_digits_fold,
                    next_w_ring_elems=next_w_ring_elems,
                    b4_model=b4_model,
                )
            )
    return transitions


def search_family(
    *,
    family: str,
    nv: int,
    max_log_basis: int,
    b4_model: str,
) -> SearchResult:
    if family not in ROOT_FAMILIES:
        raise ValueError(f"unknown family={family}")

    @lru_cache(maxsize=None)
    def best_suffix(current_w_len: int, current_log_basis: int) -> tuple[int, tuple[LevelTransition, ...], int, int]:
        best_total = packed_digits_bytes(current_w_len, current_log_basis)
        best_levels: tuple[LevelTransition, ...] = ()
        best_tail_bytes = best_total
        best_tail_basis = current_log_basis

        for transition in recursive_transitions(
            current_w_len=current_w_len,
            min_log_basis=current_log_basis,
            max_log_basis=max_log_basis,
            b4_model=b4_model,
        ):
            suffix_total, suffix_levels, tail_bytes, tail_basis = best_suffix(
                transition.next_w_len,
                transition.log_basis,
            )
            total = transition.level_total + suffix_total
            if total < best_total:
                best_total = total
                best_levels = (transition,) + suffix_levels
                best_tail_bytes = tail_bytes
                best_tail_basis = tail_basis

        return best_total, best_levels, best_tail_bytes, best_tail_basis

    best_total = None
    best_levels: tuple[LevelTransition, ...] = ()
    best_tail_bytes = 0
    best_tail_basis = 0
    best_final_w_len = 0

    for root in root_transitions(
        family=family,
        nv=nv,
        max_log_basis=max_log_basis,
        b4_model=b4_model,
    ):
        suffix_total, suffix_levels, tail_bytes, tail_basis = best_suffix(
            root.next_w_len,
            root.log_basis,
        )
        total = root.level_total + suffix_total
        if best_total is None or total < best_total:
            best_total = total
            best_levels = (root,) + suffix_levels
            best_tail_bytes = tail_bytes
            best_tail_basis = tail_basis
            best_final_w_len = best_levels[-1].next_w_len

    assert best_total is not None
    return SearchResult(
        family=family,
        b4_model=b4_model,
        nv=nv,
        no_wrapper_bytes=best_total,
        exact_proof_bytes=best_total + 5,
        final_tail_bytes=best_tail_bytes,
        final_tail_basis=best_tail_basis,
        final_w_len=best_final_w_len,
        levels=best_levels,
    )


def level_dicts(result: SearchResult) -> list[dict[str, int | str]]:
    running = 0
    out: list[dict[str, int | str]] = []
    for idx, level in enumerate(result.levels):
        running += level.level_total
        out.append(
            {
                "level": idx,
                "log_basis": level.log_basis,
                "m_vars": level.m_vars,
                "r_vars": level.r_vars,
                "num_blocks": level.num_blocks,
                "block_len": level.block_len,
                "num_digits_commit": level.num_digits_commit,
                "num_digits_open": level.num_digits_open,
                "num_digits_fold": level.num_digits_fold,
                "rounds": level.rounds,
                "next_w_len": level.next_w_len,
                "y_bytes": level.y_bytes,
                "v_bytes": level.v_bytes,
                "normcheck_kind": level.normcheck_kind,
                "normcheck_a_bytes": level.normcheck_a_bytes,
                "normcheck_b_bytes": level.normcheck_b_bytes,
                "next_commit_bytes": level.next_commit_bytes,
                "next_eval_bytes": level.next_eval_bytes,
                "level_total": level.level_total,
                "total_if_stop": running + packed_digits_bytes(level.next_w_len, level.log_basis),
            }
        )
    return out


def result_payload(result: SearchResult) -> dict[str, object]:
    return {
        "family": result.family,
        "b4_model": result.b4_model,
        "nv": result.nv,
        "no_wrapper_bytes": result.no_wrapper_bytes,
        "exact_proof_bytes": result.exact_proof_bytes,
        "final_tail_bytes": result.final_tail_bytes,
        "final_tail_basis": result.final_tail_basis,
        "final_w_len": result.final_w_len,
        "schedule": [level.log_basis for level in result.levels],
        "levels": level_dicts(result),
    }


def print_result(result: SearchResult) -> None:
    payload = result_payload(result)
    print(f"## {payload['family']} ({payload['b4_model']})")
    print()
    print(f"nv: {payload['nv']}")
    print(f"schedule: {payload['schedule']}")
    print(f"no_wrapper_bytes: {payload['no_wrapper_bytes']}")
    print(f"exact_proof_bytes: {payload['exact_proof_bytes']}")
    print(f"final_tail_basis: {payload['final_tail_basis']}")
    print(f"final_w_len: {payload['final_w_len']}")
    print(f"final_tail_bytes: {payload['final_tail_bytes']}")
    print()
    for level in payload["levels"]:
        print(
            "L{level}: basis={log_basis} (m={m_vars}, r={r_vars}) "
            "digits=({num_digits_commit},{num_digits_open},{num_digits_fold}) "
            "rounds={rounds} next_w={next_w_len} level_total={level_total} total_if_stop={total_if_stop}".format(
                **level
            )
        )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nv", type=int, default=30)
    parser.add_argument(
        "--family",
        choices=(*ROOT_FAMILIES, "all"),
        default="all",
        help="Root coefficient family to analyze.",
    )
    parser.add_argument(
        "--b4-model",
        choices=B4_MODELS,
        default="combined",
        help="How to model the b=4 normcheck payload.",
    )
    parser.add_argument(
        "--max-log-basis",
        type=int,
        default=7,
        help="Largest log_basis to consider. Search starts at 2.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    families = ROOT_FAMILIES if args.family == "all" else (args.family,)
    b4_models = ("two-stage", "combined") if args.b4_model == "both" else (args.b4_model,)

    results = [
        search_family(
            family=family,
            nv=args.nv,
            max_log_basis=args.max_log_basis,
            b4_model=b4_model,
        )
        for b4_model in b4_models
        for family in families
    ]

    if args.json:
        print(json.dumps([result_payload(result) for result in results], indent=2))
        return

    for idx, result in enumerate(results):
        if idx:
            print()
        print_result(result)


if __name__ == "__main__":
    main()
