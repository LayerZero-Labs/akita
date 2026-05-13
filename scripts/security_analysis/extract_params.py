#!/usr/bin/env python3
"""Extract SIS parameter quadruples from generated schedule tables.

Walks the six fp128_*_*.rs generated tables, parses every `GeneratedFoldStep`
struct literal, and emits a JSON document containing:

  - the per-preset stage-1 challenge family (D, omega, infinity_norm, shape),
  - every unique (a_collision_raw, a_width) and (b_collision_raw, b_width)
    parameter pair the production planner actually selects,
  - the SIS bucket each one rounds up to,
  - the rank the planner reads from sis_floor.rs,
  - residual MSIS margins (post-tensor extraction-degradation).

Run:

    python3 scripts/security_analysis/extract_params.py \
        > scripts/security_analysis/params.json
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
GENERATED_DIR = REPO / "crates" / "akita-types" / "src" / "generated"

# fp128_stage1_challenge_config (crates/akita-config/src/proof_optimized.rs).
# Each entry: (omega, infinity_norm, shape).
PRESET_FAMILY: Dict[int, Tuple[int, int, str]] = {
    32: (121, 8, "Flat"),    # BoundedL1Norm (truncated to |C| = 2^128)
    64: (54, 2, "Tensor"),   # ExactShell { count_mag1: 30, count_mag2: 12 }; omega = 30 + 2*12 = 54
    128: (32, 1, "Tensor"),  # Uniform { weight: 32, nonzero_coeffs: [-1, 1] }; omega = 32
}

# Sparse-challenge L1 mass `weight * max(|c|)`. For tensor, the
# *effective* L1 mass is omega**2 (logical block challenge after
# negacyclic product). LevelParams stores `challenge_l1_mass = omega**2`
# for tensor schedules in the generated tables; we verify that here.
def effective_l1_mass(d: int) -> int:
    omega, _, shape = PRESET_FAMILY[d]
    return omega * omega if shape == "Tensor" else omega


# 4 * omega tensor extraction-degradation; flat is 1.
def msis_extraction_degradation(d: int) -> int:
    omega, _, shape = PRESET_FAMILY[d]
    return 4 * omega if shape == "Tensor" else 1


# extraction_linf = degradation * base challenge L-inf.
def extraction_linf(d: int) -> int:
    _, inf_norm, _ = PRESET_FAMILY[d]
    return msis_extraction_degradation(d) * inf_norm


# SIS collision buckets per D (must match generated/sis_floor.rs::ceil_supported_collision).
SIS_BUCKETS: Dict[int, List[int]] = {
    32: [2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095,
         8191, 16383, 32767, 65535, 131071, 262143],
    64: [2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095,
         8191, 16383, 32767],
    128: [2, 3, 7, 15, 31, 63, 255, 511, 1023, 2047, 4095, 8191],
}


def ceil_supported_collision(d: int, value: int) -> Optional[int]:
    """Smallest bucket >= value, or None if value exceeds the largest bucket."""
    for bucket in SIS_BUCKETS[d]:
        if value <= bucket:
            return bucket
    return None


# Generated SIS rank/width table (mirrors sis_floor.rs::sis_max_widths).
# Loaded lazily from sis_floor.rs to avoid hand-duplicating numbers.
SIS_TABLE_RE = re.compile(
    r"\((\d+),\s*(\d+)\)\s*=>\s*Some\(\[\s*([\d_\s,]+?)\s*\]\)",
    re.MULTILINE | re.DOTALL,
)

# u32::checked_shl bounds; matches the generated sis_floor.rs::MAX_RANK assert.
MAX_RANK = 7


def load_sis_table() -> Dict[Tuple[int, int], List[int]]:
    text = (GENERATED_DIR / "sis_floor.rs").read_text()
    table: Dict[Tuple[int, int], List[int]] = {}
    for match in SIS_TABLE_RE.finditer(text):
        d = int(match.group(1))
        coll = int(match.group(2))
        widths = [int(w.replace("_", "").strip())
                  for w in match.group(3).split(",") if w.strip()]
        assert len(widths) == MAX_RANK, (d, coll, widths)
        table[(d, coll)] = widths
    return table


def min_rank_for_secure_width(table: Dict[Tuple[int, int], List[int]],
                              d: int, coll: int, width: int) -> Optional[int]:
    widths = table.get((d, coll))
    if widths is None:
        return None
    for i, max_w in enumerate(widths):
        if width <= max_w:
            return i + 1
    return None


@dataclass(frozen=True)
class FoldStep:
    preset: str
    max_num_vars: int
    step_index: int  # 0 = root, >=1 = recursive
    current_w_len: int
    d: int
    log_basis: int
    challenge_l1_mass: int
    m_vars: int
    r_vars: int
    n_a: int
    n_b: int
    n_d: int
    delta_open: int
    delta_fold: int
    delta_commit: int

    @property
    def is_root(self) -> bool:
        return self.step_index == 0

    @property
    def num_blocks(self) -> int:
        return 1 << self.r_vars

    @property
    def num_ring(self) -> int:
        return self.current_w_len // self.d

    @property
    def block_len(self) -> int:
        # Matches `LevelParams::with_decomp`:
        #   block_len = if num_ring > 0 { num_ring.div_ceil(num_blocks) }
        #               else { 1 << m_vars }
        # At root the planner passes num_ring = 0, so block_len = 2^m_vars.
        # At recursive levels, num_ring = current_w_len / D and the planner
        # uses the ceil-div form. Both prover and verifier rebuild the level
        # layout via `with_decomp`, so the live `inner_width` matches what we
        # compute here.
        if self.is_root:
            return 1 << self.m_vars
        return (self.num_ring + self.num_blocks - 1) // self.num_blocks

    @property
    def inner_width(self) -> int:
        return self.block_len * self.delta_commit

    @property
    def outer_width(self) -> int:
        return self.n_a * self.delta_open * self.num_blocks

    @property
    def d_matrix_width(self) -> int:
        return self.delta_open * self.num_blocks

    @property
    def bd_collision_raw(self) -> int:
        return (1 << self.log_basis) - 1

    @property
    def a_collision_raw(self) -> int:
        # `sis_derived_root_params_for_layout` uses a_raw=2 only at root
        # with `log_commit_bound == 1` (the onehot presets). The recursive
        # path in `sis_derived_recursive_params_for_layout` always uses
        # `bd_collision`. We approximate "is onehot root" by
        # (step_index == 0 and preset ends with `_onehot`).
        if self.is_root and self.preset.endswith("_onehot"):
            return 2
        return self.bd_collision_raw

    @property
    def derivation_shape(self) -> str:
        """The shape used by the planner when deriving the SIS rank floor.

        Root layouts call `apply_stage1_challenge_shape(...)` before deriving,
        so they see the production shape (Tensor for D=64/128, Flat for D=32).
        Recursive layouts in `sis_derived_recursive_params` build a tentative
        `LevelParams::params_only(...)` which defaults `stage1_challenge_shape`
        to `Flat`; the production shape is only applied *after* the rank floor
        has been picked. This mismatch is the suspected planner bug.
        """
        production = PRESET_FAMILY[self.d][2]
        if self.is_root:
            return production
        return "Flat"  # recursive levels derive under Flat assumption


FOLD_RE = re.compile(
    r"GeneratedFoldStep\s*\{\s*"
    r"current_w_len:\s*(?P<current_w_len>\d+)\s*,\s*"
    r"d:\s*(?P<d>\d+)\s*,\s*"
    r"log_basis:\s*(?P<log_basis>\d+)\s*,\s*"
    r"challenge_l1_mass:\s*(?P<challenge_l1_mass>\d+)\s*,\s*"
    r"m_vars:\s*(?P<m_vars>\d+)\s*,\s*"
    r"r_vars:\s*(?P<r_vars>\d+)\s*,\s*"
    r"n_a:\s*(?P<n_a>\d+)\s*,\s*"
    r"n_b:\s*(?P<n_b>\d+)\s*,\s*"
    r"n_d:\s*(?P<n_d>\d+)\s*,\s*"
    r"delta_open:\s*(?P<delta_open>\d+)\s*,\s*"
    r"delta_fold:\s*(?P<delta_fold>\d+)\s*,\s*"
    r"delta_commit:\s*(?P<delta_commit>\d+)\s*,"
)

# Match every entry start so we know the max_num_vars associated with each fold.
ENTRY_RE = re.compile(
    r"GeneratedScheduleTableEntry\s*\{\s*key:\s*GeneratedScheduleKey\s*\{\s*"
    r"max_num_vars:\s*(?P<max_num_vars>\d+),",
)


def parse_table(path: Path, preset: str) -> Iterator[FoldStep]:
    text = path.read_text()
    # Split the text by entry boundaries so each fold step is attributed to its
    # max_num_vars *and* its step index within the entry's `steps` array.
    entries: List[Tuple[int, str]] = []
    matches = list(ENTRY_RE.finditer(text))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entries.append((int(m.group("max_num_vars")), text[start:end]))
    for max_num_vars, body in entries:
        for step_index, fold_match in enumerate(FOLD_RE.finditer(body)):
            yield FoldStep(
                preset=preset,
                max_num_vars=max_num_vars,
                step_index=step_index,
                current_w_len=int(fold_match["current_w_len"]),
                d=int(fold_match["d"]),
                log_basis=int(fold_match["log_basis"]),
                challenge_l1_mass=int(fold_match["challenge_l1_mass"]),
                m_vars=int(fold_match["m_vars"]),
                r_vars=int(fold_match["r_vars"]),
                n_a=int(fold_match["n_a"]),
                n_b=int(fold_match["n_b"]),
                n_d=int(fold_match["n_d"]),
                delta_open=int(fold_match["delta_open"]),
                delta_fold=int(fold_match["delta_fold"]),
                delta_commit=int(fold_match["delta_commit"]),
            )


def shape_extraction_linf(d: int, shape: str) -> int:
    """Tensor extraction degradation `4*omega*infinity_norm` if Tensor, else just `infinity_norm`.

    Mirrors `LevelParams::stage1_extraction_infinity_norm` which is `degradation * infinity_norm`
    with `degradation = 1` for Flat and `4 * l1_norm` for Tensor.
    """
    omega, inf_norm, _ = PRESET_FAMILY[d]
    if shape == "Tensor":
        return 4 * omega * inf_norm
    return inf_norm


def analyse() -> Dict[str, object]:
    sis_table = load_sis_table()
    presets = [
        ("d32_full", GENERATED_DIR / "fp128_d32_full.rs"),
        ("d32_onehot", GENERATED_DIR / "fp128_d32_onehot.rs"),
        ("d64_full", GENERATED_DIR / "fp128_d64_full.rs"),
        ("d64_onehot", GENERATED_DIR / "fp128_d64_onehot.rs"),
        ("d128_full", GENERATED_DIR / "fp128_d128_full.rs"),
        ("d128_onehot", GENERATED_DIR / "fp128_d128_onehot.rs"),
    ]

    out: Dict[str, object] = {}
    for preset, path in presets:
        folds = list(parse_table(path, preset))
        # Group by (D, log_basis, n_a, n_b, n_d, delta_open, delta_fold,
        # delta_commit, m_vars, r_vars, step_index, current_w_len) so that
        # we keep step_index and recursive `block_len` (derived from
        # current_w_len) accurate per group.
        unique = {}
        for f in folds:
            key = (f.d, f.log_basis, f.n_a, f.n_b, f.n_d,
                   f.delta_open, f.delta_fold, f.delta_commit,
                   f.m_vars, f.r_vars, f.step_index, f.current_w_len)
            unique.setdefault(key, []).append(f)

        records = []
        for key, fs in sorted(unique.items()):
            f0 = fs[0]
            production_shape = PRESET_FAMILY[f0.d][2]

            # Two independent rank computations:
            # (1) "production": collision bucket using the runtime shape;
            #     this is the bucket the SIS instance actually faces at proof time.
            # (2) "derivation": collision bucket using the shape the planner
            #     saw when picking `n_a`. Equals "production" at root but
            #     `Flat` for recursive levels (the suspected bug).
            ext_linf_prod = shape_extraction_linf(f0.d, production_shape)
            ext_linf_deriv = shape_extraction_linf(f0.d, f0.derivation_shape)
            a_extraction_prod = f0.a_collision_raw * ext_linf_prod
            a_extraction_deriv = f0.a_collision_raw * ext_linf_deriv
            a_bucket_prod = ceil_supported_collision(f0.d, a_extraction_prod)
            a_bucket_deriv = ceil_supported_collision(f0.d, a_extraction_deriv)
            b_bucket = ceil_supported_collision(f0.d, f0.bd_collision_raw)
            d_bucket = b_bucket

            a_rank_prod = min_rank_for_secure_width(
                sis_table, f0.d, a_bucket_prod, f0.inner_width) if a_bucket_prod else None
            a_rank_deriv = min_rank_for_secure_width(
                sis_table, f0.d, a_bucket_deriv, f0.inner_width) if a_bucket_deriv else None
            b_rank = min_rank_for_secure_width(
                sis_table, f0.d, b_bucket, f0.outer_width) if b_bucket else None
            d_rank = min_rank_for_secure_width(
                sis_table, f0.d, d_bucket, f0.d_matrix_width) if d_bucket else None

            records.append({
                "key": {
                    "d": f0.d, "log_basis": f0.log_basis,
                    "n_a": f0.n_a, "n_b": f0.n_b, "n_d": f0.n_d,
                    "delta_open": f0.delta_open,
                    "delta_fold": f0.delta_fold,
                    "delta_commit": f0.delta_commit,
                    "m_vars": f0.m_vars,
                    "r_vars": f0.r_vars,
                    "step_index": f0.step_index,
                    "is_root": f0.is_root,
                    "current_w_len": f0.current_w_len,
                },
                "occurrences": len(fs),
                "max_num_vars_range": [
                    min(x.max_num_vars for x in fs),
                    max(x.max_num_vars for x in fs),
                ],
                "challenge_l1_mass": f0.challenge_l1_mass,
                "block_len": f0.block_len,
                "derivation_shape": f0.derivation_shape,
                "production_shape": production_shape,
                "extraction_linf_production": ext_linf_prod,
                "extraction_linf_derivation": ext_linf_deriv,
                "a_collision_raw": f0.a_collision_raw,
                "a_extraction_production": a_extraction_prod,
                "a_extraction_derivation": a_extraction_deriv,
                "a_bucket_production": a_bucket_prod,
                "a_bucket_derivation": a_bucket_deriv,
                "a_width": f0.inner_width,
                "a_rank_required_production": a_rank_prod,
                "a_rank_required_derivation": a_rank_deriv,
                "b_collision": f0.bd_collision_raw,
                "b_bucket": b_bucket,
                "b_width": f0.outer_width,
                "b_rank_required": b_rank,
                "d_collision": f0.bd_collision_raw,
                "d_bucket": d_bucket,
                "d_width": f0.d_matrix_width,
                "d_rank_required": d_rank,
                "stored_matches_derivation": {
                    "n_a": (a_rank_deriv == f0.n_a),
                    "n_b": (b_rank == f0.n_b),
                    "n_d": (d_rank == f0.n_d),
                },
                "stored_matches_production": {
                    "n_a": (a_rank_prod == f0.n_a),
                    "n_b": (b_rank == f0.n_b),
                    "n_d": (d_rank == f0.n_d),
                },
            })
        out[preset] = {
            "family": {
                "d": folds[0].d if folds else None,
                "omega": PRESET_FAMILY[folds[0].d][0] if folds else None,
                "infinity_norm": PRESET_FAMILY[folds[0].d][1] if folds else None,
                "shape": PRESET_FAMILY[folds[0].d][2] if folds else None,
                "msis_extraction_degradation": msis_extraction_degradation(folds[0].d) if folds else None,
                "extraction_linf": extraction_linf(folds[0].d) if folds else None,
                "effective_l1_mass": effective_l1_mass(folds[0].d) if folds else None,
            },
            "fold_count": len(folds),
            "unique_fold_count": len(unique),
            "max_num_vars_seen": max((f.max_num_vars for f in folds), default=None),
            "unique_folds": records,
        }
    return out


def main() -> None:
    out = analyse()
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
