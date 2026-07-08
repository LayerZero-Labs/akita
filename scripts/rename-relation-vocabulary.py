#!/usr/bin/env python3
"""One-shot relation-matrix vocabulary cutover for PR #287."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {".git", "target", "node_modules"}

# Longest / most specific replacements first.
REPLACEMENTS: list[tuple[str, str]] = [
    # Relation RHS (paper y in M·z = y + ...)
    ("assemble_relation_y_matches_generate_y", "assemble_relation_rhs_matches_generate_rhs"),
    ("assemble_relation_y", "assemble_relation_rhs"),
    ("relation_y_layout_for", "relation_rhs_layout_for"),
    ("relation_y_row_count", "relation_rhs_row_count"),
    ("relation_y_coeff_len", "relation_rhs_coeff_len"),
    ("RelationYLayout", "RelationRhsLayout"),
    ("generate_y", "generate_relation_rhs"),
    ("relation_y", "relation_rhs"),
    ("expected_y_len", "expected_rhs_coeff_len"),
    ("y_trusted", "rhs_trusted"),
    # Relation matrix row layout
    ("grouped_m_row_count_for", "grouped_relation_matrix_row_count_for"),
    ("fold_m_row_layout", "fold_relation_matrix_row_layout"),
    ("m_row_count_for", "relation_matrix_row_count_for"),
    ("MRowLayout", "RelationMatrixRowLayout"),
    ("m_row_layout", "relation_matrix_row_layout"),
    # Relation matrix column evals (prover)
    ("compute_grouped_m_evals_x", "compute_relation_matrix_col_evals"),
    ("grouped_m_evals", "relation_matrix_cols"),
    ("m_evals_x", "relation_matrix_col_evals"),
    ("m_compact", "relation_matrix_col_evals_compact"),
    # Verifier setup contribution eval (tier 2)
    ("SetupEvaluatorMode", "SetupContributionEvalMode"),
    ("SetupEvaluation", "SetupContributionEvaluation"),
    ("SetupEvaluator", "SetupContributionEvaluator"),
    ("prepare_flat", "prepare_single_group_plan"),
    ("finish_cached_static_plan", "finish_cached_static_plan"),  # idempotent anchor
    # Setup evaluator method disambiguation (applied after prepare_grouped still exists)
    ("SetupContributionEvaluator::prepare_grouped", "SetupContributionEvaluator::finish_cached_static_plan"),
    ("evaluator.prepare_grouped", "evaluator.finish_cached_static_plan"),
    # Locals / params
    ("e_setup_cols", "d_physical_cols"),
]

# RingRelationInstance field/method: apply per-file with context after bulk pass.


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def iter_files() -> list[Path]:
    out: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or should_skip(path):
            continue
        if path.suffix in {".rs", ".md", ".toml"} or path.name in {"AGENTS.md"}:
            out.append(path)
    return out


def apply_replacements(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


def patch_ring_relation_instance(text: str) -> str:
    text = re.sub(r"\by: RingVec<", "rhs: RingVec<", text)
    text = re.sub(r",\s*y: RingVec<", ", rhs: RingVec<", text)
    text = re.sub(r"pub fn y\(&self\)", "pub fn rhs(&self)", text)
    text = re.sub(r"&self\.y\b", "&self.rhs", text)
    text = re.sub(r"self\.y\b", "self.rhs", text)
    text = re.sub(r"\by\.coeff_len", "rhs.coeff_len", text)
    text = re.sub(r"\by\.can_decode_vec", "rhs.can_decode_vec", text)
    text = re.sub(r"\by\.as_ring_slice", "rhs.as_ring_slice", text)
    text = re.sub(r"ring relation y\b", "ring relation rhs", text)
    text = re.sub(r"assembled relation `y`", "assembled relation rhs", text)
    text = re.sub(r"relation RHS vector `y`", "relation rhs vector", text)
    text = re.sub(r"relation `y`", "relation rhs", text)
    text = re.sub(r"empty y\b", "empty rhs", text)
    return text


def patch_comments(text: str) -> str:
    subs = [
        ("deferred ring-switch row MLE evaluation", "prepared relation-matrix MLE evaluation"),
        ("deferred ring-switch row replay", "relation-matrix challenge replay"),
        ("ring-switch row-eval preparation fails", "relation-matrix evaluator preparation fails"),
        ("Prepare deferred verifier ring-switch row evaluation data", "Prepare relation-matrix evaluator state"),
        ("Evaluate the prepared ring-switch row table", "Evaluate the relation matrix at a point"),
        ("ring-switch row evaluation", "relation-matrix evaluation"),
        ("Evaluate the M-table MLE", "Evaluate the relation matrix MLE"),
        ("M-table column evaluation", "relation matrix column evaluation"),
        ("tau1-weighted M-row column", "tau1-weighted relation-matrix column"),
        ("M-row layout", "relation-matrix row layout"),
        ("M-row count", "relation-matrix row count"),
        ("M-row order", "relation-matrix row order"),
        ("M-row weights", "relation-matrix row weights"),
        ("row-eval state", "relation-matrix evaluator state"),
    ]
    for old, new in subs:
        text = text.replace(old, new)
    return text


def main() -> None:
    changed = 0
    for path in iter_files():
        original = path.read_text()
        updated = apply_replacements(original)
        if path.name == "ring_relation.rs":
            updated = patch_ring_relation_instance(updated)
        if path.name == "relation.rs" and "proof" in path.parts:
            updated = patch_ring_relation_instance(updated)
        if path.suffix == ".rs":
            updated = patch_comments(updated)
        if updated != original:
            path.write_text(updated)
            changed += 1
    print(f"updated {changed} files")


if __name__ == "__main__":
    main()
