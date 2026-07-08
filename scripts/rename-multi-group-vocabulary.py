#!/usr/bin/env python3
"""Bulk rename stale grouped-root / grouped-eval vocabulary."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git",
    "target",
    "book/book",
}

# Longest-first token replacements (identifiers + constants).
TOKEN_REPLACEMENTS = [
    ("GROUPED_ROOT_RECURSIVE_SETUP_UNSUPPORTED", "MULTI_GROUP_ROOT_RECURSIVE_SETUP_UNSUPPORTED"),
    ("GROUPED_ROOT_MULTI_CHUNK_UNSUPPORTED", "MULTI_GROUP_ROOT_MULTI_CHUNK_UNSUPPORTED"),
    ("GROUPED_ROOT_DENSE_UNSUPPORTED", "MULTI_GROUP_ROOT_DENSE_UNSUPPORTED"),
    ("GROUPED_ROOT_UNSUPPORTED", "MULTI_GROUP_ROOT_UNSUPPORTED"),
    ("should_reject_grouped_root", "should_reject_multi_group_root"),
    ("reject_unsupported_grouped_root", "reject_unsupported_multi_group_root"),
    ("reject_grouped_multi_chunk", "reject_multi_group_multi_chunk"),
    ("grouped_root_commit_params", "multi_group_root_commit_params"),
    ("build_trace_claim_grouped_root", "build_trace_claim_multi_group_root"),
    ("build_grouped_root_stage2_trace_table", "build_multi_group_root_stage2_trace_table"),
    ("verify_grouped_root_inner", "verify_multi_group_root_inner"),
    ("grouped_root_prover_error", "multi_group_root_prover_error"),
    ("prepare_relation_matrix_evaluator_grouped", "prepare_relation_matrix_evaluator_multi_group"),
    ("expand_to_grouped_root_level_params", "expand_to_multi_group_root_level_params"),
    ("compute_grouped_root_direct_level_params", "compute_multi_group_root_direct_level_params"),
    ("grouped_root_main_level_params_candidate", "multi_group_root_main_level_params_candidate"),
    ("grouped_root_precommitted_groups", "multi_group_root_precommitted_groups"),
    ("grouped_root_direct_cost_score", "multi_group_root_direct_cost_score"),
    ("grouped_root_segment_rings", "multi_group_root_segment_rings"),
    ("grouped_root_next_w_len", "multi_group_root_next_w_len"),
    ("sample_grouped_root_params", "sample_multi_group_root_params"),
    ("grouped_root_params", "multi_group_root_params"),
    ("grouped_plan", "plan"),
    ("grouped_packed_direct", "packed_direct"),
    ("grouped_m_row_count", "multi_group_m_row_count"),
    ("grouped_row_offsets", "multi_group_row_offsets"),
    ("grouped_root_rejects", "multi_group_root_rejects"),
    ("grouped_root_round_trip", "multi_group_root_round_trip"),
    ("grouped_root_folded", "multi_group_root_folded"),
    ("grouped_root_direct_witness", "multi_group_root_direct_witness"),
    ("grouped_root_schedule", "multi_group_root_schedule"),
    ("precommitted_grouped_root", "precommitted_multi_group_root"),
    ("compute_grouped_relation_quotient", "compute_multi_group_relation_quotient"),
    ("grouped_ring_relation_segment_lengths", "multi_group_ring_relation_segment_lengths"),
    ("grouped_relation_matrix_row_count_for", "multi_group_relation_matrix_row_count_for"),
    ("derive_grouped_stage1_challenges", "derive_multi_group_stage1_challenges"),
    ("validate_grouped_role_alpha_pows", "validate_multi_group_role_alpha_pows"),
    ("walk_grouped_generated_schedule_entry", "walk_multi_group_generated_schedule_entry"),
    ("worst_case_grouped_opening_batch_for_shape", "worst_case_multi_group_opening_batch_for_shape"),
    ("supports_grouped_final_commit", "supports_multi_group_final_commit"),
    ("grouped_extension_params", "multi_group_extension_params"),
    ("grouped_one_three_fixture", "multi_group_one_three_fixture"),
    ("grouped_segment_layout_total_matches_root_next_w_len", "multi_group_segment_layout_total_matches_root_next_w_len"),
    ("grouped_segment_layout_rejects_multi_chunk", "multi_group_segment_layout_rejects_multi_chunk"),
    ("grouped_lens", "segment_lens"),
    ("grouped_opening_layout", "multi_group_opening_layout"),
    ("grouped_same_point", "multi_group_same_point"),
    ("grouped_multi_chunk_schedule_rejects_at_effective_schedule_boundary", "multi_group_multi_chunk_schedule_rejects_at_effective_schedule_boundary"),
    ("setup_matrix_envelope_covers_grouped_batch_schedules", "setup_matrix_envelope_covers_multi_group_batch_schedules"),
    ("grouped_extension_openings_fallback_to_root_direct", "multi_group_extension_openings_fallback_to_root_direct"),
    ("opening_schedule_key_freezes_grouped_precommitteds", "opening_schedule_key_freezes_multi_group_precommitteds"),
    ("assert_grouped_fold_sizing_matches_runtime", "assert_multi_group_fold_sizing_matches_runtime"),
    ("grouped_fold_sizing_matches_runtime_for_one_three", "multi_group_fold_sizing_matches_runtime_for_one_three"),
    ("grouped_fold_sizing_matches_runtime_for_two_one", "multi_group_fold_sizing_matches_runtime_for_two_one"),
    ("grouped_sample_key", "multi_group_sample_key"),
    ("validate_generated_grouped_entry_accepts_materialized_dp_schedule", "validate_generated_multi_group_entry_accepts_materialized_dp_schedule"),
    ("grouped_single_group_supports_multi_chunk_weights", "single_group_plan_supports_multi_chunk_weights"),
    ("grouped_multi_group_packed_matches_row_fallback", "multi_group_packed_direct_matches_row_fallback"),
    ("grouped_multi_group_packed_matches_row_fallback_with_mismatched_t_cols", "multi_group_packed_direct_matches_row_fallback_with_mismatched_t_cols"),
    ("root_direct_schedule_uses_grouped_witness_len", "root_direct_schedule_uses_multi_group_witness_len"),
    ("grouped_data", "multi_group_data"),
    ("grouped_extension_openings", "multi_group_extension_openings"),
    ("grouped_key", "multi_group_key"),
    ("grouped_schedule", "multi_group_schedule"),
]

# Phrase replacements inside string literals / comments (order matters).
STRING_REPLACEMENTS = [
    ("grouped witness chunk windows", "setup witness chunk windows"),
    ("grouped chunk block coverage", "setup chunk block coverage"),
    ("grouped packed D scan missing D view", "setup packed D scan missing D view"),
    ("grouped D setup weights exceed physical D width", "setup D weights exceed physical D width"),
    ("grouped D active boundary", "setup D active boundary"),
    ("grouped D active columns", "setup D active columns"),
    ("grouped D base setup footprint", "setup D base footprint"),
    ("grouped B base setup footprint", "setup B base footprint"),
    ("grouped A base setup footprint", "setup A base footprint"),
    ("grouped D base footprint", "setup D base footprint"),
    ("grouped B base footprint", "setup B base footprint"),
    ("grouped A base footprint", "setup A base footprint"),
    ("grouped D setup footprint", "setup D footprint"),
    ("grouped B setup footprint", "setup B footprint"),
    ("grouped A setup footprint", "setup A footprint"),
    ("grouped B width", "setup B width"),
    ("grouped Z range", "setup Z range"),
    ("grouped A rows", "setup A rows"),
    ("grouped B rows", "setup B rows"),
    ("grouped D rows", "setup D rows"),
    ("grouped relation quotient", "multi-group relation quotient"),
    ("grouped witness layout does not match root group order", "multi-group witness layout does not match root group order"),
    ("grouped e width overflow", "multi-group e width overflow"),
    ("grouped inner width overflow", "multi-group inner width overflow"),
    ("grouped A-key column width is too small", "multi-group A-key column width is too small"),
    ("grouped block count overflow", "multi-group block count overflow"),
    ("grouped B vector width overflow", "multi-group B vector width overflow"),
    ("grouped row ranges do not match group matrix heights", "multi-group row ranges do not match group matrix heights"),
    ("grouped root M rows require the real root group count", "multi-group root relation rows require the real root group count"),
    ("grouped digit concatenation", "multi-group digit concatenation"),
    ("grouped digit blocks have mixed ring dimensions", "multi-group digit blocks have mixed ring dimensions"),
    ("grouped e-folded offset overflow", "multi-group e-folded offset overflow"),
    ("grouped fold grind selected different nonces across groups", "multi-group fold grind selected different nonces across groups"),
    ("grouped multi-chunk must reject", "multi-group multi-chunk must reject"),
    ("Legacy grouped-root unsupported", "Legacy multi-group-root unsupported"),
    ("Return the grouped-root rejection message", "Return the multi-group-root rejection message"),
    ("grouped-root", "multi-group-root"),
    ("grouped ring-switch requires at least one digit group", "multi-group ring-switch requires at least one digit group"),
    ("grouped ring-switch digit groups have mixed ring dimensions", "multi-group ring-switch digit groups have mixed ring dimensions"),
    ("grouped root ring-switch does not produce terminal artifacts", "multi-group root ring-switch does not produce terminal artifacts"),
    ("grouped root trace table currently requires degree-one openings", "multi-group root trace table currently requires degree-one openings"),
    ("grouped trace segment width overflow", "multi-group trace segment width overflow"),
    ("grouped trace table length overflow", "multi-group trace table length overflow"),
    ("grouped trace block width overflow", "multi-group trace block width overflow"),
    ("grouped trace plane offset overflow", "multi-group trace plane offset overflow"),
    ("grouped trace claim offset overflow", "multi-group trace claim offset overflow"),
    ("grouped trace column overflow", "multi-group trace column overflow"),
    ("grouped trace row offset overflow", "multi-group trace row offset overflow"),
    ("grouped trace row overflow", "multi-group trace row overflow"),
    ("grouped trace column bits overflow", "multi-group trace column bits overflow"),
    ("grouped trace column bound overflow", "multi-group trace column bound overflow"),
    ("grouped z width overflow", "relation matrix z width overflow"),
    ("grouped t width overflow", "relation matrix t width overflow"),
    ("grouped r width overflow", "relation matrix r width overflow"),
    ("grouped M width overflow", "relation matrix width overflow"),
    ("grouped M eval opening-point layout mismatch", "relation matrix col eval opening-point layout mismatch"),
    ("grouped M eval multiplier layout mismatch", "relation matrix col eval multiplier layout mismatch"),
    ("grouped ring-relation segment lengths require precommitted groups", "multi-group ring-relation segment lengths require precommitted groups"),
    ("grouped e-hat width overflow", "multi-group e-hat width overflow"),
    ("grouped t-hat width overflow", "multi-group t-hat width overflow"),
    ("grouped z-hat width overflow", "multi-group z-hat width overflow"),
    ("grouped e offset overflow", "multi-group e offset overflow"),
    ("grouped t offset overflow", "multi-group t offset overflow"),
    ("grouped group stride overflow", "multi-group stride overflow"),
    ("grouped root polynomial count overflow", "multi-group root polynomial count overflow"),
    ("grouped root requires precommitted groups to have at most half the final num_vars", "multi-group root requires precommitted groups to have at most half the final num_vars"),
    ("grouped root-direct schedule is missing commit params", "multi-group root-direct schedule is missing commit params"),
    ("grouped schedule has no steps", "multi-group schedule has no steps"),
    ("grouped DP regen failed", "multi-group DP regen failed"),
    ("grouped runtime schedule", "multi-group runtime schedule"),
    ("grouped polynomial count", "multi-group polynomial count"),
    ("grouped prover data", "multi-group prover data"),
    ("grouped prove", "multi-group prove"),
    ("grouped root must hand off to a suffix", "multi-group root must hand off to a suffix"),
    ("grouped main params", "multi-group main params"),
    ("grouped precommit params", "multi-group precommit params"),
    ("grouped opening batch", "multi-group opening batch"),
    ("grouped instance", "multi-group instance"),
    ("grouped segment layout", "multi-group segment layout"),
    ("grouped layout", "multi-group layout"),
    ("grouped batch", "multi-group batch"),
    ("grouped key", "multi-group key"),
    ("grouped schedule step", "multi-group schedule step"),
    ("grouped multi-chunk schedule must reject", "multi-group multi-chunk schedule must reject"),
    ("grouped same-point opening_batch", "multi-group same-point opening_batch"),
    ("grouped same-point shape should resolve to a setup envelope", "multi-group same-point shape should resolve to a setup envelope"),
    ("flat and grouped setup", "single-group and multi-group setup"),
    ("grouped root commitment layout", "multi-group root commitment layout"),
    ("grouped parameter selection", "multi-group parameter selection"),
    ("grouped internal form", "multi-group internal form"),
    ("for grouped roots", "for multi-group roots"),
    ("for grouped roots.", "for multi-group roots."),
    ("grouped [`RingRelationProver`]", "multi-group [`RingRelationProver`]"),
    ("grouped roots)", "multi-group roots)"),
    ("for grouped roots.", "for multi-group roots."),
    ("Scalar rows (`precommitteds: []`) and grouped", "Scalar rows (`precommitteds: []`) and multi-group"),
    ("Whether grouped `commit_final_group`", "Whether multi-group `commit_final_group`"),
    ("grouped final commits", "multi-group final commits"),
    ("later grouped root whose final basis", "later multi-group root whose final basis"),
    ("grouped extension openings must not select the grouped folded trace path", "multi-group extension openings must not select the multi-group folded trace path"),
    ("grouped root params", "multi-group root params"),
    ("main grouped commit params", "main multi-group commit params"),
    ("final grouped commitment", "final multi-group commitment"),
    ("the grouped root folds", "the multi-group root folds"),
    ("test/grouped-unequal", "test/multi-group-unequal"),
    ("serialize grouped proof", "serialize multi-group proof"),
    ("deserialize grouped proof", "deserialize multi-group proof"),
    ("grouped verifier claims", "multi-group verifier claims"),
    ("grouped verify", "multi-group verify"),
    ("grouped opening layout", "multi-group opening layout"),
    ("The grouped root folds", "The multi-group root folds"),
    ("Precommitted group-local params for a grouped root.", "Precommitted group-local params for a multi-group root."),
    ("in the grouped root layout", "in the multi-group root layout"),
    ("for scalar or grouped roots.", "for scalar or multi-group roots."),
    ("scalar or grouped opening-point", "scalar or multi-group opening-point"),
    ("scalar or grouped multiplier-point", "scalar or multi-group multiplier-point"),
    ("groups for grouped roots", "groups for multi-group roots"),
    ("for grouped roots. Matches", "for multi-group roots. Matches"),
    ("group lens", "segment lens"),
    ("chunked grouped M evals require a non-zero block count", "chunked relation-matrix col evals require a non-zero block count"),
    ("chunked grouped M eval chunk count is zero", "chunked relation-matrix col eval chunk count is zero"),
    ("chunked grouped M eval block window is empty", "chunked relation-matrix col eval block window is empty"),
    ("Shared grouped/singleton relation matrix column evaluation.", "Shared singleton and multi-group relation matrix column evaluation."),
    ("Unified relation matrix column evaluation for singleton and grouped root relations.", "Unified relation matrix column evaluation for singleton and multi-group root relations."),
    ("Commit the final polynomial bundle for a grouped root commitment.", "Commit the final polynomial bundle for a multi-group root commitment."),
    ("freezes precommitted layouts, and resolves the grouped root", "freezes precommitted layouts, and resolves the multi-group root"),
    ("is the multi-group-root counterpart of the succinct per-claim terms: grouped", "is the multi-group-root counterpart of the succinct per-claim terms: multi-group"),
    ("Final group shape for the grouped root commitment.", "Final group shape for the multi-group root commitment."),
    ("Build a grouped opening layout from this schedule lookup key.", "Build a multi-group opening layout from this schedule lookup key."),
    ("Per-group `z ‖ e ‖ t` widths for grouped roots in final-first witness order.", "Per-group `z ‖ e ‖ t` widths for multi-group roots in final-first witness order."),
    ("via_grouped", "via_multi_group"),
    ("grouped step", "multi-group step"),
    ("grouped schedule", "multi-group schedule"),
    ("grouped root step", "multi-group root step"),
    ("expected grouped root fold", "expected multi-group root fold"),
    ("the grouped builder", "the multi-group builder"),
    ("grouped precommit path", "multi-group precommit path"),
    ("grouped terminal root", "multi-group terminal root"),
    ("grouped folded schedules", "multi-group folded schedules"),
    ("main grouped root", "main multi-group root"),
    ("grouped root-direct", "multi-group root-direct"),
    ("grouped root-fold", "multi-group root-fold"),
    ("grouped root", "multi-group root"),
    ("grouped A-role", "multi-group A-role"),
    ("grouped B-role", "multi-group B-role"),
    ("grouped A width", "multi-group A width"),
    ("grouped D width", "multi-group D width"),
    ("grouped main root-direct", "multi-group main root-direct"),
    ("regen grouped", "regen multi-group"),
    ("needs a grouped", "needs a multi-group"),
    ("single-group grouped resolve", "single-group resolve"),
    ("grouped generated entry should validate", "multi-group generated entry should validate"),
    ("generated grouped non-terminal successor", "generated multi-group non-terminal successor"),
    ("generated grouped level byte count overflow", "generated multi-group level byte count overflow"),
    ("generated grouped proof byte total overflow", "generated multi-group proof byte total overflow"),
    ("generated grouped direct step has zero witness length", "generated multi-group direct step has zero witness length"),
    ("Pure grouped DP regeneration", "Pure multi-group DP regeneration"),
    ("combines these with grouped", "combines these with multi-group"),
    ("Pure grouped DP regeneration for", "Pure multi-group DP regeneration for"),
    ("grouped single-point batching", "multi-group single-point batching"),
]


def iter_files() -> list[Path]:
    out: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix not in {".rs", ".md"}:
            continue
        if path.name == "rename-multi-group-vocabulary.py":
            continue
        out.append(path)
    return out


def apply_replacements(text: str) -> str:
    for old, new in TOKEN_REPLACEMENTS:
        text = text.replace(old, new)
    for old, new in STRING_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def main() -> None:
    changed = 0
    for path in iter_files():
        original = path.read_text()
        updated = apply_replacements(original)
        if updated != original:
            path.write_text(updated)
            changed += 1
            print(path.relative_to(ROOT))
    print(f"updated {changed} files")


if __name__ == "__main__":
    main()
