#!/usr/bin/env bash
# Flag dead removed symbols in non-historical docs/*.md.
# README.md and AGENTS.md may cite removed names when describing the cutover;
# they are covered by review and the blast-radius comment instead.
# Historical snapshots (banner in first 8 lines) are skipped.
# See docs/documentation.md and scripts/check-spec-references.sh.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v rg >/dev/null 2>&1; then
  echo "error: ripgrep (rg) is required" >&2
  exit 2
fi

dead_patterns=(
  'akita-scheme'
  'akita-cfg'
  'akita-derive'
  'ScheduleProvider'
  'PlannerConfig'
  'WCommitmentConfig'
  'sis_offline'
  'sis_policy\.rs'
  'schedule_policy\.rs'
  '_with_policy'
)

pattern="$(IFS='|'; echo "${dead_patterns[*]}")"

removed_api_patterns=(
  'effective_batched_schedule'
  'OpeningBatch\b'
  'OpeningBatchShape'
  'OpeningGroupShape'
  'OpeningBatchLimits'
  'VerifierOpeningBatch'
  'ProverOpeningBatch'
  'ProverCommitmentGroup'
  'CommitmentGroupScheduleKey'
  'CommitmentGroupLayout'
  'GeneratedCommitmentGroup'
  'GeneratedScheduleLookupKey'
  'CleartextWitnessProof'
  'CleartextWitnessShape'
  'DirectStep'
  'GeneratedDirectStep'
  'direct_witness_bytes'
  'segment_typed_witness_shape_from_groups'
  'dispatch_ring_dim_result'
  'ExactNegacyclic \{ width, log_basis \}'
  'ChallengeShape'
  'ChallengeLabels'
  'TensorChallenges'
  'PreparedAffineFactors'
  'PreparedChallengeEvals'
  'FoldWitnessLinfCapPolicy'
  'BoundedL1Norm'
  'akita-challenges/src/tensor\.rs'
  'RingSwitchComputeBackend'
  'RingSwitchRelationRowsPlan'
  'RingSwitchQuotientRowsPlan'
  'CommitmentProver'
  'CommitmentWithHint'
  'CommittedGroupWithHint'
  'FinalCommittedGroupWithHint'
  'batched_commit'
  'commit_group\b'
  'commit_final_group'
  'commit_with_params'
  'batched_commit_with_params'
  'get_params_for_prove'
  'get_params_for_batched_commitment'
  'runtime_schedule\b'
  'committed_group_profile'
  'resolve_generated_precommitted_group_profile'
  'resolve_group_batch_schedule'
  'plan_standalone_precommit'
  'StandalonePrecommitPlan'
  'StandalonePrecommitCandidate'
  'prepare_batched_commit_inputs'
  'padded_scalar_batch_num_vars'
  'validate_scalar_point_matches_poly_arity'
  'emit_precommitted_profiles_module'
  'prior_group_profiles'
  'PriorGroupProfiles'
  'PriorGroupContext'
  'NoPriorGroups'
  'WithPriorGroups'
  'scheduler_without_prior_groups'
  'scheduler_with_prior_groups'
  'explicit_without_prior_groups'
  'explicit_with_prior_groups'
  'profile_without_prior_groups'
  'sole_profile'
  '_precommitted\.rs'
  'api/scheme\.rs'
  'RootTensorProjectionPoly\b'
  'RootTensorProjectionView\b'
  'RootTensorProjectionBatchView\b'
  'SparseRingPoly\b'
  'SparseRingView\b'
  'SparseRingBatchView\b'
  'root_tensor_projection_enabled\b'
  'root_tensor_projection_enabled_for_width\b'
  '\bProveBackendFor\b'
  'ProjectBackendFor\b'
  'CommitmentConfig::D\b'
  'uniform_ring_dimension\b'
  'setup_prefix_inner_ring_dimension\b'
  'ProtocolDispatchSlot::UniformPolicy\b'
  'validate_ring_subfield_role\b'
  'RootPolyMeta::num_ring_elems\b'
  'meta_ring_elems\b'
  'total_ring_elems\b'
  'CommittedGroupProfile'
  'PrecommittedLevelParams'
  'GeneratedRootPrecommittedGroup'
  'GeneratedSetupPrefixInput'
  'RecursiveFoldParams'
  'RootFoldParams'
  'RootFinalGroupParams'
  'RootPrecommittedGroupParams'
  'WitnessPartition\b'
  'GeneratedWitnessPartition'
  'GeneratedCommittedGroup'
  'GeneratedOpenCommitMatrix'
  '\bLevelParams\b'
  'LevelParamsLike'
  'GeneratedFoldStep'
  'GeneratedSetupPrefixGroup'
  '\bFoldStep\b'
  'TerminalWitnessPlan'
)

api_pattern="$(IFS='|'; echo "${removed_api_patterns[*]}")"

scan_file() {
  local f="$1"
  local search_pattern="$2"
  if [[ ! -f "$f" ]]; then
    return 0
  fi
  if head -n 8 "$f" | grep -qi 'historical snapshot'; then
    return 0
  fi
  rg -n "$search_pattern" "$f" 2>/dev/null || true
}

# Meta / intentionally descriptive docs (cite removed names on purpose).
skip_docs=(documentation.md crate-graph.md)

matches=""
for f in docs/*.md; do
  base="$(basename "$f")"
  for skip in "${skip_docs[@]}"; do
    if [[ "$base" == "$skip" ]]; then
      continue 2
    fi
  done
  hit="$(scan_file "$f" "$pattern")"
  if [[ -n "$hit" ]]; then
    matches+="$hit"$'\n'
  fi
done

if [[ -n "$matches" ]]; then
  echo "Dead symbol references in docs/ (non-historical). Review:" >&2
  echo >&2
  echo "$matches" >&2
  exit 1
fi

api_paths=(book/src docs specs)
for f in crates/*/README.md; do
  if [[ -f "$f" ]]; then
    api_paths+=("$f")
  fi
done

# Route this scan through `scan_file` too, so a doc marked as a historical
# snapshot is exempt from both scans rather than only the first.
api_matches=""
for f in $(rg -l \
  --glob '*.md' \
  --glob '!**/archive/**' \
  --glob '!**/generated/**' \
  "$api_pattern" "${api_paths[@]}" 2>/dev/null || true); do
  hit="$(scan_file "$f" "$api_pattern")"
  if [[ -n "$hit" ]]; then
    api_matches+="$f"$'\n'"$hit"$'\n'
  fi
done

if [[ -n "$api_matches" ]]; then
  echo "Deleted public API references in live docs. Review:" >&2
  echo >&2
  echo "$api_matches" >&2
  exit 1
fi

echo "No dead symbol references in docs/ or deleted public API references in live docs."
