#!/usr/bin/env bash
# Flag specs that reference symbols, crates, or modules removed from the codebase.
# See docs/documentation.md and specs/PRUNING.md.
#
# Usage:
#   scripts/check-spec-references.sh          # live specs only (CI default)
#   scripts/check-spec-references.sh --all    # every spec outside archive/ (audit)
#
# Exit: 0 if clean, 1 if dead references found.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

scope="live"
if [[ "${1:-}" == "--all" ]]; then
  scope="all"
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--all]" >&2
  exit 2
fi

if ! command -v rg >/dev/null 2>&1; then
  echo "error: ripgrep (rg) is required" >&2
  exit 2
fi

# Removed identifiers. Do not list Q16/Fp16 here: live specs may discuss retired
# small-field profiles by name (see crt-ntt-prime-profiles, remove-fp16).
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
)

pattern="$(IFS='|'; echo "${dead_patterns[*]}")"

# Synced with specs/PRUNING.md "Keep as live specs". CI scans only these unless --all.
live_specs=(
  specs/setup-layout-repack.md
  specs/setup-offloading-planner.md
  specs/eor-streamed-prover.md
  specs/packed-sumcheck.md
  specs/schedule-catalog-ownership.md
  specs/planner-incidence-generalization.md
  specs/multi-group-batching.md
  specs/akita-pcs-crate-decomposition.md
  specs/akita-field-refactor.md
  specs/crt-ntt-prime-profiles.md
  specs/eor-sumcheck-prover-acceleration.md
  specs/cross-repo-field-microbench.md
)
# Excluded from CI until stale `akita-scheme` / `_with_policy` refs are scrubbed:
# specs/akita-compute-backend-metal.md, specs/transcript-immediate-fixes.md

missing_live=()
for f in "${live_specs[@]}"; do
  if [[ ! -f "$f" ]]; then
    missing_live+=("$f")
  fi
done
if [[ ${#missing_live[@]} -gt 0 ]]; then
  echo "error: live_specs entries missing from tree:" >&2
  printf '  %s\n' "${missing_live[@]}" >&2
  exit 1
fi

search_paths=()
if [[ "$scope" == "live" ]]; then
  for f in "${live_specs[@]}"; do
    if [[ -f "$f" ]]; then
      search_paths+=("$f")
    fi
  done
else
  search_paths=(specs)
fi

# A spec whose header declares it a historical snapshot documents an
# implementation as it landed, so pre-cutover names in it are the record, not a
# stale reference. Same convention as scripts/check-doc-dead-symbols.sh.
is_historical_snapshot() {
  head -n 12 "$1" | grep -qiE 'historical (snapshot|spec|design record)'
}

matches=""
if [[ "$scope" == "live" ]]; then
  for f in "${search_paths[@]}"; do
    if is_historical_snapshot "$f"; then
      continue
    fi
    hit="$(rg -n --with-filename "$pattern" "$f" 2>/dev/null || true)"
    if [[ -n "$hit" ]]; then
      matches+="$hit"$'\n'
    fi
  done
else
  for f in $(rg -l --glob '!specs/archive/**' --glob '!specs/PRUNING.md' "$pattern" specs 2>/dev/null || true); do
    if is_historical_snapshot "$f"; then
      continue
    fi
    hit="$(rg -n --with-filename "$pattern" "$f" 2>/dev/null || true)"
    if [[ -n "$hit" ]]; then
      matches+="$hit"$'\n'
    fi
  done
fi

if [[ -n "$matches" ]]; then
  echo "Stale spec references (scope=$scope). Dead symbols in:" >&2
  echo >&2
  echo "$matches" >&2
  echo >&2
  echo "Update the spec, archive it (specs/PRUNING.md), or fix the reference." >&2
  exit 1
fi

echo "No dead spec references found (scope=$scope)."
