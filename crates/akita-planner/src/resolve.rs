//! Schedule resolution: catalog validation, cache-then-generate.
//!
//! [`resolve_schedule`] is the single entry point the runtime (config,
//! prover, verifier) uses to obtain a [`Schedule`] for a lookup key. When a
//! preset supplies a catalog, identity is validated and the compact entry
//! is expanded via [`schedule_from_entry`]; on a miss (or no catalog) the
//! schedule is regenerated with the offline DP search [`crate::find_schedule`].

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::{
    direct_witness_bytes, extension_opening_reduction_proof_bytes, level_proof_bytes,
    segment_typed_witness_shape, w_ring_element_count_for_chunks, AkitaScheduleInputs,
    AkitaScheduleLookupKey, CleartextWitnessShape, DirectStep, FoldStep,
    GroupBatchAkitaScheduleLookupKey, LevelParams, MRowLayout, Schedule, Step,
};

use crate::catalog_identity::validate_catalog_identity;
use crate::generated::{
    table_entry, GeneratedScheduleKey, GeneratedScheduleTable, GeneratedScheduleTableEntry,
    GeneratedStep,
};
use crate::schedule_params::validate_policy_witness_chunk;
use crate::PlannerPolicy;
use crate::{find_group_batch_schedule, find_schedule};

///
/// Convert the public runtime lookup key into a generated-table lookup key.
pub const fn generated_schedule_lookup_key(key: AkitaScheduleLookupKey) -> GeneratedScheduleKey {
    GeneratedScheduleKey {
        num_vars: key.num_vars,
        num_polynomials: key.num_polynomials,
    }
}

/// Resolve the runtime [`Schedule`] using an explicit optional catalog.
pub fn resolve_schedule(
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    catalog: Option<GeneratedScheduleTable>,
) -> Result<Schedule, AkitaError> {
    key.validate()?;
    validate_policy_witness_chunk(policy)?;
    let Some(table) = catalog else {
        return find_schedule(
            key,
            policy,
            ring_challenge_config,
            fold_challenge_shape_at_level,
        );
    };
    validate_catalog_identity(
        &table,
        policy,
        &ring_challenge_config,
        &fold_challenge_shape_at_level,
    )?;
    if let Some(entry) = table_entry(table, generated_schedule_lookup_key(key)) {
        return schedule_from_entry(
            entry,
            key,
            policy,
            ring_challenge_config,
            fold_challenge_shape_at_level,
        );
    }
    find_schedule(
        key,
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )
}

/// Resolve a grouped-root schedule without falling back to a scalar table key.
///
/// Phase 1 has no generated grouped entries yet, so catalog handling only
/// validates identity before delegating to the grouped DP fallback.
pub fn resolve_group_batch_schedule(
    key: &GroupBatchAkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    catalog: Option<GeneratedScheduleTable>,
) -> Result<Schedule, AkitaError> {
    if let Some(table) = catalog {
        validate_catalog_identity(
            &table,
            policy,
            &ring_challenge_config,
            &fold_challenge_shape_at_level,
        )?;
    }
    find_group_batch_schedule(
        key,
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )
}

fn padded_boolean_vars(len: usize) -> Result<usize, AkitaError> {
    let padded = len
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("opening witness length overflow".to_string()))?;
    Ok(padded.trailing_zeros() as usize)
}

fn extension_opening_reduction_level_bytes(
    challenge_field_bits: u32,
    extension_opening_width: usize,
    fold_level: usize,
    key: AkitaScheduleLookupKey,
    current_w_len: usize,
) -> Result<usize, AkitaError> {
    if extension_opening_width <= 1 {
        return Ok(0);
    }
    let (partials, opening_vars) = if fold_level == 0 {
        (
            extension_opening_width.saturating_mul(key.num_polynomials),
            key.num_vars,
        )
    } else {
        (extension_opening_width, padded_boolean_vars(current_w_len)?)
    };
    extension_opening_reduction_proof_bytes(
        challenge_field_bits,
        partials,
        opening_vars,
        extension_opening_width,
    )
}

/// Build the runtime [`Schedule`] for a compact generated entry.
pub fn schedule_from_entry(
    entry: &GeneratedScheduleTableEntry,
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError> {
    entry.validate()?;
    let extension_opening_width = policy.claim_ext_degree;
    let field_bits = policy.decomposition.field_bits();
    let challenge_field_bits = field_bits * policy.chal_ext_degree as u32;

    let expected_root_w_len = 1usize
        .checked_shl(key.num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("root witness length overflow".to_string()))?;
    let mut steps = Vec::with_capacity(entry.steps.len());
    let mut total = 0usize;
    let mut fold_level = 0usize;
    let mut current_w_len = expected_root_w_len;
    let mut terminal_witness_field_len: Option<usize> = None;
    let mut last_fold_lp: Option<LevelParams> = None;

    for (idx, step) in entry.steps.iter().enumerate() {
        match step {
            GeneratedStep::Fold(level) => {
                let next = entry.steps.get(idx + 1).ok_or_else(|| {
                    AkitaError::InvalidSetup(format!(
                        "generated schedule ended with a fold step at level {fold_level}"
                    ))
                })?;
                let is_terminal = matches!(next, GeneratedStep::Direct(_));
                let inputs = AkitaScheduleInputs {
                    num_vars: key.num_vars,
                    level: fold_level,
                    current_w_len,
                };
                let level_num_claims = if fold_level == 0 {
                    key.num_polynomials
                } else {
                    1
                };
                let mut lp = level.expand_to_level_params(
                    policy,
                    &ring_challenge_config,
                    fold_level,
                    current_w_len,
                    fold_challenge_shape_at_level(inputs),
                    level_num_claims,
                )?;
                // Stamp the per-level chunk layout (the expander defaults it so a
                // root-direct commit stays single-chunk); must match the DP.
                lp.witness_chunk = policy.witness_chunk_for_level(fold_level);
                let num_polynomials = if fold_level == 0 {
                    key.num_polynomials
                } else {
                    1
                };
                // Chunk count of the witness this level commits/produces. Equal to
                // the count stamped on `lp.witness_chunk`, mirroring the DP.
                let num_chunks = policy.chunks_at_level(fold_level);
                let mul_d = |ring: usize| -> Result<usize, AkitaError> {
                    ring.checked_mul(lp.ring_dimension).ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "generated next witness length overflow".to_string(),
                        )
                    })
                };
                let (next_w_len, next_lp, layout) = if is_terminal {
                    let ring = w_ring_element_count_for_chunks(
                        field_bits,
                        &lp,
                        num_polynomials,
                        MRowLayout::WithoutDBlock,
                        num_chunks,
                    )?;
                    let len = mul_d(ring)?;
                    terminal_witness_field_len = Some(len);
                    (len, None, MRowLayout::WithoutDBlock)
                } else {
                    let ring = w_ring_element_count_for_chunks(
                        field_bits,
                        &lp,
                        num_polynomials,
                        MRowLayout::WithDBlock,
                        num_chunks,
                    )?;
                    let len = mul_d(ring)?;
                    let GeneratedStep::Fold(next_level) = next else {
                        return Err(AkitaError::InvalidSetup(
                            "generated non-terminal successor must be a fold step".to_string(),
                        ));
                    };
                    let next_inputs = AkitaScheduleInputs {
                        num_vars: key.num_vars,
                        level: fold_level + 1,
                        current_w_len: len,
                    };
                    let mut next_lp = next_level.expand_to_level_params(
                        policy,
                        &ring_challenge_config,
                        fold_level + 1,
                        len,
                        fold_challenge_shape_at_level(next_inputs),
                        1,
                    )?;
                    next_lp.witness_chunk = policy.witness_chunk_for_level(fold_level + 1);
                    (len, Some(next_lp), MRowLayout::WithDBlock)
                };
                // Single commitment group at one point: one public row per level.
                let num_claims_here = 1;
                let level_bytes = level_proof_bytes(
                    field_bits,
                    challenge_field_bits,
                    &lp,
                    next_lp.as_ref(),
                    next_w_len,
                    num_claims_here,
                    layout,
                ) + extension_opening_reduction_level_bytes(
                    challenge_field_bits,
                    extension_opening_width,
                    fold_level,
                    key,
                    current_w_len,
                )?;
                total = total.checked_add(level_bytes).ok_or_else(|| {
                    AkitaError::InvalidSetup("proof byte total overflow".to_string())
                })?;
                last_fold_lp = Some(lp.clone());
                steps.push(Step::Fold(FoldStep {
                    params: lp,
                    current_w_len,
                    next_w_len,
                    level_bytes,
                }));
                fold_level += 1;
                current_w_len = next_w_len;
            }
            GeneratedStep::Direct(direct) => {
                let (witness_shape, direct_current_w_len, params) = if fold_level == 0 {
                    let params = match direct.commit {
                        Some(commit) => commit
                            .expand_to_level_params(
                                policy,
                                &ring_challenge_config,
                                0,
                                expected_root_w_len,
                                fold_challenge_shape_at_level(AkitaScheduleInputs {
                                    num_vars: key.num_vars,
                                    level: 0,
                                    current_w_len: expected_root_w_len,
                                }),
                                key.num_polynomials,
                            )
                            .ok(),
                        None => None,
                    };
                    (
                        CleartextWitnessShape::FieldElements(expected_root_w_len),
                        expected_root_w_len,
                        params,
                    )
                } else {
                    let len = terminal_witness_field_len.ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "terminal direct step missing precomputed witness length".to_string(),
                        )
                    })?;
                    let terminal_fold_level = fold_level.saturating_sub(1);
                    let terminal_lp = last_fold_lp.as_ref().ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "terminal direct step missing predecessor fold params".to_string(),
                        )
                    })?;
                    let num_polynomials = if terminal_fold_level == 0 {
                        key.num_polynomials
                    } else {
                        1
                    };
                    // The terminal-direct (cleartext) witness is single-chunk by
                    // construction: the prover emits the global folded response
                    // and one shared `r̂` tail (`build_segment_typed_witness` uses
                    // a single `z` and `num_segments = 1`). Chunking the cleartext
                    // tail is unsupported, so the last fold level must be
                    // single-chunk — the leading activated levels are chunked,
                    // never the tail. Reject loudly here instead of letting the
                    // prover hit a cryptic layout-mismatch at prove time.
                    if terminal_lp.witness_chunk.num_chunks > 1 {
                        return Err(AkitaError::InvalidSetup(
                            "terminal-direct witness does not support a multi-chunk last fold level"
                                .to_string(),
                        ));
                    }
                    let witness_shape = segment_typed_witness_shape(
                        terminal_lp,
                        field_bits,
                        num_polynomials,
                        num_polynomials,
                        1,
                        1,
                    )?;
                    (witness_shape, len, None)
                };
                let direct_bytes = direct_witness_bytes(field_bits, &witness_shape);
                total = total.checked_add(direct_bytes).ok_or_else(|| {
                    AkitaError::InvalidSetup("proof byte total overflow".to_string())
                })?;
                steps.push(Step::Direct(DirectStep {
                    current_w_len: direct_current_w_len,
                    witness_shape,
                    direct_bytes,
                    params,
                }));
            }
        }
    }

    Ok(Schedule {
        steps,
        total_bytes: total,
    })
}

pub fn estimate_proof_bytes(
    entry: &GeneratedScheduleTableEntry,
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<usize, AkitaError> {
    Ok(schedule_from_entry(
        entry,
        key,
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )?
    .total_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_types::{ChunkedWitnessCfg, DecompositionParams, SisModulusFamily};

    fn flat_policy() -> PlannerPolicy {
        PlannerPolicy {
            ring_dimension: 64,
            decomposition: DecompositionParams {
                log_basis: 3,
                log_commit_bound: 1,
                log_open_bound: Some(8),
            },
            sis_family: SisModulusFamily::Q128,
            ring_subfield_norm_bound: 1,
            claim_ext_degree: 4,
            chal_ext_degree: 4,
            basis_range: (3, 4),
            onehot_chunk_size: 1,
            tiered: false,
            witness_chunk: ChunkedWitnessCfg::default(),
        }
    }

    fn ring_challenge_config(_: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Ok(SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        })
    }

    fn fold_shape(_: AkitaScheduleInputs) -> TensorChallengeShape {
        TensorChallengeShape::Flat
    }

    #[test]
    fn resolve_schedule_none_matches_find_schedule() {
        let key = AkitaScheduleLookupKey::new(20, 1);
        let policy = flat_policy();
        let via_resolve = resolve_schedule(key, &policy, ring_challenge_config, fold_shape, None)
            .expect("resolve");
        let via_find =
            find_schedule(key, &policy, ring_challenge_config, fold_shape).expect("find");
        assert_eq!(via_resolve.total_bytes, via_find.total_bytes);
    }

    #[test]
    fn multi_chunk_schedule_ends_with_single_chunk_terminal_fold() {
        // Chunked leading levels are allowed when the planner can route through a
        // single-chunk fold before the terminal-direct tail.
        let mut policy = flat_policy();
        policy.witness_chunk = ChunkedWitnessCfg {
            num_chunks: 8,
            num_activated_levels: 2,
        };
        let key = AkitaScheduleLookupKey::new(24, 1);
        let schedule =
            find_schedule(key, &policy, ring_challenge_config, fold_shape).expect("schedule");
        let last_fold = schedule
            .steps
            .iter()
            .rev()
            .find_map(|step| match step {
                Step::Fold(fold) => Some(fold),
                _ => None,
            })
            .expect("fold-then-direct schedule");
        assert_eq!(last_fold.params.witness_chunk.num_chunks, 1);
    }

    #[test]
    fn multi_chunk_does_not_perturb_single_chunk_schedule() {
        // A policy with default (single-chunk) witness_chunk must reproduce the
        // exact schedule of today's planner for the same key.
        let key = AkitaScheduleLookupKey::new(22, 1);
        let base = flat_policy();
        let mut explicit_default = flat_policy();
        explicit_default.witness_chunk = ChunkedWitnessCfg::default();
        let a = find_schedule(key, &base, ring_challenge_config, fold_shape).expect("a");
        let b =
            find_schedule(key, &explicit_default, ring_challenge_config, fold_shape).expect("b");
        assert_eq!(a.total_bytes, b.total_bytes);
        assert_eq!(a.steps.len(), b.steps.len());
    }

    #[test]
    fn find_schedule_rejects_non_power_of_two_chunks() {
        let mut policy = flat_policy();
        policy.witness_chunk = ChunkedWitnessCfg {
            num_chunks: 6,
            num_activated_levels: 2,
        };
        let key = AkitaScheduleLookupKey::new(20, 1);
        let err = find_schedule(key, &policy, ring_challenge_config, fold_shape)
            .expect_err("non-power-of-two chunk count must be rejected");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn find_schedule_rejects_tiered_multi_chunk() {
        let mut policy = flat_policy();
        policy.tiered = true;
        policy.witness_chunk = ChunkedWitnessCfg::d64_production();
        let key = AkitaScheduleLookupKey::new(20, 1);
        let err = find_schedule(key, &policy, ring_challenge_config, fold_shape)
            .expect_err("tiered + multi-chunk must be rejected");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn resolve_schedule_rejects_zero_dimension_key() {
        let key = AkitaScheduleLookupKey::new(0, 1);
        let policy = flat_policy();

        let err = resolve_schedule(key, &policy, ring_challenge_config, fold_shape, None)
            .expect_err("zero-arity key must be rejected");

        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }
}
