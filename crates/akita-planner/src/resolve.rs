//! Schedule resolution: cache-then-generate.
//!
//! [`resolve_schedule`] is the single entry point the runtime (config,
//! prover, verifier) uses to obtain a [`Schedule`] for a lookup key. It
//! consults the shipped [`GeneratedScheduleTable`] first (the "cache") and
//! expands a matching compact entry on demand via [`schedule_from_entry`];
//! on a table miss it regenerates the schedule from scratch with the
//! offline DP search [`crate::find_schedule`]. Both paths are deterministic
//! functions of `(policy, key)` (plus the `ring_challenge_config` / `fold_shape` closures),
//! so prover and verifier resolve identical schedules.
//!
//! Every input the entry walker needs beyond the compact steps
//! (`sis_family`, `decomposition`, `challenge_field_bits`,
//! `claim_ext_degree`, `ring_subfield_norm_bound`) is a projection of
//! [`PlannerPolicy`], so the walker shares `find_schedule`'s call shape and
//! names no `CommitmentConfig` type.
//!
//! This is verifier-reachable, so every fallible step returns [`AkitaError`]
//! rather than panicking.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::LevelParams;
use akita_types::{
    direct_witness_bytes, extension_opening_reduction_proof_bytes, level_proof_bytes,
    terminal_direct_witness_shape_for_key, w_ring_element_count_with_counts_bits,
    w_ring_element_count_with_counts_for_layout_bits, AkitaScheduleInputs, AkitaScheduleLookupKey,
    CleartextWitnessShape, DirectStep, FoldStep, MRowLayout, Schedule, Step,
};

use crate::catalog_identity::validate_catalog_identity;
use crate::find_schedule;
// @generated schedule table cfg imports begin
#[cfg(not(feature = "zk"))]
use crate::generated::fp128_d64_onehot_tiered_table;
// @generated schedule table cfg imports end
use crate::generated::{
    // @generated schedule table imports begin
    fp128_d128_full_table,
    fp128_d128_onehot_table,
    fp128_d64_full_table,
    fp128_d64_onehot_table,
    fp128_d64_onehot_tensor_table,
    fp32_d128_onehot_table,
    fp32_d256_onehot_table,
    fp64_d128_onehot_table,
    fp64_d128_table,
    fp64_d256_onehot_table,
    // @generated schedule table imports end
    table_entry,
    GeneratedScheduleKey,
    GeneratedScheduleTable,
    GeneratedScheduleTableEntry,
    GeneratedStep,
    SisModulusFamily,
};
use crate::PlannerPolicy;

/// Convert the public runtime lookup key into a generated-table lookup key.
///
/// Generated tables retain the `num_commitment_groups` field in their on-disk
/// key shape, but runtime opening batches always use one commitment group.
pub const fn generated_schedule_lookup_key(key: AkitaScheduleLookupKey) -> GeneratedScheduleKey {
    GeneratedScheduleKey {
        num_vars: key.num_vars,
        num_commitment_groups: 1,
        num_t_vectors: key.num_t_vectors,
        num_w_vectors: key.num_w_vectors,
        num_z_vectors: key.num_z_vectors,
    }
}

/// The shipped generated schedule table for a `policy`, if one exists.
///
/// The planner owns every shipped table, so it owns the mapping from a
/// `PlannerPolicy` to its table. Selection keys on the SIS family and ring
/// degree, plus two binary discriminators:
///
/// - `onehot` — `decomposition.log_commit_bound == 1` (the onehot presets
///   commit balanced single-bit digits; full-field presets carry
///   `log_commit_bound == field_bits`).
/// - `root_fold_is_tensor` — whether the level-0 fold challenge is
///   tensor-shaped, which is the *only* discriminator between the otherwise
///   identical `fp128_d64_onehot` and `fp128_d64_onehot_tensor` policies.
///
/// Under the committed-fold A-role SIS pricing, `D=128` is the proof-size
/// optimum and the shipped production dimension for every field; the small
/// primes ship one-hot only (full-field small primes exceed the small-modulus
/// SIS floor at realistic `num_vars`). `(family, ring_degree, onehot)`
/// combinations with no shipped table (full-field small primes, or any
/// recursive-w derived policy whose `log_commit_bound` is its `log_basis`)
/// return `None`, so the caller regenerates from scratch.
pub fn shipped_table(
    policy: &PlannerPolicy,
    root_fold_is_tensor: bool,
) -> Option<GeneratedScheduleTable> {
    let onehot = policy.decomposition.log_commit_bound == 1;
    // Tiered policies select a dedicated tiered table whose compact entries
    // store the committed `B'`/`F` layout directly (`tier_split` + `n_f`). A
    // tiered policy never aliases a non-tiered table.
    if policy.tiered {
        #[cfg(not(feature = "zk"))]
        if !root_fold_is_tensor
            && matches!(
                (policy.sis_family, policy.ring_dimension, onehot),
                (SisModulusFamily::Q128, 64, true)
            )
        {
            return Some(fp128_d64_onehot_tiered_table());
        }
        return None;
    }
    Some(match (policy.sis_family, policy.ring_dimension, onehot) {
        (SisModulusFamily::Q128, 128, true) => fp128_d128_onehot_table(),
        (SisModulusFamily::Q128, 128, false) => fp128_d128_full_table(),
        (SisModulusFamily::Q128, 64, true) => {
            if root_fold_is_tensor {
                fp128_d64_onehot_tensor_table()
            } else {
                fp128_d64_onehot_table()
            }
        }
        (SisModulusFamily::Q128, 64, false) => fp128_d64_full_table(),
        (SisModulusFamily::Q32, 128, true) => fp32_d128_onehot_table(),
        // Full-field fp32 is unsecurable at any audited D (small-modulus SIS
        // floor), so there is no shipped table.
        (SisModulusFamily::Q32, 128, false) => return None,
        (SisModulusFamily::Q32, 256, true) => fp32_d256_onehot_table(),
        (SisModulusFamily::Q64, 128, true) => fp64_d128_onehot_table(),
        (SisModulusFamily::Q64, 128, false) => fp64_d128_table(),
        (SisModulusFamily::Q64, 256, true) => fp64_d256_onehot_table(),
        _ => return None,
    })
}

/// Resolve the runtime [`Schedule`] for `key` under `policy`: cache, then
/// generate.
///
/// This is the single entry point the runtime (config → prover/verifier)
/// uses. It first consults the planner-owned shipped-table cache
/// ([`shipped_table`]); on a table hit the matching compact entry is
/// expanded by [`schedule_from_entry`], and on a miss (or no shipped table
/// for the policy) the schedule is regenerated by the offline DP
/// [`find_schedule`]. Deterministic in `(policy, key)` plus the
/// `ring_challenge_config` / `fold_shape` closures, so prover and verifier agree and the
/// Fiat-Shamir plan digest stays consistent.
///
/// Prefer [`resolve_schedule`] when the caller supplies an explicit catalog.
///
/// # Errors
///
/// Propagates entry expansion failures and DP-search failures (invalid key
/// dimensions, witness overflow). Never panics — this is verifier-reachable.
pub fn get_schedule(
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_shape: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError> {
    // The level-0 fold shape disambiguates the tensor table; it depends only
    // on `inputs.level` for every shipped preset, so a witness-length
    // overflow at huge `num_vars` just defaults to the flat (non-tensor)
    // table and lets `find_schedule` surface the real error.
    let root_fold_is_tensor = match 1usize.checked_shl(key.num_vars as u32) {
        Some(root_w_len) => matches!(
            fold_shape(AkitaScheduleInputs {
                num_vars: key.num_vars,
                level: 0,
                current_w_len: root_w_len,
            }),
            TensorChallengeShape::Tensor
        ),
        None => false,
    };

    if let Some(table) = shipped_table(policy, root_fold_is_tensor) {
        if let Some(entry) = table_entry(table, generated_schedule_lookup_key(key)) {
            return schedule_from_entry(entry, key, policy, ring_challenge_config, fold_shape);
        }
    }
    find_schedule(key, policy, ring_challenge_config, fold_shape)
}

/// Resolve the runtime [`Schedule`] using an explicit optional catalog.
///
/// When `catalog` is `None`, delegates to [`find_schedule`] (offline DP only).
/// When `catalog` is `Some`, validates catalog identity (hard error on mismatch),
/// looks up the compact entry, expands on hit, and falls back to [`find_schedule`]
/// on a row miss.
pub fn resolve_schedule(
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_shape: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    catalog: Option<&GeneratedScheduleTable>,
) -> Result<Schedule, AkitaError> {
    let Some(table) = catalog else {
        return find_schedule(key, policy, ring_challenge_config, fold_shape);
    };
    validate_catalog_identity(table, policy, &ring_challenge_config, &fold_shape)?;
    if let Some(entry) = table_entry(*table, generated_schedule_lookup_key(key)) {
        return schedule_from_entry(entry, key, policy, ring_challenge_config, fold_shape);
    }
    find_schedule(key, policy, ring_challenge_config, fold_shape)
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
            extension_opening_width.saturating_mul(key.num_w_vectors),
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

/// Build the runtime [`Schedule`] for a compact generated entry, expanding
/// each fold level via
/// [`crate::generated::GeneratedFoldStep::expand_to_level_params`] and
/// computing each step's witness lengths and proof bytes.
///
/// This is the single canonical entry walker. The per-config inputs the
/// expansion needs are projected from `policy`; the `ring_challenge_config` / `fold_shape`
/// closures are threaded through exactly as [`find_schedule`] consumes them,
/// so the table-hit and table-miss root layouts agree byte-for-byte.
///
/// # Errors
///
/// Returns an error when the entry is structurally invalid, a fold step
/// names an unsupported ring dimension, layout expansion fails, or a
/// witness length overflows.
pub fn schedule_from_entry(
    entry: &GeneratedScheduleTableEntry,
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_shape: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
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
    #[cfg(feature = "zk")]
    let mut current_log_basis = policy.decomposition.log_basis;
    let mut terminal_witness_field_len: Option<usize> = None;
    #[cfg(not(feature = "zk"))]
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
                // The root commits `num_t_vectors` polynomials (batch factor);
                // recursive levels are always single-claim.
                let level_num_claims = if fold_level == 0 {
                    key.num_t_vectors
                } else {
                    1
                };
                let lp = level.expand_to_level_params(
                    policy,
                    &ring_challenge_config,
                    fold_level,
                    current_w_len,
                    fold_shape(inputs),
                    level_num_claims,
                )?;
                let (np, nt, nw, nz) = if fold_level == 0 {
                    (1, key.num_t_vectors, key.num_w_vectors, key.num_z_vectors)
                } else {
                    (1, 1, 1, 1)
                };
                let mul_d = |ring: usize| -> Result<usize, AkitaError> {
                    ring.checked_mul(lp.ring_dimension).ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "generated next witness length overflow".to_string(),
                        )
                    })
                };
                let (next_w_len, next_lp, layout) = if is_terminal {
                    let ring = w_ring_element_count_with_counts_for_layout_bits(
                        field_bits,
                        &lp,
                        np,
                        nt,
                        nw,
                        nz,
                        MRowLayout::WithoutDBlock,
                    )?;
                    let len = mul_d(ring)?;
                    terminal_witness_field_len = Some(len);
                    (len, None, MRowLayout::WithoutDBlock)
                } else {
                    let ring =
                        w_ring_element_count_with_counts_bits(field_bits, &lp, np, nt, nw, nz)?;
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
                    let next_lp = next_level.expand_to_level_params(
                        policy,
                        &ring_challenge_config,
                        fold_level + 1,
                        len,
                        fold_shape(next_inputs),
                        1,
                    )?;
                    (len, Some(next_lp), MRowLayout::WithDBlock)
                };
                let num_claims_here = if fold_level == 0 {
                    key.num_z_vectors
                } else {
                    1
                };
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
                #[cfg(not(feature = "zk"))]
                {
                    last_fold_lp = Some(lp.clone());
                }
                steps.push(Step::Fold(FoldStep {
                    params: lp,
                    current_w_len,
                    next_w_len,
                    level_bytes,
                }));
                fold_level += 1;
                current_w_len = next_w_len;
                #[cfg(feature = "zk")]
                {
                    current_log_basis = match next {
                        GeneratedStep::Fold(next_level) => next_level.log_basis,
                        GeneratedStep::Direct(_) => level.log_basis,
                    };
                }
            }
            GeneratedStep::Direct(direct) => {
                let (witness_shape, direct_current_w_len, params) = if fold_level == 0 {
                    // Root-direct: ship the cleartext field-element witness;
                    // carry the expanded root commit layout. The commit
                    // commits `num_t_vectors` polynomials, so the batch factor
                    // is folded straight into the B/D widths. A strict SIS
                    // audit failure (the large-`num_vars` edge) yields the
                    // *uncommittable* `params: None` rather than propagating.
                    let params = match direct.commit {
                        Some(commit) => commit
                            .expand_to_level_params(
                                policy,
                                &ring_challenge_config,
                                0,
                                expected_root_w_len,
                                fold_shape(AkitaScheduleInputs {
                                    num_vars: key.num_vars,
                                    level: 0,
                                    current_w_len: expected_root_w_len,
                                }),
                                key.num_t_vectors,
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
                    #[cfg(not(feature = "zk"))]
                    let terminal_lp = last_fold_lp.as_ref().ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "terminal direct step missing predecessor fold params".to_string(),
                        )
                    })?;
                    #[cfg(feature = "zk")]
                    let terminal_lp_stub = LevelParams::log_basis_stub(current_log_basis);
                    #[cfg(not(feature = "zk"))]
                    let terminal_log_basis = terminal_lp.log_basis;
                    #[cfg(feature = "zk")]
                    let terminal_log_basis = current_log_basis;
                    let witness_shape = terminal_direct_witness_shape_for_key(
                        #[cfg(not(feature = "zk"))]
                        terminal_lp,
                        #[cfg(feature = "zk")]
                        &terminal_lp_stub,
                        field_bits,
                        key,
                        terminal_fold_level,
                        len,
                        terminal_log_basis,
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

/// Total header-stripped proof bytes for a compact generated schedule entry.
///
/// Thin reader of [`schedule_from_entry`].
///
/// # Errors
///
/// Propagates [`schedule_from_entry`].
pub fn estimate_proof_bytes(
    entry: &GeneratedScheduleTableEntry,
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_shape: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<usize, AkitaError> {
    Ok(schedule_from_entry(entry, key, policy, ring_challenge_config, fold_shape)?.total_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_types::{DecompositionParams, SisModulusFamily};

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
    fn generated_schedule_key_uses_single_commitment_group() {
        let key = AkitaScheduleLookupKey::new(16, 4, 4, 1);
        assert_eq!(generated_schedule_lookup_key(key).num_commitment_groups, 1);
    }

    #[test]
    fn resolve_schedule_none_matches_find_schedule() {
        let key = AkitaScheduleLookupKey::new(20, 1, 1, 1);
        let policy = flat_policy();
        let via_resolve = resolve_schedule(key, &policy, ring_challenge_config, fold_shape, None)
            .expect("resolve");
        let via_find =
            find_schedule(key, &policy, ring_challenge_config, fold_shape).expect("find");
        assert_eq!(via_resolve.total_bytes, via_find.total_bytes);
    }

    #[test]
    fn resolve_schedule_row_miss_falls_back_to_dp() {
        let key = AkitaScheduleLookupKey::new(99, 1, 1, 1);
        let policy = flat_policy();
        let table = fp128_d64_onehot_table();
        let via_catalog = resolve_schedule(
            key,
            &policy,
            ring_challenge_config,
            fold_shape,
            Some(&table),
        )
        .expect("catalog miss fallback");
        let via_find =
            find_schedule(key, &policy, ring_challenge_config, fold_shape).expect("find");
        assert_eq!(via_catalog.total_bytes, via_find.total_bytes);
    }
}
