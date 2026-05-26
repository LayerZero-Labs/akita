//! Fast-verify preset support: tiered-commitment root LP layering.
//!
//! These helpers post-process the materialized root `LevelParams`
//! returned by [`crate::proof_optimized::proof_optimized_schedule_plan`]
//! to inject the tier-1 / F / ûhat-gadget metadata required by
//! `specs/tiered_commit.md` §3. A "fast-verify" preset is a production
//! preset whose root commits through a chunked B' + outer F matrix
//! (the verifier-side `setup_contribution` α-eval rectangle shrinks
//! to `chunk_width = legacy_outer_width / split_factor`, the dominant
//! verifier-cost term), at the cost of a small extra ûhat / F witness
//! segment.
//!
//! The split factor is computed *dynamically* per `LevelParams` shape
//! via [`akita_types::apply_dynamic_tier`]: the smallest divisor of
//! the outer width such that the chunked B' rectangle is no larger
//! than the inner A rectangle. When `|B| <= |A|` already, the helper
//! returns the LP unchanged (`split_factor = 1`) and the preset
//! degrades to the legacy proof-optimised path for that shape.
//!
//! Public entry points are [`fast_verifier_schedule_plan`],
//! [`fast_verifier_max_setup_matrix_size`], and the
//! [`impl_fp128_fast_verify_preset!`] macro that wires a fast-verify
//! preset into the same scaffolding as `impl_fp128_preset!` but
//! routes `schedule_plan` through the tier post-processor.

use crate::proof_optimized::{proof_optimized_schedule_plan, setup_matrix_envelope_for_shape};
use crate::CommitmentConfig;
use akita_field::AkitaError;
use akita_types::{
    apply_dynamic_tier, w_ring_element_count_with_counts_bits,
    w_ring_element_count_with_counts_for_layout_bits, AkitaPlannedStep, AkitaSchedulePlan,
    ClaimIncidenceSummary, CommitmentEnvelope, LevelParams,
};

/// Fast-verify variant of `proof_optimized_max_setup_matrix_size`.
///
/// The base function walks every committable sub-shape
/// `(num_vars', num_polys', num_points')`. For a fast-verify preset
/// the planner errors out at small `num_vars'` values whose root
/// layout cannot meet the tier constraint
/// `outer_width % split_factor == 0`. We swallow those errors as
/// "unsupported shape" (mirroring the `Ok(None)` semantics that the
/// envelope walker already understands), so the envelope just covers
/// the feasible shapes. That suffices for setup sizing because the
/// chunk-width tiered B' is strictly narrower than the legacy outer
/// width at every shape it does support, so smaller / infeasible
/// shapes cannot need a wider matrix than the supported ones.
pub fn fast_verifier_max_setup_matrix_size<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Result<(usize, usize), AkitaError> {
    if max_num_batched_polys == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    if max_num_points == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_points must be at least 1".to_string(),
        ));
    }
    if max_num_points > max_num_batched_polys {
        return Err(AkitaError::InvalidSetup(format!(
            "max_num_points ({max_num_points}) cannot exceed \
             max_num_batched_polys ({max_num_batched_polys})"
        )));
    }

    let mut max_rows: usize = 1;
    let mut max_stride: usize = 1;
    let mut saw_supported_shape = false;
    for num_vars in 1..=max_num_vars {
        let envelope = Cfg::envelope(num_vars);
        for num_polys in 1..=max_num_batched_polys {
            let upper_pts = num_polys.min(max_num_points);
            for num_points in 1..=upper_pts {
                let incidence =
                    ClaimIncidenceSummary::from_counts(num_vars, num_polys, num_points)?;
                let shape_env = match setup_matrix_envelope_for_shape::<Cfg>(&incidence, envelope) {
                    Ok(opt) => opt,
                    Err(AkitaError::InvalidSetup(_)) => None,
                    Err(err) => return Err(err),
                };
                let Some((rows, stride)) = shape_env else {
                    continue;
                };
                saw_supported_shape = true;
                max_rows = max_rows.max(rows);
                max_stride = max_stride.max(stride);
            }
        }
    }

    if !saw_supported_shape {
        return Err(AkitaError::InvalidSetup(format!(
            "fast_verifier setup matrix sizing found no generated schedules \
             for max_num_vars={max_num_vars}"
        )));
    }

    Ok((max_rows, max_stride))
}

/// Fast-verify variant of [`proof_optimized_schedule_plan`].
///
/// Reads from the Cfg's generated schedule table (sized for the
/// fast-verify shapes by the offline generator) and post-processes
/// the root step to layer fast-verify metadata on top — the on-disk
/// `GeneratedFoldStep` records only `(ring_d, log_basis, m_vars,
/// r_vars, n_a, n_b, n_d)`, so the tier metadata is re-derived here
/// dynamically from the materialised root LP shape (same rule the
/// generator used). The `n_b` stored in the table is already `n_b'`
/// (the tier-1 B' rank for the chosen split); we re-derive the same
/// `(split, n_b', n_F)` from the legacy SIS rank at the materialised
/// root `outer_width` via [`apply_dynamic_tier`].
pub fn fast_verifier_schedule_plan<Cfg: CommitmentConfig>(
    key: akita_types::AkitaScheduleLookupKey,
    envelope: CommitmentEnvelope,
) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
    let Some(mut plan) = proof_optimized_schedule_plan::<Cfg>(key, envelope)? else {
        return Ok(None);
    };
    let field_bits = Cfg::decomposition().field_bits();

    // Compute the "next step is Direct?" mask before taking the
    // mutable borrow on `plan.steps.first_mut()` (borrow checker).
    let suffix_len = plan.steps.len();
    let next_is_direct: Vec<bool> = (1..suffix_len)
        .map(|i| matches!(plan.steps.get(i + 1), Some(AkitaPlannedStep::Direct(_))))
        .collect();
    let Some(AkitaPlannedStep::Fold(root_level)) = plan.steps.first_mut() else {
        return Ok(Some(plan));
    };

    // The materialiser constructs the root LP with
    // `b_key.col_len = full_outer_width` and `b_key.row_len = n_b'`
    // (the post-split rank stored in the table). Synthesize the
    // *legacy* unchunked LP first so `apply_dynamic_tier` re-derives
    // the same split that the generator picked.
    let legacy_root = synthesize_legacy_root_lp(&root_level.lp)?;
    let tiered_lp = apply_dynamic_tier(&legacy_root, field_bits)?;

    // The base materialiser computed the root's `runtime_next_w_len`
    // against the pre-tiered LP, so we re-do it here against the
    // tiered LP — `w_ring_element_count_with_counts_for_layout_bits`
    // already adds the ûhat segment for tiered roots.
    let next_w_ring = w_ring_element_count_with_counts_for_layout_bits(
        field_bits,
        &tiered_lp,
        key.num_points,
        key.num_t_vectors,
        key.num_w_vectors,
        key.num_z_vectors,
        akita_types::MRowLayout::Intermediate,
    )?;
    let tiered_next_w_len = next_w_ring
        .checked_mul(tiered_lp.ring_dimension)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("fast_verifier root next witness length overflow".to_string())
        })?;
    root_level.lp = tiered_lp;
    root_level.next_inputs.current_w_len = tiered_next_w_len;

    // Suffix walk: at each recursive level `block_len = ceil(num_ring
    // / num_blocks)` (see `LevelParams::with_decomp`), so when the
    // root exit `num_ring` grows by the ûhat segment we must re-lay
    // out every downstream level under the new entry state.
    let root_decomp = Cfg::decomposition();
    let mut prev_w_len = tiered_next_w_len;
    for (idx, step) in plan.steps.iter_mut().enumerate().skip(1) {
        match step {
            AkitaPlannedStep::Fold(level) => {
                level.inputs.current_w_len = prev_w_len;
                let level_decomp =
                    recursive_level_decomposition_from_root(root_decomp, level.lp.log_basis);
                let num_ring = prev_w_len / level.lp.ring_dimension;
                let m_vars = level.lp.m_vars;
                let r_vars = level.lp.r_vars;
                let relayed = akita_types::layout::sis_derivation::level_layout_from_params(
                    m_vars,
                    r_vars,
                    &level.lp,
                    level_decomp,
                    num_ring,
                )?;
                level.lp = relayed;
                let next_ring = if next_is_direct[idx - 1] {
                    w_ring_element_count_with_counts_for_layout_bits(
                        field_bits,
                        &level.lp,
                        1,
                        1,
                        1,
                        1,
                        akita_types::MRowLayout::Terminal,
                    )?
                } else {
                    w_ring_element_count_with_counts_bits(field_bits, &level.lp, 1, 1, 1, 1)?
                };
                let next_len = next_ring
                    .checked_mul(level.lp.ring_dimension)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("fast_verifier suffix next-w overflow".to_string())
                    })?;
                level.next_inputs.current_w_len = next_len;
                prev_w_len = next_len;
            }
            AkitaPlannedStep::Direct(direct) => {
                direct.state.current_w_len = prev_w_len;
            }
        }
    }
    Ok(Some(plan))
}

/// Recover a legacy (non-tiered) root LP from a materialised entry's LP.
///
/// The base materialiser constructs the root LP with
/// `b_key.col_len = full_outer_width` and `b_key.row_len = n_b'`
/// (post-split). To re-derive the dynamic split we need the *legacy*
/// (unchunked) `n_b`, which we recover via the SIS floor at
/// `outer_width` with `split_factor = 1`.
fn synthesize_legacy_root_lp(materialised: &LevelParams) -> Result<LevelParams, AkitaError> {
    use akita_types::layout::sis_derivation::tiered_b_prime_rank;
    let outer_width = materialised.full_outer_width();
    let family = materialised.b_key.sis_family();
    let d = materialised.ring_dimension;
    let t_inf_bound = materialised.b_key.collision_inf();
    let legacy_n_b = tiered_b_prime_rank(family, d as u32, t_inf_bound, outer_width, 1)? as usize;
    Ok(LevelParams {
        b_key: akita_types::AjtaiKeyParams::new_unchecked(
            family,
            legacy_n_b,
            outer_width,
            t_inf_bound,
            d,
        ),
        ..materialised.clone()
    })
}

/// Derive the recursive-level decomposition from the root.
///
/// Mirrors the inline rule used by `akita_derive::current_level_layout_with_log_basis`:
/// recursive levels keep the root's `log_open_bound` but reset
/// `log_basis = log_commit_bound = log_basis`.
fn recursive_level_decomposition_from_root(
    root_decomp: akita_types::DecompositionParams,
    log_basis: u32,
) -> akita_types::DecompositionParams {
    let parent_open = root_decomp
        .log_open_bound
        .unwrap_or(root_decomp.log_commit_bound);
    akita_types::DecompositionParams {
        log_basis,
        log_commit_bound: log_basis,
        log_open_bound: Some(parent_open),
    }
}

/// Fast-verify variant of `impl_fp128_preset`. Shares 99 % of its
/// body with the legacy macro; the only differences are:
///
/// * `schedule_plan` post-processes the materialised plan so the
///   root step's `LevelParams` carries the fast-verify fields even
///   though the on-disk `GeneratedFoldStep` only records the legacy
///   fields. The split factor is chosen dynamically per LP shape
///   via [`akita_types::apply_dynamic_tier`].
/// * `max_setup_matrix_size` uses the tolerant walk that swallows
///   tier-infeasible sub-shapes as `Ok(None)`.
/// * `audited_root_rank` / `envelope` are not extended — the tiered
///   `b_key.col_len = chunk_width = legacy_outer_width / split_factor`
///   is strictly smaller than the legacy outer width, so the legacy
///   envelope safely upper-bounds it.
macro_rules! impl_fp128_fast_verify_preset {
    ($cfg:ident, $d:expr, $log_commit_bound:expr, $table:expr) => {
        impl $crate::CommitmentConfig for $cfg {
            type Field = Field;
            type ClaimField = Field;
            type ChallengeField = Field;
            const D: usize = $d;

            fn decomposition() -> akita_types::DecompositionParams {
                akita_types::DecompositionParams {
                    log_basis: 3,
                    log_commit_bound: $log_commit_bound,
                    log_open_bound: if $log_commit_bound < 128 {
                        Some(128)
                    } else {
                        None
                    },
                }
            }

            fn stage1_challenge_config(
                d: usize,
            ) -> Result<akita_challenges::SparseChallengeConfig, akita_field::AkitaError> {
                match d {
                    32 => Ok(akita_challenges::SparseChallengeConfig::BoundedL1Norm),
                    64 => Ok(akita_challenges::SparseChallengeConfig::ExactShell {
                        count_mag1: 30,
                        count_mag2: 12,
                    }),
                    128 => Ok(akita_challenges::SparseChallengeConfig::Uniform {
                        weight: 31,
                        nonzero_coeffs: vec![-1, 1],
                    }),
                    _ => Err(akita_field::AkitaError::InvalidSetup(format!(
                        "unsupported fp128 ring dim {d}"
                    ))),
                }
            }

            fn sis_modulus_family() -> akita_types::SisModulusFamily {
                akita_types::SisModulusFamily::Q128
            }

            fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
                $table
            }

            fn schedule_plan(
                key: akita_types::AkitaScheduleLookupKey,
            ) -> Result<Option<akita_types::AkitaSchedulePlan>, akita_field::AkitaError> {
                let envelope = <Self as $crate::CommitmentConfig>::envelope(key.num_vars);
                $crate::fast_verifier::fast_verifier_schedule_plan::<Self>(key, envelope)
            }

            fn audited_root_rank(role: akita_types::AjtaiRole, max_num_vars: usize) -> usize {
                let log_commit_bound =
                    <Self as $crate::CommitmentConfig>::decomposition().log_commit_bound;
                let threshold: Option<usize> = match (
                    <Self as $crate::CommitmentConfig>::D,
                    log_commit_bound,
                    role,
                ) {
                    (128, lcb, akita_types::AjtaiRole::Inner) if lcb != 1 => Some(59),
                    (128, _, akita_types::AjtaiRole::Outer) => Some(54),
                    (64, 1, akita_types::AjtaiRole::Outer) => Some(38),
                    _ => None,
                };
                1 + usize::from(threshold.is_some_and(|t| max_num_vars >= t))
            }

            fn envelope(max_num_vars: usize) -> akita_types::CommitmentEnvelope {
                $crate::proof_optimized::proof_optimized_envelope::<Self>(max_num_vars)
            }

            fn max_setup_matrix_size(
                max_num_vars: usize,
                max_num_batched_polys: usize,
                max_num_points: usize,
            ) -> Result<(usize, usize), akita_field::AkitaError> {
                $crate::fast_verifier::fast_verifier_max_setup_matrix_size::<Self>(
                    max_num_vars,
                    max_num_batched_polys,
                    max_num_points,
                )
            }

            fn log_basis_search_range(_inputs: akita_types::AkitaScheduleInputs) -> (u32, u32) {
                (
                    $crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MIN,
                    $crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MAX,
                )
            }
        }
    };
}
pub(crate) use impl_fp128_fast_verify_preset;
