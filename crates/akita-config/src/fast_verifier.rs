//! Fast-verify preset support: tiered-commitment root LP layering.
//!
//! These helpers mirror the legacy proof-optimised production helpers
//! but post-process the root `LevelParams` to inject the tier-1 / F /
//! ûhat-gadget metadata required by `specs/tiered_commit.md` §3. A
//! "fast-verify" preset is a production preset whose root commits
//! through a chunked B' + outer F matrix (the verifier-side
//! `setup_contribution` α-eval rectangle shrinks to `chunk_width =
//! legacy_outer_width / split_factor`, which is the dominant
//! verifier-cost term), at the cost of a small extra ûhat / F
//! witness segment.
//!
//! The split factor is computed *dynamically* per
//! `LevelParams` shape via
//! [`akita_types::apply_dynamic_tier`]: the smallest divisor of the
//! outer width such that the chunked B' rectangle is no larger than
//! the inner A rectangle. When `|B| <= |A|` already, the helper
//! returns the LP unchanged (`split_factor = 1`) and the preset
//! degrades to the legacy proof-optimised path for that shape.
//!
//! Public entry points are the `fast_verifier_*` helpers plus the
//! [`impl_fp128_fast_verify_preset!`] macro that wires a fast-verify
//! preset into the same scaffolding as `impl_fp128_preset!`.

use crate::proof_optimized::{
    proof_optimized_root_level_layout_with_log_basis,
    proof_optimized_root_level_params_for_layout_with_log_basis, proof_optimized_schedule_plan,
    setup_matrix_envelope_for_shape,
};
use crate::CommitmentConfig;
use akita_field::AkitaError;
use akita_types::{
    apply_dynamic_tier, AkitaPlannedStep, AkitaScheduleInputs, AkitaScheduleLookupKey,
    AkitaSchedulePlan, ClaimIncidenceSummary, LevelParams,
};

/// Layer fast-verify metadata (`split_factor`, `outer_log_basis`,
/// `num_digits_outer`, `f_key`) onto a legacy root `LevelParams`, with
/// the split factor chosen dynamically from the LP's own
/// `(|A|, |B|, outer_width)` shape via [`apply_dynamic_tier`].
///
/// When the dynamic rule returns `split_factor = 1` (i.e. `|B| <= |A|`
/// already) the helper returns an LP with cleared tier fields; the
/// preset effectively degrades to the legacy proof-optimised path for
/// that shape.
///
/// # Errors
///
/// Returns an error if no valid divisor of `outer_width` covers the
/// `ceil(|B|/|A|)` target, or if the SIS floor tables don't cover the
/// resulting `(family, D, collision, width)` tuples.
pub(crate) fn fast_verifier_apply_to_root_lp<Cfg: CommitmentConfig>(
    legacy_root: &LevelParams,
) -> Result<LevelParams, AkitaError> {
    let field_bits = Cfg::decomposition().field_bits();
    apply_dynamic_tier(legacy_root, field_bits)
}

/// Fast-verify variant of `proof_optimized_root_level_layout_with_log_basis`.
///
/// Runs the production SIS-rank convergence to derive the legacy root
/// shape, then layers fast-verify metadata on top via
/// [`fast_verifier_apply_to_root_lp`].
pub(crate) fn fast_verifier_root_level_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    log_basis: u32,
) -> Result<LevelParams, AkitaError> {
    let legacy = proof_optimized_root_level_layout_with_log_basis::<Cfg>(inputs, log_basis)?;
    fast_verifier_apply_to_root_lp::<Cfg>(&legacy)
}

/// Fast-verify variant of
/// `proof_optimized_root_level_params_for_layout_with_log_basis`.
pub(crate) fn fast_verifier_root_level_params_for_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, AkitaError> {
    let legacy = proof_optimized_root_level_params_for_layout_with_log_basis::<Cfg>(inputs, lp)?;
    fast_verifier_apply_to_root_lp::<Cfg>(&legacy)
}

/// Fast-verify variant of `proof_optimized_max_setup_matrix_size`.
///
/// The base function walks every committable sub-shape
/// `(num_vars', num_polys', num_points')` with `1 ≤ num_vars' ≤ max`.
/// For a fast-verify preset the planner errors out at small
/// `num_vars'` values whose root layout cannot meet the tier
/// constraint `outer_width % split_factor == 0`. We swallow those
/// errors as "unsupported shape" (mirroring the `Ok(None)` semantics
/// that the envelope walker already understands), so the envelope
/// just covers the feasible shapes. That suffices for setup sizing
/// because the chunk-width tiered B' is strictly narrower than the
/// legacy outer width at every shape it does support, so smaller /
/// infeasible shapes cannot need a wider matrix than the supported
/// ones.
pub(crate) fn fast_verifier_max_setup_matrix_size<Cfg: CommitmentConfig>(
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
    let setup_envelope = Cfg::envelope(max_num_vars);
    for num_vars in 1..=max_num_vars {
        for num_polys in 1..=max_num_batched_polys {
            let upper_pts = num_polys.min(max_num_points);
            for num_points in 1..=upper_pts {
                let incidence =
                    ClaimIncidenceSummary::from_counts(num_vars, num_polys, num_points)?;
                // Tolerant shape walk: planner errors at infeasible
                // tier shapes are treated as "not a supported shape"
                // rather than propagated.
                let shape_env =
                    match setup_matrix_envelope_for_shape::<Cfg>(&incidence, &setup_envelope) {
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

/// Fast-verify variant of `proof_optimized_schedule_plan`.
///
/// Reads from the Cfg's generated schedule table (sized for the
/// fast-verify shapes by the offline generator) and post-processes
/// the root step to layer fast-verify metadata on top — the on-disk
/// `GeneratedFoldStep` records only `(ring_d, log_basis, m_vars,
/// r_vars, n_a, n_b, n_d)`, so the tier metadata is re-derived here
/// dynamically from the materialised root LP shape (same rule the
/// generator used). The `n_b` stored in the table is already `n_b'`
/// (the tier-1 B' rank for the chosen split); we re-derive the same
/// `(split, n_b', n_F)` from `(n_a, num_blocks, num_digits_open,
/// block_len, num_digits_commit, log_basis)` via
/// [`apply_dynamic_tier`].
pub(crate) fn fast_verifier_schedule_plan<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
    use akita_types::{
        w_ring_element_count_with_vector_counts_bits,
        w_ring_element_count_with_vector_counts_for_layout_bits,
    };
    let Some(mut plan) = proof_optimized_schedule_plan::<Cfg>(key)? else {
        return Ok(None);
    };
    let field_bits = Cfg::decomposition().field_bits();
    // Layer fast-verify metadata onto the root LP. The base
    // materialiser (`schedule_plan_from_generated_entry`) computed
    // the root's `runtime_next_w_len` against the pre-tiered LP, so
    // we re-do it here against the tiered LP — which DOES include
    // the ûhat segment via the tier-aware
    // `w_ring_element_count_with_counts_for_layout`. We then walk
    // the suffix and update each `inputs.current_w_len` /
    // `next_inputs.current_w_len` using the EXACT same `_bits`
    // variants the base materialiser uses, so we don't introduce
    // sizing discrepancies at recursive levels.
    //
    // The generated entry's `n_b` is already `n_b'` for the chosen
    // dynamic split, so before applying the tier we synthesise the
    // legacy `n_b` from the SIS floor at the materialised root
    // `outer_width` so `apply_dynamic_tier` reproduces the same
    // split.
    let suffix_len = plan.steps.len();
    let next_is_direct: Vec<bool> = (1..suffix_len)
        .map(|i| matches!(plan.steps.get(i + 1), Some(AkitaPlannedStep::Direct(_))))
        .collect();
    let Some(AkitaPlannedStep::Fold(root_level)) = plan.steps.first_mut() else {
        return Ok(Some(plan));
    };
    let legacy_root = synthesize_legacy_root_lp(&root_level.lp)?;
    let tiered_lp = apply_dynamic_tier(&legacy_root, field_bits)?;
    let next_w_ring = akita_types::w_ring_element_count_with_counts_for_layout::<Cfg::Field>(
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
    // / num_blocks)` where `num_ring = current_w_len /
    // ring_dimension` (see `LevelParams::with_decomp` — the level's
    // `block_len` is sized to actually hold the carried witness,
    // not just `1 << m_vars`). Since fast-verify changes the root's
    // exit `num_ring` (extra ûhat segment + tier-1/F r-rows), every
    // downstream level must be re-laid-out under the new entry
    // state. We re-derive each level's LP from its `(m_vars,
    // r_vars)` + the cumulative tiered `current_w_len`, mirroring
    // exactly what `schedule_plan_from_generated_entry` does for the
    // legacy path.
    let root_decomp = Cfg::decomposition();
    let mut prev_w_len = tiered_next_w_len;
    for (idx, step) in plan.steps.iter_mut().enumerate().skip(1) {
        match step {
            AkitaPlannedStep::Fold(level) => {
                level.inputs.current_w_len = prev_w_len;
                let level_decomp =
                    akita_types::layout::sis_derivation::recursive_level_decomposition_from_root(
                        root_decomp,
                        level.lp.log_basis,
                    );
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
                    w_ring_element_count_with_vector_counts_for_layout_bits::<Cfg::Field>(
                        field_bits,
                        &level.lp,
                        1,
                        1,
                        1,
                        1,
                        akita_types::MRowLayout::Terminal,
                    )?
                } else {
                    w_ring_element_count_with_vector_counts_bits::<Cfg::Field>(
                        field_bits, &level.lp, 1, 1, 1, 1,
                    )?
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
/// The base materialiser (`schedule_plan_from_generated_entry`) constructs
/// the root LP with `b_key.col_len = full_outer_width` and
/// `b_key.row_len = n_b_from_table`. For a fast-verify table the stored
/// `n_b` is already `n_b'` (post-split). To re-derive the dynamic split
/// we need the *legacy* (unchunked) `n_b`, which we recover via the SIS
/// floor table at `outer_width` with `split_factor = 1`. This is the
/// inverse of what the table generator did.
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

/// Fast-verify variant of `impl_fp128_preset`. Shares 99 % of its
/// body with the legacy macro; the only differences are:
///
/// * `root_level_layout_with_log_basis` /
///   `root_level_params_for_layout_with_log_basis` (and their planner
///   twins) call the `fast_verifier_*` helpers that layer the tier-1
///   / F / ûhat-gadget metadata on the legacy LP they produce. The
///   split factor is chosen dynamically per LP shape via
///   [`apply_dynamic_tier`].
/// * `schedule_plan` post-processes the materialised plan so the
///   root step's `LevelParams` carries the fast-verify fields even
///   though the on-disk `GeneratedFoldStep` only records the legacy
///   fields.
/// * `audited_root_rank` / `envelope` are not extended — the tiered
///   `b_key.col_len = chunk_width = legacy_outer_width / split_factor`
///   is strictly smaller than the legacy outer width, so the legacy
///   envelope safely upper-bounds it.
macro_rules! impl_fp128_fast_verify_preset {
    ($cfg:ident, $d:expr, $log_commit_bound:expr, $table:expr) => {
        impl akita_types::ScheduleProvider for $cfg {
            fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
                $table
            }

            fn schedule_key(key: akita_types::AkitaScheduleLookupKey) -> String {
                $crate::proof_optimized::proof_optimized_schedule_key::<Self>(key)
            }

            fn schedule_plan(
                key: akita_types::AkitaScheduleLookupKey,
            ) -> Result<Option<akita_types::AkitaSchedulePlan>, akita_field::AkitaError> {
                $crate::fast_verifier::fast_verifier_schedule_plan::<Self>(key)
            }
        }

        impl $crate::CommitmentConfig for $cfg {
            type Field = Field;
            type ClaimField = Field;
            type ChallengeField = Field;
            const D: usize = $d;

            fn decomposition() -> akita_types::DecompositionParams {
                $crate::proof_optimized::fp128_decomposition($log_commit_bound, 3)
            }

            fn stage1_challenge_config(d: usize) -> akita_challenges::SparseChallengeConfig {
                $crate::proof_optimized::fp128_stage1_challenge_config(d)
            }

            fn sis_modulus_family() -> akita_types::SisModulusFamily {
                akita_types::SisModulusFamily::Q128
            }

            fn audited_root_rank(role: akita_types::AjtaiRole, max_num_vars: usize) -> usize {
                $crate::proof_optimized::fp128_audited_root_rank::<Self>(role, max_num_vars)
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

            fn level_params_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> akita_types::LevelParams {
                // Non-root levels are non-tiered (tiering applies at
                // the root only — `specs/tiered_commit.md` §1). For
                // the root we still emit a non-tiered base shape here
                // because the table-driven materialiser layers
                // fast-verify fields on top via `schedule_plan`
                // above. Production code paths that bypass the table
                // and ask for the root LP shape directly use
                // `root_level_layout_with_log_basis`, which DOES
                // apply tiering.
                $crate::proof_optimized::proof_optimized_level_params_with_log_basis::<Self>(
                    inputs, log_basis,
                )
            }

            fn root_level_params_for_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                lp: &akita_types::LevelParams,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                $crate::fast_verifier::fast_verifier_root_level_params_for_layout_with_log_basis::<
                    Self,
                >(inputs, lp)
            }

            fn root_level_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                $crate::fast_verifier::fast_verifier_root_level_layout_with_log_basis::<Self>(
                    inputs, log_basis,
                )
            }

            fn log_basis_at_level(inputs: akita_types::AkitaScheduleInputs) -> u32 {
                $crate::proof_optimized::proof_optimized_log_basis_at_level::<Self>(inputs)
            }

            fn log_basis_search_range(_inputs: akita_types::AkitaScheduleInputs) -> (u32, u32) {
                $crate::proof_optimized::proof_optimized_log_basis_search_range()
            }
        }

        #[cfg(feature = "planner")]
        impl akita_planner::PlannerConfig for $cfg {
            type PlannerField = Field;

            const PLANNER_D: usize = $d;

            fn planner_field_bits() -> u32 {
                <Self as $crate::CommitmentConfig>::decomposition().field_bits()
            }

            fn planner_challenge_field_bits() -> u32 {
                <Self as $crate::CommitmentConfig>::decomposition().field_bits()
                    * (<Self as $crate::CommitmentConfig>::CHAL_EXT_DEGREE as u32)
            }

            fn planner_extension_opening_width() -> usize {
                <Self as $crate::CommitmentConfig>::CLAIM_EXT_DEGREE
            }

            fn planner_recursive_witness_expansion() -> usize {
                1
            }

            fn planner_recursive_public_rows() -> usize {
                1
            }

            fn planner_sis_modulus_family() -> akita_types::SisModulusFamily {
                <Self as $crate::CommitmentConfig>::sis_modulus_family()
            }

            fn planner_stage1_challenge_config(
                d: usize,
            ) -> akita_challenges::SparseChallengeConfig {
                <Self as $crate::CommitmentConfig>::stage1_challenge_config(d)
            }

            fn planner_schedule_plan(
                key: akita_types::AkitaScheduleLookupKey,
            ) -> Result<Option<akita_types::AkitaSchedulePlan>, akita_field::AkitaError> {
                <Self as akita_types::ScheduleProvider>::schedule_plan(key)
            }

            fn planner_root_level_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                <Self as $crate::CommitmentConfig>::root_level_layout_with_log_basis(
                    inputs, log_basis,
                )
            }

            fn planner_current_level_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                $crate::current_level_layout_with_log_basis::<Self>(inputs, log_basis)
            }

            fn planner_direct_level_params_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                $crate::schedule_policy::direct_level_params_with_log_basis::<Self>(
                    inputs, log_basis,
                )
            }

            fn planner_root_level_params_for_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                lp: &akita_types::LevelParams,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                <Self as $crate::CommitmentConfig>::root_level_params_for_layout_with_log_basis(
                    inputs, lp,
                )
            }

            fn planner_log_basis_search_range(
                inputs: akita_types::AkitaScheduleInputs,
            ) -> (u32, u32) {
                <Self as $crate::CommitmentConfig>::log_basis_search_range(inputs)
            }
        }
    };
}
pub(crate) use impl_fp128_fast_verify_preset;
