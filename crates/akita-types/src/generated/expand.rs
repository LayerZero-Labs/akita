//! On-demand expansion of compact generated schedule steps into full
//! [`LevelParams`].
//!
//! The planner stores only the brute-forced parameters
//! (`ring_d, log_basis, m_vars, r_vars, n_a, n_b, n_d`) in
//! [`GeneratedFoldStep`]; every other `LevelParams` component is a
//! deterministic function of those plus the config-fixed policy inputs.
//! [`GeneratedFoldStep::expand_to_level_params`] is the single place that
//! reconstructs the full layout, replacing the former
//! `akita-derive` materializer.
//!
//! This is verifier-reachable (config resolves levels through it on the
//! replay path), so every fallible step returns [`AkitaError`] rather than
//! panicking.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;

use crate::generated::sis_floor::ceil_supported_collision;
use crate::generated::{
    GeneratedDirectStep, GeneratedFoldStep, GeneratedScheduleTableEntry, GeneratedStep,
    SisModulusFamily,
};
use crate::{AjtaiKeyParams, DecompositionParams, LevelParams};

/// Reconstruct the SIS-secure A and B/D collision buckets for a generated
/// fold level.
///
/// Mirrors the bucket math the offline SIS derivation used:
/// `bd_raw = 2^log_basis − 1` rounded up to the nearest audited bucket;
/// `a_raw = 2` when `log_commit_bound == 1` else `bd_raw`, multiplied by the
/// stage-1 infinity norm and the ring-subfield embedding norm, then rounded
/// up.
///
/// # Errors
///
/// Returns an error when `log_basis` overflows the bound, the collision
/// product overflows, or no audited bucket covers the requested collision.
pub fn generated_level_buckets(
    sis_family: SisModulusFamily,
    ring_dimension: usize,
    log_basis: u32,
    log_commit_bound: u32,
    stage1_inf_norm: u32,
    ring_subfield_norm_bound: u32,
) -> Result<(u32, u32), AkitaError> {
    let bd_raw = 1u32
        .checked_shl(log_basis)
        .and_then(|b| b.checked_sub(1))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "generated schedule log_basis {log_basis} overflows bd_raw"
            ))
        })?;
    let a_raw = if log_commit_bound == 1 { 2 } else { bd_raw };
    let a_collision_raw = a_raw
        .checked_mul(stage1_inf_norm)
        .and_then(|v| v.checked_mul(ring_subfield_norm_bound))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "generated schedule A-role collision overflow at log_basis={log_basis}"
            ))
        })?;
    let a_bucket = ceil_supported_collision(sis_family, ring_dimension as u32, a_collision_raw)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "no audited A-role bucket for generated schedule \
                 (family={sis_family:?}, d={ring_dimension}, collision_inf={a_collision_raw})"
            ))
        })?;
    let bd_bucket = ceil_supported_collision(sis_family, ring_dimension as u32, bd_raw)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "no audited B/D-role bucket for generated schedule \
                 (family={sis_family:?}, d={ring_dimension}, collision_inf={bd_raw})"
            ))
        })?;
    Ok((a_bucket, bd_bucket))
}

impl GeneratedFoldStep {
    /// Expand this compact fold step into the full committed
    /// [`LevelParams`] for its position in the schedule.
    ///
    /// `fold_level` is `0` at the root and `>0` at recursive levels; it
    /// selects the level-local decomposition (root inherits
    /// `root_decomp`; recursive levels collapse `log_commit_bound` to the
    /// level's own `log_basis`). `current_w_len` is the witness length in
    /// field elements entering this level, used to size `block_len` and
    /// (with `batched_root`) the scaled root layout.
    ///
    /// `batched_root` is `Some((num_t_vectors, field_bits))` only when
    /// expanding a batched root (`fold_level == 0` and the lookup key has a
    /// non-singleton shape); it scales the per-claim root layout up to the
    /// batched widths.
    ///
    /// The same method expands a root-direct commit step (the
    /// [`GeneratedDirectStep::commit`] payload): a root-direct commit is a
    /// `fold_level == 0` expansion.
    ///
    /// # Errors
    ///
    /// Returns an error when bucket resolution, layout assembly, or batched
    /// scaling fails.
    #[allow(clippy::too_many_arguments)]
    pub fn expand_to_level_params(
        &self,
        sis_family: SisModulusFamily,
        fold_level: usize,
        current_w_len: usize,
        root_decomp: DecompositionParams,
        stage1: SparseChallengeConfig,
        fold_shape: TensorChallengeShape,
        ring_subfield_norm_bound: u32,
        batched_root: Option<(usize, u32)>,
    ) -> Result<LevelParams, AkitaError> {
        let ring_d = self.ring_d as usize;
        if ring_d == 0 {
            return Err(AkitaError::InvalidSetup(
                "generated fold step has zero ring dimension".to_string(),
            ));
        }
        // Level-local decomposition: the root inherits the config's full
        // decomposition; recursive levels carry balanced-digit `w` entries
        // so `log_commit_bound` collapses to the level's own `log_basis`.
        let level_decomp = if fold_level == 0 {
            DecompositionParams {
                log_basis: self.log_basis,
                ..root_decomp
            }
        } else {
            DecompositionParams {
                log_basis: self.log_basis,
                log_commit_bound: self.log_basis,
                log_open_bound: Some(
                    root_decomp
                        .log_open_bound
                        .unwrap_or(root_decomp.log_commit_bound),
                ),
            }
        };
        let (a_bucket, bd_bucket) = generated_level_buckets(
            sis_family,
            ring_d,
            self.log_basis,
            level_decomp.log_commit_bound,
            stage1.infinity_norm(),
            ring_subfield_norm_bound,
        )?;
        // Seed the params with shipped ranks and audited buckets; `col_len`
        // stays at the `params_only` placeholder `0` until `with_layout`
        // fills it. `collision_inf` is preserved by `with_layout`, so the
        // downstream audit sees the right bucket.
        let mut params = LevelParams::params_only(
            sis_family,
            ring_d,
            self.log_basis,
            self.n_a as usize,
            self.n_b as usize,
            self.n_d as usize,
            stage1,
        )
        .with_fold_challenge_shape(fold_shape);
        params.a_key =
            AjtaiKeyParams::new_unchecked(sis_family, self.n_a as usize, 0, a_bucket, ring_d);
        params.b_key =
            AjtaiKeyParams::new_unchecked(sis_family, self.n_b as usize, 0, bd_bucket, ring_d);
        params.d_key =
            AjtaiKeyParams::new_unchecked(sis_family, self.n_d as usize, 0, bd_bucket, ring_d);

        let layout = crate::level_layout_from_params(
            self.m_vars as usize,
            self.r_vars as usize,
            &params,
            level_decomp,
            current_w_len / ring_d,
        )?;
        let mut lp = params.with_layout(&layout);
        if fold_level == 0 {
            if let Some((num_t_vectors, field_bits)) = batched_root {
                lp = crate::scale_batched_root_layout(&lp, num_t_vectors, field_bits)?;
            }
        }
        Ok(lp)
    }
}

impl GeneratedScheduleTableEntry {
    /// Number of fold levels before the terminal direct step.
    pub fn num_fold_levels(&self) -> usize {
        self.steps
            .iter()
            .filter(|step| matches!(step, GeneratedStep::Fold(_)))
            .count()
    }

    /// Whether this entry uses the root-direct fast path (its first step is
    /// a `Direct`).
    pub fn is_root_direct(&self) -> bool {
        matches!(self.steps.first(), Some(GeneratedStep::Direct(_)))
    }

    /// The root fold step, when the entry starts with one.
    pub fn root_fold_step(&self) -> Option<&GeneratedFoldStep> {
        match self.steps.first() {
            Some(GeneratedStep::Fold(step)) => Some(step),
            _ => None,
        }
    }

    /// The terminal direct step, when the entry ends with one.
    pub fn terminal_direct(&self) -> Option<&GeneratedDirectStep> {
        match self.steps.last() {
            Some(GeneratedStep::Direct(step)) => Some(step),
            _ => None,
        }
    }

    /// The brute-forced fold step that carries the root commit layout: the
    /// root fold step for fold-root entries, or the root-direct commit for
    /// root-direct entries. `None` for an uncommittable root-direct entry.
    pub fn root_commit_step(&self) -> Option<&GeneratedFoldStep> {
        match self.steps.first() {
            Some(GeneratedStep::Fold(step)) => Some(step),
            Some(GeneratedStep::Direct(direct)) => direct.commit.as_ref(),
            None => None,
        }
    }

    /// Validate the structural invariants the runtime relies on: the entry
    /// is non-empty, ends in a `Direct`, and has no non-terminal `Direct`.
    ///
    /// # Errors
    ///
    /// Returns an error when any invariant is violated.
    pub fn validate(&self) -> Result<(), AkitaError> {
        if self.steps.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "generated schedule table entry must contain at least one step".to_string(),
            ));
        }
        let last = self.steps.len() - 1;
        for (idx, step) in self.steps.iter().enumerate() {
            if matches!(step, GeneratedStep::Direct(_)) && idx != last {
                return Err(AkitaError::InvalidSetup(
                    "generated direct step must be terminal".to_string(),
                ));
            }
        }
        if !matches!(self.steps[last], GeneratedStep::Direct(_)) {
            return Err(AkitaError::InvalidSetup(
                "generated schedule must end in a terminal direct step".to_string(),
            ));
        }
        Ok(())
    }
}
