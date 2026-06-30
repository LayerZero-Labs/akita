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

use crate::generated::{
    GeneratedDirectStep, GeneratedFoldStep, GeneratedScheduleTableEntry, GeneratedStep,
};
use crate::PlannerPolicy;
use akita_types::sis::{
    committed_fold_a_role_rank, decomposed_s_block_ring_count, decomposed_t_ring_count,
    decomposed_w_ring_count, min_secure_rank, num_digits_open, num_digits_s_commit,
    rounded_up_collision_norm_t, rounded_up_collision_norm_tiered_commitment,
    rounded_up_collision_norm_w, SisModulusFamily,
};
use akita_types::{AjtaiKeyParams, DecompositionParams, LevelParams};

fn require_exact_rank(
    role: &str,
    sis_family: SisModulusFamily,
    ring_d: usize,
    collision_l2_sq: u128,
    width: usize,
    stored_rank: usize,
) -> Result<(), AkitaError> {
    let expected = min_secure_rank(sis_family, ring_d as u32, collision_l2_sq, width as u64)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "no audited {role}-role rank for generated schedule \
                 (family={sis_family:?}, d={ring_d}, collision_l2_sq={collision_l2_sq}, width={width})"
            ))
        })?;
    if stored_rank != expected {
        return Err(AkitaError::InvalidSetup(format!(
            "generated schedule {role}-rank mismatch: stored n_{role} = {stored_rank}, recomputed n_{role} = {expected}"
        )));
    }
    Ok(())
}

impl GeneratedFoldStep {
    /// Expand this compact fold step into the full committed
    /// [`LevelParams`] for its position in the schedule.
    ///
    /// `fold_level` is `0` at the root and `>0` at recursive levels; it
    /// selects the level-local decomposition (root inherits the config
    /// decomposition; recursive levels collapse `log_commit_bound` to the
    /// level's own `log_basis`). `current_w_len` is the witness length in
    /// field elements entering this level, used to size `block_len`.
    ///
    /// `num_claims` is the batch factor folded directly into the outer (B)
    /// and prover (D) matrix widths — the root commits `num_claims`
    /// polynomials. `num_claims == 1` is the singleton root (and every
    /// recursive level); a batched root passes the lookup key's
    /// `num_polynomials`. There is no separate per-claim-then-scale pass: the
    /// width helpers receive `num_claims` as the `t_vectors` factor.
    ///
    /// The A/B/D widths and audited collision buckets are derived by the
    /// shared `ajtai_a_width_bucket` / `ajtai_b_width_bucket` /
    /// `ajtai_d_width_bucket` helpers — the *same* functions the planner DP
    /// (`compute_ajtai_key_params_*`) uses — so the bucket the DP sized
    /// `(n_a, n_b, n_d)` against can never drift from the bucket reconstructed
    /// here. The only difference is the rank source: the DP computes the tight
    /// SIS-secure minimum, while expansion replays the stored rank and audits
    /// it against the same width + bucket via the fallible
    /// [`AjtaiKeyParams::try_new`].
    ///
    /// The same method expands a root-direct commit step (the
    /// [`GeneratedDirectStep::commit`] payload): a root-direct commit is a
    /// `fold_level == 0` expansion.
    ///
    /// # Errors
    ///
    /// Returns an error when the stored ring dimension disagrees with the
    /// policy, bucket/width resolution fails, or a shipped rank fails its SIS
    /// audit against the (batched) width.
    pub fn expand_to_level_params(
        &self,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
        fold_level: usize,
        current_w_len: usize,
        fold_shape: TensorChallengeShape,
        num_claims: usize,
    ) -> Result<LevelParams, AkitaError> {
        let ring_d = self.ring_d as usize;
        if ring_d == 0 || ring_d != policy.ring_dimension {
            return Err(AkitaError::InvalidSetup(format!(
                "generated fold step ring dimension {ring_d} does not match policy D={}",
                policy.ring_dimension
            )));
        }
        let is_root = fold_level == 0;
        let log_basis = self.log_basis;
        let sis_family = policy.sis_family;

        // Block geometry: the root spans `2^m_vars` ring elements per block;
        // recursive levels pack `ceil(num_ring / num_blocks)` instead.
        let m_vars = self.m_vars as usize;
        let r_vars = self.r_vars as usize;
        let num_blocks = 1usize.checked_shl(r_vars as u32).ok_or_else(|| {
            AkitaError::InvalidSetup("generated schedule 2^r_vars overflows usize".to_string())
        })?;
        let block_len = if is_root {
            1usize.checked_shl(m_vars as u32).ok_or_else(|| {
                AkitaError::InvalidSetup("generated schedule 2^m_vars overflows usize".to_string())
            })?
        } else {
            (current_w_len / ring_d).div_ceil(num_blocks)
        };

        // Per-role rounded-up collision buckets + committed widths, via the
        // `akita_types::sis` primitives. The B/D widths carry the `num_claims`
        // batch factor (the root commits `num_claims` polynomials); the stored
        // `n_a` sizes the B-role width. Unlike the planner DP, expansion audits
        // the *shipped* ranks against these (norm, width) via `try_new`.
        let no_layout = |role: &str| {
            AkitaError::InvalidSetup(format!(
                "no audited {role}-role layout for generated schedule \
                 (family={sis_family:?}, d={ring_d}, log_basis={log_basis})"
            ))
        };
        let decomp = DecompositionParams {
            log_basis,
            ..policy.decomposition
        };
        let ring_challenge_cfg = ring_challenge_config(ring_d)?;
        let num_digits_commit = num_digits_s_commit(decomp, is_root);
        let num_digits_open_val = num_digits_open(decomp);

        let inner_width = decomposed_s_block_ring_count(block_len, num_digits_commit)
            .ok_or_else(|| no_layout("A"))?;
        let (a_bucket, expected_n_a) = committed_fold_a_role_rank(
            sis_family,
            ring_d,
            decomp,
            &ring_challenge_cfg,
            fold_shape,
            is_root,
            policy.onehot_chunk_size,
            policy.ring_subfield_norm_bound,
            r_vars,
            num_claims,
            inner_width as u64,
        )
        .ok_or_else(|| no_layout("A"))?;
        if self.n_a as usize != expected_n_a {
            return Err(AkitaError::InvalidSetup(format!(
                "generated schedule A-rank mismatch: stored n_a = {}, recomputed n_a = {expected_n_a}",
                self.n_a
            )));
        }
        require_exact_rank(
            "a",
            sis_family,
            ring_d,
            a_bucket,
            inner_width,
            self.n_a as usize,
        )?;

        let b_bucket = rounded_up_collision_norm_t(sis_family, ring_d, log_basis)
            .ok_or_else(|| no_layout("B"))?;
        let outer_width = decomposed_t_ring_count(
            self.n_a as usize,
            num_digits_open_val,
            num_blocks,
            num_claims,
        )
        .ok_or_else(|| no_layout("B"))?;

        let d_bucket = rounded_up_collision_norm_w(sis_family, ring_d, log_basis)
            .ok_or_else(|| no_layout("D"))?;
        let d_matrix_width = decomposed_w_ring_count(num_digits_open_val, num_blocks, num_claims)
            .ok_or_else(|| no_layout("D"))?;

        let num_digits_open = num_digits_open_val;

        // A one-hot root (`log_commit_bound == 1`) commits a sparse witness;
        // recursive and dense levels are dense (`onehot_chunk_size = 0`).
        let onehot_chunk_size = if is_root && policy.decomposition.log_commit_bound == 1 {
            policy.onehot_chunk_size
        } else {
            0
        };

        // Tiered second tier (`B'`/`F`): the compact entry stores the committed
        // layout directly — `n_b` is the shrunk `B'` rank, `tier_split` is the
        // split factor, and `n_f` is the second-tier `F` rank. The `B'` width is
        // the full outer width divided by the split, and `F` commits
        // `tier_split · n_b · num_digits_open` digit columns at the same
        // digit-range bucket as `B`/`D`. A single-tier step stores `None`/`None`
        // and keeps the full `B` width. (No `apply_tiering` re-search: the table
        // is the frozen snapshot; the DP path owns `apply_tiering` for misses.)
        let (b_width, tier_split, f_key) = match (self.tier_split, self.n_f) {
            (None, None) => (outer_width, 1usize, None),
            (Some(f), Some(n_f)) => {
                let f = f as usize;
                if f <= 1 {
                    return Err(AkitaError::InvalidSetup(
                        "generated tiered step has tier_split <= 1".to_string(),
                    ));
                }
                if !policy.tiered {
                    return Err(AkitaError::InvalidSetup(
                        "generated tiered step is not allowed by the planner policy".to_string(),
                    ));
                }
                if outer_width == 0 || !outer_width.is_multiple_of(f) {
                    return Err(AkitaError::InvalidSetup(
                        "generated tiered B' width does not divide the full outer width"
                            .to_string(),
                    ));
                }
                let b_small_width = outer_width / f;
                let f_width = f
                    .checked_mul(self.n_b as usize)
                    .and_then(|w| w.checked_mul(num_digits_open_val))
                    .ok_or_else(|| no_layout("F"))?;
                let f_bucket =
                    rounded_up_collision_norm_tiered_commitment(sis_family, ring_d, log_basis)
                        .ok_or_else(|| no_layout("F"))?;
                require_exact_rank("f", sis_family, ring_d, f_bucket, f_width, n_f as usize)?;
                let f_key =
                    AjtaiKeyParams::try_new(sis_family, n_f as usize, f_width, f_bucket, ring_d)?;
                (b_small_width, f, Some(f_key))
            }
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "generated tiered step must set both tier_split and n_f, or neither"
                        .to_string(),
                ));
            }
        };
        require_exact_rank(
            "b",
            sis_family,
            ring_d,
            b_bucket,
            b_width,
            self.n_b as usize,
        )?;
        require_exact_rank(
            "d",
            sis_family,
            ring_d,
            d_bucket,
            d_matrix_width,
            self.n_d as usize,
        )?;

        // Audit each shipped rank against its width + bucket as we build the
        // key (verifier-reachable, so the fallible `try_new` is used instead
        // of the panicking `new`).
        let params = LevelParams {
            ring_dimension: ring_d,
            log_basis,
            a_key: AjtaiKeyParams::try_new(
                sis_family,
                self.n_a as usize,
                inner_width,
                a_bucket,
                ring_d,
            )?,
            b_key: AjtaiKeyParams::try_new(
                sis_family,
                self.n_b as usize,
                b_width,
                b_bucket,
                ring_d,
            )?,
            d_key: AjtaiKeyParams::try_new(
                sis_family,
                self.n_d as usize,
                d_matrix_width,
                d_bucket,
                ring_d,
            )?,
            num_blocks,
            block_len,
            m_vars,
            r_vars,
            stage1_config: ring_challenge_cfg,
            fold_challenge_shape: fold_shape,
            num_digits_commit,
            num_digits_open,
            onehot_chunk_size,
            tier_split,
            f_key,
            fold_linf_cap_config: akita_types::sis::FoldWitnessLinfCapConfig::worst_case_beta_only(
            ),
            num_digits_fold_one: 1,
            field_bits_hint: 0,
            cached_num_digits_fold_claims: 0,
            cached_num_digits_fold_value: 1,
            precommitted_groups: Vec::new(),
        };
        params.with_fold_linf_cap_config(policy.decomposition.field_bits(), num_claims)
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
