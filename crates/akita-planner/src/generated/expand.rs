//! On-demand expansion of compact generated schedule steps into full
//! [`LevelParams`].
//!
//! The planner stores only the brute-forced parameters
//! (`ring_d, log_basis, position_index_bits, block_index_bits, num_live_blocks, n_a, n_b, n_d`) in
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
    GeneratedDirectStep, GeneratedFoldStep, GeneratedFoldStepWithSetupMetadata,
    GeneratedScheduleTableEntry, GeneratedSetupPrefixGroup, GeneratedStep,
};
use crate::schedule_params::{
    optimize_fold_challenge_shape, optimize_num_blocks_per_chunk_granule,
};
use crate::PlannerPolicy;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    fold_witness_digit_plan, min_secure_rank, num_digits_open, num_digits_s_commit,
    num_digits_setup_prefix_commit, rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm,
    FoldChallengeNorms, FoldWitnessLinfCapConfig, FoldWitnessNorms, SisTableKey,
};
use akita_types::{
    shared_d_digit_log_basis, AjtaiKeyParams, CommitmentRingDims, DecompositionParams, LevelParams,
    PolynomialGroupLayout, PrecommittedGroupParams, PrecommittedLevelParams, SetupContributionMode,
};

fn sis_key(
    policy: &PlannerPolicy,
    role: akita_types::SisMatrixRole,
    ring_dimension: u32,
    coeff_linf_bound: u128,
) -> SisTableKey {
    SisTableKey {
        policy: policy.sis_security_policy,
        table_digest: policy.sis_table_digest,
        modulus_profile: policy.sis_modulus_profile,
        role,
        ring_dimension,
        coeff_linf_bound,
    }
}

fn require_exact_rank(
    role: &str,
    key: SisTableKey,
    width: usize,
    stored_rank: usize,
) -> Result<(), AkitaError> {
    let expected = min_secure_rank(key, width as u64).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "no audited {role}-role rank for generated schedule \
             (policy={}, profile={:?}, d={}, coeff_linf_bound={}, width={width})",
            key.policy.name(),
            key.modulus_profile,
            key.ring_dimension,
            key.coeff_linf_bound
        ))
    })?;
    if stored_rank != expected {
        return Err(AkitaError::InvalidSetup(format!(
            "generated schedule {role}-rank mismatch: stored n_{role} = {stored_rank}, recomputed n_{role} = {expected}"
        )));
    }
    Ok(())
}

impl GeneratedSetupPrefixGroup {
    fn expand_to_precommitted_group(
        self,
        policy: &PlannerPolicy,
        ring_challenge_cfg: &SparseChallengeConfig,
        fold_shape: TensorChallengeShape,
        log_basis: u32,
    ) -> Result<PrecommittedLevelParams, AkitaError> {
        let d = policy.ring_dimension;
        let sis_modulus_profile = policy.sis_modulus_profile;
        let sis_policy = policy.sis_security_policy;
        let num_live_ring_elements_per_claim = self.num_live_ring_elements_per_claim as usize;
        let num_positions_per_block = self.num_positions_per_block as usize;
        let num_live_blocks = self.num_live_blocks as usize;
        let num_blocks_per_chunk_granule = self.num_blocks_per_chunk_granule as usize;
        let fold_shape = optimize_fold_challenge_shape(fold_shape, num_live_blocks)?;
        let n_prefix = num_live_ring_elements_per_claim
            .checked_mul(d)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("generated setup-prefix length overflow".into())
            })?;
        if n_prefix == 0 || !n_prefix.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "generated setup-prefix length must be a power of two".into(),
            ));
        }
        let prefix_num_vars = n_prefix.trailing_zeros() as usize;
        let layout = PrecommittedGroupParams {
            group: PolynomialGroupLayout::singleton(prefix_num_vars),
            num_live_ring_elements_per_claim,
            num_positions_per_block,
            num_live_blocks,
            num_blocks_per_chunk_granule,
            fold_challenge_shape: self.fold_challenge_shape,
            log_basis,
            n_a: self.n_a as usize,
            conservative_n_b: self.n_b as usize,
        };
        let decomp = DecompositionParams {
            log_basis,
            ..policy.decomposition
        };
        let num_digits_commit = num_digits_setup_prefix_commit(decomp);
        let num_digits_open_val = num_digits_open(decomp);
        let no_layout = |role: &str| {
            AkitaError::InvalidSetup(format!(
                "no audited setup-prefix {role}-role layout for generated schedule \
                 (profile={sis_modulus_profile:?}, d={d}, log_basis={log_basis})"
            ))
        };
        layout.validate_root_geometry(d)?;
        if fold_shape != self.fold_challenge_shape {
            return Err(AkitaError::InvalidSetup(
                "generated setup-prefix challenge shape mismatch".into(),
            ));
        }
        let inner_width = decomposed_s_block_ring_count(num_positions_per_block, num_digits_commit)
            .ok_or_else(|| no_layout("A"))?;
        let a_bucket = rounded_up_role_a_inf_norm(
            sis_policy,
            sis_modulus_profile,
            d,
            decomp,
            ring_challenge_cfg,
            fold_shape,
            false,
            0,
            policy.ring_subfield_norm_bound,
            num_live_blocks,
            1,
            inner_width as u64,
        )
        .ok_or_else(|| no_layout("A"))?;
        require_exact_rank(
            "setup-prefix a",
            sis_key(policy, akita_types::SisMatrixRole::A, d as u32, a_bucket),
            inner_width,
            self.n_a as usize,
        )?;
        let b_bucket = rounded_up_collision_inf_norm(
            sis_policy,
            sis_modulus_profile,
            akita_types::SisMatrixRole::B,
            d,
            log_basis,
        )
        .ok_or_else(|| no_layout("B"))?;
        let outer_width =
            decomposed_t_ring_count(self.n_a as usize, num_digits_open_val, num_live_blocks, 1)
                .ok_or_else(|| no_layout("B"))?;
        require_exact_rank(
            "setup-prefix b",
            sis_key(policy, akita_types::SisMatrixRole::B, d as u32, b_bucket),
            outer_width,
            self.n_b as usize,
        )?;
        let a_key = AjtaiKeyParams::try_new(
            sis_policy,
            policy.sis_table_digest,
            sis_modulus_profile,
            akita_types::SisMatrixRole::A,
            self.n_a as usize,
            inner_width,
            a_bucket,
            d,
        )?;
        let b_key = AjtaiKeyParams::try_new(
            sis_policy,
            policy.sis_table_digest,
            sis_modulus_profile,
            akita_types::SisMatrixRole::B,
            self.n_b as usize,
            outer_width,
            b_bucket,
            d,
        )?;
        let fold_linf_cap_config = FoldWitnessLinfCapConfig::for_fold_level(
            ring_challenge_cfg,
            fold_shape,
            d,
            inner_width,
        )?;
        let challenge = FoldChallengeNorms {
            infinity_norm: fold_shape.effective_infinity_norm(ring_challenge_cfg) as u128,
            l1_norm: fold_shape.effective_l1_mass(ring_challenge_cfg) as u128,
        };
        let (num_digits_fold_one, _) = fold_witness_digit_plan(
            num_live_blocks,
            1,
            policy.decomposition.field_bits(),
            log_basis,
            challenge,
            FoldWitnessNorms::new(log_basis, d, 1, false),
            &fold_linf_cap_config,
        )?;
        Ok(PrecommittedLevelParams {
            layout,
            a_key,
            b_key,
            num_digits_commit,
            num_digits_open: num_digits_open_val,
            num_digits_fold_one,
        })
    }
}

impl GeneratedFoldStep {
    /// Expand this compact fold step into the full committed
    /// [`LevelParams`] for its position in the schedule.
    ///
    /// `fold_level` is `0` at the root and `>0` at recursive levels; it
    /// selects the level-local decomposition (root inherits the config
    /// decomposition; recursive levels collapse `log_commit_bound` to the
    /// level's own `log_basis`). `current_w_len` is the witness length in
    /// field elements entering this level, used to size `num_positions_per_block`.
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
        self.expand_to_level_params_with_setup(
            policy,
            ring_challenge_config,
            fold_level,
            current_w_len,
            fold_shape,
            num_claims,
            None,
            SetupContributionMode::Direct,
            None,
        )
    }

    /// Expand a root-direct commit payload (`GeneratedDirectStep::commit`).
    ///
    /// Root-direct commits ship the raw polynomial unchunked, matching
    /// `compute_root_direct_level_params`.
    pub fn expand_to_root_direct_commit_params(
        &self,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
        current_w_len: usize,
        fold_shape: TensorChallengeShape,
        num_claims: usize,
    ) -> Result<LevelParams, AkitaError> {
        self.expand_to_level_params_with_setup(
            policy,
            ring_challenge_config,
            0,
            current_w_len,
            fold_shape,
            num_claims,
            None,
            SetupContributionMode::Direct,
            Some(1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_to_level_params_with_setup(
        &self,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
        fold_level: usize,
        current_w_len: usize,
        fold_shape: TensorChallengeShape,
        num_claims: usize,
        setup_prefix_group: Option<GeneratedSetupPrefixGroup>,
        setup_contribution_mode: SetupContributionMode,
        chunk_count_override: Option<usize>,
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
        let sis_modulus_profile = policy.sis_modulus_profile;
        let sis_policy = policy.sis_security_policy;

        // Digit-innermost geometry keeps `M = 2^position_index_bits` at every level
        // and carries exact live `B = ceil(N / M)` separately from its Boolean domain.
        let position_index_bits = self.position_index_bits as usize;
        let block_index_bits = self.block_index_bits as usize;
        let num_positions_per_block =
            1usize
                .checked_shl(position_index_bits as u32)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "generated schedule 2^position_index_bits overflows usize".to_string(),
                    )
                })?;
        let num_live_blocks = self.num_live_blocks as usize;
        if num_live_blocks == 0
            || num_live_blocks
                .checked_next_power_of_two()
                .map(|domain| domain.trailing_zeros() as usize)
                != Some(block_index_bits)
        {
            return Err(AkitaError::InvalidSetup(
                "generated schedule exact live block count disagrees with block_index_bits"
                    .to_string(),
            ));
        }
        if current_w_len == 0 || (!is_root && !current_w_len.is_multiple_of(ring_d)) {
            return Err(AkitaError::InvalidSetup(
                "witness length is not divisible by the ring dimension".to_string(),
            ));
        }
        // Root-direct inputs may be shorter than one ring and are zero-padded
        // inside that ring. Recursive witnesses are ring-aligned by contract.
        let num_live_ring_elements_per_claim = if is_root {
            current_w_len.div_ceil(ring_d)
        } else {
            current_w_len / ring_d
        };
        let derived_num_live_blocks =
            num_live_ring_elements_per_claim.div_ceil(num_positions_per_block);
        if derived_num_live_blocks != num_live_blocks {
            return Err(AkitaError::InvalidSetup(format!(
                "generated schedule num_live_blocks={} does not match ceil(N={num_live_ring_elements_per_claim} / M={num_positions_per_block})={derived_num_live_blocks}",
                num_live_blocks,
            )));
        }
        let fold_shape = optimize_fold_challenge_shape(fold_shape, num_live_blocks)?;
        let num_chunks = chunk_count_override.unwrap_or_else(|| policy.chunks_at_level(fold_level));
        let num_blocks_per_chunk_granule =
            optimize_num_blocks_per_chunk_granule(num_live_blocks, num_chunks, fold_shape)?;

        // Per-role rounded-up collision buckets + committed widths, via the
        // `akita_types::sis` primitives. The B/D widths carry the `num_claims`
        // batch factor (the root commits `num_claims` polynomials); `n_a` is the
        // A-matrix row count. Unlike the planner DP, expansion audits the
        // *shipped* ranks against these (norm, width) via `try_new`.
        let no_layout = |role: &str| {
            AkitaError::InvalidSetup(format!(
                "no audited {role}-role layout for generated schedule \
                 (profile={sis_modulus_profile:?}, d={ring_d}, log_basis={log_basis})"
            ))
        };
        let decomp = DecompositionParams {
            log_basis,
            ..policy.decomposition
        };
        let ring_challenge_cfg = ring_challenge_config(ring_d)?;
        let num_digits_commit = num_digits_s_commit(decomp, is_root);
        let num_digits_open_val = num_digits_open(decomp);

        let inner_width = decomposed_s_block_ring_count(num_positions_per_block, num_digits_commit)
            .ok_or_else(|| no_layout("A"))?;
        let a_bucket = rounded_up_role_a_inf_norm(
            sis_policy,
            sis_modulus_profile,
            ring_d,
            decomp,
            &ring_challenge_cfg,
            fold_shape,
            is_root,
            policy.onehot_chunk_size,
            policy.ring_subfield_norm_bound,
            num_live_blocks,
            num_claims,
            inner_width as u64,
        )
        .ok_or_else(|| no_layout("A"))?;
        require_exact_rank(
            "a",
            sis_key(
                policy,
                akita_types::SisMatrixRole::A,
                ring_d as u32,
                a_bucket,
            ),
            inner_width,
            self.n_a as usize,
        )?;

        let b_bucket = rounded_up_collision_inf_norm(
            sis_policy,
            sis_modulus_profile,
            akita_types::SisMatrixRole::B,
            ring_d,
            log_basis,
        )
        .ok_or_else(|| no_layout("B"))?;
        let outer_width = decomposed_t_ring_count(
            self.n_a as usize,
            num_digits_open_val,
            num_live_blocks,
            num_claims,
        )
        .ok_or_else(|| no_layout("B"))?;

        let d_bucket = rounded_up_collision_inf_norm(
            sis_policy,
            sis_modulus_profile,
            akita_types::SisMatrixRole::D,
            ring_d,
            log_basis,
        )
        .ok_or_else(|| no_layout("D"))?;
        let main_d_width =
            decomposed_w_ring_count(num_digits_open_val, num_live_blocks, num_claims)
                .ok_or_else(|| no_layout("D"))?;
        let setup_prefix = if let Some(group) = setup_prefix_group {
            let commitment_params = group.expand_to_precommitted_group(
                policy,
                &ring_challenge_cfg,
                fold_shape,
                log_basis,
            )?;
            let n_prefix = 1usize
                .checked_shl(commitment_params.layout.group.num_vars() as u32)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("generated setup-prefix length overflow".into())
                })?;
            if group.natural_len as usize > n_prefix {
                return Err(AkitaError::InvalidSetup(
                    "generated setup-prefix natural length exceeds commitment domain".into(),
                ));
            }
            Some(akita_types::setup_prefix_slot_id(
                policy.ring_dimension,
                group.natural_len as usize,
                commitment_params,
            ))
        } else {
            None
        };
        let precommitted_groups = Vec::new();
        let precommitted_d_width = setup_prefix
            .as_ref()
            .map(|prefix| prefix.commitment_params.d_segment_width())
            .transpose()?
            .unwrap_or(0);
        let d_matrix_width = main_d_width
            .checked_add(precommitted_d_width)
            .ok_or_else(|| AkitaError::InvalidSetup("generated D width overflow".into()))?;

        let num_digits_open = num_digits_open_val;

        // A one-hot root (`log_commit_bound == 1`) commits a sparse witness;
        // recursive and dense levels are dense (`onehot_chunk_size = 0`).
        let onehot_chunk_size = if is_root && policy.decomposition.log_commit_bound == 1 {
            policy.onehot_chunk_size
        } else {
            0
        };

        // Size the committed B matrix at the full outer width.
        require_exact_rank(
            "b",
            sis_key(
                policy,
                akita_types::SisMatrixRole::B,
                ring_d as u32,
                b_bucket,
            ),
            outer_width,
            self.n_b as usize,
        )?;
        require_exact_rank(
            "d",
            sis_key(
                policy,
                akita_types::SisMatrixRole::D,
                ring_d as u32,
                d_bucket,
            ),
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
                sis_policy,
                policy.sis_table_digest,
                sis_modulus_profile,
                akita_types::SisMatrixRole::A,
                self.n_a as usize,
                inner_width,
                a_bucket,
                ring_d,
            )?,
            b_key: AjtaiKeyParams::try_new(
                sis_policy,
                policy.sis_table_digest,
                sis_modulus_profile,
                akita_types::SisMatrixRole::B,
                self.n_b as usize,
                outer_width,
                b_bucket,
                ring_d,
            )?,
            d_key: AjtaiKeyParams::try_new(
                sis_policy,
                policy.sis_table_digest,
                sis_modulus_profile,
                akita_types::SisMatrixRole::D,
                self.n_d as usize,
                d_matrix_width,
                d_bucket,
                ring_d,
            )?,
            num_live_ring_elements_per_claim,
            num_live_blocks,
            num_positions_per_block,
            num_blocks_per_chunk_granule,
            fold_challenge_config: ring_challenge_cfg,
            fold_challenge_shape: fold_shape,
            num_digits_commit,
            num_digits_open,
            onehot_chunk_size,
            fold_linf_cap_config: akita_types::sis::FoldWitnessLinfCapConfig::worst_case_beta_only(
            ),
            num_digits_fold_one: 1,
            field_bits_hint: 0,
            cached_num_digits_block_claims: 0,
            cached_num_digits_fold_value: 1,
            // The chunk layout depends on the step's role (fold vs root-direct
            // commit), which the caller knows; default here and let the caller
            // (`schedule_from_entry`) stamp the per-level value for fold steps so
            // a root-direct commit stays single-chunk.
            witness_chunk: akita_types::ChunkedWitnessCfg::default(),
            precommitted_groups,
            setup_prefix,
            role_dims: CommitmentRingDims::uniform(ring_d),
            setup_contribution_mode,
        };
        let mut params =
            params.with_fold_linf_cap_config(policy.decomposition.field_bits(), num_claims)?;
        params.stamp_role_dims_from_keys();
        Ok(params)
    }

    /// Expand a compact root step for a multi-group-root schedule.
    ///
    /// The main group's A/B layouts are claim-scaled by `main_num_polys`, while
    /// the shared D matrix has one segment for the main group plus the frozen
    /// precommitted group segments. This intentionally differs from scalar
    /// batched roots, whose D width is scaled by the total polynomial count.
    pub fn expand_to_multi_group_root_level_params(
        &self,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
        fold_shape: TensorChallengeShape,
        main_num_polys: usize,
        precommitted_groups: Vec<PrecommittedLevelParams>,
        precommitted_d_width: usize,
    ) -> Result<LevelParams, AkitaError> {
        self.expand_to_multi_group_root_level_params_with_setup(
            policy,
            ring_challenge_config,
            fold_shape,
            main_num_polys,
            precommitted_groups,
            precommitted_d_width,
            SetupContributionMode::Direct,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_to_multi_group_root_level_params_with_setup(
        &self,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
        fold_shape: TensorChallengeShape,
        main_num_polys: usize,
        precommitted_groups: Vec<PrecommittedLevelParams>,
        precommitted_d_width: usize,
        setup_contribution_mode: SetupContributionMode,
    ) -> Result<LevelParams, AkitaError> {
        let ring_d = self.ring_d as usize;
        if ring_d == 0 || ring_d != policy.ring_dimension {
            return Err(AkitaError::InvalidSetup(format!(
                "generated multi-group root ring dimension {ring_d} does not match policy D={}",
                policy.ring_dimension
            )));
        }
        if precommitted_groups.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "generated multi-group root requires precommitted groups".to_string(),
            ));
        }

        let log_basis = self.log_basis;
        let sis_modulus_profile = policy.sis_modulus_profile;
        let sis_policy = policy.sis_security_policy;
        let position_index_bits = self.position_index_bits as usize;
        let block_index_bits = self.block_index_bits as usize;
        let num_live_blocks = self.num_live_blocks as usize;
        if num_live_blocks == 0
            || num_live_blocks
                .checked_next_power_of_two()
                .map(|domain| domain.trailing_zeros() as usize)
                != Some(block_index_bits)
        {
            return Err(AkitaError::InvalidSetup(
                "generated multi-group exact live block count disagrees with block_index_bits"
                    .to_string(),
            ));
        }
        let num_positions_per_block =
            1usize
                .checked_shl(position_index_bits as u32)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "generated multi-group root 2^position_index_bits overflows usize"
                            .to_string(),
                    )
                })?;
        let fold_shape = optimize_fold_challenge_shape(fold_shape, num_live_blocks)?;
        let num_blocks_per_chunk_granule = optimize_num_blocks_per_chunk_granule(
            num_live_blocks,
            policy.chunks_at_level(0),
            fold_shape,
        )?;

        let no_layout = |role: &str| {
            AkitaError::InvalidSetup(format!(
                "no audited {role}-role layout for generated multi-group root \
                 (profile={sis_modulus_profile:?}, d={ring_d}, log_basis={log_basis})"
            ))
        };
        let decomp = DecompositionParams {
            log_basis,
            ..policy.decomposition
        };
        let ring_challenge_cfg = ring_challenge_config(ring_d)?;
        let num_digits_commit = num_digits_s_commit(decomp, true);
        let num_digits_open_val = num_digits_open(decomp);

        let inner_width = decomposed_s_block_ring_count(num_positions_per_block, num_digits_commit)
            .ok_or_else(|| no_layout("A"))?;
        let a_bucket = rounded_up_role_a_inf_norm(
            sis_policy,
            sis_modulus_profile,
            ring_d,
            decomp,
            &ring_challenge_cfg,
            fold_shape,
            true,
            policy.onehot_chunk_size,
            policy.ring_subfield_norm_bound,
            num_live_blocks,
            main_num_polys,
            inner_width as u64,
        )
        .ok_or_else(|| no_layout("A"))?;
        require_exact_rank(
            "a",
            sis_key(
                policy,
                akita_types::SisMatrixRole::A,
                ring_d as u32,
                a_bucket,
            ),
            inner_width,
            self.n_a as usize,
        )?;

        let b_bucket = rounded_up_collision_inf_norm(
            sis_policy,
            sis_modulus_profile,
            akita_types::SisMatrixRole::B,
            ring_d,
            log_basis,
        )
        .ok_or_else(|| no_layout("B"))?;
        let outer_width = decomposed_t_ring_count(
            self.n_a as usize,
            num_digits_open_val,
            num_live_blocks,
            main_num_polys,
        )
        .ok_or_else(|| no_layout("B"))?;

        let main_d_width =
            decomposed_w_ring_count(num_digits_open_val, num_live_blocks, main_num_polys)
                .ok_or_else(|| no_layout("D"))?;
        let d_matrix_width = main_d_width
            .checked_add(precommitted_d_width)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("generated multi-group D width overflow".into())
            })?;
        let d_log_basis = shared_d_digit_log_basis(log_basis, &precommitted_groups);
        let d_bucket = rounded_up_collision_inf_norm(
            sis_policy,
            sis_modulus_profile,
            akita_types::SisMatrixRole::D,
            ring_d,
            d_log_basis,
        )
        .ok_or_else(|| no_layout("D"))?;

        let onehot_chunk_size = if policy.decomposition.log_commit_bound == 1 {
            policy.onehot_chunk_size
        } else {
            0
        };

        let params = LevelParams {
            ring_dimension: ring_d,
            log_basis,
            a_key: AjtaiKeyParams::try_new(
                sis_policy,
                policy.sis_table_digest,
                sis_modulus_profile,
                akita_types::SisMatrixRole::A,
                self.n_a as usize,
                inner_width,
                a_bucket,
                ring_d,
            )?,
            b_key: AjtaiKeyParams::try_new(
                sis_policy,
                policy.sis_table_digest,
                sis_modulus_profile,
                akita_types::SisMatrixRole::B,
                self.n_b as usize,
                outer_width,
                b_bucket,
                ring_d,
            )?,
            d_key: AjtaiKeyParams::try_new(
                sis_policy,
                policy.sis_table_digest,
                sis_modulus_profile,
                akita_types::SisMatrixRole::D,
                self.n_d as usize,
                d_matrix_width,
                d_bucket,
                ring_d,
            )?,
            num_live_ring_elements_per_claim: num_live_blocks
                .checked_mul(num_positions_per_block)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("generated root source length overflow".to_string())
                })?,
            num_live_blocks,
            num_positions_per_block,
            num_blocks_per_chunk_granule,
            fold_challenge_config: ring_challenge_cfg,
            fold_challenge_shape: fold_shape,
            num_digits_commit,
            num_digits_open: num_digits_open_val,
            onehot_chunk_size,
            fold_linf_cap_config: akita_types::sis::FoldWitnessLinfCapConfig::worst_case_beta_only(
            ),
            num_digits_fold_one: 1,
            field_bits_hint: 0,
            cached_num_digits_block_claims: 0,
            cached_num_digits_fold_value: 1,
            witness_chunk: akita_types::ChunkedWitnessCfg::default(),
            precommitted_groups,
            setup_prefix: None,
            role_dims: CommitmentRingDims::uniform(ring_d),
            setup_contribution_mode,
        };
        let mut params =
            params.with_fold_linf_cap_config(policy.decomposition.field_bits(), main_num_polys)?;
        params.stamp_role_dims_from_keys();
        Ok(params)
    }
}

impl GeneratedFoldStepWithSetupMetadata {
    #[allow(clippy::too_many_arguments)]
    pub fn expand_to_level_params(
        &self,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
        fold_level: usize,
        current_w_len: usize,
        fold_shape: TensorChallengeShape,
        num_claims: usize,
    ) -> Result<LevelParams, AkitaError> {
        self.fold.expand_to_level_params_with_setup(
            policy,
            ring_challenge_config,
            fold_level,
            current_w_len,
            fold_shape,
            num_claims,
            self.setup_prefix_group,
            self.setup_contribution_mode,
            None,
        )
    }

    pub fn expand_to_multi_group_root_level_params(
        &self,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
        fold_shape: TensorChallengeShape,
        main_num_polys: usize,
        precommitted_groups: Vec<PrecommittedLevelParams>,
        precommitted_d_width: usize,
    ) -> Result<LevelParams, AkitaError> {
        self.fold
            .expand_to_multi_group_root_level_params_with_setup(
                policy,
                ring_challenge_config,
                fold_shape,
                main_num_polys,
                precommitted_groups,
                precommitted_d_width,
                self.setup_contribution_mode,
            )
    }
}

impl GeneratedScheduleTableEntry {
    /// Number of fold levels before the terminal direct step.
    pub fn num_fold_levels(&self) -> usize {
        self.steps
            .iter()
            .filter_map(GeneratedStep::fold_step)
            .count()
    }

    /// Whether this entry uses the root-direct fast path (its first step is
    /// a `Direct`).
    pub fn is_root_direct(&self) -> bool {
        matches!(self.steps.first(), Some(GeneratedStep::Direct(_)))
    }

    /// The root fold step, when the entry starts with one.
    pub fn root_fold_step(&self) -> Option<&GeneratedFoldStep> {
        self.steps.first().and_then(GeneratedStep::fold_step)
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
            Some(GeneratedStep::FoldWithSetupMetadata(step)) => Some(&step.fold),
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
        validate_generated_steps(self.steps)
    }
}

fn validate_generated_steps(steps: &[GeneratedStep]) -> Result<(), AkitaError> {
    if steps.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "generated schedule table entry must contain at least one step".to_string(),
        ));
    }
    let last = steps.len() - 1;
    for (idx, step) in steps.iter().enumerate() {
        if matches!(step, GeneratedStep::Direct(_)) && idx != last {
            return Err(AkitaError::InvalidSetup(
                "generated direct step must be terminal".to_string(),
            ));
        }
    }
    if !matches!(steps[last], GeneratedStep::Direct(_)) {
        return Err(AkitaError::InvalidSetup(
            "generated schedule must end in a terminal direct step".to_string(),
        ));
    }
    Ok(())
}
