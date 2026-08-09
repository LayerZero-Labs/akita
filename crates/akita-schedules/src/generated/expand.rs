//! On-demand expansion of compact generated schedule steps into full
//! [`CommittedGroupParams`].
//!
//! Generated rows store optimizer choices, including each exact fold digit
//! depth. Expansion derives commitment digit depths, matrix widths, collision
//! buckets, and minimum SIS-secure output ranks without rerunning honest fold
//! sizing.
//!
//! This is verifier-reachable (config resolves levels through it on the
//! replay path), so every fallible step returns [`AkitaError`] rather than
//! panicking.

use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;

use crate::generated::{
    GeneratedCommittedGroup, GeneratedFoldScheduleEntry, GeneratedOpenCommitMatrix,
    GeneratedPrecommittedProfile, GeneratedSetupPrefixInput, GeneratedTerminalFold,
};
use crate::PlannerPolicy;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    min_secure_rank, num_digits_inner, num_digits_open, num_digits_setup_prefix_commit,
    projected_role_ring_count, rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm,
    SisTableKey,
};
use akita_types::{
    shared_d_digit_log_basis, validate_role_dims, CommitmentRingDims, CommittedGroupParams,
    CommittedGroupProfile, DecompositionParams, InnerCommitMatrixParams, OpenCommitMatrixParams,
    OuterCommitMatrixParams, PolynomialGroupLayout, PrecommittedLevelParams,
    TerminalCommittedGroupParams,
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

fn secure_rank(role: &str, key: SisTableKey, width: usize) -> Result<usize, AkitaError> {
    min_secure_rank(key, width as u64).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "no audited {role}-role rank for generated schedule \
             (policy={}, profile={:?}, d={}, coeff_linf_bound={}, width={width})",
            key.policy.name(),
            key.modulus_profile,
            key.ring_dimension,
            key.coeff_linf_bound
        ))
    })
}

fn generated_count(value: u64, name: &str) -> Result<usize, AkitaError> {
    usize::try_from(value).map_err(|_| {
        AkitaError::InvalidSetup(format!("generated {name} does not fit the target platform"))
    })
}

impl GeneratedSetupPrefixInput {
    fn expand_to_precommitted_group(
        self,
        policy: &PlannerPolicy,
        ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
        log_basis_open: u32,
    ) -> Result<PrecommittedLevelParams, AkitaError> {
        super::validate_certified_bases(
            self.commitment.inner_commit_matrix.log_basis,
            self.commitment.outer_commit_matrix.log_basis,
            log_basis_open,
            policy,
            "generated setup-prefix group",
        )?;
        let dimensions = CommitmentRingDims {
            inner: self.commitment.inner_commit_matrix.ring_dimension as usize,
            outer: self.commitment.outer_commit_matrix.ring_dimension as usize,
            // A setup-prefix group is opened by its consuming fold's D matrix,
            // so only its persisted A/B dimensions are reconstructed here.
            opening: self.commitment.outer_commit_matrix.ring_dimension as usize,
        };
        validate_role_dims(dimensions)?;
        let d_a = dimensions.d_a();
        let d_b = dimensions.d_b();
        let ring_challenge_cfg = ring_challenge_config(d_a)?;
        let sis_modulus_profile = policy.sis_modulus_profile;
        let sis_policy = policy.sis_security_policy;
        let geometry = self.commitment.geometry;
        let num_live_ring_elements_per_claim = generated_count(
            geometry.live_ring_elements_per_claim,
            "live ring-element count",
        )?;
        let num_positions_per_block =
            generated_count(geometry.positions_per_block, "positions per block")?;
        let num_live_blocks = generated_count(geometry.live_blocks, "live block count")?;
        let n_prefix = num_live_ring_elements_per_claim
            .checked_mul(d_a)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("generated setup-prefix length overflow".into())
            })?;
        if n_prefix == 0 || !n_prefix.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "generated setup-prefix length must be a power of two".into(),
            ));
        }
        let prefix_num_vars = n_prefix.trailing_zeros() as usize;
        let inner_decomp = DecompositionParams {
            log_basis: self.commitment.inner_commit_matrix.log_basis,
            ..policy.decomposition
        };
        let outer_decomp = DecompositionParams {
            log_basis: self.commitment.outer_commit_matrix.log_basis,
            ..policy.decomposition
        };
        let open_decomp = DecompositionParams {
            log_basis: log_basis_open,
            ..policy.decomposition
        };
        let num_digits_inner = num_digits_setup_prefix_commit(inner_decomp);
        let num_digits_outer = num_digits_open(outer_decomp);
        let num_digits_open_val = num_digits_open(open_decomp);
        let no_layout = |role: &str| {
            AkitaError::InvalidSetup(format!(
                "no audited setup-prefix {role}-role layout for generated schedule \
                 (profile={sis_modulus_profile:?}, dims={dimensions:?}, inner={}, outer={}, open={})",
                self.commitment.inner_commit_matrix.log_basis,
                self.commitment.outer_commit_matrix.log_basis,
                log_basis_open
            ))
        };
        let inner_width = decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
            .ok_or_else(|| no_layout("A"))?;
        let num_digits_fold = usize::try_from(self.num_digits_fold).map_err(|_| {
            AkitaError::InvalidSetup(
                "generated setup-prefix fold digit depth does not fit the target platform".into(),
            )
        })?;
        if num_digits_fold == 0 {
            return Err(AkitaError::InvalidSetup(
                "generated setup-prefix fold digit depth must be nonzero".into(),
            ));
        }
        let a_bucket = rounded_up_role_a_inf_norm(
            sis_policy,
            policy.sis_table_digest,
            sis_modulus_profile,
            d_a,
            log_basis_open,
            &ring_challenge_cfg,
            num_digits_fold,
            policy.ring_subfield_norm_bound,
        )
        .ok_or_else(|| no_layout("A"))?;
        let n_a = secure_rank(
            "setup-prefix a",
            sis_key(
                policy,
                akita_types::SisMatrixRole::Inner,
                d_a as u32,
                a_bucket,
            ),
            inner_width,
        )?;
        let b_bucket = rounded_up_collision_inf_norm(
            sis_policy,
            sis_modulus_profile,
            akita_types::SisMatrixRole::Outer,
            d_b,
            log_basis_open,
        )
        .ok_or_else(|| no_layout("B"))?;
        let native_outer_width = decomposed_t_ring_count(n_a, num_digits_outer, num_live_blocks, 1)
            .ok_or_else(|| no_layout("B"))?;
        let outer_width = projected_role_ring_count(d_a, d_b, native_outer_width)
            .ok_or_else(|| no_layout("B"))?;
        let n_b = secure_rank(
            "setup-prefix b",
            sis_key(
                policy,
                akita_types::SisMatrixRole::Outer,
                d_b as u32,
                b_bucket,
            ),
            outer_width,
        )?;
        let inner_commit_matrix = InnerCommitMatrixParams::try_new(
            sis_policy,
            policy.sis_table_digest,
            sis_modulus_profile,
            n_a,
            inner_width,
            a_bucket,
            d_a,
        )?;
        let outer_commit_matrix = OuterCommitMatrixParams::try_new(
            sis_policy,
            policy.sis_table_digest,
            sis_modulus_profile,
            n_b,
            outer_width,
            b_bucket,
            d_b,
        )?;
        let layout = CommittedGroupProfile {
            version: CommittedGroupProfile::VERSION,
            group: PolynomialGroupLayout::singleton(prefix_num_vars),
            num_live_ring_elements_per_claim,
            num_positions_per_block,
            num_live_blocks,
            log_basis_inner: self.commitment.inner_commit_matrix.log_basis,
            num_digits_inner,
            inner_commit_matrix,
            log_basis_outer: self.commitment.outer_commit_matrix.log_basis,
            num_digits_outer,
            outer_commit_matrix,
        };
        layout.validate_root_geometry()?;
        Ok(PrecommittedLevelParams {
            layout,
            log_basis_open,
            fold_challenge_config: ring_challenge_cfg,
            num_digits_open: num_digits_open_val,
            num_digits_fold,
        })
    }
}

impl GeneratedPrecommittedProfile {
    /// Expand this compact generated standalone precommit descriptor into its
    /// canonical runtime profile.
    pub fn expand_to_committed_profile(
        self,
        policy: &PlannerPolicy,
    ) -> Result<CommittedGroupProfile, AkitaError> {
        self.group.validate()?;
        let geometry = self.commitment.geometry;
        let num_live_ring_elements_per_claim = generated_count(
            geometry.live_ring_elements_per_claim,
            "live ring-element count",
        )?;
        let num_positions_per_block =
            generated_count(geometry.positions_per_block, "positions per block")?;
        let num_live_blocks = generated_count(geometry.live_blocks, "live block count")?;
        let d_a = self.commitment.inner_commit_matrix.ring_dimension as usize;
        let d_b = self.commitment.outer_commit_matrix.ring_dimension as usize;
        validate_role_dims(CommitmentRingDims {
            inner: d_a,
            outer: d_b,
            opening: d_b,
        })?;
        if self.commitment.outer_commit_matrix.slice_count != 1 {
            return Err(AkitaError::InvalidSetup(
                "generated precommit B matrix must use one slice".to_string(),
            ));
        }
        let num_digits_inner = generated_count(self.num_digits_inner as u64, "inner digit depth")?;
        let num_digits_outer = generated_count(self.num_digits_outer as u64, "outer digit depth")?;
        if num_digits_inner == 0 || num_digits_outer == 0 {
            return Err(AkitaError::InvalidSetup(
                "generated precommit digit depths must be nonzero".to_string(),
            ));
        }
        let inner_width = decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("generated precommit A width overflow".to_string())
            })?;
        let n_a = generated_count(self.inner_output_rank as u64, "A output rank")?;
        let inner_commit_matrix = InnerCommitMatrixParams::try_new(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            n_a,
            inner_width,
            self.inner_coeff_linf_bound,
            d_a,
        )?;
        let native_outer_width = decomposed_t_ring_count(
            n_a,
            num_digits_outer,
            num_live_blocks,
            self.group.num_polynomials(),
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup("generated precommit native B width overflow".to_string())
        })?;
        let outer_width =
            projected_role_ring_count(d_a, d_b, native_outer_width).ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "generated precommit projected B width overflow".to_string(),
                )
            })?;
        let n_b = generated_count(self.outer_output_rank as u64, "B output rank")?;
        let outer_commit_matrix = OuterCommitMatrixParams::try_new(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            n_b,
            outer_width,
            self.outer_coeff_linf_bound,
            d_b,
        )?;
        let profile = CommittedGroupProfile {
            version: CommittedGroupProfile::VERSION,
            group: self.group,
            num_live_ring_elements_per_claim,
            num_positions_per_block,
            num_live_blocks,
            log_basis_inner: self.commitment.inner_commit_matrix.log_basis,
            num_digits_inner,
            inner_commit_matrix,
            log_basis_outer: self.commitment.outer_commit_matrix.log_basis,
            num_digits_outer,
            outer_commit_matrix,
        };
        profile.validate_frozen_precommit(policy.decomposition.field_bits())?;
        Ok(profile)
    }
}

impl GeneratedCommittedGroup {
    /// Expand this compact fold step into the full committed
    /// [`CommittedGroupParams`] for its position in the schedule.
    ///
    /// `fold_level` is `0` at the root and `>0` at recursive levels; it
    /// selects the level-local decomposition (root inherits the config
    /// decomposition; recursive levels collapse `log_commit_bound` to the
    /// level's own `log_basis`). `input_witness_len` is the witness length in
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
    /// the role-specific commit-matrix parameter constructor.
    ///
    /// # Errors
    ///
    /// Returns an error when a stored role dimension is invalid,
    /// bucket/width resolution fails, or a generated rank fails its SIS audit
    /// against the batched width.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn expand_to_level_params_with_setup(
        &self,
        policy: &PlannerPolicy,
        payload_mode: akita_types::CommitmentPayloadMode,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
        fold_level: usize,
        exact_num_digits_inner: Option<u32>,
        generated_num_digits_fold: u32,
        input_witness_len: usize,
        num_claims: usize,
        open_commit_matrix: GeneratedOpenCommitMatrix,
        setup_prefix_group: Option<GeneratedSetupPrefixInput>,
    ) -> Result<CommittedGroupParams, AkitaError> {
        let dimensions = CommitmentRingDims {
            inner: self.inner_commit_matrix.ring_dimension as usize,
            outer: self.outer_commit_matrix.ring_dimension as usize,
            opening: open_commit_matrix.ring_dimension as usize,
        };
        validate_role_dims(dimensions)?;
        let ring_d = dimensions.d_a();
        let is_root = fold_level == 0;
        let log_basis_inner = self.inner_commit_matrix.log_basis;
        let log_basis_outer = self.outer_commit_matrix.log_basis;
        let log_basis_open = open_commit_matrix.log_basis;
        let sis_modulus_profile = policy.sis_modulus_profile;
        let sis_policy = policy.sis_security_policy;

        // Digit-innermost geometry keeps `M = 2^position_index_bits` at every level
        // and carries exact live `B = ceil(N / M)` separately from its Boolean domain.
        let num_positions_per_block =
            generated_count(self.geometry.positions_per_block, "positions per block")?;
        let num_live_blocks = generated_count(self.geometry.live_blocks, "live block count")?;
        let block_index_bits = num_live_blocks
            .checked_next_power_of_two()
            .map_or(0, |domain| domain.trailing_zeros() as usize);
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
        if input_witness_len == 0 {
            return Err(AkitaError::InvalidSetup(
                "witness length must be nonzero".to_string(),
            ));
        }
        // Every exact live prefix may end in a partial ring. The commitment
        // view supplies the one implicit-zero suffix.
        let num_live_ring_elements_per_claim = input_witness_len.div_ceil(ring_d);
        let derived_num_live_blocks =
            num_live_ring_elements_per_claim.div_ceil(num_positions_per_block);
        if derived_num_live_blocks != num_live_blocks {
            return Err(AkitaError::InvalidSetup(format!(
                "generated schedule num_live_blocks={} does not match ceil(N={num_live_ring_elements_per_claim} / M={num_positions_per_block})={derived_num_live_blocks}",
                num_live_blocks,
            )));
        }

        // Per-role rounded-up collision buckets + committed widths, via the
        // `akita_types::sis` primitives. The B/D widths carry the `num_claims`
        // batch factor (the root commits `num_claims` polynomials); `n_a` is the
        // A-matrix row count. Unlike the planner DP, expansion audits the
        // generated ranks against these (norm, width) via `try_new`.
        let no_layout = |role: &str| {
            AkitaError::InvalidSetup(format!(
                "no audited {role}-role layout for generated schedule \
                 (profile={sis_modulus_profile:?}, dims={dimensions:?}, inner={log_basis_inner}, outer={log_basis_outer}, open={log_basis_open})"
            ))
        };
        let outer_decomp = DecompositionParams {
            log_basis: log_basis_outer,
            ..policy.decomposition
        };
        let witness_decomp = DecompositionParams {
            log_basis: log_basis_inner,
            log_commit_bound: policy.decomposition.field_bits(),
            log_open_bound: Some(policy.decomposition.field_bits()),
        };
        let open_decomp = DecompositionParams {
            log_basis: log_basis_open,
            ..policy.decomposition
        };
        let ring_challenge_cfg = ring_challenge_config(ring_d)?;
        let num_digits_inner = if let Some(num_digits_inner) = exact_num_digits_inner {
            usize::try_from(num_digits_inner).map_err(|_| {
                AkitaError::InvalidSetup(
                    "generated root inner digit depth does not fit the target platform".into(),
                )
            })?
        } else {
            num_digits_inner(witness_decomp, is_root)
        };
        let num_digits_outer = num_digits_open(outer_decomp);
        let num_digits_open_val = num_digits_open(open_decomp);

        let inner_width = decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
            .ok_or_else(|| no_layout("A"))?;
        let num_digits_fold = usize::try_from(generated_num_digits_fold).map_err(|_| {
            AkitaError::InvalidSetup(
                "generated fold digit depth does not fit the target platform".into(),
            )
        })?;
        if num_digits_fold == 0 {
            return Err(AkitaError::InvalidSetup(
                "generated fold digit depth must be nonzero".into(),
            ));
        }
        let a_bucket = rounded_up_role_a_inf_norm(
            sis_policy,
            policy.sis_table_digest,
            sis_modulus_profile,
            ring_d,
            log_basis_open,
            &ring_challenge_cfg,
            num_digits_fold,
            policy.ring_subfield_norm_bound,
        )
        .ok_or_else(|| no_layout("A"))?;
        let n_a = secure_rank(
            "a",
            sis_key(
                policy,
                akita_types::SisMatrixRole::Inner,
                ring_d as u32,
                a_bucket,
            ),
            inner_width,
        )?;

        let b_bucket = rounded_up_collision_inf_norm(
            sis_policy,
            sis_modulus_profile,
            akita_types::SisMatrixRole::Outer,
            dimensions.d_b(),
            log_basis_outer,
        )
        .ok_or_else(|| no_layout("B"))?;
        let native_outer_width =
            decomposed_t_ring_count(n_a, num_digits_outer, num_live_blocks, num_claims)
                .ok_or_else(|| no_layout("B"))?;
        let outer_width =
            projected_role_ring_count(dimensions.d_a(), dimensions.d_b(), native_outer_width)
                .ok_or_else(|| no_layout("B"))?;

        let d_bucket = rounded_up_collision_inf_norm(
            sis_policy,
            sis_modulus_profile,
            akita_types::SisMatrixRole::Open,
            dimensions.d_d(),
            log_basis_open,
        )
        .ok_or_else(|| no_layout("D"))?;
        let native_main_d_width =
            decomposed_w_ring_count(num_digits_open_val, num_live_blocks, num_claims)
                .ok_or_else(|| no_layout("D"))?;
        let main_d_width =
            projected_role_ring_count(dimensions.d_a(), dimensions.d_d(), native_main_d_width)
                .ok_or_else(|| no_layout("D"))?;
        let setup_prefix = if let Some(group) = setup_prefix_group {
            let commitment_params = group.expand_to_precommitted_group(
                policy,
                &ring_challenge_config,
                log_basis_open,
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
                group.natural_len as usize,
                commitment_params,
            ))
        } else {
            None
        };
        let precommitted_groups = Vec::new();
        let precommitted_d_width = setup_prefix
            .as_ref()
            .map(|prefix| prefix.commitment_params.d_segment_width(dimensions.d_d()))
            .transpose()?
            .unwrap_or(0);
        let d_matrix_width = main_d_width
            .checked_add(precommitted_d_width)
            .ok_or_else(|| AkitaError::InvalidSetup("generated D width overflow".into()))?;

        let num_digits_open = num_digits_open_val;

        // Size the committed B matrix at the full outer width.
        let n_b = secure_rank(
            "b",
            sis_key(
                policy,
                akita_types::SisMatrixRole::Outer,
                dimensions.d_b() as u32,
                b_bucket,
            ),
            outer_width,
        )?;
        let n_d = secure_rank(
            "d",
            sis_key(
                policy,
                akita_types::SisMatrixRole::Open,
                dimensions.d_d() as u32,
                d_bucket,
            ),
            d_matrix_width,
        )?;

        // Audit each generated rank against its width + bucket as we build the
        // key (verifier-reachable, so the fallible `try_new` is used instead
        // of the panicking `new`).
        let params = CommittedGroupParams {
            payload_mode,
            log_basis_inner,
            log_basis_outer,
            log_basis_open,
            inner_commit_matrix: InnerCommitMatrixParams::try_new(
                sis_policy,
                policy.sis_table_digest,
                sis_modulus_profile,
                n_a,
                inner_width,
                a_bucket,
                ring_d,
            )?,
            outer_commit_matrix: OuterCommitMatrixParams::try_new(
                sis_policy,
                policy.sis_table_digest,
                sis_modulus_profile,
                n_b,
                outer_width,
                b_bucket,
                dimensions.d_b(),
            )?,
            open_commit_matrix: OpenCommitMatrixParams::try_new(
                sis_policy,
                policy.sis_table_digest,
                sis_modulus_profile,
                n_d,
                d_matrix_width,
                d_bucket,
                dimensions.d_d(),
            )?,
            num_live_ring_elements_per_claim,
            num_live_blocks,
            num_positions_per_block,
            fold_challenge_config: ring_challenge_cfg,
            num_digits_inner,
            num_digits_outer,
            num_digits_open,
            num_digits_fold,
            // The caller stamps the configured per-level chunk policy after
            // expansion; this neutral default keeps parameter construction pure.
            witness_chunk: akita_types::ChunkedWitnessCfg::default(),
            precommitted_groups,
            setup_prefix,
        };
        Ok(params)
    }

    /// Expand a compact root step for a multi-group-root schedule.
    ///
    /// The main group's A/B layouts are claim-scaled by `main_num_polys`, while
    /// the shared D matrix has one segment for the main group plus the frozen
    /// precommitted group segments. This intentionally differs from scalar
    /// batched roots, whose D width is scaled by the total polynomial count.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn expand_to_multi_group_root_level_params_with_setup(
        &self,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
        main_num_polys: usize,
        num_digits_inner: u32,
        num_digits_fold: u32,
        precommitted_groups: Vec<PrecommittedLevelParams>,
        precommitted_d_width: usize,
        open_commit_matrix: GeneratedOpenCommitMatrix,
    ) -> Result<CommittedGroupParams, AkitaError> {
        let dimensions = CommitmentRingDims {
            inner: self.inner_commit_matrix.ring_dimension as usize,
            outer: self.outer_commit_matrix.ring_dimension as usize,
            opening: open_commit_matrix.ring_dimension as usize,
        };
        validate_role_dims(dimensions)?;
        let ring_d = dimensions.d_a();
        if precommitted_groups.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "generated multi-group root requires precommitted groups".to_string(),
            ));
        }

        let log_basis_inner = self.inner_commit_matrix.log_basis;
        let log_basis_outer = self.outer_commit_matrix.log_basis;
        let log_basis_open = open_commit_matrix.log_basis;
        let sis_modulus_profile = policy.sis_modulus_profile;
        let sis_policy = policy.sis_security_policy;
        let num_live_blocks = generated_count(self.geometry.live_blocks, "live block count")?;
        let block_index_bits = num_live_blocks
            .checked_next_power_of_two()
            .map_or(0, |domain| domain.trailing_zeros() as usize);
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
            generated_count(self.geometry.positions_per_block, "positions per block")?;

        let no_layout = |role: &str| {
            AkitaError::InvalidSetup(format!(
                "no audited {role}-role layout for generated multi-group root \
                 (profile={sis_modulus_profile:?}, d={ring_d}, inner={log_basis_inner}, outer={log_basis_outer}, open={log_basis_open})"
            ))
        };
        let outer_decomp = DecompositionParams {
            log_basis: log_basis_outer,
            ..policy.decomposition
        };
        let open_decomp = DecompositionParams {
            log_basis: log_basis_open,
            ..policy.decomposition
        };
        let ring_challenge_cfg = ring_challenge_config(ring_d)?;
        let num_digits_inner = usize::try_from(num_digits_inner).map_err(|_| {
            AkitaError::InvalidSetup(
                "generated root inner digit depth does not fit the target platform".into(),
            )
        })?;
        let num_digits_outer = num_digits_open(outer_decomp);
        let num_digits_open_val = num_digits_open(open_decomp);

        let inner_width = decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
            .ok_or_else(|| no_layout("A"))?;
        let num_digits_fold = usize::try_from(num_digits_fold).map_err(|_| {
            AkitaError::InvalidSetup(
                "generated root fold digit depth does not fit the target platform".into(),
            )
        })?;
        let a_bucket = rounded_up_role_a_inf_norm(
            sis_policy,
            policy.sis_table_digest,
            sis_modulus_profile,
            ring_d,
            log_basis_open,
            &ring_challenge_cfg,
            num_digits_fold,
            policy.ring_subfield_norm_bound,
        )
        .ok_or_else(|| no_layout("A"))?;
        let n_a = secure_rank(
            "a",
            sis_key(
                policy,
                akita_types::SisMatrixRole::Inner,
                ring_d as u32,
                a_bucket,
            ),
            inner_width,
        )?;

        let b_bucket = rounded_up_collision_inf_norm(
            sis_policy,
            sis_modulus_profile,
            akita_types::SisMatrixRole::Outer,
            dimensions.d_b(),
            log_basis_outer,
        )
        .ok_or_else(|| no_layout("B"))?;
        let native_outer_width =
            decomposed_t_ring_count(n_a, num_digits_outer, num_live_blocks, main_num_polys)
                .ok_or_else(|| no_layout("B"))?;
        let outer_width =
            projected_role_ring_count(dimensions.d_a(), dimensions.d_b(), native_outer_width)
                .ok_or_else(|| no_layout("B"))?;

        let native_main_d_width =
            decomposed_w_ring_count(num_digits_open_val, num_live_blocks, main_num_polys)
                .ok_or_else(|| no_layout("D"))?;
        let main_d_width =
            projected_role_ring_count(dimensions.d_a(), dimensions.d_d(), native_main_d_width)
                .ok_or_else(|| no_layout("D"))?;
        let d_matrix_width = main_d_width
            .checked_add(precommitted_d_width)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("generated multi-group D width overflow".into())
            })?;
        let d_log_basis = shared_d_digit_log_basis(log_basis_open, &precommitted_groups);
        let d_bucket = rounded_up_collision_inf_norm(
            sis_policy,
            sis_modulus_profile,
            akita_types::SisMatrixRole::Open,
            dimensions.d_d(),
            d_log_basis,
        )
        .ok_or_else(|| no_layout("D"))?;

        let n_b = secure_rank(
            "b",
            sis_key(
                policy,
                akita_types::SisMatrixRole::Outer,
                self.outer_commit_matrix.ring_dimension,
                b_bucket,
            ),
            outer_width,
        )?;
        let n_d = secure_rank(
            "d",
            sis_key(
                policy,
                akita_types::SisMatrixRole::Open,
                open_commit_matrix.ring_dimension,
                d_bucket,
            ),
            d_matrix_width,
        )?;
        let params = CommittedGroupParams {
            payload_mode: akita_types::CommitmentPayloadMode::Compressed,
            log_basis_inner,
            log_basis_outer,
            log_basis_open,
            inner_commit_matrix: InnerCommitMatrixParams::try_new(
                sis_policy,
                policy.sis_table_digest,
                sis_modulus_profile,
                n_a,
                inner_width,
                a_bucket,
                ring_d,
            )?,
            outer_commit_matrix: OuterCommitMatrixParams::try_new(
                sis_policy,
                policy.sis_table_digest,
                sis_modulus_profile,
                n_b,
                outer_width,
                b_bucket,
                dimensions.d_b(),
            )?,
            open_commit_matrix: OpenCommitMatrixParams::try_new(
                sis_policy,
                policy.sis_table_digest,
                sis_modulus_profile,
                n_d,
                d_matrix_width,
                d_bucket,
                dimensions.d_d(),
            )?,
            num_live_ring_elements_per_claim: num_live_blocks
                .checked_mul(num_positions_per_block)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("generated root group length overflow".to_string())
                })?,
            num_live_blocks,
            num_positions_per_block,
            fold_challenge_config: ring_challenge_cfg,
            num_digits_inner,
            num_digits_outer,
            num_digits_open: num_digits_open_val,
            num_digits_fold,
            witness_chunk: akita_types::ChunkedWitnessCfg::default(),
            precommitted_groups,
            setup_prefix: None,
        };
        Ok(params)
    }
}

impl GeneratedTerminalFold {
    pub(crate) fn expand_to_level_params(
        &self,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
        _fold_level: usize,
        input_witness_len: usize,
    ) -> Result<TerminalCommittedGroupParams, AkitaError> {
        let ring_dimension = self.inner_commit_matrix.ring_dimension as usize;
        if ring_dimension == 0 {
            return Err(AkitaError::InvalidSetup(
                "generated terminal inner ring dimension must be nonzero".to_string(),
            ));
        }
        if input_witness_len == 0 {
            return Err(AkitaError::InvalidSetup(
                "terminal witness length must be nonzero".to_string(),
            ));
        }
        let num_live_ring_elements_per_claim = input_witness_len.div_ceil(ring_dimension);
        let num_positions_per_block =
            generated_count(self.geometry.positions_per_block, "positions per block")?;
        let num_live_blocks = generated_count(self.geometry.live_blocks, "live block count")?;
        let generated_live_ring_elements = generated_count(
            self.geometry.live_ring_elements_per_claim,
            "live ring-element count",
        )?;
        if num_positions_per_block == 0
            || !num_positions_per_block.is_power_of_two()
            || generated_live_ring_elements != num_live_ring_elements_per_claim
            || num_live_ring_elements_per_claim.div_ceil(num_positions_per_block) != num_live_blocks
        {
            return Err(AkitaError::InvalidSetup(
                "generated terminal geometry does not match its input witness".to_string(),
            ));
        }
        let log_basis_inner = self.inner_commit_matrix.log_basis;
        let num_digits_inner = usize::try_from(self.num_digits_inner).map_err(|_| {
            AkitaError::InvalidSetup(
                "generated terminal inner digit depth does not fit the target platform".into(),
            )
        })?;
        if num_digits_inner == 0 {
            return Err(AkitaError::InvalidSetup(
                "generated terminal inner digit depth must be nonzero".into(),
            ));
        }
        let inner_width = decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal A width overflow".to_string()))?;
        let sparse = ring_challenge_config(ring_dimension)?;
        let output_rank = usize::try_from(self.inner_output_rank).map_err(|_| {
            AkitaError::InvalidSetup(
                "generated terminal inner rank does not fit the target platform".into(),
            )
        })?;
        if output_rank == 0 || self.inner_coeff_linf_bound == 0 {
            return Err(AkitaError::InvalidSetup(
                "generated terminal matrix contract must be nonzero".into(),
            ));
        }
        let inner_commit_matrix = InnerCommitMatrixParams::try_new(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            output_rank,
            inner_width,
            self.inner_coeff_linf_bound,
            ring_dimension,
        )?;
        let terminal = TerminalCommittedGroupParams {
            log_basis_inner,
            inner_commit_matrix,
            num_live_ring_elements_per_claim,
            num_positions_per_block,
            num_live_blocks,
            num_digits_inner,
        };
        if self.z_admission_linf_cap == 0
            || self.z_admission_linf_cap > terminal.certified_response_linf_cap(&sparse)?
            || self.z_rice_low_bits >= 64
            || self.z_payload_bytes == 0
        {
            return Err(AkitaError::InvalidSetup(
                "generated terminal response contract is invalid".into(),
            ));
        }
        Ok(terminal)
    }
}

impl GeneratedFoldScheduleEntry {
    /// Number of fold levels before the terminal direct step.
    pub fn num_fold_levels(&self) -> usize {
        self.recursive_folds.len() + 2
    }

    /// Validate the structural invariants the runtime relies on.
    ///
    /// # Errors
    ///
    /// Returns an error when any invariant is violated.
    pub fn validate(&self) -> Result<(), AkitaError> {
        if self.root.final_group.layout.num_polynomials() == 0 {
            return Err(AkitaError::UnsupportedSchedule(
                "generated root final group must be nonempty".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::{PlannerCostModelId, RingDimensionScheduleMode, SelectionPolicyId};
    use akita_types::{
        ChunkedWitnessCfg, SisModulusProfileId, SisSecurityPolicyId, SisTableDigest,
    };

    fn recursive_fp128_policy() -> PlannerPolicy {
        PlannerPolicy {
            cost_model: PlannerCostModelId::ExactPayloadAndSetupEnvelope,
            selection_policy: SelectionPolicyId::MinFirstDirectSetupThenPayload,
            setup_field_budget: None,
            min_offloaded_witness_contraction: 3,
            uniform_ring_dimension: 64,
            setup_prefix_inner_ring_dimension: 128,
            ring_dimension_schedule_mode: RingDimensionScheduleMode::UniformDimension {
                ring_dimension: 64,
            },
            decomposition: DecompositionParams {
                log_basis: 3,
                log_commit_bound: 1,
                log_open_bound: Some(128),
            },
            sis_modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
            sis_security_policy: SisSecurityPolicyId::Quantum128BitADPS16,
            sis_table_digest: SisTableDigest::CURRENT,
            ring_subfield_norm_bound: 1,
            claim_ext_degree: 1,
            chal_ext_degree: 1,
            inner_basis_range: (3, 16),
            opening_basis_range: (3, 6),
            witness_chunk: ChunkedWitnessCfg::default(),
            recursive_setup_planning: true,
        }
    }

    #[test]
    fn setup_prefix_expansion_preserves_independent_a_b_dimensions() {
        let input = GeneratedSetupPrefixInput {
            natural_len: 1 << 16,
            num_digits_fold: 4,
            commitment: GeneratedCommittedGroup {
                geometry: crate::generated::GeneratedBlockGeometry {
                    live_ring_elements_per_claim: 512,
                    positions_per_block: 32,
                    live_blocks: 16,
                },
                inner_commit_matrix: crate::generated::GeneratedInnerCommitMatrix {
                    ring_dimension: 128,
                    log_basis: 3,
                },
                outer_commit_matrix: crate::generated::GeneratedOuterCommitMatrix {
                    ring_dimension: 64,
                    log_basis: 3,
                    slice_count: 1,
                },
            },
        };
        let requested_dimensions = RefCell::new(Vec::new());
        let ring_challenge_config = |d| {
            requested_dimensions.borrow_mut().push(d);
            SparseChallengeConfig::production_for_ring_dim(d).ok_or_else(|| {
                AkitaError::InvalidSetup(format!("unsupported test ring dimension {d}"))
            })
        };

        let expanded = input
            .expand_to_precommitted_group(&recursive_fp128_policy(), &ring_challenge_config, 3)
            .expect("audited mixed-dimension setup-prefix layout");

        assert_eq!(&*requested_dimensions.borrow(), &[128]);
        assert_eq!(expanded.layout.inner_commit_matrix.ring_dimension(), 128);
        assert_eq!(expanded.layout.inner_commit_matrix.input_width(), 1376);
        assert_eq!(expanded.layout.outer_commit_matrix.ring_dimension(), 64);
        assert_eq!(expanded.layout.outer_commit_matrix.input_width(), 4128);
        assert_eq!(
            expanded.fold_challenge_config,
            SparseChallengeConfig::production_for_ring_dim(128)
                .expect("production D128 challenge config")
        );
    }
}
