//! On-demand expansion of compact generated schedule steps into full
//! [`CommittedGroupParams`].
//!
//! Generated rows store optimizer choices. Expansion derives all digit depths,
//! matrix widths, collision buckets, and minimum SIS-secure output ranks.
//!
//! This is verifier-reachable (config resolves levels through it on the
//! replay path), so every fallible step returns [`AkitaError`] rather than
//! panicking.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;

use crate::generated::{
    GeneratedCommittedGroup, GeneratedFoldScheduleEntry, GeneratedOpenCommitMatrix,
    GeneratedSetupPrefixInput, GeneratedTerminalFold,
};
use crate::schedule_params::optimize_fold_challenge_shape;
use crate::PlannerPolicy;
use akita_types::sis::{
    ceil_supported_linf_bound, decomposed_s_block_ring_count, decomposed_t_ring_count,
    decomposed_w_ring_count, fold_witness_digit_plan, fold_witness_unsnapped_linf_cap,
    min_secure_rank, num_digits_inner, num_digits_open, num_digits_setup_prefix_commit,
    role_a_collision_inf_norm_for_response_bound, rounded_up_collision_inf_norm,
    rounded_up_role_a_inf_norm, FoldChallengeNorms, FoldWitnessLinfCapConfig, FoldWitnessNorms,
    SisTableKey,
};
use akita_types::{
    shared_d_digit_log_basis, CommittedGroupParams, DecompositionParams, InnerCommitMatrixParams,
    OpenCommitMatrixParams, OuterCommitMatrixParams, PolynomialGroupLayout,
    PrecommittedGroupDescriptor, PrecommittedLevelParams, TerminalCommittedGroupParams,
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
        ring_challenge_cfg: &SparseChallengeConfig,
        fold_shape: TensorChallengeShape,
        log_basis_open: u32,
    ) -> Result<PrecommittedLevelParams, AkitaError> {
        super::validate_certified_bases(
            self.commitment.inner_commit_matrix.log_basis,
            self.commitment.outer_commit_matrix.log_basis,
            log_basis_open,
            policy,
            "generated setup-prefix group",
        )?;
        let d = self.d_setup as usize;
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
        let mut layout = PrecommittedGroupDescriptor {
            group: PolynomialGroupLayout::singleton(prefix_num_vars),
            num_live_ring_elements_per_claim,
            num_positions_per_block,
            num_live_blocks,
            log_basis_inner: self.commitment.inner_commit_matrix.log_basis,
            log_basis_outer: self.commitment.outer_commit_matrix.log_basis,
            n_a: 1,
            a_coeff_linf_bound: 1,
            n_b: 1,
            b_coeff_linf_bound: 1,
        };
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
                 (profile={sis_modulus_profile:?}, d={d}, inner={}, outer={}, open={})",
                self.commitment.inner_commit_matrix.log_basis,
                self.commitment.outer_commit_matrix.log_basis,
                log_basis_open
            ))
        };
        layout.validate_root_geometry(d)?;
        if fold_shape != TensorChallengeShape::Flat {
            return Err(AkitaError::InvalidSetup(
                "generated setup-prefix challenge shape mismatch".into(),
            ));
        }
        let inner_width = decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
            .ok_or_else(|| no_layout("A"))?;
        let a_bucket = rounded_up_role_a_inf_norm(
            sis_policy,
            sis_modulus_profile,
            d,
            inner_decomp,
            log_basis_open,
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
        let n_a = secure_rank(
            "setup-prefix a",
            sis_key(
                policy,
                akita_types::SisMatrixRole::Inner,
                d as u32,
                a_bucket,
            ),
            inner_width,
        )?;
        let b_bucket = rounded_up_collision_inf_norm(
            sis_policy,
            sis_modulus_profile,
            akita_types::SisMatrixRole::Outer,
            d,
            log_basis_open,
        )
        .ok_or_else(|| no_layout("B"))?;
        let outer_width = decomposed_t_ring_count(n_a, num_digits_outer, num_live_blocks, 1)
            .ok_or_else(|| no_layout("B"))?;
        let n_b = secure_rank(
            "setup-prefix b",
            sis_key(
                policy,
                akita_types::SisMatrixRole::Outer,
                d as u32,
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
            d,
        )?;
        let outer_commit_matrix = OuterCommitMatrixParams::try_new(
            sis_policy,
            policy.sis_table_digest,
            sis_modulus_profile,
            n_b,
            outer_width,
            b_bucket,
            d,
        )?;
        layout.n_a = n_a;
        layout.n_b = n_b;
        layout.a_coeff_linf_bound = a_bucket;
        layout.b_coeff_linf_bound = b_bucket;
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
            log_basis_open,
            challenge,
            FoldWitnessNorms::new(self.commitment.inner_commit_matrix.log_basis, d, 1, false),
            &fold_linf_cap_config,
        )?;
        Ok(PrecommittedLevelParams {
            layout,
            inner_commit_matrix,
            outer_commit_matrix,
            log_basis_open,
            num_digits_inner,
            num_digits_outer,
            num_digits_open: num_digits_open_val,
            num_digits_fold_one,
        })
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
    /// Returns an error when the stored ring dimension disagrees with the
    /// policy, bucket/width resolution fails, or a shipped rank fails its SIS
    /// audit against the (batched) width.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn expand_to_level_params_with_setup(
        &self,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
        fold_level: usize,
        input_witness_len: usize,
        fold_shape: TensorChallengeShape,
        num_claims: usize,
        open_commit_matrix: GeneratedOpenCommitMatrix,
        setup_prefix_group: Option<GeneratedSetupPrefixInput>,
    ) -> Result<CommittedGroupParams, AkitaError> {
        let ring_d = self.inner_commit_matrix.ring_dimension as usize;
        if ring_d == 0 || ring_d != policy.ring_dimension {
            return Err(AkitaError::InvalidSetup(format!(
                "generated fold step ring dimension {ring_d} does not match policy D={}",
                policy.ring_dimension
            )));
        }
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
        if input_witness_len == 0 || (!is_root && !input_witness_len.is_multiple_of(ring_d)) {
            return Err(AkitaError::InvalidSetup(
                "witness length is not divisible by the ring dimension".to_string(),
            ));
        }
        // Root inputs may be shorter than one ring and are zero-padded inside
        // that ring. Recursive witnesses are ring-aligned by contract.
        let num_live_ring_elements_per_claim = if is_root {
            input_witness_len.div_ceil(ring_d)
        } else {
            input_witness_len / ring_d
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

        // Per-role rounded-up collision buckets + committed widths, via the
        // `akita_types::sis` primitives. The B/D widths carry the `num_claims`
        // batch factor (the root commits `num_claims` polynomials); `n_a` is the
        // A-matrix row count. Unlike the planner DP, expansion audits the
        // *shipped* ranks against these (norm, width) via `try_new`.
        let no_layout = |role: &str| {
            AkitaError::InvalidSetup(format!(
                "no audited {role}-role layout for generated schedule \
                 (profile={sis_modulus_profile:?}, d={ring_d}, inner={log_basis_inner}, outer={log_basis_outer}, open={log_basis_open})"
            ))
        };
        let outer_decomp = DecompositionParams {
            log_basis: log_basis_outer,
            ..policy.decomposition
        };
        let witness_decomp = DecompositionParams {
            log_basis: log_basis_inner,
            ..policy.decomposition
        };
        let open_decomp = DecompositionParams {
            log_basis: log_basis_open,
            ..policy.decomposition
        };
        let ring_challenge_cfg = ring_challenge_config(ring_d)?;
        let num_digits_inner = num_digits_inner(witness_decomp, is_root);
        let num_digits_outer = num_digits_open(outer_decomp);
        let num_digits_open_val = num_digits_open(open_decomp);

        let inner_width = decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
            .ok_or_else(|| no_layout("A"))?;
        let a_bucket = rounded_up_role_a_inf_norm(
            sis_policy,
            sis_modulus_profile,
            ring_d,
            witness_decomp,
            log_basis_open,
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
            ring_d,
            log_basis_outer,
        )
        .ok_or_else(|| no_layout("B"))?;
        let outer_width =
            decomposed_t_ring_count(n_a, num_digits_outer, num_live_blocks, num_claims)
                .ok_or_else(|| no_layout("B"))?;

        let d_bucket = rounded_up_collision_inf_norm(
            sis_policy,
            sis_modulus_profile,
            akita_types::SisMatrixRole::Open,
            ring_d,
            log_basis_open,
        )
        .ok_or_else(|| no_layout("D"))?;
        let main_d_width =
            decomposed_w_ring_count(num_digits_open_val, num_live_blocks, num_claims)
                .ok_or_else(|| no_layout("D"))?;
        let setup_prefix = if let Some(group) = setup_prefix_group {
            let commitment_params = group.expand_to_precommitted_group(
                policy,
                &ring_challenge_cfg,
                TensorChallengeShape::Flat,
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
        let n_b = secure_rank(
            "b",
            sis_key(
                policy,
                akita_types::SisMatrixRole::Outer,
                ring_d as u32,
                b_bucket,
            ),
            outer_width,
        )?;
        let n_d = secure_rank(
            "d",
            sis_key(
                policy,
                akita_types::SisMatrixRole::Open,
                ring_d as u32,
                d_bucket,
            ),
            d_matrix_width,
        )?;

        // Audit each shipped rank against its width + bucket as we build the
        // key (verifier-reachable, so the fallible `try_new` is used instead
        // of the panicking `new`).
        let params = CommittedGroupParams {
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
                ring_d,
            )?,
            open_commit_matrix: OpenCommitMatrixParams::try_new(
                sis_policy,
                policy.sis_table_digest,
                sis_modulus_profile,
                n_d,
                d_matrix_width,
                d_bucket,
                ring_d,
            )?,
            num_live_ring_elements_per_claim,
            num_live_blocks,
            num_positions_per_block,
            fold_challenge_config: ring_challenge_cfg,
            fold_challenge_shape: fold_shape,
            num_digits_inner,
            num_digits_outer,
            num_digits_open,
            onehot_chunk_size,
            fold_linf_cap_config: akita_types::sis::FoldWitnessLinfCapConfig::worst_case_beta_only(
            ),
            num_digits_fold_one: 1,
            field_bits_hint: 0,
            cached_num_digits_block_claims: 0,
            cached_num_digits_fold_value: 1,
            // The caller stamps the configured per-level chunk policy after
            // expansion; this neutral default keeps parameter construction pure.
            witness_chunk: akita_types::ChunkedWitnessCfg::default(),
            precommitted_groups,
            setup_prefix,
        };
        let params =
            params.with_fold_linf_cap_config(policy.decomposition.field_bits(), num_claims)?;
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
        fold_shape: TensorChallengeShape,
        main_num_polys: usize,
        precommitted_groups: Vec<PrecommittedLevelParams>,
        precommitted_d_width: usize,
        open_commit_matrix: GeneratedOpenCommitMatrix,
    ) -> Result<CommittedGroupParams, AkitaError> {
        let ring_d = self.inner_commit_matrix.ring_dimension as usize;
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
        let fold_shape = optimize_fold_challenge_shape(fold_shape, num_live_blocks)?;

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
        let witness_decomp = DecompositionParams {
            log_basis: log_basis_inner,
            ..policy.decomposition
        };
        let open_decomp = DecompositionParams {
            log_basis: log_basis_open,
            ..policy.decomposition
        };
        let ring_challenge_cfg = ring_challenge_config(ring_d)?;
        let num_digits_inner = num_digits_inner(witness_decomp, true);
        let num_digits_outer = num_digits_open(outer_decomp);
        let num_digits_open_val = num_digits_open(open_decomp);

        let inner_width = decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
            .ok_or_else(|| no_layout("A"))?;
        let a_bucket = rounded_up_role_a_inf_norm(
            sis_policy,
            sis_modulus_profile,
            ring_d,
            witness_decomp,
            log_basis_open,
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
            ring_d,
            log_basis_outer,
        )
        .ok_or_else(|| no_layout("B"))?;
        let outer_width =
            decomposed_t_ring_count(n_a, num_digits_outer, num_live_blocks, main_num_polys)
                .ok_or_else(|| no_layout("B"))?;

        let main_d_width =
            decomposed_w_ring_count(num_digits_open_val, num_live_blocks, main_num_polys)
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
            ring_d,
            d_log_basis,
        )
        .ok_or_else(|| no_layout("D"))?;

        let onehot_chunk_size = if policy.decomposition.log_commit_bound == 1 {
            policy.onehot_chunk_size
        } else {
            0
        };

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
                ring_d,
            )?,
            open_commit_matrix: OpenCommitMatrixParams::try_new(
                sis_policy,
                policy.sis_table_digest,
                sis_modulus_profile,
                n_d,
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
            fold_challenge_config: ring_challenge_cfg,
            fold_challenge_shape: fold_shape,
            num_digits_inner,
            num_digits_outer,
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
        };
        let params =
            params.with_fold_linf_cap_config(policy.decomposition.field_bits(), main_num_polys)?;
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
    ) -> Result<(TerminalCommittedGroupParams, u128), AkitaError> {
        let ring_dimension = self.inner_commit_matrix.ring_dimension as usize;
        if ring_dimension == 0 || ring_dimension != policy.ring_dimension {
            return Err(AkitaError::InvalidSetup(
                "generated terminal inner ring dimension does not match policy".to_string(),
            ));
        }
        if input_witness_len == 0 || !input_witness_len.is_multiple_of(ring_dimension) {
            return Err(AkitaError::InvalidSetup(
                "terminal witness length is not inner-ring aligned".to_string(),
            ));
        }
        let num_live_ring_elements_per_claim = input_witness_len / ring_dimension;
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
        let witness_decomposition = DecompositionParams {
            log_basis: log_basis_inner,
            ..policy.decomposition
        };
        let num_digits_inner = num_digits_inner(witness_decomposition, false);
        let inner_width = decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal A width overflow".to_string()))?;
        let sparse = ring_challenge_config(ring_dimension)?;
        let challenge = FoldChallengeNorms::new(&sparse, TensorChallengeShape::Flat);
        let fold_linf_cap_config = FoldWitnessLinfCapConfig::for_fold_level(
            &sparse,
            TensorChallengeShape::Flat,
            ring_dimension,
            inner_width,
        )?;
        let witness_norms = FoldWitnessNorms::new(log_basis_inner, ring_dimension, 1, false);
        let (honest_response_linf_cap, _) = fold_witness_unsnapped_linf_cap(
            num_live_blocks,
            1,
            challenge,
            witness_norms,
            &fold_linf_cap_config,
        )?;
        let collision = role_a_collision_inf_norm_for_response_bound(
            challenge.l1_norm,
            policy.ring_subfield_norm_bound,
            honest_response_linf_cap,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("terminal A collision overflow".to_string()))?;
        let collision_bucket = ceil_supported_linf_bound(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            akita_types::SisMatrixRole::Inner,
            ring_dimension as u32,
            collision,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup("terminal A collision exceeds the SIS table".to_string())
        })?;
        let key = sis_key(
            policy,
            akita_types::SisMatrixRole::Inner,
            ring_dimension as u32,
            collision_bucket,
        );
        let output_rank = secure_rank("terminal A", key, inner_width)?;
        let inner_commit_matrix = InnerCommitMatrixParams::try_new(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            output_rank,
            inner_width,
            collision_bucket,
            ring_dimension,
        )?;
        Ok((
            TerminalCommittedGroupParams {
                log_basis_inner,
                inner_commit_matrix,
                num_live_ring_elements_per_claim,
                num_positions_per_block,
                num_live_blocks,
                num_digits_inner,
                fold_linf_cap_config,
            },
            honest_response_linf_cap,
        ))
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
