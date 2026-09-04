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
use akita_error::AkitaError;

use crate::candidate::{selective_l2_inner_matrix, SelectiveL2CandidateGeometry};
use crate::generated::{
    GeneratedFoldScheduleEntry, GeneratedFrozenGroup, GeneratedGroup, GeneratedMatrix,
    GeneratedSetupPrefix, GeneratedTerminalFold,
};
use crate::PlannerPolicy;
use akita_types::sis::{
    decomposed_s_block_ring_count, min_secure_rank, num_digits_inner, num_digits_open,
    rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, SisTableKey,
};
use akita_types::{
    validate_role_dims, CommitmentRingDims, CommitmentSliceCount, CommitmentSliceGeometry,
    CommittedGroupParams, CommittedSourceEncoding, DecompositionParams, GroupOpenPhaseParams,
    InnerCommitMatrixParams, OpenCommitMatrixParams, OuterCommitMatrixParams, TerminalFoldParams,
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

/// Where a group sits, and where its live length comes from.
///
/// This is the whole difference between what used to be two near-identical
/// expansion functions. Everything else they did was the same, including the
/// shared-D digit basis: `shared_d_digit_log_basis` ignores its group argument
/// and returns the main basis, so the two paths were already computing one
/// bucket.
#[allow(clippy::large_enum_variant)]
pub(crate) enum GroupLengthSource {
    /// Live length follows from the witness arriving at this level, and may end
    /// in a partial ring.
    IncomingWitness {
        input_witness_len: usize,
        num_claims: usize,
        setup_prefix: Option<GeneratedSetupPrefix>,
    },
    /// A grouped root pins its live length in its own geometry, and its shared D
    /// matrix carries the frozen precommitted segments alongside its own.
    PinnedGrouped {
        num_claims: usize,
        precommitted_groups: Vec<GroupOpenPhaseParams>,
    },
}

impl GroupLengthSource {
    fn num_claims(&self) -> usize {
        match self {
            Self::IncomingWitness { num_claims, .. } | Self::PinnedGrouped { num_claims, .. } => {
                *num_claims
            }
        }
    }
}

pub(crate) enum GeneratedFoldExpansionRole {
    Root {
        num_digits_inner: u32,
    },
    Recursive {
        fold_level: usize,
        response_l2_sq_cap: Option<u128>,
    },
}

pub(crate) struct GeneratedGroupExpansion {
    pub role: GeneratedFoldExpansionRole,
    pub payload_mode: akita_types::CommitmentPayloadMode,
    pub ring_relation_mode: akita_types::RingRelationMode,
    pub open_commit_matrix: GeneratedMatrix,
    pub group: akita_types::PolynomialGroupLayout,
    pub source: GroupLengthSource,
}

impl GeneratedFrozenGroup {
    fn expand_to_group(
        self,
        setup_natural_len: Option<usize>,
        policy: &PlannerPolicy,
        ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
        log_basis_open: u32,
        num_response_chunks: usize,
    ) -> Result<GroupOpenPhaseParams, AkitaError> {
        let d_a = self.profile.inner.matrix.ring_dimension();
        // The consuming fold supplies the shared opening basis and the challenge
        // family; a prefix cannot pin its own. This used to be a stored
        // `GroupOpeningPlan` re-derived here and rejected on disagreement, which
        // meant three audits guarding a field that carried no information. The
        // plan is now derived only, so there is nothing left to disagree.
        let challenge_dimension = match self.opening_method {
            akita_types::OpeningMethod::EvaluationTrace => d_a,
            akita_types::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension,
            } => challenge_subring_dimension,
        };
        let fold_challenge_config = ring_challenge_config(challenge_dimension)?;
        let admission_policy = akita_types::PrecommittedGroupAdmissionPolicy {
            decomposition: policy.decomposition,
            sis_security_policy: policy.sis_security_policy,
            sis_table_digest: policy.sis_table_digest,
            sis_modulus_profile: policy.sis_modulus_profile,
            num_response_chunks,
        };
        let num_digits_fold = generated_count(
            u64::from(self.num_digits_fold),
            "generated group fold digits",
        )?;
        let params = if let Some(natural_len) = setup_natural_len {
            GroupOpenPhaseParams::admit_setup_prefix(
                self.profile,
                natural_len,
                num_digits_fold,
                admission_policy,
                self.opening_method,
                fold_challenge_config,
                log_basis_open,
            )?
        } else {
            GroupOpenPhaseParams::admit(
                self.profile,
                num_digits_fold,
                admission_policy,
                self.opening_method,
                fold_challenge_config,
                log_basis_open,
            )?
        };
        params.validate()?;
        Ok(params)
    }
}

impl GeneratedSetupPrefix {
    fn expand_to_group(
        self,
        policy: &PlannerPolicy,
        ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
        log_basis_open: u32,
        num_response_chunks: usize,
    ) -> Result<GroupOpenPhaseParams, AkitaError> {
        let natural_len = generated_count(self.natural_len, "setup-prefix natural length")?;
        self.group.expand_to_group(
            Some(natural_len),
            policy,
            ring_challenge_config,
            log_basis_open,
            num_response_chunks,
        )
    }
}

impl GeneratedGroup {
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
    /// Expand a fold's incoming setup prefix, if it carries one.
    ///
    /// Split out so the one expansion body can take its prefix from a fold or
    /// its precommitted segments from a grouped root without branching twice.
    fn expand_setup_prefix(
        prefix: Option<GeneratedSetupPrefix>,
        policy: &PlannerPolicy,
        ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
        log_basis_open: u32,
        num_response_chunks: usize,
    ) -> Result<Option<GroupOpenPhaseParams>, AkitaError> {
        let Some(group) = prefix else {
            return Ok(None);
        };

        let commitment_params = group.expand_to_group(
            policy,
            &ring_challenge_config,
            log_basis_open,
            num_response_chunks,
        )?;
        let n_prefix = 1usize
            .checked_shl(commitment_params.profile.group.num_vars() as u32)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("generated setup-prefix length overflow".into())
            })?;
        let group_natural_len = generated_count(group.natural_len, "setup-prefix natural length")?;
        akita_types::validate_setup_prefix_domain(group_natural_len, n_prefix)?;
        Ok(Some(akita_types::scheduled_setup_prefix(
            group_natural_len,
            commitment_params,
        )))
    }

    pub(crate) fn expand_group(
        &self,
        policy: &PlannerPolicy,
        ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
        expansion: GeneratedGroupExpansion,
    ) -> Result<CommittedGroupParams, AkitaError> {
        let GeneratedGroupExpansion {
            role,
            payload_mode,
            ring_relation_mode,
            open_commit_matrix,
            group,
            source,
        } = expansion;
        let (fold_level, exact_num_digits_inner, response_l2_sq_cap) = match role {
            GeneratedFoldExpansionRole::Root { num_digits_inner } => {
                (0, Some(num_digits_inner), None)
            }
            GeneratedFoldExpansionRole::Recursive {
                fold_level,
                response_l2_sq_cap,
            } => (fold_level, None, response_l2_sq_cap),
        };
        let opening_method = self.opening_method;
        let generated_num_digits_fold = self.num_digits_fold;
        let num_claims = source.num_claims();
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
        let num_response_chunks = policy.chunks_at_level(fold_level);
        let sis_modulus_profile = policy.sis_modulus_profile;
        let sis_policy = policy.sis_security_policy;

        // Digit-innermost geometry keeps `M = 2^position_index_bits` at every level
        // and carries exact live `B = ceil(N / M)` separately from its Boolean domain.
        let num_positions_per_block = self.geometry.positions_per_block;
        let num_live_blocks = self.geometry.live_blocks;
        let outer_slice_count = CommitmentSliceCount::try_new(self.outer_slice_count as usize)?;
        outer_slice_count.validate_for_commitment(fold_level, payload_mode, num_live_blocks)?;
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
        let (num_live_ring_elements_per_claim, producer_num_vars) = match &source {
            GroupLengthSource::IncomingWitness {
                input_witness_len, ..
            } => {
                if *input_witness_len == 0 {
                    return Err(AkitaError::InvalidSetup(
                        "witness length must be nonzero".to_string(),
                    ));
                }
                // Every exact live prefix may end in a partial ring. The
                // commitment view supplies the one implicit-zero suffix.
                let live = input_witness_len.div_ceil(ring_d);
                let derived_num_live_blocks = live.div_ceil(num_positions_per_block);
                if derived_num_live_blocks != num_live_blocks {
                    return Err(AkitaError::InvalidSetup(format!(
                        "generated schedule num_live_blocks={num_live_blocks} does not match ceil(N={live} / M={num_positions_per_block})={derived_num_live_blocks}"
                    )));
                }
                (live, input_witness_len.trailing_zeros() as usize)
            }
            GroupLengthSource::PinnedGrouped { .. } => (
                num_live_blocks
                    .checked_mul(num_positions_per_block)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("generated root group length overflow".to_string())
                    })?,
                self.geometry
                    .live_ring_elements_per_claim
                    .checked_mul(ring_d)
                    .filter(|len| len.is_power_of_two())
                    .map_or(0, |len| len.trailing_zeros() as usize),
            ),
        };

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
        let ring_challenge_cfg = match opening_method {
            akita_types::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension,
            } => akita_challenges::SparseChallengeConfig::production_for_ring_dim(
                challenge_subring_dimension,
            )
            .ok_or_else(|| no_layout("A"))?,
            akita_types::OpeningMethod::EvaluationTrace if response_l2_sq_cap.is_some() => {
                akita_challenges::selective_l2_challenge_config(ring_d)
                    .unwrap_or(ring_challenge_config(ring_d)?)
            }
            akita_types::OpeningMethod::EvaluationTrace => ring_challenge_config(ring_d)?,
        };
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
            num_response_chunks,
        )
        .ok_or_else(|| no_layout("A"))?;
        let linf_n_a = secure_rank(
            "a",
            sis_key(
                policy,
                akita_types::SisMatrixRole::Inner,
                ring_d as u32,
                a_bucket,
            ),
            inner_width,
        )?;
        let inner_commit_matrix = if let Some(response_l2_sq_cap) = response_l2_sq_cap {
            let fold_basis = 1usize
                .checked_shl(log_basis_open)
                .ok_or_else(|| AkitaError::InvalidSetup("generated L2 basis overflow".into()))?;
            let matrix = selective_l2_inner_matrix(
                policy,
                SelectiveL2CandidateGeometry {
                    fold_level,
                    num_claims,
                    num_chunks: policy.chunks_at_level(fold_level),
                    inner_width,
                    ring_dimension: ring_d,
                    fold_basis,
                    fold_digit_count: num_digits_fold,
                    fold_challenge_config: &ring_challenge_cfg,
                    response_l2_sq_cap: Some(response_l2_sq_cap),
                    norm_proof_shape: None,
                },
            )?
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "generated L2 route is not admitted by the calibrated response model".into(),
                )
            })?;
            if !matches!(
                matrix.security_route(),
                akita_types::InnerCommitSecurityRoute::L2 {
                    response_l2_sq_cap: canonical_cap,
                    ..
                } if canonical_cap == response_l2_sq_cap
            ) {
                return Err(AkitaError::InvalidSetup(
                    "generated L2 cap disagrees with canonical candidate policy".into(),
                ));
            }
            matrix
        } else {
            InnerCommitMatrixParams::try_new(
                sis_policy,
                policy.sis_table_digest,
                sis_modulus_profile,
                linf_n_a,
                inner_width,
                a_bucket,
                ring_d,
            )?
        };
        let n_a = inner_commit_matrix.output_rank();

        let b_bucket = rounded_up_collision_inf_norm(
            sis_policy,
            sis_modulus_profile,
            akita_types::SisMatrixRole::Outer,
            dimensions.d_b(),
            log_basis_outer,
        )
        .ok_or_else(|| no_layout("B"))?;
        let outer_width = CommitmentSliceGeometry::try_new(
            outer_slice_count,
            num_live_blocks,
            num_claims,
            n_a,
            num_digits_outer,
            dimensions.d_a(),
            dimensions.d_b(),
        )?
        .physical_input_width();

        let d_bucket = rounded_up_collision_inf_norm(
            sis_policy,
            sis_modulus_profile,
            akita_types::SisMatrixRole::Open,
            dimensions.d_d(),
            log_basis_open,
        )
        .ok_or_else(|| no_layout("D"))?;
        let main_d_width = akita_types::opening_d_segment_width(
            opening_method,
            policy.claim_ext_degree,
            dimensions.d_a(),
            dimensions.d_d(),
            num_digits_open_val,
            num_live_blocks,
            num_claims,
        )?;
        let (precommitted_groups, precommitted_d_width, setup_prefix) = match source {
            GroupLengthSource::PinnedGrouped {
                precommitted_groups,
                ..
            } => {
                if precommitted_groups.is_empty() {
                    return Err(AkitaError::InvalidSetup(
                        "generated multi-group root requires precommitted groups".to_string(),
                    ));
                }
                let mut precommitted_d_width = 0usize;
                for group in &precommitted_groups {
                    precommitted_d_width = precommitted_d_width
                        .checked_add(
                            group.d_segment_width(policy.claim_ext_degree, dimensions.d_d())?,
                        )
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "generated multi-group D width overflow".to_string(),
                            )
                        })?;
                }
                (precommitted_groups, precommitted_d_width, None)
            }
            GroupLengthSource::IncomingWitness { setup_prefix, .. } => {
                let setup_prefix = Self::expand_setup_prefix(
                    setup_prefix,
                    policy,
                    ring_challenge_config,
                    log_basis_open,
                    num_response_chunks,
                )?;
                let width = setup_prefix
                    .as_ref()
                    .map(|prefix| prefix.d_segment_width(policy.claim_ext_degree, dimensions.d_d()))
                    .transpose()?
                    .unwrap_or(0);
                (Vec::new(), width, setup_prefix)
            }
        };
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
        let source_encoding = CommittedSourceEncoding::for_producer(
            opening_method,
            policy.claim_ext_degree,
            ring_d,
            producer_num_vars,
            is_root,
        );
        let open_matrix = OpenCommitMatrixParams::try_new(
            sis_policy,
            policy.sis_table_digest,
            sis_modulus_profile,
            n_d,
            d_matrix_width,
            d_bucket,
            dimensions.d_d(),
        )?;
        let groups = setup_prefix
            .into_iter()
            .chain(precommitted_groups)
            .chain(std::iter::once(GroupOpenPhaseParams {
                profile: akita_types::GroupCommitPhaseParams {
                    version: akita_types::GroupCommitPhaseParams::VERSION,
                    group,
                    blocks: akita_types::BlockGeometry::new(
                        num_live_ring_elements_per_claim,
                        num_positions_per_block,
                        num_live_blocks,
                    ),
                    outer_slice_count,
                    inner: akita_types::RoleParams::new(
                        akita_types::GadgetDigits::new(log_basis_inner, num_digits_inner),
                        inner_commit_matrix,
                    ),
                    outer: akita_types::RoleParams::new(
                        akita_types::GadgetDigits::new(log_basis_outer, num_digits_outer),
                        OuterCommitMatrixParams::try_new(
                            sis_policy,
                            policy.sis_table_digest,
                            sis_modulus_profile,
                            n_b,
                            outer_width,
                            b_bucket,
                            dimensions.d_b(),
                        )?,
                    ),
                },
                opening: akita_types::GroupOpeningPlan {
                    opening_method,
                    fold_challenge_config: ring_challenge_cfg,
                    log_basis_open,
                    num_digits_open,
                    num_digits_fold,
                },
                setup_natural_len: None,
            }))
            .collect();
        let params = CommittedGroupParams::try_new(
            groups,
            open_matrix,
            payload_mode,
            ring_relation_mode,
            source_encoding,
            // The caller stamps the configured per-level chunk policy after
            // expansion; this neutral default keeps parameter construction pure.
            akita_types::ChunkedWitnessCfg::default(),
        )?;
        Ok(params)
    }
}

impl GeneratedTerminalFold {
    pub(crate) fn expand_to_level_params(
        &self,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
        fold_level: usize,
        input_witness_len: usize,
    ) -> Result<TerminalFoldParams, AkitaError> {
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
        let num_positions_per_block = self.geometry.positions_per_block;
        let num_live_blocks = self.geometry.live_blocks;
        let generated_live_ring_elements = self.geometry.live_ring_elements_per_claim;
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
        let fold_digit_count = usize::try_from(self.fold_digit_count).map_err(|_| {
            AkitaError::InvalidSetup(
                "generated terminal fold digit count does not fit the target platform".into(),
            )
        })?;
        if self.fold_log_basis == 0 || fold_digit_count == 0 {
            return Err(AkitaError::InvalidSetup(
                "generated terminal fold basis and digit count must be nonzero".into(),
            ));
        }
        let inner_width = decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal A width overflow".to_string()))?;
        let sparse = if self.response_l2_sq_cap.is_some() {
            akita_challenges::selective_l2_challenge_config(ring_dimension).ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "generated terminal L2 route has no certified operator-norm challenge".into(),
                )
            })?
        } else {
            ring_challenge_config(ring_dimension)?
        };
        let output_rank = usize::try_from(self.inner_output_rank).map_err(|_| {
            AkitaError::InvalidSetup(
                "generated terminal inner rank does not fit the target platform".into(),
            )
        })?;
        if output_rank == 0
            || (self.response_l2_sq_cap.is_none() && self.inner_coeff_linf_bound == 0)
        {
            return Err(AkitaError::InvalidSetup(
                "generated terminal matrix contract must be nonzero".into(),
            ));
        }
        let inner_commit_matrix = if let Some(response_l2_sq_cap) = self.response_l2_sq_cap {
            let fold_basis = 1usize
                .checked_shl(self.fold_log_basis)
                .ok_or_else(|| AkitaError::InvalidSetup("terminal L2 basis overflow".into()))?;
            let matrix = selective_l2_inner_matrix(
                policy,
                SelectiveL2CandidateGeometry {
                    fold_level,
                    num_claims: 1,
                    num_chunks: 1,
                    inner_width,
                    ring_dimension,
                    fold_basis,
                    fold_digit_count,
                    fold_challenge_config: &sparse,
                    response_l2_sq_cap: Some(response_l2_sq_cap),
                    norm_proof_shape: Some(akita_types::PhysicalL2NormProofShape::Direct {
                        physical_response_len: inner_width.checked_mul(ring_dimension).ok_or_else(
                            || {
                                AkitaError::InvalidSetup(
                                    "terminal L2 response length overflow".into(),
                                )
                            },
                        )?,
                    }),
                },
            )?
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "generated terminal L2 route has no canonical source model".into(),
                )
            })?;
            let cap_matches = matches!(
                matrix.security_route(),
                akita_types::InnerCommitSecurityRoute::L2 {
                    response_l2_sq_cap: cap,
                    ..
                } if cap == response_l2_sq_cap
            );
            if matrix.output_rank() != output_rank || !cap_matches {
                return Err(AkitaError::InvalidSetup(
                    "generated terminal L2 matrix disagrees with its canonical source model".into(),
                ));
            }
            matrix
        } else {
            InnerCommitMatrixParams::try_new(
                policy.sis_security_policy,
                policy.sis_table_digest,
                policy.sis_modulus_profile,
                output_rank,
                inner_width,
                self.inner_coeff_linf_bound,
                ring_dimension,
            )?
        };
        let terminal = TerminalFoldParams {
            fold_challenge_config: sparse,
            response_shape: akita_types::TerminalResponseShape {
                layout: akita_types::TailSegmentLayout {
                    ring_dimension,
                    groups: Vec::new(),
                    logical_num_elems: 0,
                },
            },
            input_witness_len: 0,
            blocks: akita_types::BlockGeometry::new(
                num_live_ring_elements_per_claim,
                num_positions_per_block,
                num_live_blocks,
            ),
            inner: akita_types::RoleParams::new(
                akita_types::GadgetDigits::new(log_basis_inner, num_digits_inner),
                inner_commit_matrix,
            ),
            fold: akita_types::GadgetDigits::new(self.fold_log_basis, fold_digit_count),
        };
        if terminal
            .validate_terminal_linf_cap(self.z_linf_cap)
            .is_err()
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
        if self.final_group.num_polynomials() == 0 {
            return Err(AkitaError::UnsupportedSchedule(
                "generated root final group must be nonempty".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "fp128-onehot-recursive"))]
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
            selective_l2_response_model: crate::SelectiveL2ResponseModelId::Disabled,
            selection_policy: SelectionPolicyId::MinFirstDirectSetupThenPayloadV2,
            recursive_setup_search_policy: crate::RecursiveSetupSearchPolicy::Exhaustive,
            recursive_split_search_policy: crate::RecursiveSplitSearchPolicy::Exhaustive,
            setup_field_budget: None,
            min_offloaded_witness_contraction: 3,
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
            sis_l2_table_digest: akita_types::SisL2TableDigest::CURRENT,
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
        // The shared opening basis belongs to the consuming fold, so take both.
        let (fold, input) =
            crate::generated::fp128_onehot_recursive::FP128_ONEHOT_RECURSIVE_SCHEDULES
                .iter()
                .flat_map(|entry| entry.recursive_folds)
                .find_map(|fold| fold.setup_prefix.map(|prefix| (fold, prefix)))
                .expect("generated recursive setup-prefix fixture");
        let requested_dimensions = RefCell::new(Vec::new());
        let ring_challenge_config = |d| {
            requested_dimensions.borrow_mut().push(d);
            SparseChallengeConfig::production_for_ring_dim(d).ok_or_else(|| {
                AkitaError::InvalidSetup(format!("unsupported test ring dimension {d}"))
            })
        };

        let expanded = input
            .expand_to_group(
                &recursive_fp128_policy(),
                &ring_challenge_config,
                fold.core.open_commit_matrix.log_basis,
                1,
            )
            .expect("audited mixed-dimension setup-prefix layout");

        let expected_challenge_dimension = match input.group.opening_method {
            akita_types::OpeningMethod::EvaluationTrace => {
                input.group.profile.inner.matrix.ring_dimension()
            }
            akita_types::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension,
            } => challenge_subring_dimension,
        };
        assert_eq!(
            &*requested_dimensions.borrow(),
            &[expected_challenge_dimension]
        );
        assert_eq!(expanded.profile, input.group.profile);
        // The plan is derived, so assert it carries the two inputs the row still
        // stores and the basis its consuming fold supplied.
        assert_eq!(expanded.opening.opening_method, input.group.opening_method);
        assert_eq!(
            expanded.opening.num_digits_fold,
            input.group.num_digits_fold as usize
        );
        assert_eq!(
            expanded.opening.log_basis_open,
            fold.core.open_commit_matrix.log_basis
        );
    }

    #[test]
    fn setup_prefix_expansion_rejects_frozen_profile_mutation() {
        let (fold, mut input) =
            crate::generated::fp128_onehot_recursive::FP128_ONEHOT_RECURSIVE_SCHEDULES
                .iter()
                .flat_map(|entry| entry.recursive_folds)
                .find_map(|fold| fold.setup_prefix.map(|prefix| (fold, prefix)))
                .expect("generated recursive setup-prefix fixture");
        input.group.profile.blocks.live_blocks += 1;
        let ring_challenge_config = |d| {
            SparseChallengeConfig::production_for_ring_dim(d).ok_or_else(|| {
                AkitaError::InvalidSetup(format!("unsupported test ring dimension {d}"))
            })
        };

        input
            .expand_to_group(
                &recursive_fp128_policy(),
                &ring_challenge_config,
                fold.core.open_commit_matrix.log_basis,
                1,
            )
            .expect_err("frozen setup-prefix profile mutation must reject");
    }

    #[test]
    fn setup_prefix_expansion_rejects_zero_natural_length() {
        let (fold, mut input) =
            crate::generated::fp128_onehot_recursive::FP128_ONEHOT_RECURSIVE_SCHEDULES
                .iter()
                .flat_map(|entry| entry.recursive_folds)
                .find_map(|fold| fold.setup_prefix.map(|prefix| (fold, prefix)))
                .expect("generated recursive setup-prefix fixture");
        input.natural_len = 0;
        let ring_challenge_config = |d| {
            SparseChallengeConfig::production_for_ring_dim(d).ok_or_else(|| {
                AkitaError::InvalidSetup(format!("unsupported test ring dimension {d}"))
            })
        };

        input
            .expand_to_group(
                &recursive_fp128_policy(),
                &ring_challenge_config,
                fold.core.open_commit_matrix.log_basis,
                1,
            )
            .expect_err("zero setup-prefix natural length must reject");
    }
}
