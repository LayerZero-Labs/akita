use super::*;

/// Final-group layouts shared by all root precommit opening assignments at one
/// fixed dimension tuple, opening family, and pair of decomposition bases.
pub(crate) struct PreparedRootFinalGroup {
    pub(super) candidates_by_split: Vec<Vec<CommittedGroupParams>>,
    opening_batch: OpeningClaimsLayout,
    field_bits: u32,
    extension_degree: usize,
}

impl PreparedRootFinalGroup {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        key: &ScheduleLookupKey,
        final_source_contract: CommittedSourceContract,
        policy: &PlannerPolicy,
        dimensions: CommitmentRingDims,
        opening: PlannerOpeningCandidate,
        candidate_log_basis_inner: u32,
        candidate_log_basis_open: u32,
        guide: Option<crate::schedule_params::CandidateLayoutGuide>,
    ) -> Result<Self, AkitaError> {
        dimensions.validate_role_projection()?;
        opening.validate_for(policy.claim_ext_degree, dimensions)?;
        let field_bits = policy.decomposition.field_bits();
        let alpha = dimensions.d_a().trailing_zeros() as usize;
        let reduced_vars = key.final_group.num_vars().saturating_sub(alpha);
        if reduced_vars == 0 {
            return Ok(Self {
                candidates_by_split: Vec::new(),
                opening_batch: key.opening_layout()?,
                field_bits,
                extension_degree: policy.claim_ext_degree,
            });
        }
        let candidate_ctx = MultiGroupRootCandidateCtx {
            policy,
            dimensions,
            opening,
            final_source_contract,
            final_num_vars: key.final_group.num_vars(),
            main_num_polys: key.final_group.num_polynomials(),
            source: crate::schedule_params::root_inner_basis_source(final_source_contract),
        };
        let opening_batch = key.opening_layout()?;
        let min_block_index_bits: usize = if reduced_vars >= 3 { 1 } else { 0 };
        let max_block_index_bits: usize = (reduced_vars - 1).min(usize::BITS as usize - 1);
        let num_ring_elems = checked::pow2(reduced_vars).ok_or_else(|| {
            AkitaError::InvalidSetup("root reduced-variable domain is too large".into())
        })?;
        let delta_commit = candidate_ctx
            .source
            .num_digits_inner(policy.decomposition, candidate_log_basis_inner)?;
        let delta_open = num_digits_open(DecompositionParams {
            log_basis: candidate_log_basis_open,
            ..policy.decomposition
        });
        let mut split_domain = if let Some(guide) = guide {
            reduced_vars
                .checked_sub(guide.position_index_bits)
                .into_iter()
                .collect()
        } else {
            recursive_split_search_domain(
                policy.recursive_split_search_policy,
                num_ring_elems,
                reduced_vars,
                delta_commit,
                delta_open,
                policy.chunks_at_level(0),
            )
        };
        if guide.is_none() && min_block_index_bits == 0 {
            split_domain.push(0);
        }
        split_domain
            .retain(|&split| min_block_index_bits <= split && split <= max_block_index_bits);
        split_domain.sort_unstable_by(|left, right| right.cmp(left));
        split_domain.dedup();

        let mut candidates_by_split = Vec::new();

        for &block_index_bits in &split_domain {
            let position_index_bits = reduced_vars - block_index_bits;
            let num_live_blocks = 1usize << block_index_bits;
            let mut slice_candidates = Vec::new();
            for outer_slice_count in akita_params::CommitmentSliceCount::ALL {
                if guide.is_some_and(|guide| outer_slice_count != guide.outer_slice_count) {
                    continue;
                }
                if outer_slice_count
                    .validate_for_commitment(
                        0,
                        akita_params::CommitmentPayloadMode::Compressed,
                        num_live_blocks,
                    )
                    .is_err()
                {
                    continue;
                }
                let Some(mut candidate_params) = root_final_group_level_params_candidate(
                    &candidate_ctx,
                    RootFinalGroupCandidateInput {
                        log_basis_inner: candidate_log_basis_inner,
                        log_basis_open: candidate_log_basis_open,
                        position_index_bits,
                        block_index_bits,
                        outer_slice_count,
                    },
                )?
                else {
                    continue;
                };
                candidate_params.witness_chunk = policy.witness_chunk_at_level(0);
                slice_candidates.push(candidate_params);
            }
            candidates_by_split.push(slice_candidates);
        }
        Ok(Self {
            candidates_by_split,
            opening_batch,
            field_bits,
            extension_degree: policy.claim_ext_degree,
        })
    }

    /// Combine cached final-group layouts with admitted producers. Witness
    /// ranking disables setup-based slice pruning so it sees every valid size.
    pub(crate) fn candidates(
        &self,
        policy: &PlannerPolicy,
        producers: &PreparedRootProducers,
        prune_slices: bool,
    ) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
        let mut candidates = Vec::new();
        for prototypes in &self.candidates_by_split {
            let mut slices = Vec::new();
            for main in prototypes {
                let Some(params) = producers.combine_with(main)? else {
                    continue;
                };
                if params.compression_sources_supported()? {
                    slices.push(params);
                }
            }
            let slices = if prune_slices {
                crate::schedule_params::prune_root_slices_by_first_direct_setup(
                    policy,
                    &self.opening_batch,
                    slices,
                )?
            } else {
                slices
            };
            for params in slices {
                if let Some(len) = root_batch_next_w_len(
                    self.field_bits,
                    self.extension_degree,
                    &params,
                    &self.opening_batch,
                )? {
                    candidates.push((params, len));
                }
            }
        }
        Ok(candidates)
    }
}

pub(super) struct MultiGroupRootCandidateCtx<'a> {
    pub(super) policy: &'a PlannerPolicy,
    pub(super) dimensions: CommitmentRingDims,
    pub(super) opening: PlannerOpeningCandidate,
    pub(super) final_source_contract: CommittedSourceContract,
    pub(super) final_num_vars: usize,
    pub(super) main_num_polys: usize,
    pub(super) source: crate::InnerBasisSource,
}

pub(super) struct RootFinalGroupCandidateInput {
    pub(super) log_basis_inner: u32,
    pub(super) log_basis_open: u32,
    pub(super) position_index_bits: usize,
    pub(super) block_index_bits: usize,
    pub(super) outer_slice_count: akita_params::CommitmentSliceCount,
}

pub(super) fn root_final_group_level_params_candidate(
    ctx: &MultiGroupRootCandidateCtx<'_>,
    input: RootFinalGroupCandidateInput,
) -> Result<Option<CommittedGroupParams>, AkitaError> {
    let RootFinalGroupCandidateInput {
        log_basis_inner,
        log_basis_open,
        position_index_bits,
        block_index_bits,
        outer_slice_count,
    } = input;
    let policy = ctx.policy;
    let dimensions = ctx.dimensions;
    let d_a = dimensions.d_a();
    let decomp = ctx.policy.decomposition;
    let level_decomp = DecompositionParams {
        log_basis: log_basis_open,
        ..decomp
    };
    let num_digits_inner = ctx
        .source
        .num_digits_inner(ctx.policy.decomposition, log_basis_inner)?;
    let num_digits_outer = num_digits_open(level_decomp);
    let num_digits_open = num_digits_outer;
    let Some(num_live_blocks) = checked::pow2(block_index_bits) else {
        return Ok(None);
    };
    let Some(num_positions_per_block) = checked::pow2(position_index_bits) else {
        return Ok(None);
    };
    let Some(num_live_ring_elements_per_claim) =
        num_live_blocks.checked_mul(num_positions_per_block)
    else {
        return Ok(None);
    };
    let Some(width_s) = decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
    else {
        return Ok(None);
    };
    let num_chunks = policy.chunks_at_level(0);
    let witness_norms = ctx
        .final_source_contract
        .source_norms(
            log_basis_inner,
            num_digits_inner,
            d_a,
            akita_error::checked::pow2(ctx.final_num_vars)
                .and_then(|len| len.checked_mul(ctx.main_num_polys))
                .ok_or_else(|| AkitaError::InvalidSetup("source length overflow".into()))?,
        )?
        .fold_witness;
    let Some(ab_candidate) = derive_ab_commitment_candidate(AbCommitmentCandidateRequest {
        policy,
        fold_policy: &ctx
            .final_source_contract
            .class()
            .honest_fold_policy(decomp.field_bits()),
        ring_challenge_cfg: &ctx.opening.challenge_config(),
        challenge_dimension: ctx.opening.challenge_dimension(d_a),
        dimensions,
        payload_mode: akita_params::CommitmentPayloadMode::Compressed,
        num_claims: ctx.main_num_polys,
        num_live_ring_elements_per_claim,
        num_live_blocks,
        num_positions_per_block,
        num_chunks,
        outer_slice_count,
        witness_norms,
        log_basis_open,
        width_s,
        num_digits_outer,
        modeled_linf_cap: None,
    })?
    else {
        return Ok(None);
    };
    let num_digits_fold = ab_candidate.num_digits_fold;
    let inner_commit_matrix = ab_candidate.inner_commit_matrix;
    let outer_commit_matrix = ab_candidate.outer_commit_matrix;

    let Ok(main_d_width) = akita_params::opening_d_segment_width(
        ctx.opening.method(),
        policy.claim_ext_degree,
        d_a,
        dimensions.d_d(),
        num_digits_open,
        num_live_blocks,
        ctx.main_num_polys,
    ) else {
        return Ok(None);
    };
    let Some((open_key, main_d_width)) = projected_collision_role_price(
        policy,
        akita_params::SisMatrixRole::Open,
        dimensions.d_d(),
        dimensions.d_d(),
        main_d_width,
        log_basis_open,
    ) else {
        return Ok(None);
    };
    let Ok(open_commit_matrix) =
        OpenCommitMatrixParams::try_new_with_min_rank(open_key, main_d_width)
    else {
        return Ok(None);
    };

    let groups = vec![akita_params::GroupOpenPhaseParams {
        profile: akita_params::GroupCommitPhaseParams {
            version: akita_params::GroupCommitPhaseParams::VERSION,
            group: akita_params::PolynomialGroupLayout::new(ctx.final_num_vars, ctx.main_num_polys),
            blocks: akita_params::BlockGeometry::new(
                num_live_ring_elements_per_claim,
                num_positions_per_block,
                num_live_blocks,
            ),
            outer_slice_count,
            inner: akita_params::RoleParams::new(
                akita_params::GadgetDigits::new(log_basis_inner, num_digits_inner),
                inner_commit_matrix,
            ),
            outer: akita_params::RoleParams::new(
                akita_params::GadgetDigits::new(log_basis_open, num_digits_outer),
                outer_commit_matrix,
            ),
        },
        opening: akita_params::GroupOpeningPlan {
            opening_method: ctx.opening.method(),
            fold_challenge_config: ctx.opening.challenge_config(),
            log_basis_open,
            num_digits_open,
            num_digits_fold,
        },
        setup_natural_len: None,
    }];
    let params = CommittedGroupParams::try_new(
        groups,
        open_commit_matrix,
        akita_params::CommitmentPayloadMode::Compressed,
        akita_params::RingRelationMode::QuotientLift,
        akita_params::CommittedSourceEncoding::for_producer(
            ctx.opening.method(),
            policy.claim_ext_degree,
            d_a,
            ctx.final_num_vars,
            true,
        ),
        // Root folds use the ordinary single-chunk precommit path before the
        // schedule-level chunk policy is applied.
        akita_params::ChunkedWitnessCfg::default(),
    )?;

    Ok(Some(params))
}
