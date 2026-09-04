use super::*;

/// Enumerate every root split and slice for the single-group oracle fixture.
/// Candidate materialization stays canonical, while this reference domain is
/// independent of production split bounds and local slice pruning.
pub(crate) fn exhaustive_root_candidates_for_reference(
    key: &AkitaScheduleLookupKey,
    final_honest_fold_policy: HonestFoldPolicySpec,
    policy: &PlannerPolicy,
    dimensions: CommitmentRingDims,
    opening: PlannerOpeningCandidate,
    candidate_log_basis_inner: u32,
    candidate_log_basis_open: u32,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    key.validate(policy.decomposition.field_bits())?;
    dimensions.validate_role_projection()?;
    opening.validate_for(0, policy.claim_ext_degree, dimensions)?;
    if !key.precommitteds.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "the exhaustive root reference supports one uncommitted group".into(),
        ));
    }
    let alpha = dimensions.d_a().trailing_zeros() as usize;
    let reduced_vars = key.final_group.num_vars().saturating_sub(alpha);
    if reduced_vars == 0 {
        return Ok(Vec::new());
    }
    let candidate_ctx = MultiGroupRootCandidateCtx {
        policy,
        dimensions,
        opening,
        final_honest_fold_policy,
        final_num_vars: key.final_group.num_vars(),
        main_num_polys: key.final_group.num_polynomials(),
        source: crate::schedule_params::root_inner_basis_source(
            final_honest_fold_policy,
            policy.decomposition.log_commit_bound,
        ),
    };
    let opening_batch = key.opening_layout()?;
    let min_split = usize::from(reduced_vars >= 3);
    let max_split = (reduced_vars - 1).min(usize::BITS as usize - 1);
    let mut candidates = Vec::new();
    for block_index_bits in (min_split..=max_split).rev() {
        let position_index_bits = reduced_vars - block_index_bits;
        let num_live_blocks = 1usize << block_index_bits;
        for outer_slice_count in akita_types::CommitmentSliceCount::ALL {
            if outer_slice_count
                .validate_for_commitment(
                    0,
                    akita_types::CommitmentPayloadMode::Compressed,
                    num_live_blocks,
                )
                .is_err()
            {
                continue;
            }
            let Some(mut params) = root_final_group_level_params_candidate(
                &candidate_ctx,
                RootFinalGroupCandidateInput {
                    log_basis_inner: candidate_log_basis_inner,
                    log_basis_open: candidate_log_basis_open,
                    position_index_bits,
                    block_index_bits,
                    outer_slice_count,
                    precommitted_groups: &[],
                    precommitted_d_width: 0,
                },
            )?
            else {
                continue;
            };
            params.witness_chunk = crate::policy::witness_chunk_at_level(policy, 0);
            let Some(output_witness_len) = root_batch_next_w_len(
                policy.decomposition.field_bits(),
                policy.claim_ext_degree,
                &params,
                &opening_batch,
            )?
            else {
                continue;
            };
            candidates.push((params, output_witness_len));
        }
    }
    Ok(candidates)
}

#[test]
fn interchangeable_groups_follow_the_materialized_descriptor_order() {
    let fold_challenge_config = akita_challenges::SparseChallengeConfig::pm1_only(3);
    let params = CommittedGroupParams::params_only(
        akita_types::SisModulusProfileId::Q128OffsetA7F7,
        256,
        3,
        2,
        2,
        2,
        fold_challenge_config,
    )
    .with_decomp(2, 2, 2, 2, 2)
    .expect("descriptor-order fixture");
    let profile = GroupCommitPhaseParams {
        version: GroupCommitPhaseParams::VERSION,
        group: PolynomialGroupLayout::singleton(16),
        blocks: params.blocks(),
        outer_slice_count: params.outer_slice_count(),
        inner: akita_types::RoleParams::new(
            akita_types::GadgetDigits::new(
                params.inner().digits.log_basis,
                params.inner().digits.num_digits,
            ),
            params.inner().matrix,
        ),
        outer: akita_types::RoleParams::new(
            akita_types::GadgetDigits::new(
                params.outer().digits.log_basis,
                params.outer().digits.num_digits,
            ),
            params.outer().matrix,
        ),
    };
    let group = |challenge_subring_dimension| GroupOpenPhaseParams {
        profile,
        opening: akita_types::GroupOpeningPlan {
            opening_method: akita_types::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension,
            },
            fold_challenge_config:
                akita_challenges::SparseChallengeConfig::production_for_ring_dim(
                    challenge_subring_dimension,
                )
                .expect("production challenge dimension"),
            log_basis_open: params.open().digits.log_basis,
            num_digits_open: params.open().digits.num_digits,
            num_digits_fold: params.num_digits_fold(),
        },
        setup_natural_len: None,
    };
    let mut groups = vec![group(64), group(256)];
    assert!(
        groups[1].canonical_descriptor_bytes() < groups[0].canonical_descriptor_bytes(),
        "little-endian descriptor order must expose the numeric-order regression"
    );

    canonicalize_interchangeable_precommitted_groups(&mut groups, &[vec![0, 1]])
        .expect("equal-width group descriptors");

    let mut reversed = vec![group(256), group(64)];
    canonicalize_interchangeable_precommitted_groups(&mut reversed, &[vec![0, 1]])
        .expect("equal-width group descriptors");
    assert_eq!(
        groups, reversed,
        "canonicalization must not depend on candidate traversal order"
    );

    let dimensions = groups
        .iter()
        .map(|group| match group.opening.opening_method {
            akita_types::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension,
            } => challenge_subring_dimension,
            akita_types::OpeningMethod::EvaluationTrace => {
                panic!("fixture must remain coefficient packing")
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(dimensions, vec![256, 64]);

    let middle = GroupOpenPhaseParams {
        profile: GroupCommitPhaseParams {
            group: PolynomialGroupLayout::singleton(32),
            ..profile
        },
        ..group(128)
    };
    let root_group_bytes = |groups: &[GroupOpenPhaseParams]| {
        groups
            .iter()
            .flat_map(|group| {
                let mut bytes = group.profile.canonical_descriptor_bytes();
                bytes.extend(group.canonical_descriptor_bytes());
                bytes
            })
            .collect::<Vec<_>>()
    };
    let mut nonadjacent = vec![group(64), middle, group(256)];
    let swapped = vec![group(256), middle, group(64)];
    let expected = root_group_bytes(&nonadjacent).min(root_group_bytes(&swapped));

    canonicalize_interchangeable_precommitted_groups(&mut nonadjacent, &[vec![0, 2], vec![1]])
        .expect("equal-width non-adjacent group descriptors");
    assert_eq!(
        root_group_bytes(&nonadjacent),
        expected,
        "non-adjacent interchangeable groups must minimize the complete root-group descriptor",
    );

    let mut variable_width = vec![group(64), group(256)];
    variable_width[1].opening.opening_method = akita_types::OpeningMethod::EvaluationTrace;
    assert!(matches!(
        canonicalize_interchangeable_precommitted_groups(&mut variable_width, &[vec![0, 1]]),
        Err(AkitaError::InvalidSetup(message))
            if message.contains("descriptors must have one width")
    ));
}
