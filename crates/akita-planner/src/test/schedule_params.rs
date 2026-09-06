use super::*;

fn exhaustive_parent_alignment_cmp(
    left: PackedProofCost,
    right: PackedProofCost,
    compare: impl Fn(usize, usize) -> bool,
) -> bool {
    (0..8).all(|parent_remainder| {
        left.checked_proof_bytes_with_parent_remainder(parent_remainder)
            .zip(right.checked_proof_bytes_with_parent_remainder(parent_remainder))
            .is_some_and(|(left, right)| compare(left, right))
    })
}

#[test]
fn packed_proof_cost_alignment_order_matches_exhaustive_comparison() {
    let mut costs = Vec::new();
    for payload_bytes in 0..=3 {
        for nonce_bits in 0..=24 {
            costs.push(PackedProofCost::new(payload_bytes, nonce_bits, 0).unwrap());
        }
    }
    costs.push(PackedProofCost::new(usize::MAX, 0, 0).unwrap());

    for &left in &costs {
        for &right in &costs {
            assert_eq!(
                left.never_worse_for_every_parent(right),
                exhaustive_parent_alignment_cmp(left, right, |left, right| left <= right),
            );
            assert_eq!(
                left.strictly_better_for_every_parent(right),
                exhaustive_parent_alignment_cmp(left, right, |left, right| left < right),
            );
        }
    }
}

#[test]
fn packed_proof_cost_rejects_only_query_budget_exhaustion() {
    let limit = akita_types::TRANSCRIPT_GRINDING_QUERY_LIMIT;
    let empty = PackedProofCost::new(0, 0, 0).unwrap();
    assert_eq!(empty.checked_prepend(0, 0, limit).unwrap(), None);

    let individually_valid_suffix = PackedProofCost::new(0, 0, limit - 2).unwrap();
    assert_eq!(
        individually_valid_suffix.checked_prepend(0, 0, 2).unwrap(),
        None,
        "two individually valid edge totals can exhaust the complete budget"
    );
    assert!(matches!(
        individually_valid_suffix.checked_prepend(0, 0, u64::MAX),
        Err(AkitaError::InvalidSetup(_))
    ));
    assert!(matches!(
        PackedProofCost::new(usize::MAX, 0, 0)
            .unwrap()
            .checked_prepend(1, 0, limit),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn oversized_candidate_is_skipped_while_valid_alternative_is_retained() {
    let limit = akita_types::TRANSCRIPT_GRINDING_QUERY_LIMIT;
    let suffix = PackedProofCost::new(0, 0, 0).unwrap();
    let selected = [(10, limit), (20, 1)]
        .into_iter()
        .filter_map(|(payload_bytes, queries)| {
            suffix.checked_prepend(payload_bytes, 0, queries).unwrap()
        })
        .min_by_key(|cost| cost.proof_bytes());

    assert_eq!(selected, Some(PackedProofCost::new(20, 0, 1).unwrap()));
    assert!([limit, limit + 1]
        .into_iter()
        .all(|queries| suffix.checked_prepend(0, 0, queries).unwrap().is_none()));
}

#[test]
fn packed_proof_cost_dominance_requires_no_more_queries() {
    let smaller_proof_more_queries = PackedProofCost::new(9, 0, 11).unwrap();
    let larger_proof_fewer_queries = PackedProofCost::new(10, 0, 10).unwrap();

    assert!(!smaller_proof_more_queries.never_worse_for_every_parent(larger_proof_fewer_queries));
    assert!(
        !smaller_proof_more_queries.strictly_better_for_every_parent(larger_proof_fewer_queries)
    );
    assert!(!larger_proof_fewer_queries.never_worse_for_every_parent(smaller_proof_more_queries));
}

#[test]
fn dyadic_chunk_geometry_prices_exact_work_and_residual_imbalance() {
    assert_eq!(
        layout_candidate_score(100, 13, 4).unwrap(),
        (127, 100, 13, 1)
    );
    assert_eq!(
        layout_candidate_score(100, 12, 4).unwrap(),
        (124, 100, 12, 0)
    );
    assert_eq!(layout_candidate_score(100, 4, 8).unwrap(), (109, 100, 4, 1));
    for (blocks, chunks) in [(0, 1), (8, 0), (8, 3), (8, 128)] {
        assert!(layout_candidate_score(100, blocks, chunks).is_err());
    }
}

#[test]
fn ring_dimension_domain_is_canonical_and_rejects_invalid_carriers() {
    let domain = RingDimensionSearchDomain::new([
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        },
        CommitmentRingDims::uniform(64),
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        },
    ])
    .unwrap();
    assert_eq!(
        domain.candidates(),
        &[
            CommitmentRingDims::uniform(64),
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 64
            },
        ]
    );
    assert!(RingDimensionSearchDomain::new([CommitmentRingDims {
        inner: 64,
        outer: 128,
        opening: 64
    }])
    .is_err());
    assert!(RingDimensionSearchDomain::new([CommitmentRingDims::uniform(256)]).is_ok());
}

#[cfg(feature = "catalog-gen")]
#[test]
fn setup_first_slice_pruning_uses_the_padded_direct_prefix() {
    use akita_config::{policy_of, proof_optimized::fp32::OneHot};
    use akita_types::{CommitmentSliceCount, SisModulusProfileId};

    let mut policy = policy_of::<OneHot>();
    policy.selection_policy = crate::SelectionPolicyId::MinFirstDirectSetupThenPayloadV2;
    let params_for = |outer_slice_count| {
        let mut params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q32Offset99,
            64,
            3,
            2,
            8,
            2,
            SparseChallengeConfig::pm1_only(3),
        );
        params.own_group_mut().profile.outer_slice_count = outer_slice_count;
        params.with_decomp(1, 64, 2, 2, 2).expect("slice candidate")
    };
    let opening_layout = OpeningClaimsLayout::new(6, 1).expect("opening layout");
    let candidates = [
        CommitmentSliceCount::FOUR,
        CommitmentSliceCount::ONE,
        CommitmentSliceCount::TWO,
    ]
    .map(params_for)
    .into_iter()
    .collect();

    let selected =
        prune_locally_unprofitable_slices(&policy, &opening_layout, candidates).expect("pruning");
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].outer_slice_count(), CommitmentSliceCount::FOUR);
    let selected_capacity = padded_setup_prefix_len(
        active_setup_field_len(&selected[0], &opening_layout).expect("selected setup prefix"),
    );
    for other in [CommitmentSliceCount::ONE, CommitmentSliceCount::TWO] {
        let other_params = params_for(other);
        let other_capacity = padded_setup_prefix_len(
            active_setup_field_len(&other_params, &opening_layout).expect("other setup prefix"),
        );
        assert!(selected_capacity <= other_capacity);
    }
}
