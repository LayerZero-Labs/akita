use super::*;

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
    policy.selection_policy = crate::SelectionPolicyId::MinFirstDirectSetupThenPayload;
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
