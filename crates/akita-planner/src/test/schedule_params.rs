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
