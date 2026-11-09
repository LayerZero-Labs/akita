use super::*;
use akita_config::{policy_of, proof_optimized::fp64::Dense, CommitmentConfig};
use akita_params::{
    sis::CommittedSourceContract, BlockGeometry, ChunkedWitnessCfg, CommittedGroupParams,
    PolynomialGroupLayout,
};

fn fixture(
    a_dimensions: &[usize],
) -> (
    PlannerPolicy,
    ScheduleLookupKey,
    Vec<CommittedSourceContract>,
) {
    let policy = policy_of::<Dense>(ChunkedWitnessCfg::default_non_chunked());
    // Only opening-domain enumeration consumes these synthetic profiles.
    let precommitteds = a_dimensions
        .iter()
        .enumerate()
        .map(|(index, &dimension)| {
            let params = CommittedGroupParams::params_only(
                policy.sis_modulus_profile,
                dimension,
                3,
                2,
                2,
                2,
                akita_challenges::SparseChallengeConfig::production_for_ring_dim(dimension)
                    .unwrap(),
            );
            let mut profile = params.own_group().profile;
            profile.group = PolynomialGroupLayout::new(12, index + 1);
            profile.blocks = BlockGeometry::new(4, 2, 2);
            profile
        })
        .collect();
    let key = ScheduleLookupKey {
        root_accumulated_setup_prefix_len: 0,
        final_group: PolynomialGroupLayout::singleton(16),
        precommitteds,
    };
    let contracts = vec![Dense::committed_source_contract().unwrap(); a_dimensions.len()];
    (policy, key, contracts)
}

fn subrings(products: Vec<Vec<PlannerOpeningCandidate>>) -> Vec<Vec<usize>> {
    products
        .into_iter()
        .map(|product| {
            product
                .into_iter()
                .map(|opening| match opening.method() {
                    akita_params::OpeningMethod::SubringCoefficientPacking {
                        challenge_subring_dimension,
                    } => challenge_subring_dimension,
                    _ => unreachable!("packing assignments must use packing"),
                })
                .collect()
        })
        .collect()
}

#[test]
fn exactly_256_assignments_keep_exhaustive_order() {
    let (policy, key, contracts) = fixture(&[256; 8]);
    let actual = subrings(
        packing_precommit_opening_products(
            &policy,
            CommitmentRingDims::uniform(64),
            &key,
            &contracts,
        )
        .unwrap()
        .unwrap(),
    );
    let expected = (0..256)
        .map(|bits| {
            (0..8)
                .rev()
                .map(|i| 64 << ((bits >> i) & 1))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn oversized_and_overflowing_products_defer_to_witness_sizing() {
    for count in [9, usize::BITS as usize + 1] {
        let (policy, key, contracts) = fixture(&vec![256; count]);
        assert!(packing_precommit_opening_products(
            &policy,
            CommitmentRingDims::uniform(64),
            &key,
            &contracts
        )
        .unwrap()
        .is_none());
    }
}

#[test]
fn empty_and_incompatible_domains_keep_their_meaning() {
    for (a_dimensions, expected) in [(vec![], vec![vec![]]), (vec![512, 64], vec![])] {
        let (policy, key, contracts) = fixture(&a_dimensions);
        assert_eq!(
            packing_precommit_opening_products(
                &policy,
                CommitmentRingDims::uniform(64),
                &key,
                &contracts,
            )
            .unwrap(),
            Some(expected),
        );
    }
}
