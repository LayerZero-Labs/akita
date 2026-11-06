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
fn oversized_product_matches_exhaustive_width_ranking_after_shared_seeds() {
    let dimensions = [512, 1024, 512, 256, 512, 512];
    let (policy, mut key, mut contracts) = fixture(&dimensions);
    // Repeated groups do not add choice axes, but must affect the width score.
    for _ in 0..3 {
        key.precommitteds.push(key.precommitteds[0]);
        contracts.push(contracts[0]);
    }
    let actual = subrings(
        packing_precommit_opening_products(
            &policy,
            CommitmentRingDims::uniform(64),
            &key,
            &contracts,
        )
        .unwrap(),
    );

    let domains = dimensions.map(|a| {
        [64, 128, 256, 512]
            .into_iter()
            .filter(|&s| 2 * s <= a)
            .collect::<Vec<_>>()
    });
    let total: usize = domains.iter().map(Vec::len).product();
    assert!(total > MAX_PRECOMMIT_OPENING_PRODUCTS);
    let mut exhaustive = Vec::new();
    for mut ordinal in 0..total {
        let mut assignment = vec![0; domains.len()];
        for (index, domain) in domains.iter().enumerate().rev() {
            assignment[index] = domain[ordinal % domain.len()];
            ordinal /= domain.len();
        }
        // Two live blocks per polynomial, and four copies of the first class.
        let score: usize = assignment
            .iter()
            .enumerate()
            .map(|(i, &s)| (2 * s / 64) * 2 * (i + 1) * if i == 0 { 4 } else { 1 })
            .sum();
        exhaustive.push((score, assignment));
    }
    exhaustive.sort_unstable();
    let mut expected = vec![vec![64; 6], vec![128; 6]];
    for (_, assignment) in exhaustive {
        if !expected.contains(&assignment) {
            expected.push(assignment);
        }
        if expected.len() == MAX_PRECOMMIT_OPENING_PRODUCTS {
            break;
        }
    }
    assert_eq!(actual, expected);
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
fn overflowing_product_count_keeps_bounded_distinct_assignments() {
    let count = usize::BITS as usize + 1;
    let (policy, key, contracts) = fixture(&vec![256; count]);
    let actual = subrings(
        packing_precommit_opening_products(
            &policy,
            CommitmentRingDims::uniform(64),
            &key,
            &contracts,
        )
        .unwrap(),
    );
    assert_eq!(actual.len(), MAX_PRECOMMIT_OPENING_PRODUCTS);
    assert_eq!(actual[0], vec![64; count]);
    assert_eq!(actual[1], vec![128; count]);
    let unique = actual.iter().collect::<std::collections::BTreeSet<_>>();
    assert_eq!(unique.len(), actual.len());
    assert!(actual.iter().all(|assignment| {
        assignment.len() == count && assignment.iter().all(|&s| s == 64 || s == 128)
    }));
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
            expected,
        );
    }
}
