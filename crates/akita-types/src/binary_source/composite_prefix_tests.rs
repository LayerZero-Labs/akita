use super::*;
use crate::binary_source::{binary_ordinary_profile_fixture, binary_word_manifest_fixture};
use akita_algebra::eq_poly::EqPolynomial;
use akita_params::binary_setup::BinaryFieldProfile;
use akita_params::sis::{InnerCommitMatrixParams, OuterCommitMatrixParams};
use akita_params::{BlockGeometry, CommitmentSliceGeometry, PolynomialGroupLayout};
use jolt_algebra::{Ext2, ExtField, Prime64Offset59 as F, Ring, Zero};

type E = Ext2<F>;

pub(crate) fn slot(domain: usize) -> SetupPrefixSlotId {
    let mut profile = binary_ordinary_profile_fixture(BinaryFieldProfile::P64);
    profile.group = PolynomialGroupLayout::singleton(domain.trailing_zeros() as usize);
    profile.inner.digits.num_digits = 22;
    profile.inner.matrix = InnerCommitMatrixParams::try_new_with_min_rank(
        profile
            .inner
            .matrix
            .sis_table_key()
            .expect("fixture inner SIS key"),
        44,
    )
    .unwrap();
    profile.blocks = BlockGeometry::new(domain / 128, 2, domain / 256);
    let width = CommitmentSliceGeometry::try_new(
        profile.outer_slice_count,
        profile.blocks.live_blocks,
        1,
        profile.inner.matrix.output_rank(),
        22,
        128,
        64,
    )
    .unwrap()
    .physical_input_width();
    profile.outer.matrix =
        OuterCommitMatrixParams::try_new_with_min_rank(profile.outer.matrix.sis_table_key(), width)
            .unwrap();
    SetupPrefixSlotId {
        natural_len: domain,
        commitment_profile: profile,
    }
}

fn blocks(domains: &[usize]) -> Vec<CompositePrefixBlock> {
    domains
        .iter()
        .enumerate()
        .map(|(index, &domain_len)| CompositePrefixBlock {
            kind: CompositePrefixKind::Binary,
            index,
            setup_identity: [index as u8; 32],
            local_identity: vec![index as u8, 7],
            domain_len,
        })
        .collect()
}

#[test]
fn placement_selectors_gaps_and_padding_match_dense_extension() {
    let blocks = blocks(&[256, 1024, 512, 256]);
    let plan = CompositePrefixLayout::new(
        &blocks,
        b"root layout with translated addresses",
        &slot(4096),
    )
    .unwrap();
    assert_eq!(
        plan.placements()
            .iter()
            .map(|p| p.offset)
            .collect::<Vec<_>>(),
        [0, 1024, 2048, 2560]
    );
    assert!(
        (plan.domain_len() as u128) < 4 * blocks.iter().map(|b| b.domain_len as u128).sum::<u128>()
    );
    let local = blocks
        .iter()
        .map(|block| {
            (0..block.domain_len)
                .map(|i| F::from_u64((1 + i + 13 * block.index) as u64))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let tables = local.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let table = plan.assemble(&tables, 4096).unwrap();
    assert!(table[256..1024]
        .iter()
        .chain(&table[2816..])
        .all(|&x| x == F::zero()));
    let point = (0..12)
        .map(|i| E::from_base_fn(|c| F::from_u64(17 + i * 31 + c as u64 * 7)))
        .collect::<Vec<_>>();
    let local_evaluations = local
        .iter()
        .zip(&blocks)
        .map(|(table, block)| {
            let weights =
                EqPolynomial::evals(&point[..block.domain_len.trailing_zeros() as usize]).unwrap();
            table
                .iter()
                .zip(weights)
                .fold(E::zero(), |sum, (&x, eq)| sum + E::lift_base(x) * eq)
        })
        .collect::<Vec<_>>();
    let dense = table
        .iter()
        .zip(EqPolynomial::evals(&point).unwrap())
        .fold(E::zero(), |sum, (&x, eq)| sum + E::lift_base(x) * eq);
    assert_eq!(plan.evaluate(&point, &local_evaluations).unwrap(), dense);
    assert!(plan.evaluate(&point[..11], &local_evaluations).is_err());
    assert!(plan.assemble(&tables, 4095).is_err());
    assert!(plan.assemble(&tables[..3], 4096).is_err());
}

#[test]
fn placement_bound_holds_across_heterogeneous_domain_sequences() {
    let mut state = 17u64;
    for count in 1..=64 {
        let domains = (0..count)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                256usize << (state % 7)
            })
            .collect::<Vec<_>>();
        let blocks = blocks(&domains);
        // Independent placement using division rather than the layout helper.
        let end = domains.iter().fold(0usize, |end, &domain| {
            end.div_ceil(domain) * domain + domain
        });
        let plan =
            CompositePrefixLayout::new(&blocks, b"root", &slot(end.next_power_of_two())).unwrap();
        assert!(plan.domain_len() < 4 * domains.iter().sum::<usize>());
        assert_eq!(plan.domain_len(), end.next_power_of_two());
    }
}

#[test]
fn identities_bind_seed_order_role_subset_rank_and_destination() {
    let manifest = binary_word_manifest_fixture(BinaryFieldProfile::P64, 2);
    let typed = BinaryTypedPrefix::new(&manifest, true, 0, 64).unwrap();
    let local_slot = slot(typed.domain_len);
    let first = CompositePrefixBlock::binary(0, &[1; 32], &manifest, &typed, &local_slot).unwrap();
    assert_eq!(
        first.local_identity,
        typed.identity(&manifest, &local_slot).unwrap()
    );
    let mut second = first.clone();
    second.index = 1;
    second.setup_identity = [2; 32];
    let source = [first.clone(), second];
    let composite_slot = slot(typed.domain_len * 2);
    let plan =
        CompositePrefixLayout::new(&source, b"ring64;segments0,16384", &composite_slot).unwrap();
    let different_seed =
        CompositePrefixBlock::binary(0, &[3; 32], &manifest, &typed, &local_slot).unwrap();
    let mut mutations = Vec::new();
    let mut seed = source.clone();
    seed[0] = different_seed;
    mutations.push(seed);
    let mut order = source.clone();
    order.swap(0, 1);
    order[0].index = 0;
    order[1].index = 1;
    mutations.push(order);
    let mut role = source.clone();
    role[1].kind = CompositePrefixKind::Ordinary;
    mutations.push(role);
    for changed in mutations {
        assert_ne!(
            plan.identity(),
            CompositePrefixLayout::new(&changed, b"ring64;segments0,16384", &composite_slot)
                .unwrap()
                .identity()
        );
    }
    for (word, planes, ring) in [(true, 2, 64), (false, 4, 64)] {
        let changed = BinaryTypedPrefix::new(&manifest, word, planes, ring).unwrap();
        let block = CompositePrefixBlock::binary(
            0,
            &[1; 32],
            &manifest,
            &changed,
            &slot(changed.domain_len),
        )
        .unwrap();
        assert_ne!(first.local_identity, block.local_identity);
        let changed_plan =
            CompositePrefixLayout::new(&[block], b"changed root", &slot(changed.domain_len))
                .unwrap();
        assert_ne!(plan.identity(), changed_plan.identity());
    }
    let larger = binary_word_manifest_fixture(BinaryFieldProfile::P64, 160);
    assert_ne!(manifest.b_rows(), larger.b_rows());
    let larger_prefix = BinaryTypedPrefix::new(&larger, true, 0, 64).unwrap();
    let larger_slot = slot(larger_prefix.domain_len);
    let larger_block =
        CompositePrefixBlock::binary(0, &[1; 32], &larger, &larger_prefix, &larger_slot).unwrap();
    assert_ne!(first.local_identity, larger_block.local_identity);
    assert_ne!(
        plan.identity(),
        CompositePrefixLayout::new(&[larger_block], b"root", &larger_slot)
            .unwrap()
            .identity()
    );
    let singleton =
        CompositePrefixLayout::new(std::slice::from_ref(&first), b"root", &local_slot).unwrap();
    assert_eq!(singleton.domain_len(), typed.domain_len);
    assert_eq!(singleton.placements()[0].offset, 0);
    assert_eq!(
        singleton.placements()[0].block.local_identity,
        typed.identity(&manifest, &local_slot).unwrap()
    );
    let table = typed
        .setup_coefficients::<F>(&[1; 32], &manifest, typed.domain_len)
        .unwrap();
    assert_eq!(
        singleton.assemble(&[&table], typed.domain_len).unwrap(),
        table
    );
}

#[test]
fn canonical_root_binding_includes_destination_domains_and_all_group_maps() {
    use crate::root_repacking::{RootCoefficientRepacking, RootRepackingSegment};
    let segments = vec![
        RootRepackingSegment {
            geometry: akita_params::RelationRowGeometry::native(128).unwrap(),
            elements: 2,
            digits: 1,
        };
        2
    ];
    let root = RootCoefficientRepacking::new(segments.clone(), 64).unwrap();
    let maps: &[&[usize]] = &[&[0], &[1]];
    let identity = composite_prefix_root_identity(&root, maps, &[&[0], &[]]).unwrap();
    assert_ne!(
        identity,
        composite_prefix_root_identity(&root, maps, &[&[1], &[]]).unwrap()
    );
    assert!(composite_prefix_root_identity(&root, maps, &[&[0, 0], &[]]).is_err());
    assert!(composite_prefix_root_identity(&root, maps, &[&[4096], &[]]).is_err());
    let ring = RootCoefficientRepacking::new(segments, 128).unwrap();
    assert_ne!(
        identity,
        composite_prefix_root_identity(&ring, maps, &[&[0], &[]]).unwrap()
    );
    let padded = root.clone().with_domain_len(root.domain_len() * 2).unwrap();
    assert_ne!(
        identity,
        composite_prefix_root_identity(&padded, maps, &[&[0], &[]]).unwrap()
    );
    assert_ne!(
        identity,
        composite_prefix_root_identity(&root, &[&[1], &[0]], &[&[0], &[]]).unwrap()
    );
    assert!(composite_prefix_root_identity(&root, &[&[0], &[0]], &[&[0], &[]]).is_err());
    assert!(composite_prefix_root_identity(&root, &[&[0]], &[&[0]]).is_err());
    assert!(composite_prefix_root_identity(&root, &[&[0], &[2]], &[&[0], &[]]).is_err());
}

#[test]
fn malformed_rosters_domains_caps_and_counts_reject_without_tables() {
    let valid_slot = slot(256);
    for domain in [0, 255, 257] {
        let invalid = blocks(&[domain]);
        assert!(CompositePrefixLayout::new(&invalid, b"root", &valid_slot).is_err());
    }
    assert!(CompositePrefixLayout::new(&[], b"root", &valid_slot).is_err());
    assert!(CompositePrefixLayout::new(&blocks(&[256; 65]), b"root", &valid_slot).is_err());
    assert!(CompositePrefixLayout::new(&blocks(&[usize::MAX]), b"root", &valid_slot).is_err());
    assert!(CompositePrefixLayout::new(
        &blocks(&[1usize << 32, 1usize << 32]),
        b"root",
        &valid_slot
    )
    .is_err());
    assert!(CompositePrefixLayout::new(&blocks(&[256]), b"", &valid_slot).is_err());
    let mut invalid = blocks(&[256]);
    invalid[0].index = 1;
    assert!(CompositePrefixLayout::new(&invalid, b"root", &valid_slot).is_err());
    invalid[0].index = 0;
    invalid[0].kind = CompositePrefixKind::Ordinary;
    assert!(CompositePrefixLayout::new(&invalid, b"root", &valid_slot).is_err());
    invalid[0].kind = CompositePrefixKind::Binary;
    invalid[0].local_identity = vec![0; MAX_IDENTITY_BYTES - 1];
    assert!(CompositePrefixLayout::new(&invalid, b"r", &valid_slot).is_err());
    invalid[0].local_identity = vec![0; MAX_IDENTITY_BYTES];
    assert!(CompositePrefixLayout::new(&invalid, b"root", &valid_slot).is_err());
    let mut invalid = blocks(&[256, 256, 256]);
    invalid[1].kind = CompositePrefixKind::Ordinary;
    assert!(CompositePrefixLayout::new(&invalid, b"root", &slot(1024)).is_err());
}

#[test]
fn corrupted_local_lengths_reject_before_composite_preparation() {
    let manifest = binary_word_manifest_fixture(BinaryFieldProfile::P64, 1);
    let typed = BinaryTypedPrefix::new(&manifest, true, 0, 64).unwrap();
    let mut shortened = typed.clone();
    shortened.domain_len = 256;
    assert!(CompositePrefixBlock::binary(0, &[1; 32], &manifest, &shortened, &slot(256)).is_err());
    let mut truncated = typed.clone();
    truncated.natural_len -= 1;
    assert!(CompositePrefixBlock::binary(
        0,
        &[1; 32],
        &manifest,
        &truncated,
        &slot(typed.domain_len)
    )
    .is_err());
}
