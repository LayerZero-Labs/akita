use super::*;
use crate::binary_source::{
    grouped_binding_fixture, grouped_sources_fixture, BinarySourceDescriptor, BinaryTypedPrefix,
    GroupedTypedPrefix,
};

#[test]
fn staged_four_group_p64_capacity_binding() {
    let (binary, ordinary) = grouped_sources_fixture();
    let plan = GroupedNativeRootPlan::new(&binary, &ordinary, 128).unwrap();
    let typed = grouped_binding_fixture(&plan);
    typed.validate_plan(&plan).unwrap();
    plan.validate_layout(plan.repacking()).unwrap();
    assert_eq!(plan.binary().len(), 4);
    let mut next_segment = 0;
    for (i, group) in plan.binary().iter().enumerate() {
        assert_eq!(group.source().setup_group, i as u64);
        assert_eq!(group.first_segment(), next_segment);
        assert_eq!(group.source().prime.claims().len(), 2);
        assert_eq!(group.source().prime.planes().len(), 4);
        assert!(group
            .source()
            .prime
            .planes()
            .iter()
            .all(|plane| plane.value == 0));
        let quotient = group.local().quotient_segment().unwrap();
        assert!(!group.local().opening_segments().contains(&quotient));
        for (local, shape) in group.local().repacking().segments().iter().enumerate() {
            let common = group.first_segment() + local;
            assert_eq!(&plan.repacking().segments()[common], shape);
            assert_eq!(
                group.local().repacking().segment_range(local).unwrap().start
                    + group.coefficient_offset(),
                plan.repacking().segment_range(common).unwrap().start,
            );
        }
        next_segment += group.local().repacking().segments().len();
    }
    let batch = plan.ordinary().unwrap();
    for (group, (bits, slots)) in batch
        .plans()
        .iter()
        .zip([(2, 3), (9, 3), (18, 5), (33, 1)])
    {
        assert_eq!(group.first_segment(), next_segment);
        assert_eq!(group.source_width().unwrap().bits(), bits);
        assert_eq!(group.profile().group.num_polynomials(), slots);
        assert_eq!(group.profile().outer.digits.num_digits, 22);
        assert_eq!(group.segments().len(), 5);
        assert_eq!(
            &plan.repacking().segments()[next_segment..next_segment + 5],
            group.segments()
        );
        next_segment += 5;
    }
    assert_eq!(plan.repacking().segments().len(), next_segment);
    assert_eq!(
        typed.outgoing().domain_len(),
        1 << typed.successor().group.num_vars()
    );
    assert_eq!(typed.natural_len(), typed.first().natural_len);
    assert_eq!(typed.identity().len(), 32);
}

#[test]
fn grouped_rejects_reordering() {
    let (mut binary, mut ordinary) = grouped_sources_fixture();
    let plan = GroupedNativeRootPlan::new(&binary, &ordinary, 128).unwrap();
    let typed = grouped_binding_fixture(&plan);
    binary.swap(0, 1);
    let reordered = GroupedNativeRootPlan::new(&binary, &ordinary, 128).unwrap();
    assert!(typed.validate_plan(&reordered).is_err());
    binary.swap(0, 1);
    ordinary.swap(0, 1);
    let reordered = GroupedNativeRootPlan::new(&binary, &ordinary, 128).unwrap();
    assert!(typed.validate_plan(&reordered).is_err());
    ordinary.swap(0, 1);
    let mut claims = binary[0].prime.claims().to_vec();
    claims.reverse();
    binary[0].prime = PrimePlaneManifest::with_field_profile(
        binary[0].manifest.descriptor(),
        claims,
        BinaryFieldProfile::P64,
    )
    .unwrap();
    let reordered = GroupedNativeRootPlan::new(&binary, &ordinary, 128).unwrap();
    assert!(typed.validate_plan(&reordered).is_err());
}

#[test]
fn grouped_rejects_omitted_segment() {
    let (binary, ordinary) = grouped_sources_fixture();
    let plan = GroupedNativeRootPlan::new(&binary, &ordinary, 128).unwrap();
    let typed = grouped_binding_fixture(&plan);
    let omitted = GroupedNativeRootPlan::new(&binary[1..], &ordinary, 128).unwrap();
    assert!(typed.validate_plan(&omitted).is_err());
    let omitted = GroupedNativeRootPlan::new(&binary, &ordinary[1..], 128).unwrap();
    assert!(typed.validate_plan(&omitted).is_err());
    for omitted in 0..plan.repacking().segments().len() {
        let mut segments = plan.repacking().segments().to_vec();
        segments.remove(omitted);
        let supplied = RootCoefficientRepacking::new(segments, 128).unwrap();
        assert!(plan.validate_layout(&supplied).is_err(), "omitted {omitted}");
    }
}

#[test]
fn grouped_rejects_wrong_field() {
    let (mut binary, ordinary) = grouped_sources_fixture();
    binary[0].prime = PrimePlaneManifest::with_field_profile(
        binary[0].manifest.descriptor(),
        [],
        BinaryFieldProfile::P128,
    )
    .unwrap();
    assert!(GroupedNativeRootPlan::new(&binary, &ordinary, 128).is_err());
    let (binary, mut ordinary) = grouped_sources_fixture();
    ordinary[0].0 = crate::binary_source::binary_ordinary_profile_fixture(
        BinaryFieldProfile::P128,
        Some((2, 3)),
    );
    assert!(GroupedNativeRootPlan::new(&binary, &ordinary, 128).is_err());
}

#[test]
fn grouped_rejects_wrong_carrier() {
    let (mut binary, ordinary) = grouped_sources_fixture();
    binary[0].carrier = ordinary[0].0;
    assert!(GroupedNativeRootPlan::new(&binary, &ordinary, 128).is_err());
}

#[test]
fn grouped_rejects_wrong_owner() {
    let (mut binary, ordinary) = grouped_sources_fixture();
    let source = &binary[1];
    let descriptor = BinarySourceDescriptor::new(
        source.manifest.descriptor().index_map().clone(),
        binary[0].manifest.descriptor().polynomials().to_vec(),
        source.manifest.descriptor().profiles().clone(),
    )
    .unwrap();
    binary[1].manifest = BinaryCommittedGroupManifest::with_field_profile(
        descriptor,
        source.setup_group,
        &source.carrier,
        BinaryFieldProfile::P64,
    )
    .unwrap();
    assert!(GroupedNativeRootPlan::new(&binary, &ordinary, 128).is_err());
    let (mut binary, ordinary) = grouped_sources_fixture();
    binary[0].setup_group = 1;
    assert!(GroupedNativeRootPlan::new(&binary, &ordinary, 128).is_err());
}

#[test]
fn grouped_rejects_short_successor() {
    let (binary, ordinary) = grouped_sources_fixture();
    let plan = GroupedNativeRootPlan::new(&binary, &ordinary, 128).unwrap();
    let typed = grouped_binding_fixture(&plan);
    let short = binary[0].carrier;
    short.validate_frozen_precommit(64).unwrap();
    assert!(GroupedTypedPrefix::new(
        &plan,
        short,
        typed.first().clone(),
        typed.setup().clone(),
    )
    .is_err());
}

#[test]
fn grouped_rejects_short_prefix() {
    let (binary, ordinary) = grouped_sources_fixture();
    let plan = GroupedNativeRootPlan::new(&binary, &ordinary, 128).unwrap();
    let typed = grouped_binding_fixture(&plan);
    let mut short = typed.first().clone();
    short.natural_len -= 1;
    assert!(GroupedTypedPrefix::new(
        &plan,
        *typed.successor(),
        short,
        typed.setup().clone(),
    )
    .is_err());
}

#[test]
fn grouped_rejects_short_setup() {
    let (binary, ordinary) = grouped_sources_fixture();
    let plan = GroupedNativeRootPlan::new(&binary, &ordinary, 128).unwrap();
    let typed = grouped_binding_fixture(&plan);
    let mut short = typed.setup().clone();
    short.num_field_elements = 1;
    assert!(GroupedTypedPrefix::new(
        &plan,
        *typed.successor(),
        typed.first().clone(),
        short,
    )
    .is_err());
}

#[test]
fn grouped_identity_binds_zero_coordinates_and_resources() {
    let (mut binary, ordinary) = grouped_sources_fixture();
    let plan = GroupedNativeRootPlan::new(&binary, &ordinary, 128).unwrap();
    let typed = grouped_binding_fixture(&plan);
    let mut claims = binary[0].prime.claims().to_vec();
    claims[1].evaluations[0].value = [0, 1].into();
    binary[0].prime = PrimePlaneManifest::with_field_profile(
        binary[0].manifest.descriptor(),
        claims,
        BinaryFieldProfile::P64,
    )
    .unwrap();
    let changed = GroupedNativeRootPlan::new(&binary, &ordinary, 128).unwrap();
    assert!(typed.validate_plan(&changed).is_err());
    let changed = grouped_binding_fixture(&changed);
    assert_ne!(typed.identity(), changed.identity());
    let mut setup = typed.setup().clone();
    setup.setup_seed = [42; 32].into();
    let changed =
        GroupedTypedPrefix::new(&plan, *typed.successor(), typed.first().clone(), setup).unwrap();
    assert_ne!(typed.identity(), changed.identity());
    let sum = plan
        .binary()
        .iter()
        .map(|group| {
            BinaryTypedPrefix::from_plan(&group.source().manifest, group.local())
                .unwrap()
                .natural_len
        })
        .sum::<usize>();
    assert_eq!(typed.natural_len(), sum);
}
