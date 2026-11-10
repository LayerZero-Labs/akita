use super::*;
use crate::binary::{
    binary_ordinary_profile_fixture, BinaryEvaluationBounds, BinaryIndexMap, BinaryQuotientBounds,
    BinaryRootWitnessPlan, BinarySourceDescriptor, OrdinaryNativeRootPlan,
};
use akita_params::binary_setup::{BinaryFieldProfile, BINARY_SETUP_P64};

fn ordinary_profile(
    field: BinaryFieldProfile,
    variables: usize,
    owners: usize,
) -> akita_params::GroupCommitPhaseParams {
    let mut profile = binary_ordinary_profile_fixture(field);
    profile.group = akita_params::PolynomialGroupLayout::new(variables, owners);
    let rings = (1usize << variables) / profile.inner.matrix.ring_dimension();
    profile.blocks = akita_params::BlockGeometry::new(rings, 2, rings / 2);
    let width = profile
        .derive_slice_geometry()
        .unwrap()
        .physical_input_width();
    profile.outer.matrix = akita_params::sis::OuterCommitMatrixParams::try_new_with_min_rank(
        profile.outer.matrix.sis_table_key(),
        width,
    )
    .unwrap();
    profile
}

fn owner_id(group: usize, owner: usize) -> [u8; 32] {
    let mut id = [0; 32];
    id[..8].copy_from_slice(&(group as u64).to_le_bytes());
    id[8..16].copy_from_slice(&(owner as u64).to_le_bytes());
    id
}

fn occurrence(
    group: usize,
    m: usize,
    columns: usize,
    owners: usize,
    field: BinaryFieldProfile,
    word: bool,
    planes: usize,
) -> BinaryRootGroupOccurrence {
    let carriers = 4 * m * columns;
    let index = BinaryIndexMap::new(128, carriers, carriers, columns).unwrap();
    let profiles =
        BinaryCommittedGroupManifest::profiles_for_field(&index, owners, 0, field).unwrap();
    let descriptor = BinarySourceDescriptor::new(
        index,
        (0..owners).map(|owner| owner_id(group, owner)).collect(),
        profiles,
    )
    .unwrap();
    BinaryRootGroupOccurrence {
        manifest: BinaryCommittedGroupManifest::with_field_profile(
            descriptor,
            0,
            &binary_ordinary_profile_fixture(field),
            field,
        )
        .unwrap(),
        word,
        prime_coordinate_planes: planes,
        registered_commitment: None,
    }
}

fn ordinary(group: usize, field: BinaryFieldProfile) -> OrdinaryRootGroupOccurrence {
    OrdinaryRootGroupOccurrence {
        plan: OrdinaryNativeRootPlan::new(binary_ordinary_profile_fixture(field), 0).unwrap(),
        polynomials: vec![owner_id(group, 0)],
        registered_commitment: None,
    }
}

#[test]
fn admitted_envelope_uses_the_exact_coefficientwise_integer_bounds() {
    let quotient = BinaryQuotientBounds::new(8192, 256).unwrap();
    assert_eq!(quotient.residual_bound, 132_788_824_280_985_600);
    assert_eq!(quotient.digits, 12);
    assert_eq!(quotient.no_wrap_bound, 692_495_571_806_665_380);
    let evaluation =
        BinaryEvaluationBounds::new(4 * 8192, 256, u128::from(BINARY_SETUP_P64)).unwrap();
    assert_eq!(evaluation.parity_honest_carry, 31_921_152);
    assert_eq!(evaluation.parity_no_wrap_bound, 217_233_992);
    assert_eq!(evaluation.value_honest_carry, 62_208);
    assert_eq!(evaluation.value_no_wrap_bound, 797_257);
    assert!(BinaryEvaluationBounds::new(4 * 8192, 256, 217_233_992).is_err());
    assert!(BinaryEvaluationBounds::new(4, 256, 797_257).is_err());
}

#[test]
fn ordered_rosters_share_one_contiguous_layout_and_one_row_vector() {
    for (binary_count, mixed_views) in [
        (2, false),
        (3, false),
        (4, false),
        (2, true),
        (3, true),
        (4, true),
    ] {
        for ordinary_count in [0, 1, 3] {
            let binary = (0..binary_count)
                .map(|group| {
                    occurrence(
                        100 - group, // Declared order is deliberately not ID order.
                        1 << group,
                        1 << (group % 3),
                        group + 1,
                        BinaryFieldProfile::P64,
                        !mixed_views || group % 3 != 1,
                        if mixed_views && group % 3 == 2 {
                            0
                        } else {
                            2 * (group + 1)
                        },
                    )
                })
                .collect::<Vec<_>>();
            let ordinary = (0..ordinary_count)
                .map(|group| ordinary(binary_count + group, BinaryFieldProfile::P64))
                .collect::<Vec<_>>();
            let plan = BinaryRootGroupPlan::new(&binary, &ordinary, 256).unwrap();
            let binary_owners = binary
                .iter()
                .map(|group| group.manifest.descriptor().polynomials().len() as u128)
                .sum::<u128>();
            let ordinary_owners = ordinary
                .iter()
                .map(|group| group.plan.profile().group.num_polynomials() as u128)
                .sum::<u128>();
            assert_eq!(plan.totals().owners, binary_owners + ordinary_owners);
            assert_eq!(plan.totals().residual_coefficients, 1944 * binary_owners);
            assert_eq!(
                plan.totals().binary_input_carriers,
                binary
                    .iter()
                    .map(|group| {
                        let descriptor = group.manifest.descriptor();
                        descriptor.polynomials().len() as u128
                            * descriptor.index_map().padded_carriers() as u128
                    })
                    .sum::<u128>()
            );
            assert_eq!(
                plan.totals().ordinary_input_coefficients,
                ordinary
                    .iter()
                    .map(|group| {
                        let profile = group.plan.profile();
                        profile.group.num_polynomials() as u128
                            * (1u128 << profile.group.num_vars())
                    })
                    .sum::<u128>()
            );
            assert_eq!(
                plan.totals().fold_coordinates,
                binary
                    .iter()
                    .map(|group| {
                        let descriptor = group.manifest.descriptor();
                        descriptor.polynomials().len() as u128
                            * descriptor.index_map().columns() as u128
                    })
                    .sum::<u128>()
            );
            assert_eq!(
                plan.totals().prime_claims,
                binary
                    .iter()
                    .map(|group| (group.prime_coordinate_planes / 2) as u128)
                    .sum::<u128>()
            );
            let repacking = plan.repacking();
            let mut next_segment = 0;
            let mut expected_live = 0;
            let mut next_row = 0;
            for (index, group) in plan.binary().iter().enumerate() {
                assert_eq!(group.global_index, index);
                assert_eq!(
                    group.local_plan.descriptor(),
                    binary[index].manifest.descriptor()
                );
                let local = group.local_plan.repacking();
                let legacy_len = group
                    .local_plan
                    .quotient_segment()
                    .unwrap_or(local.segments().len());
                assert_eq!(
                    group.legacy_segments,
                    next_segment..next_segment + legacy_len
                );
                assert_eq!(
                    &repacking.segments()[group.legacy_segments.clone()],
                    &local.segments()[..legacy_len]
                );
                next_segment = group.legacy_segments.end;
                assert_eq!(group.rows.coefficients.start, next_row);
                assert_eq!(group.rows.coefficients.len(), 1944 * group.owners);
                next_row = group.rows.coefficients.end;
                let depth = group
                    .local_plan
                    .projection_bounds()
                    .quotient()
                    .unwrap()
                    .digits;
                // Closed form from the witness-layout table, independently
                // checked against the physical segment sum below.
                let length = 4096 * group.owners * group.ring_rows
                    + 24576 * group.owners * group.columns
                    + usize::from(group.word) * 256 * group.owners * (group.columns + 9 + 6)
                    + 5632 * group.prime_coordinate_planes * group.columns
                    + 3072 * group.owners * depth;
                assert_eq!(group.physical_len, length);
                expected_live += length;
            }
            for (index, group) in plan.ordinary().iter().enumerate() {
                assert_eq!(group.global_index, binary_count + index);
                assert_eq!(group.polynomials, ordinary[index].polynomials);
                assert_eq!(group.segments, next_segment..next_segment + 4);
                next_segment += 4;
                let p = group.plan.profile();
                let compression = p.outer_compression_plan().unwrap();
                let maps = compression.maps();
                let length = p.group.num_polynomials()
                    * p.blocks.live_blocks
                    * p.blocks.positions_per_block
                    * p.inner.digits.num_digits
                    * p.inner.matrix.ring_dimension()
                    + p.group.num_polynomials()
                        * p.blocks.live_blocks
                        * p.inner.matrix.output_rank()
                        * (p.inner.matrix.ring_dimension() / p.outer.matrix.ring_dimension())
                        * p.outer.digits.num_digits
                        * p.outer.matrix.ring_dimension()
                    + maps[0].input_width() * maps[0].ring_dimension()
                    + maps[1].input_width() * maps[1].ring_dimension();
                assert_eq!(group.physical_len, length);
                expected_live += length;
            }
            for group in plan.binary() {
                assert_eq!(group.quotient_segment, Some(next_segment));
                next_segment += 1;
                for (range, count) in [
                    (&group.rows.word, 2 * usize::from(group.word)),
                    (&group.rows.b, binary[group.global_index].manifest.b_rows()),
                    (
                        &group.rows.d,
                        binary[group.global_index]
                            .manifest
                            .d_label(group.word, group.prime_coordinate_planes)
                            .unwrap()
                            .rank as usize,
                    ),
                    (
                        &group.rows.prime,
                        2 * group.prime_coordinate_planes
                            / group.local_plan.field_profile().extension_degree(),
                    ),
                ] {
                    assert_eq!(range.start, next_row);
                    assert_eq!(range.len(), count);
                    next_row = range.end;
                }
            }
            for group in plan.ordinary() {
                assert_eq!(group.rows.linear.start, next_row);
                assert_eq!(group.rows.linear.len(), group.plan.linear_batch_count());
                next_row = group.rows.linear.end;
                assert_eq!(group.rows.negative_bits.start, next_row);
                assert_eq!(
                    group.rows.negative_bits.len(),
                    group
                        .plan
                        .negative_binary_addresses(repacking)
                        .unwrap()
                        .len()
                );
                next_row = group.rows.negative_bits.end;
            }
            assert_eq!(next_row, plan.batch_count());
            assert_eq!(next_segment, repacking.segments().len());
            let mut end = 0;
            for (segment, shape) in repacking.segments().iter().enumerate() {
                let range = repacking.segment_range(segment).unwrap();
                assert_eq!(range.start, end);
                assert_eq!(
                    range.len(),
                    shape.elements * shape.digits * shape.geometry.physical_coefficient_width()
                );
                end = range.end;
            }
            assert_eq!(end, expected_live);
            assert_eq!(
                repacking.domain_len(),
                expected_live.max(256).next_power_of_two()
            );
        }
    }
}

#[test]
fn translations_cross_boundaries_with_unaligned_group_and_quotient_starts() {
    let binary = [
        occurrence(0, 1, 2, 1, BinaryFieldProfile::P64, true, 2),
        occurrence(1, 4, 1, 2, BinaryFieldProfile::P64, false, 2),
        // Keep the quotient bank nonaligned with two coordinate planes per claim.
        occurrence(2, 2, 2, 1, BinaryFieldProfile::P64, true, 0),
    ];
    let plan =
        BinaryRootGroupPlan::new(&binary, &[ordinary(3, BinaryFieldProfile::P64)], 256).unwrap();
    let mut unaligned = false;
    let mut crosses_boundary = false;
    for group in plan.binary() {
        let local = group.local_plan.repacking();
        for (segment, shape) in local.segments().iter().enumerate() {
            let local_range = local.segment_range(segment).unwrap();
            let start = plan
                .coefficient_address(group.global_index, segment, 0, 0, 0)
                .unwrap();
            unaligned |= !start.is_multiple_of(1024);
            let last = plan
                .coefficient_address(
                    group.global_index,
                    segment,
                    shape.elements - 1,
                    shape.digits - 1,
                    shape.geometry.polynomial_modulus_dimension() - 1,
                )
                .unwrap();
            crosses_boundary |= start > 0 && start.ilog2() != last.ilog2();
            for element in [0, shape.elements - 1] {
                for digit in [0, shape.digits - 1] {
                    for coefficient in [0, shape.geometry.polynomial_modulus_dimension() - 1] {
                        let expected = start
                            + local
                                .coefficient_address(segment, element, digit, coefficient)
                                .unwrap()
                            - local_range.start;
                        let address = plan
                            .coefficient_address(
                                group.global_index,
                                segment,
                                element,
                                digit,
                                coefficient,
                            )
                            .unwrap();
                        assert_eq!(address, expected);
                        assert!(!plan.repacking().is_padding_address(address).unwrap());
                    }
                }
            }
        }
    }
    assert!(unaligned);
    assert!(crosses_boundary);
    let ordinary = &plan.ordinary()[0];
    for (segment, shape) in ordinary.plan.segments().iter().enumerate() {
        assert_eq!(
            plan.coefficient_address(ordinary.global_index, segment, 0, 0, 0)
                .unwrap(),
            plan.repacking()
                .segment_range(ordinary.segments.start + segment)
                .unwrap()
                .start
        );
        assert!(plan
            .coefficient_address(ordinary.global_index, segment, shape.elements, 0, 0)
            .is_err());
    }
    let mut next = plan
        .repacking()
        .segment_range(ordinary.segments.end - 1)
        .unwrap()
        .end;
    for group in plan.binary() {
        let range = plan
            .repacking()
            .segment_range(group.quotient_segment.unwrap())
            .unwrap();
        assert_eq!(range.start, next);
        assert!(!range.start.is_multiple_of(1024));
        next = range.end;
    }
    for (group, segment, element, digit, coefficient) in [
        (4, 0, 0, 0, 0),
        (0, usize::MAX, 0, 0, 0),
        (0, 0, usize::MAX, 0, 0),
        (0, 0, 0, usize::MAX, 0),
        (0, 0, 0, 0, 648),
    ] {
        assert!(plan
            .coefficient_address(group, segment, element, digit, coefficient)
            .is_err());
    }
}

#[test]
fn singleton_layout_and_batch_accounting_equal_the_predecessor_in_both_fields() {
    for field in [BinaryFieldProfile::P128, BinaryFieldProfile::P64] {
        for (m, columns, owners) in [(1, 1, 1), (1, 2, 2), (4, 8, 3), (128, 32, 1)] {
            for (word, claims) in [(true, 0), (false, 1), (true, 1)] {
                let planes = claims * field.extension_degree();
                let input = occurrence(0, m, columns, owners, field, word, planes);
                for mixed in [false, true] {
                    let mut predecessor = BinaryRootWitnessPlan::for_profile(
                        input.manifest.descriptor(),
                        input.manifest.a_rank(),
                        256,
                        field,
                        word,
                        planes,
                    )
                    .unwrap();
                    let ordinary = mixed.then(|| {
                        let first = predecessor
                            .quotient_segment()
                            .unwrap_or(predecessor.repacking().segments().len());
                        OrdinaryRootGroupOccurrence {
                            plan: OrdinaryNativeRootPlan::new(
                                binary_ordinary_profile_fixture(field),
                                first,
                            )
                            .unwrap(),
                            polynomials: vec![owner_id(1, 0)],
                            registered_commitment: None,
                        }
                    });
                    if let Some(ordinary) = &ordinary {
                        predecessor = predecessor.with_ordinary(&ordinary.plan).unwrap();
                    }
                    let group = BinaryRootGroupPlan::new(
                        std::slice::from_ref(&input),
                        ordinary.as_slice(),
                        256,
                    )
                    .unwrap();
                    assert_eq!(group.repacking(), predecessor.repacking());
                    assert_eq!(
                        group.binary()[0].quotient_segment,
                        predecessor.quotient_segment()
                    );
                    assert_eq!(
                        group.binary()[0].rows.coefficients.len(),
                        predecessor.coefficient_batch_count().unwrap()
                    );
                    let d = input.manifest.d_label(word, planes).unwrap().rank as usize;
                    let mut expected = predecessor.coefficient_batch_count().unwrap()
                        + 2 * usize::from(word)
                        + input.manifest.b_rows()
                        + d
                        + 2 * claims;
                    if let Some(ordinary) = ordinary {
                        expected += ordinary.plan.linear_batch_count()
                            + ordinary
                                .plan
                                .negative_binary_addresses(predecessor.repacking())
                                .unwrap()
                                .len();
                    }
                    assert_eq!(group.batch_count(), expected);
                    let expanded = group
                        .clone()
                        .with_domain_len(group.repacking().domain_len() * 2)
                        .unwrap();
                    assert_eq!(expanded.batch_count(), group.batch_count());
                    assert_eq!(expanded.minimum_domain_len(), group.minimum_domain_len());
                    for segment in 0..group.repacking().segments().len() {
                        assert_eq!(
                            expanded.repacking().segment_range(segment).unwrap(),
                            group.repacking().segment_range(segment).unwrap()
                        );
                    }
                }
            }
        }
    }
}

fn total_fields(totals: &mut BinaryRootGroupTotals) -> [&mut u128; 9] {
    [
        &mut totals.owners,
        &mut totals.binary_input_carriers,
        &mut totals.ordinary_input_coefficients,
        &mut totals.prime_claims,
        &mut totals.fold_coordinates,
        &mut totals.residual_coefficients,
        &mut totals.segment_cap,
        &mut totals.witness_domain,
        &mut totals.batches,
    ]
}

#[test]
fn aggregate_caps_accept_each_boundary_and_reject_its_successor() {
    // Some resource boundaries are shadowed by W <= 2^32 for a complete
    // witness. Exercise each independently at the common admission boundary.
    for (field, cap) in [
        4096,
        1 << 28,
        1 << 28,
        4096,
        1 << 20,
        7_962_624,
        448,
        1 << 32,
        1 << 32,
    ]
    .into_iter()
    .enumerate()
    {
        let mut totals = BinaryRootGroupTotals::default();
        *total_fields(&mut totals)[field] = cap;
        assert!(totals.validate().is_ok());
        *total_fields(&mut totals)[field] = cap + 1;
        assert!(totals.validate().is_err());
        *total_fields(&mut totals)[field] = u128::MAX;
        assert!(totals.validate().is_err());
    }
    let field = BinaryFieldProfile::P64;
    let binary = (0..65)
        .map(|group| occurrence(group, 1, 1, 1, field, true, 2))
        .collect::<Vec<_>>();
    let maximum = BinaryRootGroupPlan::new(&binary[..64], &[], 256).unwrap();
    assert_eq!(maximum.repacking().segments().len(), 448);
    assert_eq!(maximum.totals().segment_cap, 448);
    assert!(BinaryRootGroupPlan::new(&binary, &[], 256).is_err());
    let maximum = occurrence(0, 1, 1, 4096, field, true, 8192);
    let maximum = BinaryRootGroupPlan::new(&[maximum], &[], 256).unwrap();
    assert_eq!(maximum.totals().owners, 4096);
    assert_eq!(maximum.totals().prime_claims, 4096);
    assert_eq!(maximum.binary()[0].prime_coordinate_planes, 8192);
    assert_eq!(maximum.binary()[0].rows.prime.len(), 8192);
    assert_eq!(maximum.totals().residual_coefficients, 7_962_624);
}

#[test]
fn malformed_presence_counts_dimensions_and_domains_reject() {
    let field = BinaryFieldProfile::P64;
    let valid = occurrence(0, 1, 2, 1, field, true, 0);
    assert!(BinaryRootGroupPlan::new(&[], &[], 256).is_err());
    assert!(BinaryRootGroupPlan::new(&[], &[ordinary(1, field)], 256).is_err());
    for (word, planes) in [
        (false, 0),
        (false, 1),
        (true, 1),
        (false, 4),
        (true, 4),
        (true, usize::MAX),
    ] {
        let mut invalid = valid.clone();
        invalid.word = word;
        invalid.prime_coordinate_planes = planes;
        assert!(BinaryRootGroupPlan::new(&[invalid], &[], 256).is_err());
    }
    for (m, columns) in [(8192, 256), (1, 256)] {
        BinaryRootGroupPlan::new(&[occurrence(0, m, columns, 1, field, true, 0)], &[], 256)
            .unwrap();
    }
    for (m, columns) in [(16384, 256), (1, 512)] {
        assert!(BinaryRootGroupPlan::new(
            &[occurrence(0, m, columns, 1, field, true, 0)],
            &[],
            256
        )
        .is_err());
    }
    for destination in [0, 3, usize::MAX] {
        assert!(BinaryRootGroupPlan::new(std::slice::from_ref(&valid), &[], destination).is_err());
    }
    let plan = BinaryRootGroupPlan::new(std::slice::from_ref(&valid), &[], 256).unwrap();
    assert!(plan.clone().with_domain_len(1 << 32).is_ok());
    assert!(plan.clone().with_domain_len(1 << 33).is_err());
    assert!(plan
        .clone()
        .with_domain_len(plan.repacking().domain_len() - 1)
        .is_err());
    let mut invalid = ordinary(1, field);
    invalid.polynomials.clear();
    assert!(BinaryRootGroupPlan::new(std::slice::from_ref(&valid), &[invalid], 256).is_err());
    assert!(BinaryRootGroupPlan::new(
        std::slice::from_ref(&valid),
        &[ordinary(1, BinaryFieldProfile::P128)],
        256
    )
    .is_err());
    assert!(BinaryRootGroupPlan::new(
        &[
            valid,
            occurrence(1, 1, 2, 1, BinaryFieldProfile::P128, true, 0)
        ],
        &[],
        256
    )
    .is_err());
}

#[test]
fn repeated_owner_names_require_the_identical_registered_commitment_occurrence() {
    let field = BinaryFieldProfile::P64;
    let mut first = occurrence(0, 1, 2, 1, field, true, 0);
    assert!(BinaryRootGroupPlan::new(&[first.clone(), first.clone()], &[], 256).is_err());
    first.registered_commitment = Some([31; 32]);
    BinaryRootGroupPlan::new(&[first.clone(), first.clone()], &[], 256).unwrap();
    let mut other = first.clone();
    other.registered_commitment = Some([32; 32]);
    assert!(BinaryRootGroupPlan::new(&[first.clone(), other], &[], 256).is_err());
    let mut changed = occurrence(0, 2, 2, 1, field, true, 0);
    changed.registered_commitment = first.registered_commitment;
    assert!(BinaryRootGroupPlan::new(&[first.clone(), changed], &[], 256).is_err());
    let mut other = ordinary(1, field);
    other.polynomials = first.manifest.descriptor().polynomials().to_vec();
    other.registered_commitment = first.registered_commitment;
    assert!(BinaryRootGroupPlan::new(std::slice::from_ref(&first), &[other], 256).is_err());
    let mut other = ordinary(1, field);
    assert!(BinaryRootGroupPlan::new(
        std::slice::from_ref(&first),
        &[other.clone(), other.clone()],
        256
    )
    .is_err());
    other.registered_commitment = Some([44; 32]);
    BinaryRootGroupPlan::new(&[first], &[other.clone(), other], 256).unwrap();
}

#[test]
fn overflowing_physical_lengths_and_row_sums_return_typed_errors() {
    let input = occurrence(0, 1, 2, 1, BinaryFieldProfile::P64, true, 0);
    let plan = BinaryRootGroupPlan::new(&[input], &[], 256).unwrap();
    let mut segment = plan.repacking().segments()[0];
    segment.elements = usize::MAX;
    segment.digits = usize::MAX;
    assert!(physical_len(&[segment]).is_err());
    assert!(BinaryRootRowCounts {
        coefficients: usize::MAX,
        word: 1,
        b: 0,
        d: 0,
        prime: 0
    }
    .total()
    .is_err());
}

#[test]
fn row_counts_reject_inexact_coordinate_plane_division() {
    let input = occurrence(0, 1, 2, 1, BinaryFieldProfile::P64, true, 0);
    let group = BinaryRootGroupPlan::new(&[input], &[], 256).unwrap();
    let mut plan = group.binary()[0].local_plan.clone();
    // Corrupt the internal count to exercise the shared row boundary directly.
    plan.prime_planes = 1;
    assert!(matches!(
        plan.row_counts(2, 1),
        Err(akita_error::AkitaError::InvalidProof)
    ));
    plan.prime_planes = usize::MAX;
    assert!(plan.row_counts(2, 1).is_err());
}

#[test]
fn singleton_batch_count_matches_the_existing_public_compiler() {
    use crate::binary::{
        BinaryRootPublic, BinaryTerminalClaim, PrimeClaimValue, PrimeGroupClaim, PrimePlaneClaim,
        PrimePlaneManifest,
    };
    let setup = crate::AkitaSetupDescriptor {
        max_num_vars: 0,
        max_num_batched_polys: 0,
        num_field_elements: crate::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
        setup_seed: [41; 32].into(),
    };
    for field in [BinaryFieldProfile::P128, BinaryFieldProfile::P64] {
        for (word, claims) in [(true, 0), (false, 1), (true, 1), (false, 2), (true, 2)] {
            let planes = claims * field.extension_degree();
            let owners = claims.max(1);
            let input = occurrence(0, 1, 2, owners, field, word, planes);
            let descriptor = input.manifest.descriptor();
            let terminal = word.then(|| {
                BinaryTerminalClaim::new(
                    descriptor,
                    descriptor.polynomials()[0],
                    vec![[0; 21]; 2],
                    vec![[0; 21]],
                    [0; 21],
                )
                .unwrap()
            });
            let element = match field {
                BinaryFieldProfile::P128 => PrimeClaimValue::Base(0),
                BinaryFieldProfile::P64 => PrimeClaimValue::Extension([0, 1]),
            };
            let prime = PrimePlaneManifest::with_field_profile(
                descriptor,
                (planes > 0).then(|| PrimeGroupClaim {
                    point: vec![element; 11],
                    evaluations: (0..claims)
                        .map(|polynomial| PrimePlaneClaim {
                            polynomial,
                            value: element,
                        })
                        .collect(),
                }),
                field,
            )
            .unwrap();
            for mixed in [false, true] {
                let ordinary = mixed.then(|| ordinary(1, field));
                let public = BinaryRootPublic::new(
                    &input.manifest,
                    terminal.as_ref(),
                    &prime,
                    &setup,
                    1 << 22,
                    256,
                    ordinary.as_ref().map(|group| *group.plan.profile()),
                )
                .unwrap();
                let group = BinaryRootGroupPlan::new(
                    std::slice::from_ref(&input),
                    ordinary.as_slice(),
                    256,
                )
                .unwrap()
                .with_domain_len(public.plan().repacking().domain_len())
                .unwrap();
                // The singleton retains dev's order: ordinary rows precede prime rows.
                let coefficients = match field {
                    BinaryFieldProfile::P128 => 7,
                    BinaryFieldProfile::P64 => 1944 * owners,
                };
                let word_end = coefficients + 2 * usize::from(word);
                let b_end = word_end + input.manifest.b_rows();
                let d_end = b_end + input.manifest.d_label(word, planes).unwrap().rank as usize;
                let ordinary_end =
                    d_end + ordinary.as_ref().map_or(0, |g| g.plan.linear_batch_count());
                let negative_end = ordinary_end
                    + ordinary
                        .as_ref()
                        .map_or(0, |g| g.plan.negative_binary_count().unwrap());
                let prime_end = negative_end + 2 * claims;
                assert_eq!(
                    public.batch_ranges().unwrap(),
                    [
                        coefficients..word_end,
                        word_end..b_end,
                        b_end..d_end,
                        d_end..ordinary_end,
                        ordinary_end..negative_end,
                        negative_end..prime_end,
                    ]
                );
                assert_eq!(public.batch_count(), prime_end);
                assert_eq!(group.binary()[0].rows.prime.len(), 2 * claims);
                assert_eq!(group.batch_count(), public.batch_count());
                assert_eq!(group.repacking(), public.plan().repacking());
            }
        }
    }
}

#[test]
fn ordinary_exponent_and_mixed_group_boundaries_are_admitted_before_allocation() {
    let field = BinaryFieldProfile::P64;
    let binary = occurrence(0, 1, 2, 1, field, true, 0);
    for variables in [25, 26] {
        let source = OrdinaryRootGroupOccurrence {
            plan: OrdinaryNativeRootPlan::new(ordinary_profile(field, variables, 1), 0).unwrap(),
            polynomials: vec![owner_id(1, 0)],
            registered_commitment: None,
        };
        let result = BinaryRootGroupPlan::new(std::slice::from_ref(&binary), &[source], 256);
        if variables == 25 {
            let plan = result.unwrap();
            assert_eq!(plan.totals().ordinary_input_coefficients, 1 << 25);
        } else {
            assert!(result.is_err());
        }
    }
    let ordinary = (1..=64)
        .map(|group| ordinary(group, field))
        .collect::<Vec<_>>();
    let plan =
        BinaryRootGroupPlan::new(std::slice::from_ref(&binary), &ordinary[..63], 256).unwrap();
    assert_eq!(plan.ordinary().last().unwrap().global_index, 63);
    assert!(BinaryRootGroupPlan::new(std::slice::from_ref(&binary), &ordinary, 256).is_err());
    assert!(BinaryRootGroupPlan::new(std::slice::from_ref(&binary), &[], 1 << 33).is_err());
    // I itself meets its cap, but its T digits exceed the independent table cap.
    let maximum_i = occurrence(0, 1, 256, 4096, field, true, 0);
    assert!(BinaryRootGroupPlan::new(&[maximum_i], &[], 256).is_err());
}

#[test]
fn input_word_capacity_and_width_are_validated_before_roster_construction() {
    let field = BinaryFieldProfile::P64;
    let carrier = binary_ordinary_profile_fixture(field);
    for width in [1, 2, 8, 128] {
        let capacity = 8 * (128 / width);
        for logical in [1, capacity] {
            let index = BinaryIndexMap::new(width, logical, 8, 2).unwrap();
            let profiles =
                BinaryCommittedGroupManifest::profiles_for_field(&index, 1, 0, field).unwrap();
            let descriptor =
                BinarySourceDescriptor::new(index, vec![owner_id(0, 0)], profiles).unwrap();
            let input = BinaryRootGroupOccurrence {
                manifest: BinaryCommittedGroupManifest::with_field_profile(
                    descriptor, 0, &carrier, field,
                )
                .unwrap(),
                word: true,
                prime_coordinate_planes: 0,
                registered_commitment: None,
            };
            let plan = BinaryRootGroupPlan::new(&[input], &[], 256).unwrap();
            assert_eq!(plan.binary()[0].word_width, width);
            assert_eq!(plan.binary()[0].logical_len, logical);
        }
        assert!(BinaryIndexMap::new(width, 0, 8, 2).is_err());
        assert!(BinaryIndexMap::new(width, capacity + 1, 8, 2).is_err());
    }
    for width in [0, 3, 256, usize::MAX] {
        assert!(BinaryIndexMap::new(width, 1, 8, 2).is_err());
    }
}
