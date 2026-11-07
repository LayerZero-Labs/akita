use super::*;
use crate::binary_source::root_public::reference::residual_envelope;
use crate::binary_source::{BinaryCommitmentProfiles, BinaryIndexMap, CANONICAL_IMAGE_WINDOW};
use akita_challenges::{Challenges, SparseChallenge};
use jolt_field::{One, Prime128Offset275, Ring, Zero};

type F = Prime128Offset275;

fn descriptor() -> BinarySourceDescriptor {
    BinarySourceDescriptor::new(
        BinaryIndexMap::new(128, 8, 8, 2).unwrap(),
        vec![[1; 32]],
        BinaryCommitmentProfiles {
            a: [2; 32],
            b: [3; 32],
        },
    )
    .unwrap()
}

#[test]
fn five_segments_share_addresses_when_scheduled_domain_grows() {
    let descriptor = descriptor();
    let plan =
        BinaryRootWitnessPlan::for_profile(&descriptor, 2, 256, BinaryFieldProfile::P128, true, 0)
            .unwrap();
    let index = descriptor.index_map();
    let rows = index.padded_carriers() / index.columns() / 4;
    assert_eq!(
        plan.repacking()
            .segments()
            .iter()
            .map(|shape| (shape.elements, shape.digits))
            .collect::<Vec<_>>(),
        vec![
            (descriptor.polynomials().len() * rows, 4),
            (
                descriptor.polynomials().len() * index.columns() * plan.rank(),
                8
            ),
            (descriptor.polynomials().len() * index.columns(), 1),
            (descriptor.polynomials().len(), BINARY_PARITY_CARRY_DIGITS),
            (descriptor.polynomials().len(), BINARY_VALUE_CARRY_DIGITS),
        ]
    );
    let live = plan
        .repacking()
        .segments()
        .iter()
        .map(|shape| shape.elements * shape.digits * shape.geometry.physical_coefficient_width())
        .sum::<usize>();
    assert_eq!(plan.repacking().domain_len(), live.next_power_of_two());
    let larger = plan.clone().with_domain_len(131072).unwrap();
    for segment in 0..5 {
        assert_eq!(
            plan.repacking().segment_range(segment).unwrap(),
            larger.repacking().segment_range(segment).unwrap()
        );
    }
    assert!(larger.repacking().is_padding_address(131071).unwrap());
    assert!(plan.clone().with_domain_len(32768).is_err());
    assert!(plan.with_domain_len(65537).is_err());
    assert!(BinaryRootWitnessPlan::for_profile(
        &descriptor,
        0,
        256,
        BinaryFieldProfile::P128,
        true,
        0
    )
    .is_err());
}

#[test]
fn direct_weights_join_a_and_terminal_carries_but_not_extended_padding() {
    let descriptor = descriptor();
    let plan =
        BinaryRootWitnessPlan::for_profile(&descriptor, 2, 256, BinaryFieldProfile::P128, true, 0)
            .unwrap()
            .with_domain_len(131072)
            .unwrap();
    let claim = BinaryTerminalClaim::new(
        &descriptor,
        [1; 32],
        vec![[0; 21]; 2],
        vec![[0; 21]],
        [0; 21],
    )
    .unwrap();
    let challenges = Challenges::from_sparse(
        vec![
            SparseChallenge {
                positions: Default::default(),
                coeffs: Default::default()
            };
            2
        ],
        2,
        1,
    )
    .unwrap();
    let matrix = vec![vec![[0i128; 648]]; 2];
    let weights = plan
        .relation_weights(
            Some(&claim),
            Some(BinaryAOperand::new::<F>(&matrix)),
            &challenges,
            F::from_u64(17),
            &[[F::one(); 648]; 2],
            [F::from_u64(19), F::from_u64(23)],
        )
        .unwrap();
    let table = vec![F::zero(); plan.repacking().domain_len()];
    weights.check_table(&table).unwrap();
    for address in [
        plan.repacking().coefficient_address(2, 0, 0, 0).unwrap(),
        plan.repacking().coefficient_address(3, 0, 0, 0).unwrap(),
        plan.repacking().coefficient_address(4, 0, 0, 0).unwrap(),
    ] {
        let mut altered = table.clone();
        altered[address] = F::one();
        assert!(weights.check_table(&altered).is_err());
    }
    // No root row reads the enlarged domain's padding.
    assert!(plan.repacking().is_padding_address(131071).unwrap());
    let mut altered = table.clone();
    altered[131071] = F::one();
    weights.check_table(&altered).unwrap();
    let mut invalid_matrix = matrix.clone();
    invalid_matrix[0][0][0] = i128::from(CANONICAL_IMAGE_WINDOW.1) + 1;
    assert!(binary_direct_a_weights(
        &plan,
        Some(BinaryAOperand::new::<F>(&invalid_matrix)),
        &challenges,
        &[[F::one(); 648]; 2],
    )
    .is_err());
}

#[test]
fn p64_quotient_bounds_derive_minimal_depth_and_full_accepted_lift() {
    use crate::binary_source::{BinaryQuotientBounds, BINARY_SOURCE_Q};
    use akita_params::binary_setup::{BinaryFieldProfile, BINARY_SETUP_P64};
    let reach = |digits: usize, digit: u128| {
        (0..digits)
            .map(|i| digit * 8u128.pow(i as u32))
            .sum::<u128>()
    };
    for (m, c) in [(1, 2), (8192, 256)] {
        let l = residual_envelope(m, c);
        let h = l / u128::from(BINARY_SOURCE_Q);
        let depth = (1..).find(|&d| reach(d, 3) >= h).unwrap();
        let bound = reach(depth, 4);
        let lift = l + u128::from(BINARY_SOURCE_Q) * bound;
        let bounds = BinaryQuotientBounds::new(m, c).unwrap();
        assert_eq!(bounds.residual_bound, l);
        assert_eq!(bounds.quotient_floor, h);
        assert_eq!(bounds.digits, depth);
        assert_eq!(bounds.positive_reach, reach(depth, 3));
        assert_eq!(bounds.quotient_bound, bound);
        assert_eq!(bounds.no_wrap_bound, lift);
        assert!(lift < u128::from(BINARY_SETUP_P64));
        assert!(reach(depth - 1, 3) < h);
        assert!(reach(depth, 3) >= h);
        // Owner count does not enter the coefficientwise lift or the y cap.
        for owners in [1, 2, 4096] {
            let selected =
                BinaryResidualBounds::for_profile(m, c, 3, owners, BinaryFieldProfile::P64)
                    .unwrap();
            assert_eq!(selected.quotient(), Some(&bounds));
            assert_eq!(selected.projection_count(), 0);
            assert!(crate::binary_source::encode_binary_projection(0, &selected).is_err());
            assert!(crate::binary_source::decode_binary_projection(&[0; 12], &selected).is_err());
        }
    }
    for log_m in 0..=13 {
        for log_c in 0..=8 {
            let (m, c) = (1 << log_m, 1 << log_c);
            BinaryEvaluationBounds::new(4 * m, c, u128::from(BINARY_SETUP_P64)).unwrap();
            BinaryQuotientBounds::new(m, c).unwrap();
        }
    }
    for (m, c) in [
        (0, 2),
        (1, 0),
        (usize::MAX, 2),
        (1, usize::MAX),
        (1 << 30, 256),
    ] {
        assert!(BinaryQuotientBounds::new(m, c).is_err());
    }
}

fn descriptor_geometry(m: usize, columns: usize, owners: usize) -> BinarySourceDescriptor {
    BinarySourceDescriptor::new(
        BinaryIndexMap::new(128, 8, 4 * m * columns, columns).unwrap(),
        (0..owners)
            .map(|owner| {
                let mut id = [0; 32];
                id[..8].copy_from_slice(&(owner as u64).to_le_bytes());
                id
            })
            .collect(),
        BinaryCommitmentProfiles {
            a: [2; 32],
            b: [3; 32],
        },
    )
    .unwrap()
}

#[test]
fn p64_quotients_are_last_and_preserve_existing_natural_addresses() {
    use akita_params::binary_setup::BinaryFieldProfile;
    for (m, columns) in [(1, 2), (8192, 256)] {
        let required =
            residual_envelope(m, columns) / u128::from(crate::binary_source::BINARY_SOURCE_Q);
        let depth = (1..)
            .find(|&d| (0..d).map(|i| 3 * 8u128.pow(i)).sum::<u128>() >= required)
            .unwrap() as usize;
        for owners in [1, 2, 4096] {
            let descriptor = descriptor_geometry(m, columns, owners);
            for (word, planes) in [(true, 0), (false, 2), (true, 2)] {
                let plan = BinaryRootWitnessPlan::for_profile(
                    &descriptor,
                    3,
                    256,
                    BinaryFieldProfile::P64,
                    word,
                    planes,
                )
                .unwrap();
                let segment = plan.quotient_segment().unwrap();
                assert_eq!(segment, if word { 5 } else { 2 } + usize::from(planes > 0));
                let shape = plan.repacking().segments()[segment];
                assert_eq!((shape.elements, shape.digits), (3 * owners, depth));
                assert_eq!(shape.geometry.polynomial_modulus_dimension(), 648);
                assert_eq!(shape.geometry.physical_coefficient_width(), 1024);
                let range = plan.repacking().segment_range(segment).unwrap();
                if word && planes == 0 {
                    let response = m * 4 * 1024;
                    let image = columns * 3 * 8 * 1024;
                    let word = columns * 256;
                    let carries = (BINARY_PARITY_CARRY_DIGITS + BINARY_VALUE_CARRY_DIGITS) * 256;
                    assert_eq!(range.start, (response + image + word + carries) * owners);
                }
                assert_eq!(range.len(), 3 * owners * depth * 1024);
                assert_eq!(plan.coefficient_batch_count().unwrap(), 1944 * owners);
                assert_eq!(plan.projection_bounds().projection_count(), 0);
                assert!(!plan.opening_segments().contains(&segment));
                for owner in [0, owners - 1] {
                    for row in 0..3 {
                        for digit in [0, depth - 1] {
                            for coefficient in [0, 647] {
                                let element = owner * 3 + row;
                                let address = plan
                                    .repacking()
                                    .coefficient_address(segment, element, digit, coefficient)
                                    .unwrap();
                                assert_eq!(
                                    address,
                                    range.start + (element * depth + digit) * 1024 + coefficient
                                );
                                assert!(!plan.repacking().is_padding_address(address).unwrap());
                            }
                        }
                    }
                }
                assert!(plan
                    .repacking()
                    .coefficient_address(segment, 3 * owners, 0, 0)
                    .is_err());
                assert!(plan
                    .repacking()
                    .coefficient_address(segment, 0, depth, 0)
                    .is_err());
                assert!(plan
                    .repacking()
                    .coefficient_address(segment, 0, 0, 648)
                    .is_err());
            }
        }
    }
    let legacy = BinaryRootWitnessPlan::for_profile(
        &descriptor(),
        3,
        256,
        BinaryFieldProfile::P128,
        true,
        0,
    )
    .unwrap();
    assert_eq!(legacy.field_profile(), BinaryFieldProfile::P128);
    assert_eq!(legacy.quotient_segment(), None);
    assert_eq!(
        legacy.projection_bounds().projection_count(),
        crate::binary_source::BINARY_PROJECTION_COUNT
    );
    assert_eq!(
        legacy.coefficient_batch_count().unwrap(),
        crate::binary_source::BINARY_PROJECTION_COUNT
    );
    let mut offset = 0;
    for (segment, shape) in legacy.repacking().segments().iter().enumerate() {
        let length = shape.elements * shape.digits * shape.geometry.physical_coefficient_width();
        assert_eq!(
            legacy.repacking().segment_range(segment).unwrap(),
            offset..offset + length
        );
        offset += length;
    }
    assert_eq!(legacy.repacking().domain_len(), offset.next_power_of_two());
    let residual = residual_envelope(1, 2);
    assert_eq!(legacy.projection_bounds().residual_bound(), residual);
    assert_eq!(
        legacy.projection_bounds().projection_bound(),
        u128::from(crate::binary_source::BINARY_SOURCE_Q - 1)
            * (legacy.rank() * 648) as u128
            * residual
    );
}

#[test]
fn p64_public_batch_layout_has_coefficients_and_no_projection_messages() {
    use crate::binary_source::{
        BinaryCommittedGroupManifest, BinaryRootPublic, OrdinaryNativeRootPlan, PrimeGroupClaim,
        PrimePlaneClaim, PrimePlaneManifest,
    };
    use crate::AkitaSetupDescriptor;
    use akita_params::binary_setup::BinaryFieldProfile;
    use jolt_field::{Ext2, Prime64Offset59};
    type E = Ext2<Prime64Offset59>;
    let setup = AkitaSetupDescriptor {
        max_num_vars: 0,
        max_num_batched_polys: 0,
        num_field_elements: crate::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
        setup_seed: [41; 32].into(),
    };
    for field_profile in [BinaryFieldProfile::P128, BinaryFieldProfile::P64] {
        let carrier = super::binary_ordinary_profile_fixture(field_profile);
        let ordinary_carrier = carrier;
        let index = BinaryIndexMap::new(128, 8, 8, 2).unwrap();
        let profiles =
            BinaryCommittedGroupManifest::profiles_for_field(&index, 1, 0, field_profile).unwrap();
        let descriptor = BinarySourceDescriptor::new(index, vec![[1; 32]], profiles).unwrap();
        let manifest = BinaryCommittedGroupManifest::with_field_profile(
            descriptor.clone(),
            0,
            &carrier,
            field_profile,
        )
        .unwrap();
        let encoded = manifest.canonical_bytes();
        let decoded = BinaryCommittedGroupManifest::from_bytes(&encoded).unwrap();
        assert_eq!(decoded.field_profile(), field_profile);
        let mut wrong_version = encoded;
        let version = b"AkitaBinaryCommitment".len();
        wrong_version[version] = if field_profile == BinaryFieldProfile::P64 {
            1
        } else {
            2
        };
        assert!(BinaryCommittedGroupManifest::from_bytes(&wrong_version).is_err());
        for (word, planes, mixed) in [(true, 0, false), (false, 1, false), (true, 1, true)] {
            let planes = planes * field_profile.extension_degree();
            let zero = if field_profile == BinaryFieldProfile::P64 {
                crate::binary_source::PrimeClaimValue::from([0, 0])
            } else {
                0.into()
            };
            let claim = word.then(|| {
                BinaryTerminalClaim::new(
                    &descriptor,
                    [1; 32],
                    vec![[0; 21]; 2],
                    vec![[0; 21]],
                    [0; 21],
                )
                .unwrap()
            });
            let prime = PrimePlaneManifest::with_field_profile(
                &descriptor,
                (planes > 0).then(|| PrimeGroupClaim {
                    point: vec![zero; 11],
                    evaluations: vec![PrimePlaneClaim {
                        polynomial: 0,
                        value: zero,
                    }],
                }),
                field_profile,
            )
            .unwrap();
            let public = BinaryRootPublic::new(
                &manifest,
                claim.as_ref(),
                &prime,
                &setup,
                1 << 22,
                256,
                mixed.then_some(ordinary_carrier),
            )
            .unwrap();
            let plan = public.plan();
            let coefficient_count = if field_profile == BinaryFieldProfile::P64 {
                1944
            } else {
                7
            };
            let ordinary = mixed.then(|| {
                OrdinaryNativeRootPlan::new(
                    ordinary_carrier,
                    if word { 5 } else { 2 } + usize::from(planes > 0),
                )
                .unwrap()
            });
            let other = 2 * usize::from(word)
                + manifest.b_rows()
                + manifest.d_label(word, planes).unwrap().rank as usize
                + 2 * prime.claim().map_or(0, |claim| claim.evaluations.len())
                + ordinary
                    .as_ref()
                    .map_or(0, OrdinaryNativeRootPlan::linear_batch_count)
                + ordinary
                    .as_ref()
                    .map(|ordinary| {
                        ordinary
                            .negative_binary_addresses(plan.repacking())
                            .unwrap()
                            .len()
                    })
                    .unwrap_or(0);
            assert_eq!(public.batch_count(), coefficient_count + other);
            let batches = vec![E::one(); public.batch_count()];
            let [b, d] = public.authentication_batches(&batches).unwrap();
            assert_eq!(b.len(), manifest.b_rows());
            assert_eq!(
                d.len(),
                manifest.d_label(word, planes).unwrap().rank as usize
            );
            if field_profile == BinaryFieldProfile::P64 {
                let segment = plan.quotient_segment().unwrap();
                assert_eq!(
                    segment,
                    if word { 5 } else { 2 }
                        + usize::from(planes > 0)
                        + ordinary
                            .as_ref()
                            .map_or(0, |ordinary| ordinary.segments().len())
                );
                assert!(!plan.opening_segments().contains(&segment));
                if let Some(ordinary) = ordinary {
                    let range = plan.repacking().segment_range(segment - 1).unwrap();
                    assert_eq!(
                        plan.repacking().segment_range(segment).unwrap().start,
                        range.end
                    );
                    assert_eq!(
                        plan.repacking().segments()[segment - 1],
                        *ordinary.segments().last().unwrap()
                    );
                }
                if word && planes == 0 && !mixed {
                    let challenges = Challenges::from_sparse(
                        vec![
                            SparseChallenge {
                                positions: Default::default(),
                                coeffs: Default::default()
                            };
                            2
                        ],
                        2,
                        1,
                    )
                    .unwrap();
                    let alpha = E::new(Prime64Offset59::from_u64(3), Prime64Offset59::from_u64(5));
                    let mut batches = vec![E::zero(); public.batch_count()];
                    for (index, batch) in batches.iter_mut().take(1944).enumerate() {
                        *batch = E::new(
                            Prime64Offset59::from_u64(index as u64 + 1),
                            Prime64Offset59::from_u64(index as u64 + 7),
                        );
                    }
                    let b = vec![E::zero(); manifest.payload_coefficients().unwrap()];
                    let d =
                        vec![E::zero(); manifest.d_label(word, planes).unwrap().rank as usize * 64];
                    let (weights, negative) = public
                        .relation(&challenges, alpha, &batches, &[], &[], &b, &d, None, None)
                        .unwrap();
                    assert!(negative.is_empty());
                    assert_eq!(weights.claim(), E::zero());
                    let rho = batches[..1944]
                        .chunks_exact(648)
                        .map(|row| row.try_into().unwrap())
                        .collect::<Vec<[E; 648]>>();
                    let direct = binary_direct_a_weights(plan, None, &challenges, &rho).unwrap();
                    assert_eq!(
                        weights.materialize().unwrap(),
                        direct.materialize().unwrap()
                    );
                    assert!(public
                        .relation(&challenges, alpha, &batches, &rho, &[], &b, &d, None, None)
                        .is_err());
                    assert!(public
                        .relation(
                            &challenges,
                            alpha,
                            &batches,
                            &[],
                            &[0; 7],
                            &b,
                            &d,
                            None,
                            None
                        )
                        .is_err());
                }
                assert!(BinaryFieldProfile::P64
                    .admit_root(field_profile.modulus(), 2)
                    .is_err());
            } else {
                assert_eq!(plan.quotient_segment(), None);
            }
        }
    }
}
