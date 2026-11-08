use super::*;
use crate::binary_source::{binary_word_manifest_fixture, BinaryRootWitnessPlan};

#[test]
fn extension_prime_encoding_rejections_and_capacity() {
    let field = BinaryFieldProfile::P64;
    let manifest = binary_word_manifest_fixture(field, 2);
    let source = manifest.descriptor();
    let points = (0..2)
        .map(|i| PrimeGroupClaim {
            point: vec![[2 + i, 3]; 11],
            evaluations: vec![
                PrimePlaneClaim {
                    polynomial: 1,
                    value: [5, 7],
                },
                PrimePlaneClaim {
                    polynomial: 0,
                    value: [11, 13],
                },
            ],
        })
        .collect::<Vec<_>>();
    let build = |points| {
        PrimePlaneManifest::with_field_profile(source, Some(PrimeClaims::Extension(points)), field)
    };
    let prime = build(points.clone()).unwrap();
    assert_eq!(prime.planes().len(), 8);
    let mut expected = b"AkitaPrimePlanes/e64\x04".to_vec();
    expected.extend_from_slice(&2u64.to_le_bytes());
    for i in 0..2u64 {
        expected.extend_from_slice(&11u64.to_le_bytes());
        for _ in 0..11 {
            expected.extend_from_slice(&(2 + i).to_le_bytes());
            expected.extend_from_slice(&3u64.to_le_bytes());
        }
        expected.extend_from_slice(&2u64.to_le_bytes());
        for (owner, value) in [(0u64, [11u64, 13]), (1, [5, 7])] {
            expected.extend_from_slice(&owner.to_le_bytes());
            for x in value {
                expected.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    assert_eq!(prime.canonical_bytes(), expected);
    assert_eq!(
        prime
            .planes()
            .iter()
            .map(|p| p.polynomial)
            .collect::<Vec<_>>(),
        vec![0, 0, 1, 1, 0, 0, 1, 1]
    );
    for mutation in 0..9 {
        let mut bad = points.clone();
        match mutation {
            0 => bad.clear(),
            1 => bad.push(bad[0].clone()),
            2 => bad[1].point = bad[0].point.clone(),
            3 => {
                bad[0].point.pop();
            }
            4 => bad[0].point[0][1] = field.modulus() as u64,
            5 => bad[0].evaluations[0].value[1] = field.modulus() as u64,
            6 => bad[0].evaluations.clear(),
            7 => bad[0].evaluations[0].polynomial = 2,
            _ => bad[0].evaluations[0].polynomial = 0,
        }
        assert!(
            matches!(build(bad), Err(AkitaError::InvalidInput(_))),
            "{mutation}"
        );
    }
    assert!(PrimePlaneManifest::with_field_profile(
        source,
        Some(PrimeClaims::Extension(points)),
        BinaryFieldProfile::P128,
    )
    .is_err());
    // The shape, not whether the coordinates happen to be base-valued, costs two planes.
    let zero_extension = PrimeGroupClaim {
        point: vec![[0, 0]; 11],
        evaluations: vec![PrimePlaneClaim {
            polynomial: 0,
            value: [0, 0],
        }],
    };
    assert_eq!(build(vec![zero_extension]).unwrap().planes().len(), 2);
    for word in [false, true] {
        assert!(BinaryRootWitnessPlan::for_profile(source, 3, 64, field, word, 8).is_ok());
        assert!(manifest.d_label(word, 8).is_ok());
        assert!(BinaryRootWitnessPlan::for_profile(source, 3, 64, field, word, 9).is_err());
        assert!(manifest.d_label(word, 9).is_err());
    }
    let factors = prime
        .plane_factors::<jolt_field::Prime64Offset59>(source)
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(&factors[0].rows, &factors[2].rows));
    assert!(!std::sync::Arc::ptr_eq(&factors[0].rows, &factors[4].rows));
    for plane in 0..4 {
        assert!(std::sync::Arc::ptr_eq(
            &factors[0].omega,
            &factors[plane].omega
        ));
        assert!(std::sync::Arc::ptr_eq(
            &factors[0].columns,
            &factors[plane].columns
        ));
        assert_eq!(factors[plane].point_index, 0);
        assert_eq!(factors[plane + 4].point_index, 1);
    }
    // Only this private test can desynchronize the immutable manifest. The
    // factor boundary must reject it before any consumer can truncate a zip.
    let mut malformed = prime;
    malformed.planes.pop();
    assert!(matches!(
        malformed.plane_factors::<jolt_field::Prime64Offset59>(source),
        Err(AkitaError::InvalidSize {
            expected: 7,
            actual: 8,
        })
    ));
}

#[test]
fn extension_prime_reading_columns_match_direct_coordinate_product() {
    use jolt_field::{Ext2, ExtField, One, Prime64Offset59 as F, Ring, Zero};
    type E = Ext2<F>;
    let manifest = binary_word_manifest_fixture(BinaryFieldProfile::P64, 1);
    let claim = PrimeGroupClaim {
        point: (0..11).map(|i| [3 + i, 5 + 2 * i]).collect(),
        evaluations: vec![PrimePlaneClaim {
            polynomial: 0,
            value: [0, 0],
        }],
    };
    let prime = PrimePlaneManifest::with_field_profile(
        manifest.descriptor(),
        Some(PrimeClaims::Extension(vec![claim.clone()])),
        BinaryFieldProfile::P64,
    )
    .unwrap();
    let ext = |x: [u64; 2]| E::from_base_fn(|i| F::from_u64(x[i]));
    let batches = [ext([2, 3]), ext([5, 7]), ext([11, 13]), ext([17, 19])];
    for (plane, factors) in prime
        .plane_factors::<E>(manifest.descriptor())
        .unwrap()
        .into_iter()
        .enumerate()
    {
        let columns = factors.reading_columns(&batches).unwrap();
        for (column, cw) in columns.enumerate() {
            assert_eq!(factors.reading_column(&batches, column).unwrap(), cw);
            for coefficient in 0..162 {
                // Compute the actual E64 equality product, without the factored
                // coordinate compiler or its tensor-contraction implementation.
                let mut w = E::one();
                for (axis, &x) in claim.point[..9].iter().enumerate() {
                    let bit = if axis < 8 {
                        (coefficient >> axis) & 1
                    } else {
                        column
                    };
                    let x = ext(x);
                    w *= if bit == 0 { E::one() - x } else { x };
                }
                if plane == 1 {
                    w *= ext([0, 1]);
                }
                let expected = batches[1] * E::lift_base(w.base_coefficient(0))
                    + batches[3] * E::lift_base(w.base_coefficient(1));
                let omega = factors.omega[coefficient];
                assert_eq!(cw[0] * omega[0] + cw[1] * omega[1], expected);
            }
        }
        assert!(factors
            .reading_column(&batches, factors.columns.len())
            .is_err());
        assert!(factors.reading_column(&batches[..1], 0).is_err());
        assert_ne!(&*factors.rows, &[E::zero(); 4]);
    }
}

#[test]
fn extension_response_profiles_match_flat_planes_with_cancellation() {
    use akita_challenges::{Challenges, SparseChallenge};
    use jolt_field::{Ext2, ExtField, One, Prime64Offset59 as F, Ring, Zero};
    type E = Ext2<F>;
    let field = BinaryFieldProfile::P64;
    let manifest = binary_word_manifest_fixture(field, 2);
    let source = manifest.descriptor();
    let points = (0..2)
        .map(|pass| {
            let mut point = vec![[2, 3]; 11];
            // Distinct coefficient points with identical row tensors permit
            // exact cancellation between the two passes' response weights.
            point[0][0] += pass;
            PrimeGroupClaim {
                point,
                evaluations: (0..2)
                    .map(|polynomial| PrimePlaneClaim {
                        polynomial,
                        value: [5, 7],
                    })
                    .collect(),
            }
        })
        .collect();
    let prime =
        PrimePlaneManifest::with_field_profile(source, Some(PrimeClaims::Extension(points)), field)
            .unwrap();
    let plan = BinaryRootWitnessPlan::for_profile(source, 3, 64, field, false, 8).unwrap();
    let challenges = Challenges::from_sparse(
        vec![
            SparseChallenge {
                positions: vec![0, 161].into(),
                coeffs: vec![1, -1].into(),
            };
            4
        ],
        2,
        2,
    )
    .unwrap();
    let ext = |n| E::from_base_fn(|i| F::from_u64(n + 3 * i as u64));
    for cancel in [false, true] {
        let mut batches = (1..=16).map(ext).collect::<Vec<_>>();
        if cancel {
            for plane in 0..4 {
                batches[2 * (plane + 4)] = -batches[2 * plane];
            }
        }
        for alpha in [E::zero(), E::one(), ext(17)] {
            let weights = crate::binary_source::prime_weights::prime_plane_relation_weights(
                source,
                &prime,
                plan.repacking(),
                0,
                plan.prime_segment().unwrap(),
                &challenges,
                alpha,
                &batches,
            )
            .unwrap();
            let reference = crate::binary_source::root_public::reference::prime_plane_weights(
                source,
                &prime,
                plan.repacking(),
                0,
                plan.prime_segment().unwrap(),
                &challenges,
                alpha,
                &batches,
            )
            .unwrap()
            .materialize()
            .unwrap();
            for bits in [6, 8, 10] {
                assert_eq!(
                    weights.profiles(bits).unwrap().take_dense(),
                    reference,
                    "cancel={cancel}, bits={bits}, alpha={alpha:?}"
                );
            }
        }
    }
}
