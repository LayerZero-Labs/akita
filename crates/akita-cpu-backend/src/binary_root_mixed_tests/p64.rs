//! Fixture-sized P64 prime and mixed roots with isolated three-stage replay.
use crate::binary_root::{prime_plane_digits, BinaryRootRelation};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::trinomial::{
    image::Q24Image, module_map::unpack_signed_components, signed::signed_trinomial_product,
};
use akita_algebra::MinusTrinomial;
use akita_challenges::{Challenges, FoldChallengeDistribution, SparseChallenge};
use akita_params::binary_setup::BinaryFieldProfile;
use akita_types::binary_source::{
    binary_ordinary_profile_fixture, binary_public_digit_commitment, binary_root_matrix,
    binary_word_manifest_fixture, BinaryRootPublic, OrdinaryNativeRootPlan, PrimeClaims,
    PrimeGroupClaim, PrimePlaneClaim, PrimePlaneManifest, BINARY_SOURCE_Q,
};
use jolt_field::{CanonicalEncoding, Ext2, ExtField, Prime64Offset59 as F, Ring, Zero};

#[test]
fn binary_root_p64_prime_and_mixed_replay_all_three_stages() {
    for (mixed, copies) in [
        (false, 0),
        (true, 0),
        (false, 1),
        (true, 1),
        (false, 2),
        (true, 2),
    ] {
        let field = BinaryFieldProfile::P64;
        let manifest = binary_word_manifest_fixture(field, 1);
        let descriptor = manifest.descriptor();
        let seed = [41; 32];
        let matrix = binary_root_matrix(&seed, &manifest).unwrap();
        let sources = (0..2)
            .map(|column| {
                std::array::from_fn(|i| {
                    if i < 512 && (i + column * 7) % 31 == 0 {
                        if i % 2 == 0 {
                            1
                        } else {
                            -1
                        }
                    } else {
                        0
                    }
                })
            })
            .collect::<Vec<[i128; 648]>>();
        let images = sources
            .iter()
            .flat_map(|source| {
                matrix.iter().map(|row| {
                    signed_trinomial_product::<648, MinusTrinomial>(&row[0], source)
                        .unwrap()
                        .map(|value| {
                            Q24Image::from_residue(
                                value.rem_euclid(i128::from(BINARY_SOURCE_Q)) as u32
                            )
                            .unwrap()
                        })
                })
            })
            .collect::<Vec<_>>();
        let challenges = Challenges::from_sparse(
            (0..2)
                .map(|column| SparseChallenge {
                    positions: (0..46)
                        .map(|i| ((i * 3 + column) % 162) as u32)
                        .collect::<Vec<_>>()
                        .into(),
                    coeffs: vec![1; 46].into(),
                })
                .collect(),
            2,
            1,
        )
        .unwrap();
        let relation = BinaryRootRelation::build(
            descriptor,
            &matrix,
            &sources,
            &images,
            &challenges,
            FoldChallengeDistribution::BinaryRoot,
        )
        .unwrap();
        let point = (0..11).map(|i| 19 + i * 7).collect::<Vec<u128>>();
        // Independent evaluation of the signed source, in scalar row/column/
        // coefficient order, including the virtual zero coefficients.
        let low = EqPolynomial::evals(
            &point[..8]
                .iter()
                .copied()
                .map(F::from_u128)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let high = EqPolynomial::evals(
            &point[8..]
                .iter()
                .copied()
                .map(F::from_u128)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut value = F::zero();
        for (column, source) in sources.iter().enumerate() {
            let components = unpack_signed_components(source).unwrap();
            for (row, component) in components.iter().enumerate() {
                for (coefficient, &v) in component.iter().enumerate() {
                    value += high[column + 2 * row] * low[coefficient] * F::from_i128(v);
                }
            }
        }
        let prime_claim = PrimeGroupClaim {
            point,
            evaluations: vec![PrimePlaneClaim {
                polynomial: 0,
                value: value.to_u128_checked().unwrap(),
            }],
        };
        let prime_claim = if copies == 0 {
            PrimeClaims::Base(prime_claim)
        } else {
            PrimeClaims::Extension(
                (0..copies)
                    .map(|copy| {
                        let point = (0..11)
                            .map(|i| [19 + i * 7, 3 + copy * 11 + i])
                            .collect::<Vec<[u64; 2]>>();
                        let extension = point
                            .iter()
                            .map(|x| Ext2::<F>::from_base_fn(|i| F::from_u64(x[i])))
                            .collect::<Vec<_>>();
                        let eq = EqPolynomial::evals(&extension).unwrap();
                        let mut value = Ext2::<F>::zero();
                        for (column, source) in sources.iter().enumerate() {
                            for (row, component) in
                                unpack_signed_components(source).unwrap().iter().enumerate()
                            {
                                for (coefficient, &v) in component.iter().enumerate() {
                                    value += eq[coefficient + 256 * (column + 2 * row)]
                                        * Ext2::<F>::from_i128(v);
                                }
                            }
                        }
                        assert_ne!(value.base_coefficient(1), F::zero());
                        PrimeGroupClaim {
                            point,
                            evaluations: vec![PrimePlaneClaim {
                                polynomial: 0,
                                value: std::array::from_fn(|i| {
                                    value.base_coefficient(i).to_u128_checked().unwrap() as u64
                                }),
                            }],
                        }
                    })
                    .collect(),
            )
        };
        let prime =
            PrimePlaneManifest::with_field_profile(descriptor, Some(prime_claim.clone()), field)
                .unwrap();
        let profile = mixed.then(|| binary_ordinary_profile_fixture(field, None));
        let setup = akita_types::AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: akita_types::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
            setup_seed: seed.into(),
        };
        let public =
            BinaryRootPublic::new(&manifest, false, &prime, &setup, 1 << 17, 64, profile).unwrap();
        let plan = public.plan();
        let mut natural = vec![
            BinaryRootRelation::response_digits(&relation.response).unwrap(),
            BinaryRootRelation::image_digits(&images).unwrap(),
            prime_plane_digits::<F>(descriptor, &prime, &sources).unwrap(),
        ];
        assert_eq!(natural[2].len(), prime.planes().len() * 2 * 22 * 162);
        let ordinary = profile.map(|profile| {
            let ordinary = OrdinaryNativeRootPlan::new(profile, 3).unwrap();
            let compression = profile.outer_compression_plan().unwrap();
            assert_eq!(compression.field_bits(), 64);
            assert_eq!(
                compression
                    .maps()
                    .iter()
                    .map(|map| map.ring_dimension())
                    .collect::<Vec<_>>(),
                vec![64, 32]
            );
            let (segments, point, evaluations, payload) =
                crate::opaque::binary_ordinary_p64_fixture::ordinary_preimage(profile, &ordinary);
            assert_eq!(
                segments[2].len(),
                compression.maps()[0].padded_digit_count()
            );
            assert_eq!(
                segments[3].len(),
                compression.maps()[1].padded_digit_count()
            );
            natural.extend(segments);
            assert_eq!(evaluations.len(), 1);
            (point, evaluations[0], payload)
        });
        natural.push(
            BinaryRootRelation::quotient_digits(
                &relation
                    .rows
                    .iter()
                    .map(|row| row.remainder)
                    .collect::<Vec<_>>(),
                plan.projection_bounds().quotient().unwrap(),
            )
            .unwrap(),
        );
        let witness = plan
            .repacking()
            .repack(&natural.iter().map(Vec::as_slice).collect::<Vec<_>>())
            .unwrap();
        let b =
            binary_public_digit_commitment::<F>(&seed, manifest.b_label(), 0, &natural[1]).unwrap();
        let d = binary_public_digit_commitment::<F>(
            &seed,
            manifest.d_label(false, prime.planes().len()).unwrap(),
            0,
            &natural[2],
        )
        .unwrap();
        let mut bad = prime_claim;
        match &mut bad {
            PrimeClaims::Base(claim) => {
                claim.evaluations[0].value = (claim.evaluations[0].value + 1) % field.modulus();
            }
            PrimeClaims::Extension(points) => {
                points[0].evaluations[0].value[1] =
                    (points[0].evaluations[0].value[1] + 1) % field.modulus() as u64;
            }
        }
        let bad = PrimePlaneManifest::with_field_profile(descriptor, Some(bad), field).unwrap();
        let false_public =
            BinaryRootPublic::new(&manifest, false, &bad, &setup, 1 << 17, 64, profile).unwrap();
        crate::opaque::sumcheck::stage3::p64_root_transcript_tests::root_round_trip(
            &manifest,
            &public,
            &matrix,
            &challenges,
            &witness,
            &b,
            &d,
            ordinary
                .as_ref()
                .map(|(point, value, payload)| (point.as_slice(), *value, payload.as_slice())),
            Some(&false_public),
        );
    }
}
