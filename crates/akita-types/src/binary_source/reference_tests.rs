//! Differential tests: structured root weights against the flat reference.

use super::*;
use crate::binary_source::{BinaryCommittedGroupManifest, BinaryIndexMap, PrimePlaneClaim};
use crate::AkitaSetupDescriptor;
use akita_challenges::SparseChallenge;
use akita_params::sis::{
    sis_table_key_for_linf_bound, InnerCommitMatrixParams, OuterCommitMatrixParams, SisMatrixRole,
    SisTableDigest, DEFAULT_SIS_SECURITY_POLICY,
};
use akita_params::{
    BlockGeometry, CommitmentSliceCount, CommitmentSliceGeometry, GadgetDigits,
    GroupCommitPhaseParams, PolynomialGroupLayout, RoleParams, SisModulusProfileId,
};
use jolt_field::{Prime128Offset275 as F, Ring, Zero};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// One small ordinary group with real audited table cells, as the mixed root
/// end-to-end tests select.
fn ordinary_profile() -> GroupCommitPhaseParams {
    let key = |role, degree| {
        sis_table_key_for_linf_bound(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q128Offset275,
            role,
            degree,
            8,
        )
        .unwrap()
    };
    let inner =
        InnerCommitMatrixParams::try_new_with_min_rank(key(SisMatrixRole::Inner, 128), 2).unwrap();
    let geometry = CommitmentSliceGeometry::try_new(
        CommitmentSliceCount::ONE,
        4,
        1,
        inner.output_rank(),
        43,
        128,
        64,
    )
    .unwrap();
    let outer = OuterCommitMatrixParams::try_new_with_min_rank(
        key(SisMatrixRole::Outer, 64),
        geometry.physical_input_width(),
    )
    .unwrap();
    let profile = GroupCommitPhaseParams {
        version: GroupCommitPhaseParams::VERSION,
        group: PolynomialGroupLayout::singleton(10),
        blocks: BlockGeometry::new(8, 2, 4),
        outer_slice_count: CommitmentSliceCount::ONE,
        inner: RoleParams::new(GadgetDigits::new(3, 1), inner),
        outer: RoleParams::new(GadgetDigits::new(3, 43), outer),
    };
    profile.validate_frozen_precommit(128).unwrap();
    profile
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    word_width: usize,
    owners: usize,
    word_claim: bool,
    prime_planes: usize,
    extra_rings: usize,
    mixed: bool,
}

fn field(rng: &mut StdRng) -> F {
    F::from_u128(rng.gen::<u128>() >> 1)
}

fn binary_field_bytes(rng: &mut StdRng) -> [u8; 21] {
    let mut bytes: [u8; 21] = rng.gen();
    bytes[20] &= 0x03;
    bytes
}

fn public(shape: Shape, rng: &mut StdRng) -> BinaryRootPublic {
    let index = BinaryIndexMap::new(shape.word_width, 8 * (128 / shape.word_width), 8, 2).unwrap();
    let profiles =
        BinaryCommittedGroupManifest::commitment_profiles(&index, shape.owners, 0).unwrap();
    let polynomials = (0..shape.owners)
        .map(|owner| [29 + owner as u8; 32])
        .collect::<Vec<_>>();
    let descriptor = BinarySourceDescriptor::new(index, polynomials.clone(), profiles).unwrap();
    let carrier = ordinary_profile();
    let manifest = BinaryCommittedGroupManifest::new(descriptor.clone(), 0, &carrier).unwrap();
    let rows = descriptor.index_map().padded_carriers() / descriptor.index_map().columns();
    let claim = shape.word_claim.then(|| {
        BinaryTerminalClaim::new(
            &descriptor,
            polynomials[shape.owners - 1],
            (0..rows.trailing_zeros())
                .map(|_| binary_field_bytes(rng))
                .collect(),
            (0..descriptor.index_map().columns().trailing_zeros())
                .map(|_| binary_field_bytes(rng))
                .collect(),
            binary_field_bytes(rng),
        )
        .unwrap()
    });
    let arity = descriptor.index_map().padded_carriers().trailing_zeros() as usize + 8;
    let claims = (0..shape.prime_planes)
        .map(|plane| PrimePlaneClaim {
            polynomial: plane % shape.owners,
            point: (0..arity)
                .map(|_| rng.gen::<u128>() >> 1)
                .collect::<Vec<_>>(),
            tensor_component: (plane % 4) as u8,
            value: rng.gen::<u128>() >> 1,
        })
        .collect::<Vec<_>>();
    let prime = PrimePlaneManifest::new(&descriptor, &claims).unwrap();
    let setup = AkitaSetupDescriptor {
        max_num_vars: 0,
        max_num_batched_polys: 0,
        num_field_elements: crate::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
        setup_seed: [41u8; 32].into(),
    };
    let ordinary = shape.mixed.then(ordinary_profile);
    let build = |logical_len| {
        BinaryRootPublic::new(
            &manifest,
            claim.as_ref(),
            &prime,
            &setup,
            logical_len,
            64,
            ordinary,
        )
    };
    let natural_end = build(1 << 22)
        .unwrap()
        .plan()
        .repacking()
        .natural_end()
        .unwrap();
    assert!(build(natural_end - 1).is_err());
    build(natural_end + shape.extra_rings * 64).unwrap()
}

/// One structured and one flat compilation of a root relation.
struct Compiled {
    structured: NativeRootWeights<F>,
    negative: Vec<(usize, F)>,
    flat: NativeRootWeights<F>,
    flat_negative: Vec<(usize, F)>,
}

/// Compile one shape's root relation both ways, without and with A, and draw
/// two evaluation points on its domain.
fn compile(shape: Shape, seed: u64) -> (BinaryRootPublic, Vec<Compiled>, Vec<Vec<F>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let public = public(shape, &mut rng);
    let descriptor = public.plan().descriptor().clone();
    let index = descriptor.index_map();
    let owners = descriptor.polynomials().len();
    let columns = index.columns();
    let challenges = Challenges::from_sparse(
        (0..columns * owners)
            .map(|_| {
                let mut positions = Vec::new();
                while positions.len() < 46 {
                    let position = rng.gen_range(0..162u32);
                    if !positions.contains(&position) {
                        positions.push(position);
                    }
                }
                positions.sort_unstable();
                SparseChallenge {
                    positions: positions.into(),
                    coeffs: (0..46)
                        .map(|_| if rng.gen() { 1i8 } else { -1 })
                        .collect::<Vec<_>>()
                        .into(),
                }
            })
            .collect(),
        columns,
        owners,
    )
    .unwrap();
    let alpha = field(&mut rng);
    let batches = (0..public.batch_count())
        .map(|_| field(&mut rng))
        .collect::<Vec<_>>();
    let rho = (0..owners * public.plan().rank())
        .map(|_| std::array::from_fn(|_| field(&mut rng)))
        .collect::<Vec<[F; 648]>>();
    let projected: [i128; 7] = std::array::from_fn(|_| rng.gen::<i64>().into());
    let payload_len = public.manifest.b_rows() * public.manifest.b_ring_dimension();
    let commitment = (0..payload_len)
        .map(|_| field(&mut rng))
        .collect::<Vec<_>>();
    let d_payload = (0..payload_len)
        .map(|_| field(&mut rng))
        .collect::<Vec<_>>();
    let ordinary = ordinary_profile();
    let ordinary_point = (0..ordinary.group.num_vars())
        .map(|_| field(&mut rng))
        .collect::<Vec<_>>();
    let ordinary_payload = (0..ordinary
        .outer_compression_plan()
        .unwrap()
        .terminal_coefficients())
        .map(|_| field(&mut rng))
        .collect::<Vec<_>>();
    let ordinary_claim = shape
        .mixed
        .then(|| (&ordinary_point[..], field(&mut rng), &ordinary_payload[..]));
    let a_rows = index.padded_carriers() / (4 * columns);
    let matrix = (0..public.plan().rank())
        .map(|_| {
            (0..a_rows)
                .map(|_| std::array::from_fn(|_| rng.gen_range(-7_126_704i128..=7_126_704)))
                .collect()
        })
        .collect::<Vec<Vec<[i128; 648]>>>();
    let domain_len = public.plan().repacking().domain_len();
    let num_vars = domain_len.trailing_zeros() as usize;
    let points = (0..2)
        .map(|_| (0..num_vars).map(|_| field(&mut rng)).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let compiled = [None, Some(&matrix[..])]
        .into_iter()
        .map(|matrix| {
            let (structured, negative) = public
                .relation(
                    &challenges,
                    alpha,
                    &batches,
                    &rho,
                    &projected,
                    &commitment,
                    &d_payload,
                    ordinary_claim,
                    matrix,
                )
                .unwrap();
            let (flat, flat_negative) = root_relation(
                &public,
                &challenges,
                alpha,
                &batches,
                &rho,
                &projected,
                &commitment,
                &d_payload,
                ordinary_claim,
                matrix,
            )
            .unwrap();
            Compiled {
                structured,
                negative,
                flat,
                flat_negative,
            }
        })
        .collect();
    (public, compiled, points)
}

fn check(shape: Shape, seed: u64) {
    let (public, compiled, points) = compile(shape, seed);
    let domain_len = public.plan().repacking().domain_len();
    for Compiled {
        structured,
        negative,
        flat,
        flat_negative,
    } in compiled
    {
        assert_eq!(negative, flat_negative, "{shape:?}");
        assert_eq!(structured.claim(), flat.claim(), "{shape:?}");
        assert_eq!(structured.table_len(), domain_len);
        let mut dense = vec![F::zero(); domain_len];
        for &(address, weight) in flat.terms() {
            dense[address] += weight;
        }
        assert!(
            structured.materialize().unwrap() == dense,
            "{shape:?}: materialized weights differ"
        );
        for point in &points {
            assert_eq!(
                structured.evaluate(point).unwrap(),
                flat.evaluate(point).unwrap(),
                "{shape:?}"
            );
        }
    }
}

/// The end-to-end shapes: word widths 128 and 8, one and two owners, word
/// claim on and off, zero to five prime planes, an enlarged outgoing witness
/// and the mixed ordinary root.
fn shapes() -> [Shape; 8] {
    let base = Shape {
        word_width: 128,
        owners: 1,
        word_claim: true,
        prime_planes: 0,
        extra_rings: 0,
        mixed: false,
    };
    [
        base,
        Shape {
            word_width: 8,
            ..base
        },
        Shape {
            word_claim: false,
            owners: 2,
            prime_planes: 2,
            ..base
        },
        Shape {
            prime_planes: 1,
            ..base
        },
        Shape {
            owners: 2,
            prime_planes: 5,
            ..base
        },
        Shape {
            word_claim: false,
            prime_planes: 3,
            extra_rings: 5,
            ..base
        },
        Shape {
            mixed: true,
            ..base
        },
        Shape {
            mixed: true,
            word_width: 8,
            prime_planes: 1,
            extra_rings: 3,
            ..base
        },
    ]
}

#[test]
fn structured_root_weights_match_the_flat_reference_on_every_end_to_end_shape() {
    for (seed, shape) in shapes().into_iter().enumerate() {
        check(shape, seed as u64);
    }
}

/// The deletion rule's support lemma: every compiled root row, including the
/// negative-bit rows, is zero on every padding address.
#[test]
fn compiled_root_weights_vanish_on_every_padding_address() {
    for (seed, shape) in shapes().into_iter().enumerate() {
        let (public, compiled, _) = compile(shape, seed as u64);
        let repacking = public.plan().repacking();
        let padding = (0..repacking.domain_len())
            .filter(|&address| repacking.is_padding_address(address).unwrap())
            .collect::<Vec<_>>();
        assert!(!padding.is_empty(), "{shape:?}");
        for Compiled {
            structured,
            negative,
            flat,
            ..
        } in compiled
        {
            let table = structured.materialize().unwrap();
            let mut reference = vec![F::zero(); repacking.domain_len()];
            for &(address, weight) in flat.terms() {
                reference[address] += weight;
            }
            for &address in &padding {
                assert!(
                    table[address].is_zero() && reference[address].is_zero(),
                    "{shape:?}: weight on padding address {address}"
                );
            }
            for &(address, _) in &negative {
                assert!(!repacking.is_padding_address(address).unwrap(), "{shape:?}");
            }
        }
    }
}
