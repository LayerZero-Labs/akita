//! Differential tests: structured root weights against the flat reference.

use super::*;
use crate::binary_source::{
    BinaryAOperand, BinaryCommittedGroupManifest, BinaryIndexMap, PrimePlaneClaim,
};
use crate::AkitaSetupDescriptor;
use jolt_crypto::challenge::SparseChallenge;
use akita_params::sis::{
    sis_table_key_for_linf_bound, InnerCommitMatrixParams, OuterCommitMatrixParams, SisMatrixRole,
    SisTableDigest, DEFAULT_SIS_SECURITY_POLICY,
};
use akita_params::{
    BlockGeometry, CommitmentSliceCount, CommitmentSliceGeometry, GadgetDigits,
    GroupCommitPhaseParams, PolynomialGroupLayout, RoleParams, SisModulusProfileId,
};
use jolt_algebra::{Prime128Offset275 as F, Ring, Zero};
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
    carriers: usize,
    columns: usize,
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
    let index = BinaryIndexMap::new(
        shape.word_width,
        shape.carriers * (128 / shape.word_width),
        shape.carriers,
        shape.columns,
    )
    .unwrap();
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
    let compiled = [None, Some(BinaryAOperand::from_canonical(&matrix))]
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
                matrix.map(|operand| operand.matrix),
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

/// Nineteen root shapes, shared by the differential test and the padding
/// regression. They include every claim, plane and owner combination of the
/// prime-plane end-to-end tests.
///
/// Sixteen use the end-to-end geometry: 8 padded carriers and 2 columns, hence
/// 4 scalar rows and one response quad. Without the mixed group: the word claim
/// at width 128 with zero, one, two or five planes and at width 8 with zero
/// planes; two owners with the word claim and zero or five planes; and
/// prime-only claims with two planes over two owners and with three planes and
/// five extra zero rings. With the mixed group: the word claim with zero, two
/// or five planes, and at width 8 with one plane and three extra rings; and one,
/// two or five prime-only planes. The last three use 32 carriers and 2 columns,
/// or 64 carriers and 4 columns, hence 16 scalar rows and four response quads:
/// one owner with the word claim alone; two owners with two planes and the claim
/// on the second owner; and the mixed group at width 8 with the word claim, five
/// planes and two extra rings. No shape reaches a production-size geometry.
fn shapes() -> Vec<Shape> {
    let base = Shape {
        word_width: 128,
        carriers: 8,
        columns: 2,
        owners: 1,
        word_claim: true,
        prime_planes: 0,
        extra_rings: 0,
        mixed: false,
    };
    let mixed = Shape {
        mixed: true,
        ..base
    };
    vec![
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
            prime_planes: 2,
            ..base
        },
        Shape {
            prime_planes: 5,
            ..base
        },
        Shape { owners: 2, ..base },
        mixed,
        Shape {
            word_width: 8,
            prime_planes: 1,
            extra_rings: 3,
            ..mixed
        },
        Shape {
            word_claim: false,
            prime_planes: 1,
            ..mixed
        },
        Shape {
            word_claim: false,
            prime_planes: 2,
            ..mixed
        },
        Shape {
            word_claim: false,
            prime_planes: 5,
            ..mixed
        },
        Shape {
            prime_planes: 2,
            ..mixed
        },
        Shape {
            prime_planes: 5,
            ..mixed
        },
        Shape {
            carriers: 32,
            ..base
        },
        Shape {
            carriers: 64,
            columns: 4,
            owners: 2,
            prime_planes: 2,
            ..base
        },
        Shape {
            word_width: 8,
            carriers: 32,
            prime_planes: 5,
            extra_rings: 2,
            ..mixed
        },
    ]
}

/// Compare both sides on every shape in [`shapes`].
#[test]
fn structured_root_weights_match_the_flat_reference() {
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

/// The typed prefix carries the B/D rows exactly: its stored setup against
/// the prover's dense weights is the direct authentication weight at the root
/// point, and the verifier's terminal weight is the dense weights' extension.
#[test]
fn typed_prefix_offloads_the_digit_authentication_rows() {
    use crate::binary_source::{
        binary_digit_authentication_weights, BinaryAProduct, BinaryTypedPrefix,
    };
    use akita_algebra::eq_poly::EqPolynomial;
    use akita_params::binary_setup::{SetupStreamLabel, SetupStreamRole};
    for (seed, shape) in shapes().into_iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(1000 + seed as u64);
        let public = public(shape, &mut rng);
        let plan = public.plan();
        let manifest = &public.manifest;
        let prefix = BinaryTypedPrefix::from_plan(manifest, plan).unwrap();
        assert_eq!(
            BinaryTypedPrefix::new(manifest, shape.word_claim, shape.prime_planes, 64).unwrap(),
            prefix,
            "{shape:?}"
        );
        let prefix_len = prefix.domain_len;
        let root_point = (0..plan.repacking().domain_len().trailing_zeros())
            .map(|_| field(&mut rng))
            .collect::<Vec<_>>();
        let alpha = field(&mut rng);
        let batches = [0, 1].map(|_| (0..2).map(|_| field(&mut rng)).collect::<Vec<_>>());
        let seed_bytes = public.setup.setup_seed.seed;
        let labels = [
            manifest.b_label(),
            SetupStreamLabel {
                role: SetupStreamRole::WitnessCommitment,
                ..manifest.b_label()
            },
        ];
        let mut direct = F::zero();
        for (role, segments) in [vec![1], plan.opening_segments()].into_iter().enumerate() {
            direct += binary_digit_authentication_weights(
                plan,
                &segments,
                &seed_bytes,
                labels[role],
                &[F::zero(); 128],
                alpha,
                &batches[role],
            )
            .unwrap()
            .evaluate(&root_point)
            .unwrap();
        }
        let owners = plan.descriptor().polynomials().len();
        let rho = (0..owners * plan.rank())
            .map(|_| core::array::from_fn(|_| field(&mut rng)))
            .collect::<Vec<[F; 648]>>();
        let a = BinaryAProduct::new(plan, &rho, &root_point, prefix_len).unwrap();
        let setup = prefix
            .setup_coefficients::<F>(&seed_bytes, manifest, prefix_len)
            .unwrap();
        let batch_refs = [batches[0].as_slice(), batches[1].as_slice()];
        let weights = prefix
            .prover_weights(&a, &root_point, alpha, batch_refs, prefix_len)
            .unwrap();
        // Digit copies may fill the A tensor's zero padding, so compare
        // against the A-only prefix: zero batches leave only the A weights.
        let a_setup = crate::binary_source::binary_a_setup_coefficients::<F>(
            &seed_bytes,
            manifest,
            prefix_len,
        )
        .unwrap();
        let zero = [F::zero(); 2];
        let a_weights = prefix
            .prover_weights(&a, &root_point, alpha, [&zero, &zero], prefix_len)
            .unwrap();
        let inner = |s: &[F], t: &[F]| s.iter().zip(t).fold(F::zero(), |sum, (&s, &t)| sum + s * t);
        let offloaded = inner(&setup, &weights) - inner(&a_setup, &a_weights);
        assert_eq!(offloaded, direct, "{shape:?}");
        // The block form expands to the dense weight, and the setup is zero
        // outside its live blocks.
        let blocks = prefix
            .prover_weight_blocks(&a, &root_point, alpha, batch_refs, prefix_len)
            .unwrap();
        assert_eq!(blocks.domain_len(), prefix_len, "{shape:?}");
        let mut expanded = vec![F::zero(); prefix_len];
        let mut live_setup = vec![F::zero(); prefix_len];
        for (live, &block) in blocks.blocks().iter().enumerate() {
            let cells = block * 64..(block + 1) * 64;
            for (scale, row) in blocks.block_terms(live).unwrap() {
                for (value, &weight) in expanded[cells.clone()].iter_mut().zip(row) {
                    *value += scale * weight;
                }
            }
            live_setup[cells.clone()].copy_from_slice(&setup[cells]);
        }
        assert_eq!(expanded, weights, "{shape:?}");
        assert_eq!(live_setup, setup, "{shape:?}");
        // Binding a bit folds the blocks as it folds the dense table.
        let challenge = field(&mut rng);
        let mut bound = blocks.clone();
        bound.bind(challenge).unwrap();
        for (live, &block) in bound.blocks().iter().enumerate() {
            let mut folded = [F::zero(); 32];
            for (scale, row) in bound.block_terms(live).unwrap() {
                for (value, &weight) in folded.iter_mut().zip(row) {
                    *value += scale * weight;
                }
            }
            for (offset, &value) in folded.iter().enumerate() {
                let (low, high) = (
                    weights[block * 64 + 2 * offset],
                    weights[block * 64 + 2 * offset + 1],
                );
                assert_eq!(value, low + challenge * (high - low), "{shape:?}");
            }
        }
        let point = (0..prefix_len.trailing_zeros())
            .map(|_| field(&mut rng))
            .collect::<Vec<_>>();
        let extension = EqPolynomial::evals(&point)
            .unwrap()
            .into_iter()
            .zip(&weights)
            .fold(F::zero(), |sum, (eq, &t)| sum + eq * t);
        let terminal = a.evaluate(&point).unwrap()
            + prefix
                .digit_weight(&point, &root_point, alpha, batch_refs)
                .unwrap();
        assert_eq!(terminal, extension, "{shape:?}");
    }
}
