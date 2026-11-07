//! Differential tests: structured root weights against the flat reference.

use super::*;
use crate::binary_source::root_weight_fixtures::{RootWeightScenario, Shape};
use crate::binary_source::BinaryAOperand;
use jolt_field::{Prime128Offset275 as F, Zero};

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
    let scenario = RootWeightScenario::new(shape, seed);
    let RootWeightScenario {
        public,
        challenges,
        alpha,
        batches,
        rho,
        projected,
        commitment,
        d_payload,
        ordinary_claim,
        matrix,
        mut stream,
        ..
    } = scenario;
    let ordinary_claim = ordinary_claim
        .as_ref()
        .map(|(point, value, payload)| (point.as_slice(), *value, payload.as_slice()));
    let ordinary_points = ordinary_claim.map(|(point, value, _)| (point, [value]));
    let ordinary_points = ordinary_points.as_ref().map(|(point, values)| {
        [crate::opening_claims::OrdinaryPointEvaluation { point, evaluations: values }]
    });
    let native_ordinary = ordinary_points.as_ref().zip(ordinary_claim)
        .map(|(points, (_, _, payload))| (&points[..], payload));
    let num_vars = public.plan().repacking().domain_len().trailing_zeros() as usize;
    let points = (0..2)
        .map(|_| (0..num_vars).map(|_| stream.field()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let compiled = [None, Some(BinaryAOperand::new::<F>(&matrix))]
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
                    native_ordinary,
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
/// The shapes cover word-only, prime-only and mixed ordinary openings, one
/// point with values on several owners, wider source rows and extra zero rings.
/// Every plane has a distinct owner; no shape reaches production size.
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
    for (seed, mut shape) in shapes().into_iter().enumerate() {
        shape.owners = shape.owners.max(shape.prime_planes);
        check(shape, seed as u64);
    }
}

/// The deletion rule's support lemma: every compiled root row, including the
/// negative-bit rows, is zero on every padding address.
#[test]
fn compiled_root_weights_vanish_on_every_padding_address() {
    for (seed, mut shape) in shapes().into_iter().enumerate() {
        shape.owners = shape.owners.max(shape.prime_planes);
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
    for (seed, mut shape) in shapes().into_iter().enumerate() {
        shape.owners = shape.owners.max(shape.prime_planes);
        let scenario = RootWeightScenario::new(shape, 1000 + seed as u64);
        let public = scenario.public;
        let mut stream = scenario.stream;
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
            .map(|_| stream.field())
            .collect::<Vec<_>>();
        let alpha = stream.field();
        let batches = [0, 1].map(|_| (0..2).map(|_| stream.field()).collect::<Vec<_>>());
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
        let rho = scenario.rho;
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
            &crate::binary_source::binary_root_matrix(&seed_bytes, manifest).unwrap(),
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
        let challenge = stream.field();
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
            .map(|_| stream.field())
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
