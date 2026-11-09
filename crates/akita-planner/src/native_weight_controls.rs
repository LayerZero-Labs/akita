//! Independent assembled reading coefficients and translated setup products.

use super::*;
use akita_algebra::{CanonicalEncoding, One, Ring, Zero};
use akita_config::proof_optimized::fp64::ExtensionField as E;
use akita_types::binary_source::*;
use akita_types::opening_claims::OrdinaryPointEvaluation;
use native_open_fixture::{ext, opening};

fn equality(point: &[E], index: usize) -> E {
    point
        .iter()
        .enumerate()
        .fold(E::one(), |weight, (bit, &r)| {
            weight
                * if index >> bit & 1 == 1 {
                    r
                } else {
                    E::one() - r
                }
        })
}

pub(super) fn compare(readings_only: bool) {
    let opening = opening();
    let binding = &opening.binding;
    let plan = binding.plan();
    let root = (0..binding.outgoing().domain_len().trailing_zeros())
        .map(|axis| ext::<E>([11 + axis as u64, 17 + 3 * axis as u64]))
        .collect::<Vec<_>>();
    let alpha = ext::<E>([23, 29]);
    let terminals = plan
        .binary()
        .iter()
        .map(|local| {
            let descriptor = local.source().manifest.descriptor();
            let mut one = [0; 21];
            one[0] = 1;
            Some(
                BinaryTerminalClaim::new(
                    descriptor,
                    descriptor.polynomials()[0],
                    vec![one; 2],
                    vec![one; 1],
                    one,
                )
                .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let public = GroupedRootPublic::new(binding.clone(), &terminals).unwrap();
    let locals = plan
        .binary()
        .iter()
        .zip(&terminals)
        .map(|(local, terminal)| {
            let source = local.source();
            BinaryRootPublic::new(
                &source.manifest,
                terminal.as_ref(),
                &source.prime,
                binding.setup(),
                local.local().repacking().domain_len(),
                plan.repacking().destination_ring_dimension(),
                None,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut channel = akita_transcript::new_prover_channel(b"grouped-weight-oracle");
    let challenges = (0..4)
        .map(|group| {
            binary_root_challenges(
                &mut akita_challenges::ProverFoldDraw::new(&mut channel, 0, group),
                2,
                1,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let payloads = plan
        .binary()
        .iter()
        .map(|local| {
            let source = local.source();
            (
                vec![ext::<E>([31, 37]); source.manifest.b_rows() * 64],
                vec![
                    ext::<E>([41, 43]);
                    source
                        .manifest
                        .d_label(true, source.prime.planes().len())
                        .unwrap()
                        .rank as usize
                        * 64
                ],
            )
        })
        .collect::<Vec<_>>();
    let binary = challenges
        .iter()
        .zip(&payloads)
        .map(|(challenges, (b, d))| GroupedBinaryRelation {
            challenges,
            commitment: b,
            d_payload: d,
            matrix: None,
        })
        .collect::<Vec<_>>();
    let ordinary = plan.ordinary().unwrap();
    // Zero batches still require the compressed terminal payload shape.
    let signed_payloads = ordinary
        .plans()
        .iter()
        .map(|source| {
            vec![
                ext::<E>([47, 53]);
                source
                    .profile()
                    .outer_compression_plan()
                    .unwrap()
                    .terminal_coefficients()
            ]
        })
        .collect::<Vec<_>>();
    let first = ordinary
        .plans()
        .iter()
        .enumerate()
        .map(|(group, source)| {
            let reading = opening.claims.groups()[4 + group]
                .point_evaluations()
                .next()
                .unwrap();
            OrdinaryNativeRootClaim {
                profile: source.profile(),
                width: source.source_width().unwrap(),
                point: reading.point,
                evaluations: reading.evaluations,
                payload: &signed_payloads[group],
            }
        })
        .collect::<Vec<_>>();
    let second = (4..8)
        .map(|group| {
            let reading = opening.claims.groups()[group]
                .point_evaluations()
                .nth(1)
                .unwrap();
            OrdinaryPointEvaluation {
                point: reading.point,
                evaluations: reading.evaluations,
            }
        })
        .collect::<Vec<_>>();
    if readings_only {
        let mut offset = 0;
        for (group, local) in plan.binary().iter().enumerate() {
            for point_index in 0..2 {
                let mut batches = vec![E::zero(); public.batch_count()];
                batches[offset + locals[group].batch_count() - 4 + 2 * point_index + 1] = E::one();
                let (weights, _) = public
                    .relation::<Field, E>(&binary, &first, &second, alpha, &batches)
                    .unwrap();
                let claim = &local.source().prime.claims()[point_index];
                let point = claim
                    .point
                    .iter()
                    .map(|x| {
                        ext::<E>(
                            x.coordinates()
                                .iter()
                                .map(|&x| x as u64)
                                .collect::<Vec<_>>()
                                .try_into()
                                .unwrap(),
                        )
                    })
                    .collect::<Vec<_>>();
                let mut expected = E::zero();
                // Read the physical planes directly: each is a base-coordinate
                // row partial; its scalar reading uses only coefficient/column axes.
                for coordinate in 0..2 {
                    let mut coordinate_weight = E::zero();
                    let plane = local
                        .source()
                        .prime
                        .planes()
                        .iter()
                        .position(|plane| {
                            plane.point == point_index && plane.coordinate == coordinate
                        })
                        .unwrap();
                    for column in 0..2 {
                        for coefficient in 0..162 {
                            let mut power = E::one();
                            for digit in 0..22 {
                                let address = local.coefficient_offset()
                                    + local
                                        .local()
                                        .repacking()
                                        .coefficient_address(
                                            local.local().prime_segment().unwrap(),
                                            plane * 2 + column,
                                            digit,
                                            coefficient,
                                        )
                                        .unwrap();
                                let basis = ext::<E>(if coordinate == 0 { [1, 0] } else { [0, 1] });
                                let reading = equality(&point[..9], coefficient + 256 * column);
                                coordinate_weight +=
                                    basis * reading * power * equality(&root, address);
                                power *= E::from_u64(8);
                            }
                        }
                    }
                    assert_ne!(coordinate_weight, E::zero());
                    expected += coordinate_weight;
                }
                assert_ne!(expected, E::zero());
                assert_eq!(weights.evaluate(&root).unwrap(), expected);
                assert_eq!(
                    weights.claim(),
                    ext::<E>(
                        claim.evaluations[0]
                            .value
                            .coordinates()
                            .iter()
                            .map(|&x| x as u64)
                            .collect::<Vec<_>>()
                            .try_into()
                            .unwrap()
                    )
                );
            }
            offset += locals[group].batch_count();
        }
    } else {
        let batches = (0..public.batch_count())
            .map(|i| ext::<E>([59 + i as u64, 61]))
            .collect::<Vec<_>>();
        let weights = public.stage3_weights(&root, alpha, &batches).unwrap();
        let setup = binding.setup_coefficients::<Field>().unwrap();
        let mut actual = E::zero();
        for (live, &block) in weights.blocks().iter().enumerate() {
            for (scale, row) in weights.block_terms(live).unwrap() {
                for (slot, &weight) in row.iter().enumerate() {
                    let coefficient = setup[64 * block + slot].to_u128_checked().unwrap() as u64;
                    actual += scale * weight * ext::<E>([coefficient, 0]);
                }
            }
        }
        let mut expected = E::zero();
        let mut offset = 0;
        for (group, local) in plan.binary().iter().enumerate() {
            let matrix =
                binary_root_matrix(&binding.setup().setup_seed.seed, &local.source().manifest)
                    .unwrap();
            let batches = &batches[offset..offset + locals[group].batch_count()];
            offset += batches.len();
            let compile = |matrix| {
                locals[group]
                    .relation::<Field, E>(
                        &challenges[group],
                        alpha,
                        batches,
                        &[],
                        &[],
                        &payloads[group].0,
                        &payloads[group].1,
                        None,
                        matrix,
                    )
                    .unwrap()
                    .0
                    .materialize()
                    .unwrap()
            };
            let full = compile(Some(BinaryAOperand::new::<Field>(&matrix)));
            let online = compile(None);
            // Independent assembly: evaluate every local coefficient at its
            // global address, without the translated prefix or translated oracle.
            let mut owner_weight = E::zero();
            for (cell, (&full, online)) in full.iter().zip(online).enumerate() {
                if full != online {
                    let address = local.coefficient_offset() + cell;
                    owner_weight += (full - online) * equality(&root, address);
                }
            }
            assert_ne!(owner_weight, E::zero());
            expected += owner_weight;
        }
        assert_ne!(expected, E::zero());
        assert_eq!(actual, expected);
    }
}
