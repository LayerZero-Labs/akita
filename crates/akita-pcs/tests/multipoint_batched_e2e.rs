#![allow(missing_docs)]
#![cfg(all(feature = "planner", not(feature = "zk")))]

mod common;

use akita_field::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::CommitmentProver;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::Blake2bTranscript;
use akita_types::{AkitaBatchedProof, AkitaRootBatchSummary, RingCommitment};
use akita_verifier::CommitmentVerifier;
use common::*;
use std::sync::Mutex;

static E2E_TEST_LOCK: Mutex<()> = Mutex::new(());

type OneHotTestPoly = OneHotPoly<F, ONEHOT_D, u8>;
type OneHotIndexData = Vec<Option<u8>>;
type PointOneHotPolyData = Vec<Vec<(OneHotTestPoly, OneHotIndexData)>>;

fn batch_summary(total_claims: usize, point_group_counts: &[usize]) -> AkitaRootBatchSummary {
    AkitaRootBatchSummary::new(
        total_claims,
        point_group_counts.iter().sum(),
        point_group_counts.len(),
    )
    .expect("valid batch summary")
}

fn dense_layout(nv: usize, total_claims: usize, point_group_counts: &[usize]) -> LevelParams {
    DenseCfg::get_params_for_batched_commitment(
        nv,
        nv,
        batch_summary(total_claims, point_group_counts),
    )
    .expect("dense layout")
}

fn onehot_layout(nv: usize, total_claims: usize, point_group_counts: &[usize]) -> LevelParams {
    OneHotCfg::get_params_for_batched_commitment(
        nv,
        nv,
        batch_summary(total_claims, point_group_counts),
    )
    .expect("onehot layout")
}

fn make_onehot_poly_from_ring_elems(
    total_ring_elems: usize,
    seed: u64,
) -> (OneHotPoly<F, ONEHOT_D, u8>, Vec<Option<u8>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_ring_elems)
        .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
        .collect();
    let poly = OneHotPoly::<F, ONEHOT_D, u8>::new(ONEHOT_K, indices.clone()).expect("onehot poly");
    (poly, indices)
}

fn onehot_lagrange_opening(indices: &[Option<u8>], point: &[F]) -> F {
    assert_eq!(indices.len() * ONEHOT_K, 1usize << point.len());
    indices
        .iter()
        .enumerate()
        .filter_map(|(chunk_idx, hot_idx)| {
            hot_idx.map(|hot_idx| chunk_idx * ONEHOT_K + hot_idx as usize)
        })
        .fold(F::zero(), |acc, field_pos| {
            acc + point
                .iter()
                .enumerate()
                .fold(F::one(), |weight, (bit, &r)| {
                    if ((field_pos >> bit) & 1) == 1 {
                        weight * r
                    } else {
                        weight * (F::one() - r)
                    }
                })
        })
}

#[test]
fn multipoint_dense_round_trip_with_mixed_groups() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 10;
        let point_group_sizes = [vec![2], vec![2], vec![2]];
        let total_claims: usize = point_group_sizes.iter().flatten().sum();
        let point_group_counts: Vec<usize> = point_group_sizes.iter().map(Vec::len).collect();
        let layout = dense_layout(NV, total_claims, &point_group_counts);

        let point_polys: Vec<Vec<DensePoly<F, DENSE_D>>> = point_group_sizes
            .iter()
            .enumerate()
            .map(|(point_idx, groups)| {
                (0..groups.iter().sum())
                    .map(|poly_idx| {
                        make_dense_poly(
                            NV,
                            0xd3e5_1000 + (point_idx as u64) * 100 + poly_idx as u64,
                        )
                    })
                    .collect()
            })
            .collect();
        let opening_points_owned: Vec<Vec<F>> = (0..point_group_sizes.len())
            .map(|point_idx| random_point(NV, 0xaaaa_1000 + point_idx as u64))
            .collect();
        let openings_by_point: Vec<Vec<F>> = point_polys
            .iter()
            .zip(opening_points_owned.iter())
            .map(|(polys, point)| {
                polys
                    .iter()
                    .map(|poly| opening_from_poly(poly, point, &layout))
                    .collect()
            })
            .collect();

        let polys_by_point: Vec<&[DensePoly<F, DENSE_D>]> =
            point_polys.iter().map(Vec::as_slice).collect();
        let openings_by_point: Vec<&[F]> = openings_by_point.iter().map(Vec::as_slice).collect();
        let opening_points: Vec<&[F]> = opening_points_owned.iter().map(Vec::as_slice).collect();

        let setup =
            <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::setup_prover(
                NV,
                total_claims,
                point_group_sizes.len(),
            );
        let verifier_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::setup_verifier(&setup);

        let (commitments_by_point, hints_by_point) =
            <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::batched_commit(
                &polys_by_point,
                &point_group_counts,
                &setup,
            )
            .expect("multipoint batched commit");

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/dense");
        let proof =
            <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::batched_prove(
                &setup,
                prove_inputs_from_groups(&opening_points, &polys_by_point, &commitments_by_point, hints_by_point),
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("multipoint batched prove");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize");
        let decoded = AkitaBatchedProof::<F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/dense");
        let result = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentVerifier<F, DENSE_D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_inputs_from_groups(&opening_points, &openings_by_point, &commitments_by_point),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_ok(),
            "dense multipoint verification failed: {:?}",
            result.err()
        );
    });
}

#[cfg(not(feature = "zk"))]
mod non_zk_single_point {
    use super::*;

    #[test]
    fn single_point_dense_round_trip_with_uneven_groups() {
        init_rayon_pool();
        let _guard = E2E_TEST_LOCK.lock().unwrap();
        run_on_large_stack(|| {
            const NV: usize = 10;
            let point = random_point(NV, 0xaaaa_3000);
            let group_a = vec![make_dense_poly(NV, 0xd3e5_3000)];
            let group_b = vec![
                make_dense_poly(NV, 0xd3e5_3001),
                make_dense_poly(NV, 0xd3e5_3002),
            ];
            let poly_groups: Vec<&[DensePoly<F, DENSE_D>]> = vec![&group_a, &group_b];
            let point_group_counts = [poly_groups.len()];
            let total_claims: usize = poly_groups.iter().map(|group| group.len()).sum();
            let layout = dense_layout(NV, total_claims, &point_group_counts);
            let openings_a = vec![opening_from_poly(&group_a[0], &point, &layout)];
            let openings_b = group_b
                .iter()
                .map(|poly| opening_from_poly(poly, &point, &layout))
                .collect::<Vec<_>>();

            let setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::setup_prover(NV, total_claims, 1);
            let verifier_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::setup_verifier(&setup);
            let (commitments, hints) = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::batched_commit(
            &poly_groups, &point_group_counts, &setup
        )
        .expect("uneven grouped batched commit");

            let mut hints = hints.into_iter();
            let prover_claims = vec![(
                point.as_slice(),
                vec![
                    CommittedPolynomials {
                        polynomials: group_a.as_slice(),
                        commitment: &commitments[0],
                        hint: hints.next().unwrap(),
                    },
                    CommittedPolynomials {
                        polynomials: group_b.as_slice(),
                        commitment: &commitments[1],
                        hint: hints.next().unwrap(),
                    },
                ],
            )];
            let mut prover_transcript =
                Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/uneven-dense");
            let proof = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::batched_prove(
                &setup,
                prover_claims,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("uneven grouped batched prove");

            let verifier_claims = vec![(
                point.as_slice(),
                vec![
                    CommittedOpenings {
                        openings: openings_a.as_slice(),
                        commitment: &commitments[0],
                    },
                    CommittedOpenings {
                        openings: openings_b.as_slice(),
                        commitment: &commitments[1],
                    },
                ],
            )];
            let mut verifier_transcript =
                Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/uneven-dense");
            <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentVerifier<F, DENSE_D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verifier_claims,
            BasisMode::Lagrange,
        )
        .expect("uneven grouped batched verify");
        });
    }
}

#[test]
fn multipoint_onehot_round_trip_with_mixed_groups() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 15;
        let point_group_sizes = [vec![2], vec![2], vec![2]];
        let total_claims: usize = point_group_sizes.iter().flatten().sum();
        let point_group_counts: Vec<usize> = point_group_sizes.iter().map(Vec::len).collect();
        let layout = onehot_layout(NV, total_claims, &point_group_counts);

        let total_ring = layout.num_blocks * layout.block_len;
        let point_poly_data: PointOneHotPolyData = point_group_sizes
            .iter()
            .enumerate()
            .map(|(point_idx, groups)| {
                (0..groups.iter().sum())
                    .map(|poly_idx| {
                        make_onehot_poly_from_ring_elems(
                            total_ring,
                            0xa66e_2000 + (point_idx as u64) * 100 + poly_idx as u64,
                        )
                    })
                    .collect()
            })
            .collect();
        let point_polys: Vec<Vec<OneHotPoly<F, ONEHOT_D, u8>>> = point_poly_data
            .iter()
            .map(|polys| polys.iter().map(|(poly, _)| poly.clone()).collect())
            .collect();
        let opening_points_owned: Vec<Vec<F>> = (0..point_group_sizes.len())
            .map(|point_idx| random_point(NV, 0xf00d_2000 + point_idx as u64))
            .collect();
        let openings_by_point: Vec<Vec<F>> = point_polys
            .iter()
            .zip(point_poly_data.iter())
            .zip(opening_points_owned.iter())
            .map(|((_, poly_data), point)| {
                poly_data
                    .iter()
                    .map(|(_, indices)| onehot_lagrange_opening(indices, point))
                    .collect()
            })
            .collect();

        let polys_by_point: Vec<&[OneHotPoly<F, ONEHOT_D, u8>]> =
            point_polys.iter().map(Vec::as_slice).collect();
        let openings_by_point: Vec<&[F]> = openings_by_point.iter().map(Vec::as_slice).collect();
        let opening_points: Vec<&[F]> = opening_points_owned.iter().map(Vec::as_slice).collect();

        let setup =
            <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::setup_prover(
                NV,
                total_claims,
                point_group_sizes.len(),
            );
        let verifier_setup = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::setup_verifier(&setup);

        let (commitments_by_point, hints_by_point) = <AkitaCommitmentScheme<
            ONEHOT_D,
            OneHotCfg,
        > as CommitmentProver<F, ONEHOT_D>>::batched_commit(
            &polys_by_point,
            &point_group_counts,
            &setup,
        )
        .expect("multipoint batched commit");

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/onehot");
        let proof =
            <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::batched_prove(
                &setup,
                prove_inputs_from_groups(&opening_points, &polys_by_point, &commitments_by_point, hints_by_point),
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("multipoint batched prove");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize");
        let decoded = AkitaBatchedProof::<F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/onehot");
        let result = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentVerifier<
            F,
            ONEHOT_D,
        >>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_inputs_from_groups(&opening_points, &openings_by_point, &commitments_by_point),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_ok(),
            "onehot multipoint verification failed: {:?}",
            result.err()
        );
    });
}

#[cfg(not(feature = "zk"))]
mod non_zk_negative_cases {
    use super::*;

    #[test]
    fn multipoint_dense_verify_rejects_swapped_points() {
        init_rayon_pool();
        let _guard = E2E_TEST_LOCK.lock().unwrap();
        run_on_large_stack(|| {
            const NV: usize = 10;
            let point_group_sizes = [vec![2], vec![2]];
            let total_claims = 4usize;
            let point_group_counts: Vec<usize> = point_group_sizes.iter().map(Vec::len).collect();
            let layout = dense_layout(NV, total_claims, &point_group_counts);

            let point_polys: Vec<Vec<DensePoly<F, DENSE_D>>> = point_group_sizes
                .iter()
                .enumerate()
                .map(|(point_idx, groups)| {
                    (0..groups.iter().sum())
                        .map(|poly_idx| {
                            make_dense_poly(
                                NV,
                                0xd3e5_3000 + (point_idx as u64) * 100 + poly_idx as u64,
                            )
                        })
                        .collect()
                })
                .collect();
            let opening_points_owned: Vec<Vec<F>> =
                vec![random_point(NV, 0xaaaa_3000), random_point(NV, 0xaaaa_3001)];
            let openings_by_point: Vec<Vec<F>> = point_polys
                .iter()
                .zip(opening_points_owned.iter())
                .map(|(polys, point)| {
                    polys
                        .iter()
                        .map(|poly| opening_from_poly(poly, point, &layout))
                        .collect()
                })
                .collect();

            let polys_by_point: Vec<&[DensePoly<F, DENSE_D>]> =
                point_polys.iter().map(Vec::as_slice).collect();
            let openings_by_point: Vec<&[F]> =
                openings_by_point.iter().map(Vec::as_slice).collect();
            let opening_points: Vec<&[F]> =
                opening_points_owned.iter().map(Vec::as_slice).collect();

            let setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::setup_prover(NV, total_claims, point_group_sizes.len());
            let verifier_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::setup_verifier(&setup);

            let (commitments_by_point, hints_by_point) =
            <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::batched_commit(
                &polys_by_point,
                &point_group_counts,
                &setup,
            )
            .expect("multipoint batched commit");

            let mut prover_transcript =
                Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/dense_wrong_point");
            let proof = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::batched_prove(
                &setup,
                prove_inputs_from_groups(
                    &opening_points,
                    &polys_by_point,
                    &commitments_by_point,
                    hints_by_point,
                ),
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("multipoint batched prove");

            let swapped_points = vec![opening_points[1], opening_points[0]];
            let mut verifier_transcript =
                Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/dense_wrong_point");
            let result = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentVerifier<
                F,
                DENSE_D,
            >>::batched_verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                verify_inputs_from_groups(
                    &swapped_points,
                    &openings_by_point,
                    &commitments_by_point,
                ),
                BasisMode::Lagrange,
            );
            assert!(result.is_err(), "swapped opening points must be rejected");
        });
    }

    #[test]
    fn multipoint_onehot_verify_rejects_wrong_opening_count() {
        init_rayon_pool();
        let _guard = E2E_TEST_LOCK.lock().unwrap();
        run_on_large_stack(|| {
            const NV: usize = 15;
            let point_group_sizes = [vec![2], vec![2]];
            let total_claims: usize = point_group_sizes.iter().flatten().sum();
            let point_group_counts: Vec<usize> = point_group_sizes.iter().map(Vec::len).collect();
            let layout = onehot_layout(NV, total_claims, &point_group_counts);

            let total_ring = layout.num_blocks * layout.block_len;
            let point_poly_data: PointOneHotPolyData = point_group_sizes
                .iter()
                .enumerate()
                .map(|(point_idx, groups)| {
                    (0..groups.iter().sum())
                        .map(|poly_idx| {
                            make_onehot_poly_from_ring_elems(
                                total_ring,
                                0xa66e_4000 + (point_idx as u64) * 100 + poly_idx as u64,
                            )
                        })
                        .collect()
                })
                .collect();
            let point_polys: Vec<Vec<OneHotPoly<F, ONEHOT_D, u8>>> = point_poly_data
                .iter()
                .map(|polys| polys.iter().map(|(poly, _)| poly.clone()).collect())
                .collect();
            let opening_points_owned: Vec<Vec<F>> =
                vec![random_point(NV, 0xf00d_4000), random_point(NV, 0xf00d_4001)];
            let openings_by_point: Vec<Vec<F>> = point_polys
                .iter()
                .zip(point_poly_data.iter())
                .zip(opening_points_owned.iter())
                .map(|((_, poly_data), point)| {
                    poly_data
                        .iter()
                        .map(|(_, indices)| onehot_lagrange_opening(indices, point))
                        .collect()
                })
                .collect();

            let polys_by_point: Vec<&[OneHotPoly<F, ONEHOT_D, u8>]> =
                point_polys.iter().map(Vec::as_slice).collect();
            let opening_points: Vec<&[F]> =
                opening_points_owned.iter().map(Vec::as_slice).collect();

            let setup = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
                F,
                ONEHOT_D,
            >>::setup_prover(NV, total_claims, point_group_sizes.len());
            let verifier_setup = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
                F,
                ONEHOT_D,
            >>::setup_verifier(&setup);

            let (commitments_by_point, hints_by_point) = <AkitaCommitmentScheme<
            ONEHOT_D,
            OneHotCfg,
        > as CommitmentProver<F, ONEHOT_D>>::batched_commit(
            &polys_by_point,
            &point_group_counts,
            &setup,
        )
        .expect("multipoint batched commit");

            let mut prover_transcript =
                Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/onehot_wrong_opening_count");
            let proof = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
                F,
                ONEHOT_D,
            >>::batched_prove(
                &setup,
                prove_inputs_from_groups(
                    &opening_points,
                    &polys_by_point,
                    &commitments_by_point,
                    hints_by_point,
                ),
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("multipoint batched prove");

            let wrong_openings_by_point =
                vec![&openings_by_point[0][..1], &openings_by_point[1][..]];

            let mut verifier_transcript =
                Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/onehot_wrong_opening_count");
            let result = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentVerifier<
                F,
                ONEHOT_D,
            >>::batched_verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                verify_inputs_from_groups(
                    &opening_points,
                    &wrong_openings_by_point,
                    &commitments_by_point,
                ),
                BasisMode::Lagrange,
            );
            assert!(
                result.is_err(),
                "wrong verifier-side opening count must be rejected"
            );
        });
    }

    #[test]
    fn multipoint_batched_prove_rejects_capacity_overflow() {
        init_rayon_pool();
        let _guard = E2E_TEST_LOCK.lock().unwrap();
        run_on_large_stack(|| {
            const NV: usize = 10;
            let point_group_sizes = [vec![4], vec![1]];
            let total_claims: usize = point_group_sizes.iter().flatten().sum();

            let point_polys: Vec<Vec<DensePoly<F, DENSE_D>>> = point_group_sizes
                .iter()
                .enumerate()
                .map(|(point_idx, groups)| {
                    (0..groups.iter().sum())
                        .map(|poly_idx| {
                            make_dense_poly(
                                NV,
                                0xd3e5_5000 + (point_idx as u64) * 100 + poly_idx as u64,
                            )
                        })
                        .collect()
                })
                .collect();
            let polys_by_point: Vec<&[DensePoly<F, DENSE_D>]> =
                point_polys.iter().map(Vec::as_slice).collect();
            let opening_points_owned: Vec<Vec<F>> = (0..point_group_sizes.len())
                .map(|point_idx| random_point(NV, 0xaaaa_5000 + point_idx as u64))
                .collect();
            let opening_points: Vec<&[F]> =
                opening_points_owned.iter().map(Vec::as_slice).collect();

            let commit_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::setup_prover(
                NV, total_claims, point_group_sizes.len()
            );
            let prove_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::setup_prover(
                NV, total_claims - 1, point_group_sizes.len()
            );
            let point_group_counts: Vec<usize> = point_group_sizes.iter().map(Vec::len).collect();
            let (commitments_by_point, hints_by_point) =
            <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::batched_commit(
                &polys_by_point,
                &point_group_counts,
                &commit_setup,
            )
            .expect("multipoint batched commit should fit with matching setup");
            let mut transcript =
                Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/capacity-overflow");
            let result = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::batched_prove(
                &prove_setup,
                prove_inputs_from_groups(
                    &opening_points,
                    &polys_by_point,
                    &commitments_by_point,
                    hints_by_point,
                ),
                &mut transcript,
                BasisMode::Lagrange,
            );
            assert!(result.is_err(), "capacity overflow must be rejected");
        });
    }
}

// End-to-end regression coverage for the upstream bug:
// `OneHotPoly` `block_len` mismatch in `AkitaCommitmentScheme::batched_prove`
// at `max_num_vars >= 19` with shared commitments.
//
// Scenario:
//   - One `OneHotPoly` is committed once via the new
//     `commit_for_multipoint(polys, num_opening_points, setup)` API,
//     which selects the layout that matches the prove-time
//     `(num_groups=1, num_points=num_opening_points)` shape.
//   - Two distinct opening points reference the same commitment + same poly
//     slice + cloned hint via `ProverClaims`.
//   - `prover_claims_to_incidence` and `verifier_claims_to_incidence`
//     deduplicate the two `(point, group)` entries by commitment pointer
//     into one logical group, so the planner's prove-time schedule sees
//     `(num_claims=2, num_groups=1, num_points=2)` matching the commit-time
//     layout.
fn run_shared_onehot_two_points_round_trip(nv: usize, transcript_label: &[u8]) {
    let total_claims: usize = 2;
    let num_points: usize = 2;

    // OneHotPoly's total field elements = 2^NV; total ring elements
    // = 2^NV / D. The poly length is independent of layout choice.
    let total_ring = (1usize << nv) / ONEHOT_D;
    let (poly, indices) = make_onehot_poly_from_ring_elems(total_ring, 0xdead_beef);

    let setup =
        <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::setup_prover(
            nv,
            total_claims,
            num_points,
        );
    let verifier_setup = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
        F,
        ONEHOT_D,
    >>::setup_verifier(&setup);

    // Commit ONCE with the layout matching the eventual `(num_groups=1,
    // num_points=2)` prove shape.
    let polys_singleton: Vec<OneHotPoly<F, ONEHOT_D, u8>> = vec![poly];
    let (commitment, hint) = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
        F,
        ONEHOT_D,
    >>::commit_for_multipoint(&polys_singleton, num_points, &setup)
    .expect("multipoint onehot commit");

    // Two distinct opening points and their canonical openings.
    let pt0_owned = random_point(nv, 0xaaaa_1900);
    let pt1_owned = random_point(nv, 0xbbbb_1900);
    let y0 = onehot_lagrange_opening(&indices, &pt0_owned);
    let y1 = onehot_lagrange_opening(&indices, &pt1_owned);

    // Share the SAME poly slice + SAME commitment between two points; the
    // incidence flatteners dedup these into a single group with two points.
    let polys_slice: &[OneHotPoly<F, ONEHOT_D, u8>] = polys_singleton.as_slice();
    let polys_by_point: Vec<&[OneHotPoly<F, ONEHOT_D, u8>]> = vec![polys_slice, polys_slice];
    let commitments_by_point: Vec<RingCommitment<F, ONEHOT_D>> =
        vec![commitment.clone(), commitment.clone()];
    let hints_by_point: Vec<_> = vec![hint.clone(), hint.clone()];
    let opening_points: Vec<&[F]> = vec![&pt0_owned, &pt1_owned];

    let mut prover_transcript = Blake2bTranscript::<F>::new(transcript_label);
    let proof =
        <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::batched_prove(
            &setup,
            prove_inputs_from_groups(
                &opening_points,
                &polys_by_point,
                &commitments_by_point,
                hints_by_point,
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap_or_else(|e| panic!("shared OneHotPoly batched_prove failed at NV={nv}: {e:?}"));

    // Round-trip the proof through serialization and verify.
    let openings_by_point_owned: Vec<Vec<F>> = vec![vec![y0], vec![y1]];
    let openings_by_point: Vec<&[F]> = openings_by_point_owned.iter().map(Vec::as_slice).collect();

    let mut serialized = Vec::new();
    let proof_shape = proof.shape();
    proof
        .serialize_compressed(&mut serialized)
        .expect("serialize");
    let decoded = AkitaBatchedProof::<F>::deserialize_compressed(
        &mut std::io::Cursor::new(serialized),
        &proof_shape,
    )
    .expect("deserialize");

    let mut verifier_transcript = Blake2bTranscript::<F>::new(transcript_label);
    let verify_result = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentVerifier<
        F,
        ONEHOT_D,
    >>::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        verify_inputs_from_groups(&opening_points, &openings_by_point, &commitments_by_point),
        BasisMode::Lagrange,
    );
    assert!(
        verify_result.is_ok(),
        "shared OneHotPoly verification failed at NV={nv}: {:?}",
        verify_result.err()
    );
}

#[test]
fn shared_onehot_commitment_two_points_at_nv19_round_trip() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        run_shared_onehot_two_points_round_trip(19, b"shared_onehot_two_points_nv19");
    });
}

// Sweep across NV={18, 19, 20, 21} to confirm the bug report's "odd NV
// fails / even NV works" pattern is gone for the shared-OneHotPoly +
// two-point opening flow. Each NV must round-trip end-to-end.
#[test]
fn shared_onehot_commitment_two_points_round_trip_sweep() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        for nv in [18usize, 19, 20, 21] {
            let label = format!("shared_onehot_two_points_sweep_nv{nv}");
            run_shared_onehot_two_points_round_trip(nv, label.as_bytes());
        }
    });
}

// Reproduce the asymmetric-dedup pitfall: the prover stores commitments
// in a `Vec` of clones and passes distinct `&commitment` pointers (so
// dedup does NOT fire on the prover, `num_groups=2`), but the verifier
// passes a single `&commitment` pointer twice (so dedup DOES fire,
// `num_groups=1`). The two sides absorb different incidence shapes into
// the Fiat-Shamir transcript and `batched_verify` must reject without
// panicking.
//
// This documents the contract: prover and verifier must use the SAME
// pointer-aliasing strategy for shared commitments. Keeping the test in
// the suite guards against a future change to the dedup semantics that
// would silently make the asymmetric case "work" (and quietly drift the
// transcript binding).
#[test]
fn shared_onehot_commitment_dedup_asymmetry_rejects() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 19;
        let num_points: usize = 2;

        let total_ring = (1usize << NV) / ONEHOT_D;
        let (poly, indices) = make_onehot_poly_from_ring_elems(total_ring, 0xbad1_d00d);

        let setup =
            <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::setup_prover(
                NV, 2, num_points,
            );
        let verifier_setup = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::setup_verifier(&setup);

        let polys_singleton: Vec<OneHotPoly<F, ONEHOT_D, u8>> = vec![poly];
        let (commitment, hint) = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::commit_for_multipoint(
            &polys_singleton, num_points, &setup
        )
        .expect("multipoint onehot commit");

        let pt0 = random_point(NV, 0xdead_0001);
        let pt1 = random_point(NV, 0xdead_0002);
        let y0 = onehot_lagrange_opening(&indices, &pt0);
        let y1 = onehot_lagrange_opening(&indices, &pt1);

        // Prover side: Vec of clones -> two distinct &commitment pointers
        // -> dedup does NOT fire -> num_groups = 2.
        let polys_slice: &[OneHotPoly<F, ONEHOT_D, u8>] = polys_singleton.as_slice();
        let commitments_vec: Vec<RingCommitment<F, ONEHOT_D>> =
            vec![commitment.clone(), commitment.clone()];
        let prover_claims: ProverClaims<F, OneHotPoly<F, ONEHOT_D, u8>, _, _> = vec![
            (
                pt0.as_slice(),
                vec![CommittedPolynomials {
                    polynomials: polys_slice,
                    commitment: &commitments_vec[0],
                    hint: hint.clone(),
                }],
            ),
            (
                pt1.as_slice(),
                vec![CommittedPolynomials {
                    polynomials: polys_slice,
                    commitment: &commitments_vec[1],
                    hint: hint.clone(),
                }],
            ),
        ];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"shared_onehot_dedup_asymmetry");
        let proof = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::batched_prove(&setup, prover_claims, &mut prover_transcript, BasisMode::Lagrange)
        .expect("prover succeeds at its own shape");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize");
        let decoded = AkitaBatchedProof::<F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize");

        // Verifier side: single owned commitment shared between both
        // entries -> dedup DOES fire -> num_groups = 1. Asymmetric.
        let commitment_owned: RingCommitment<F, ONEHOT_D> = commitment;
        let openings_at_pt0 = [y0];
        let openings_at_pt1 = [y1];
        let verifier_claims: VerifierClaims<F, RingCommitment<F, ONEHOT_D>> = vec![
            (
                pt0.as_slice(),
                vec![CommittedOpenings {
                    openings: &openings_at_pt0,
                    commitment: &commitment_owned,
                }],
            ),
            (
                pt1.as_slice(),
                vec![CommittedOpenings {
                    openings: &openings_at_pt1,
                    commitment: &commitment_owned,
                }],
            ),
        ];

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"shared_onehot_dedup_asymmetry");
        let result = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentVerifier<
            F,
            ONEHOT_D,
        >>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verifier_claims,
            BasisMode::Lagrange,
        );
        assert!(
            result.is_err(),
            "asymmetric prover/verifier dedup must be rejected; instead got Ok"
        );
    });
}

// Negative test for the prover-side polys-slice consistency check on
// shared commitments: two `ProverClaims` entries that share a
// `&commitment` pointer but reference DIFFERENT polynomial slices must be
// rejected with a clean `AkitaError::InvalidInput`. This guards the
// safety check in `prover_claims_to_incidence` that prevents a divergent
// witness shape from silently producing a miscommitment.
#[test]
fn prover_claims_reject_shared_commitment_with_divergent_polys() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 15;
        let num_points: usize = 2;

        let total_ring = (1usize << NV) / ONEHOT_D;
        let (poly_a, _) = make_onehot_poly_from_ring_elems(total_ring, 0xabcd_0001);
        let (poly_b, _) = make_onehot_poly_from_ring_elems(total_ring, 0xabcd_0002);

        let setup =
            <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::setup_prover(
                NV, 2, num_points,
            );

        // Commit one of the polys via the multipoint API; the value is
        // arbitrary because the prove call must abort *before* using it.
        let polys_singleton: Vec<OneHotPoly<F, ONEHOT_D, u8>> = vec![poly_a];
        let (commitment, hint) = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::commit_for_multipoint(
            &polys_singleton, num_points, &setup
        )
        .expect("multipoint onehot commit");

        let pt0 = random_point(NV, 0x1111_1111);
        let pt1 = random_point(NV, 0x2222_2222);

        // Two distinct polys slices; the prover-side dedup must reject
        // even though both entries point at the same `&commitment`.
        let polys_a_slice: &[OneHotPoly<F, ONEHOT_D, u8>] = polys_singleton.as_slice();
        let polys_b_owned: Vec<OneHotPoly<F, ONEHOT_D, u8>> = vec![poly_b];
        let polys_b_slice: &[OneHotPoly<F, ONEHOT_D, u8>] = polys_b_owned.as_slice();

        let commitment_owned: RingCommitment<F, ONEHOT_D> = commitment;
        let prover_claims: ProverClaims<F, OneHotPoly<F, ONEHOT_D, u8>, _, _> = vec![
            (
                pt0.as_slice(),
                vec![CommittedPolynomials {
                    polynomials: polys_a_slice,
                    commitment: &commitment_owned,
                    hint: hint.clone(),
                }],
            ),
            (
                pt1.as_slice(),
                vec![CommittedPolynomials {
                    polynomials: polys_b_slice,
                    commitment: &commitment_owned,
                    hint: hint.clone(),
                }],
            ),
        ];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"shared_onehot_polys_divergence");
        let result = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::batched_prove(
            &setup,
            prover_claims,
            &mut prover_transcript,
            BasisMode::Lagrange,
        );
        match result {
            Err(AkitaError::InvalidInput(msg)) => {
                assert!(
                    msg.contains("shared commitment") && msg.contains("identical polynomial slice"),
                    "expected the polys-slice mismatch error, got: {msg}"
                );
            }
            other => panic!(
                "expected Err(InvalidInput(...)) for divergent-polys shared commitment, \
                 got {other:?}"
            ),
        }
    });
}
