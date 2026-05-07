#![allow(missing_docs)]
#![cfg(feature = "planner")]

mod common;

use akita_config::akita_batched_root_layout;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::CommitmentProver;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::Blake2bTranscript;
use akita_types::AkitaBatchedProof;
use akita_verifier::CommitmentVerifier;
use common::*;
use std::sync::Mutex;

static E2E_TEST_LOCK: Mutex<()> = Mutex::new(());

type OneHotTestPoly = OneHotPoly<F, ONEHOT_D, u8>;
type OneHotIndexData = Vec<Option<u8>>;
type PointOneHotPolyData = Vec<Vec<(OneHotTestPoly, OneHotIndexData)>>;

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
        let layout = akita_batched_root_layout::<DenseCfg>(NV, total_claims).unwrap();

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

        let point_group_counts: Vec<usize> = point_group_sizes.iter().map(Vec::len).collect();
        let (commitments_by_point, hints_by_point) =
            <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::batched_commit(
                &polys_by_point,
                &point_group_counts,
                &setup,
            )
            .expect("multipoint batched commit");

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/dense");
        let (statement, prove_polys, prove_hints) = prove_inputs_from_groups(
            &opening_points,
            &openings_by_point,
            &polys_by_point,
            &commitments_by_point,
            hints_by_point,
        );
        let proof =
            <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::batched_prove(
                &setup,
                statement,
                prove_polys,
                prove_hints,
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
        let layout = akita_batched_root_layout::<DenseCfg>(NV, total_claims).unwrap();
        let openings_a = vec![opening_from_poly(&group_a[0], &point, &layout)];
        let openings_b = group_b
            .iter()
            .map(|poly| opening_from_poly(poly, &point, &layout))
            .collect::<Vec<_>>();

        let setup =
            <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::setup_prover(
                NV,
                total_claims,
                1,
            );
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

        let statement = OpeningStatement::new(
            vec![point.as_slice()],
            commitments.clone(),
            openings_a
                .iter()
                .chain(openings_b.iter())
                .copied()
                .collect(),
            vec![
                vec![PointToPolynomialMap {
                    point_idx: 0,
                    polynomial_idx: 0,
                }],
                vec![
                    PointToPolynomialMap {
                        point_idx: 0,
                        polynomial_idx: 1,
                    },
                    PointToPolynomialMap {
                        point_idx: 0,
                        polynomial_idx: 2,
                    },
                ],
            ],
        )
        .unwrap();
        let prove_polys = vec![&group_a[0], &group_b[0], &group_b[1]];
        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/uneven-dense");
        let proof =
            <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::batched_prove(
                &setup,
                statement.clone(),
                prove_polys,
                hints,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("uneven grouped batched prove");

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/uneven-dense");
        <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentVerifier<F, DENSE_D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            statement,
            BasisMode::Lagrange,
        )
        .expect("uneven grouped batched verify");
    });
}

#[test]
fn multipoint_onehot_round_trip_with_mixed_groups() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 15;
        let point_group_sizes = [vec![2], vec![2], vec![2]];
        let total_claims: usize = point_group_sizes.iter().flatten().sum();
        let layout = akita_batched_root_layout::<OneHotCfg>(NV, total_claims).unwrap();

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

        let point_group_counts: Vec<usize> = point_group_sizes.iter().map(Vec::len).collect();
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
        let (statement, prove_polys, prove_hints) = prove_inputs_from_groups(
            &opening_points,
            &openings_by_point,
            &polys_by_point,
            &commitments_by_point,
            hints_by_point,
        );
        let proof =
            <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::batched_prove(
                &setup,
                statement,
                prove_polys,
                prove_hints,
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

#[test]
fn multipoint_dense_verify_rejects_swapped_points() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 10;
        let point_group_sizes = [vec![2], vec![2]];
        let total_claims = 4usize;
        let layout = akita_batched_root_layout::<DenseCfg>(NV, total_claims).unwrap();

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

        let point_group_counts: Vec<usize> = point_group_sizes.iter().map(Vec::len).collect();
        let (commitments_by_point, hints_by_point) =
            <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::batched_commit(
                &polys_by_point,
                &point_group_counts,
                &setup,
            )
            .expect("multipoint batched commit");

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/dense_wrong_point");
        let (statement, prove_polys, prove_hints) = prove_inputs_from_groups(
            &opening_points,
            &openings_by_point,
            &polys_by_point,
            &commitments_by_point,
            hints_by_point,
        );
        let proof =
            <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::batched_prove(
                &setup,
                statement,
                prove_polys,
                prove_hints,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("multipoint batched prove");

        let swapped_points = vec![opening_points[1], opening_points[0]];
        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/dense_wrong_point");
        let result = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentVerifier<F, DENSE_D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_inputs_from_groups(&swapped_points, &openings_by_point, &commitments_by_point),
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
        let layout = akita_batched_root_layout::<OneHotCfg>(NV, total_claims).unwrap();

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
        let openings_by_point_refs: Vec<&[F]> =
            openings_by_point.iter().map(Vec::as_slice).collect();
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

        let point_group_counts: Vec<usize> = point_group_sizes.iter().map(Vec::len).collect();
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
        let (statement, prove_polys, prove_hints) = prove_inputs_from_groups(
            &opening_points,
            &openings_by_point_refs,
            &polys_by_point,
            &commitments_by_point,
            hints_by_point,
        );
        let proof =
            <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::batched_prove(
                &setup,
                statement,
                prove_polys,
                prove_hints,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("multipoint batched prove");

        let wrong_openings_by_point = vec![&openings_by_point[0][..1], &openings_by_point[1][..]];

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
        let layout = akita_batched_root_layout::<DenseCfg>(NV, total_claims).unwrap();
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
        let openings_by_point: Vec<&[F]> = openings_by_point.iter().map(Vec::as_slice).collect();
        let opening_points: Vec<&[F]> = opening_points_owned.iter().map(Vec::as_slice).collect();

        let commit_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::setup_prover(NV, total_claims, point_group_sizes.len());
        let prove_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::setup_prover(NV, total_claims - 1, point_group_sizes.len());
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
        let (statement, prove_polys, prove_hints) = prove_inputs_from_groups(
            &opening_points,
            &openings_by_point,
            &polys_by_point,
            &commitments_by_point,
            hints_by_point,
        );
        let result = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::batched_prove(
            &prove_setup,
            statement,
            prove_polys,
            prove_hints,
            &mut transcript,
            BasisMode::Lagrange,
        );
        assert!(result.is_err(), "capacity overflow must be rejected");
    });
}
