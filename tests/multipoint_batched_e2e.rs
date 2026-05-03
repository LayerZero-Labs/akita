#![allow(missing_docs)]

mod common;

use akita_transcript::Blake2bTranscript;
use common::*;
use hachi_pcs::protocol::commitment::hachi_batched_root_layout;
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::proof::HachiBatchedProof;
use hachi_pcs::{
    CommitmentProver, CommitmentVerifier, FieldCore, HachiDeserialize, HachiSerialize, Transcript,
};
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
        let layout = hachi_batched_root_layout::<DenseCfg>(NV, total_claims).unwrap();

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
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::setup_prover(
                NV,
                total_claims,
                point_group_sizes.len(),
            );
        let verifier_setup = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::setup_verifier(&setup);

        let point_group_counts: Vec<usize> = point_group_sizes.iter().map(Vec::len).collect();
        let (commitments_by_point, hints_by_point) =
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::batched_commit(
                &polys_by_point,
                &point_group_counts,
                &setup,
            )
            .expect("multipoint batched commit");

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/dense");
        let proof =
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::batched_prove(
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
        let decoded = HachiBatchedProof::<F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/dense");
        let result = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentVerifier<F, DENSE_D>>::batched_verify(
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
        let layout = hachi_batched_root_layout::<DenseCfg>(NV, total_claims).unwrap();
        let openings_a = vec![opening_from_poly(&group_a[0], &point, &layout)];
        let openings_b = group_b
            .iter()
            .map(|poly| opening_from_poly(poly, &point, &layout))
            .collect::<Vec<_>>();

        let setup =
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::setup_prover(
                NV,
                total_claims,
                1,
            );
        let verifier_setup = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::setup_verifier(&setup);
        let (commitments, hints) = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
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
        let proof =
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::batched_prove(
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
        <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentVerifier<F, DENSE_D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verifier_claims,
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
        let layout = hachi_batched_root_layout::<OneHotCfg>(NV, total_claims).unwrap();

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
            <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::setup_prover(
                NV,
                total_claims,
                point_group_sizes.len(),
            );
        let verifier_setup = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::setup_verifier(&setup);

        let point_group_counts: Vec<usize> = point_group_sizes.iter().map(Vec::len).collect();
        let (commitments_by_point, hints_by_point) = <HachiCommitmentScheme<
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
            <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::batched_prove(
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
        let decoded = HachiBatchedProof::<F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/onehot");
        let result = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentVerifier<
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
        let layout = hachi_batched_root_layout::<DenseCfg>(NV, total_claims).unwrap();

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
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::setup_prover(
                NV,
                total_claims,
                point_group_sizes.len(),
            );
        let verifier_setup = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::setup_verifier(&setup);

        let point_group_counts: Vec<usize> = point_group_sizes.iter().map(Vec::len).collect();
        let (commitments_by_point, hints_by_point) =
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::batched_commit(
                &polys_by_point,
                &point_group_counts,
                &setup,
            )
            .expect("multipoint batched commit");

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/dense_wrong_point");
        let proof =
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::batched_prove(
                &setup,
                prove_inputs_from_groups(&opening_points, &polys_by_point, &commitments_by_point, hints_by_point),
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("multipoint batched prove");

        let swapped_points = vec![opening_points[1], opening_points[0]];
        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/dense_wrong_point");
        let result = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentVerifier<F, DENSE_D>>::batched_verify(
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
        let layout = hachi_batched_root_layout::<OneHotCfg>(NV, total_claims).unwrap();

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
        let opening_points: Vec<&[F]> = opening_points_owned.iter().map(Vec::as_slice).collect();

        let setup =
            <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::setup_prover(
                NV,
                total_claims,
                point_group_sizes.len(),
            );
        let verifier_setup = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::setup_verifier(&setup);

        let point_group_counts: Vec<usize> = point_group_sizes.iter().map(Vec::len).collect();
        let (commitments_by_point, hints_by_point) = <HachiCommitmentScheme<
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
        let proof =
            <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::batched_prove(
                &setup,
                prove_inputs_from_groups(&opening_points, &polys_by_point, &commitments_by_point, hints_by_point),
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("multipoint batched prove");

        let wrong_openings_by_point = vec![&openings_by_point[0][..1], &openings_by_point[1][..]];

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/onehot_wrong_opening_count");
        let result = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentVerifier<
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
        let opening_points: Vec<&[F]> = opening_points_owned.iter().map(Vec::as_slice).collect();

        let commit_setup = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::setup_prover(NV, total_claims, point_group_sizes.len());
        let prove_setup = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::setup_prover(NV, total_claims - 1, point_group_sizes.len());
        let point_group_counts: Vec<usize> = point_group_sizes.iter().map(Vec::len).collect();
        let (commitments_by_point, hints_by_point) =
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::batched_commit(
                &polys_by_point,
                &point_group_counts,
                &commit_setup,
            )
            .expect("multipoint batched commit should fit with matching setup");
        let mut transcript =
            Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/capacity-overflow");
        let result = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::batched_prove(
            &prove_setup,
            prove_inputs_from_groups(&opening_points, &polys_by_point, &commitments_by_point, hints_by_point),
            &mut transcript,
            BasisMode::Lagrange,
        );
        assert!(result.is_err(), "capacity overflow must be rejected");
    });
}
