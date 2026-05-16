#![allow(missing_docs)]
#![cfg(all(feature = "planner", not(feature = "zk")))]

mod common;

use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{commit_with_params, CommitmentProver};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::Blake2bTranscript;
use akita_types::{AkitaBatchedProof, ClaimIncidenceSummary};
use akita_verifier::CommitmentVerifier;
use common::*;
use std::sync::Mutex;

static E2E_TEST_LOCK: Mutex<()> = Mutex::new(());

type OneHotTestPoly = OneHotPoly<F, ONEHOT_D, u8>;

fn make_onehot_poly_from_ring_elems(
    total_ring_elems: usize,
    seed: u64,
) -> (OneHotTestPoly, Vec<Option<u8>>) {
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
fn multipoint_dense_round_trip_with_bundles_per_point() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 10;
        // Three opening points, each bundling 2 polynomials in its commitment.
        let num_polys_per_point = [2usize, 2, 2];
        let total_claims: usize = num_polys_per_point.iter().sum();

        // Commitments at each point must use the same layout the prover root
        // chooses for the full multipoint batch. Fallback to the singleton
        // commit layout if the planner picks a root-direct schedule (no fold).
        let incidence = ClaimIncidenceSummary::from_point_polys(NV, num_polys_per_point.to_vec())
            .expect("valid dense incidence");
        let schedule = DenseCfg::get_params_for_prove(&incidence).expect("dense prove schedule");
        let layout = match schedule.steps.first() {
            Some(akita_types::Step::Fold(root)) => root.params.clone(),
            _ => DenseCfg::commitment_layout(NV).expect("dense singleton commit layout"),
        };

        let polys_per_point: Vec<Vec<DensePoly<F, DENSE_D>>> = num_polys_per_point
            .iter()
            .enumerate()
            .map(|(point_idx, &count)| {
                (0..count)
                    .map(|poly_idx| {
                        make_dense_poly(
                            NV,
                            0xd3e5_1000 + (point_idx as u64) * 100 + poly_idx as u64,
                        )
                    })
                    .collect()
            })
            .collect();
        let opening_points_owned: Vec<Vec<F>> = (0..num_polys_per_point.len())
            .map(|point_idx| random_point(NV, 0xaaaa_1000 + point_idx as u64))
            .collect();
        let openings_per_point: Vec<Vec<F>> = polys_per_point
            .iter()
            .zip(opening_points_owned.iter())
            .map(|(polys, point)| {
                polys
                    .iter()
                    .map(|poly| opening_from_poly(poly, point, &layout))
                    .collect()
            })
            .collect();

        let polys_per_point_refs: Vec<&[DensePoly<F, DENSE_D>]> =
            polys_per_point.iter().map(Vec::as_slice).collect();
        let openings_per_point_refs: Vec<&[F]> =
            openings_per_point.iter().map(Vec::as_slice).collect();
        let opening_points: Vec<&[F]> = opening_points_owned.iter().map(Vec::as_slice).collect();

        let setup =
            <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::setup_prover(
                NV,
                total_claims,
                num_polys_per_point.len(),
            );
        let verifier_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::setup_verifier(&setup);

        // Commit each point's polynomial bundle separately using the shared
        // prove-root layout so the batched prover sees consistent rows.
        let mut commitments = Vec::with_capacity(num_polys_per_point.len());
        let mut hints = Vec::with_capacity(num_polys_per_point.len());
        for polys in &polys_per_point {
            let (commitment, hint) =
                commit_with_params(polys.as_slice(), &setup, &layout).expect("per-point commit");
            commitments.push(commitment);
            hints.push(hint);
        }

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/dense");
        let proof =
            <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::batched_prove(
                &setup,
                prove_inputs_from_groups(
                    &opening_points,
                    &polys_per_point_refs,
                    &commitments,
                    hints,
                ),
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("multipoint batched prove");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/dense");
        let result = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentVerifier<F, DENSE_D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_inputs_from_groups(&opening_points, &openings_per_point_refs, &commitments),
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
fn multipoint_onehot_round_trip_with_bundles_per_point() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 15;
        let num_polys_per_point = [2usize, 2, 2];
        let total_claims: usize = num_polys_per_point.iter().sum();
        let incidence = ClaimIncidenceSummary::from_point_polys(NV, num_polys_per_point.to_vec())
            .expect("valid onehot incidence");
        let schedule = OneHotCfg::get_params_for_prove(&incidence).expect("onehot prove schedule");
        let layout = match schedule.steps.first() {
            Some(akita_types::Step::Fold(root)) => root.params.clone(),
            _ => OneHotCfg::commitment_layout(NV).expect("onehot singleton commit layout"),
        };

        let total_ring = layout.num_blocks * layout.block_len;
        let poly_data_per_point: Vec<Vec<(OneHotTestPoly, Vec<Option<u8>>)>> = num_polys_per_point
            .iter()
            .enumerate()
            .map(|(point_idx, &count)| {
                (0..count)
                    .map(|poly_idx| {
                        make_onehot_poly_from_ring_elems(
                            total_ring,
                            0xa66e_2000 + (point_idx as u64) * 100 + poly_idx as u64,
                        )
                    })
                    .collect()
            })
            .collect();
        let polys_per_point: Vec<Vec<OneHotTestPoly>> = poly_data_per_point
            .iter()
            .map(|polys| polys.iter().map(|(poly, _)| poly.clone()).collect())
            .collect();
        let opening_points_owned: Vec<Vec<F>> = (0..num_polys_per_point.len())
            .map(|point_idx| random_point(NV, 0xf00d_2000 + point_idx as u64))
            .collect();
        let openings_per_point: Vec<Vec<F>> = poly_data_per_point
            .iter()
            .zip(opening_points_owned.iter())
            .map(|(poly_data, point)| {
                poly_data
                    .iter()
                    .map(|(_, indices)| onehot_lagrange_opening(indices, point))
                    .collect()
            })
            .collect();

        let polys_per_point_refs: Vec<&[OneHotTestPoly]> =
            polys_per_point.iter().map(Vec::as_slice).collect();
        let openings_per_point_refs: Vec<&[F]> =
            openings_per_point.iter().map(Vec::as_slice).collect();
        let opening_points: Vec<&[F]> = opening_points_owned.iter().map(Vec::as_slice).collect();

        let setup = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::setup_prover(
            NV,
            total_claims,
            num_polys_per_point.len(),
        );
        let verifier_setup = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::setup_verifier(&setup);

        let mut commitments = Vec::with_capacity(num_polys_per_point.len());
        let mut hints = Vec::with_capacity(num_polys_per_point.len());
        for polys in &polys_per_point {
            let (commitment, hint) = commit_with_params(polys.as_slice(), &setup, &layout)
                .expect("per-point onehot commit");
            commitments.push(commitment);
            hints.push(hint);
        }

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/onehot");
        let proof =
            <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::batched_prove(
                &setup,
                prove_inputs_from_groups(
                    &opening_points,
                    &polys_per_point_refs,
                    &commitments,
                    hints,
                ),
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("multipoint batched prove");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
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
            verify_inputs_from_groups(&opening_points, &openings_per_point_refs, &commitments),
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
            let num_polys_per_point = [2usize, 2];
            let total_claims: usize = num_polys_per_point.iter().sum();
            let incidence =
                ClaimIncidenceSummary::from_point_polys(NV, num_polys_per_point.to_vec())
                    .expect("valid dense incidence");
            let schedule =
                DenseCfg::get_params_for_prove(&incidence).expect("dense prove schedule");
            let layout = match schedule.steps.first() {
                Some(akita_types::Step::Fold(root)) => root.params.clone(),
                _ => DenseCfg::commitment_layout(NV).expect("dense singleton commit layout"),
            };

            let polys_per_point: Vec<Vec<DensePoly<F, DENSE_D>>> = num_polys_per_point
                .iter()
                .enumerate()
                .map(|(point_idx, &count)| {
                    (0..count)
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
            let openings_per_point: Vec<Vec<F>> = polys_per_point
                .iter()
                .zip(opening_points_owned.iter())
                .map(|(polys, point)| {
                    polys
                        .iter()
                        .map(|poly| opening_from_poly(poly, point, &layout))
                        .collect()
                })
                .collect();

            let polys_per_point_refs: Vec<&[DensePoly<F, DENSE_D>]> =
                polys_per_point.iter().map(Vec::as_slice).collect();
            let openings_per_point_refs: Vec<&[F]> =
                openings_per_point.iter().map(Vec::as_slice).collect();
            let opening_points: Vec<&[F]> =
                opening_points_owned.iter().map(Vec::as_slice).collect();

            let setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::setup_prover(NV, total_claims, num_polys_per_point.len());
            let verifier_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::setup_verifier(&setup);

            let mut commitments = Vec::with_capacity(num_polys_per_point.len());
            let mut hints = Vec::with_capacity(num_polys_per_point.len());
            for polys in &polys_per_point {
                let (commitment, hint) = commit_with_params(polys.as_slice(), &setup, &layout)
                    .expect("per-point commit");
                commitments.push(commitment);
                hints.push(hint);
            }

            let mut prover_transcript =
                Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/dense_wrong_point");
            let proof = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::batched_prove(
                &setup,
                prove_inputs_from_groups(
                    &opening_points,
                    &polys_per_point_refs,
                    &commitments,
                    hints,
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
                verify_inputs_from_groups(&swapped_points, &openings_per_point_refs, &commitments),
                BasisMode::Lagrange,
            );
            assert!(result.is_err(), "swapped opening points must be rejected");
        });
    }

    #[test]
    fn multipoint_batched_prove_rejects_capacity_overflow() {
        init_rayon_pool();
        let _guard = E2E_TEST_LOCK.lock().unwrap();
        run_on_large_stack(|| {
            const NV: usize = 10;
            let num_polys_per_point = [4usize, 1];
            let total_claims: usize = num_polys_per_point.iter().sum();

            let polys_per_point: Vec<Vec<DensePoly<F, DENSE_D>>> = num_polys_per_point
                .iter()
                .enumerate()
                .map(|(point_idx, &count)| {
                    (0..count)
                        .map(|poly_idx| {
                            make_dense_poly(
                                NV,
                                0xd3e5_5000 + (point_idx as u64) * 100 + poly_idx as u64,
                            )
                        })
                        .collect()
                })
                .collect();
            let polys_per_point_refs: Vec<&[DensePoly<F, DENSE_D>]> =
                polys_per_point.iter().map(Vec::as_slice).collect();
            let opening_points_owned: Vec<Vec<F>> = (0..num_polys_per_point.len())
                .map(|point_idx| random_point(NV, 0xaaaa_5000 + point_idx as u64))
                .collect();
            let opening_points: Vec<&[F]> =
                opening_points_owned.iter().map(Vec::as_slice).collect();

            let commit_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::setup_prover(
                NV, total_claims, num_polys_per_point.len()
            );
            // Prove setup with strictly smaller capacity than total_claims.
            let prove_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::setup_prover(
                NV, total_claims - 1, num_polys_per_point.len()
            );
            // Use commit_setup's commit layout — but with one-commit-per-point
            // we need a layout that matches what the prover will run at
            // root. Build it from a "fits-in-setup" incidence.
            let commit_incidence =
                ClaimIncidenceSummary::from_point_polys(NV, num_polys_per_point.to_vec())
                    .expect("commit incidence");
            let commit_schedule =
                DenseCfg::get_params_for_prove(&commit_incidence).expect("commit schedule");
            let commit_layout = match commit_schedule.steps.first() {
                Some(akita_types::Step::Fold(root)) => root.params.clone(),
                _ => DenseCfg::commitment_layout(NV).expect("dense singleton commit layout"),
            };
            let mut commitments = Vec::with_capacity(num_polys_per_point.len());
            let mut hints = Vec::with_capacity(num_polys_per_point.len());
            for polys in &polys_per_point {
                let (commitment, hint) =
                    commit_with_params(polys.as_slice(), &commit_setup, &commit_layout)
                        .expect("per-point commit");
                commitments.push(commitment);
                hints.push(hint);
            }
            let mut transcript =
                Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/capacity-overflow");
            let result = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::batched_prove(
                &prove_setup,
                prove_inputs_from_groups(
                    &opening_points,
                    &polys_per_point_refs,
                    &commitments,
                    hints,
                ),
                &mut transcript,
                BasisMode::Lagrange,
            );
            assert!(result.is_err(), "capacity overflow must be rejected");
        });
    }
}
