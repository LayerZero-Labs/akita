#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

mod common;

use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{commit_with_params, CommitmentProver, ComputeBackendSetup, CpuBackend};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
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

        // Mirror the production multipoint commit layout for opening
        // evaluation; `batched_commit` will use exactly this layout internally.
        let incidence = ClaimIncidenceSummary::from_point_polys(NV, num_polys_per_point.to_vec())
            .expect("valid dense incidence");
        let layout = DenseCfg::get_params_for_batched_commitment(&incidence)
            .expect("dense batched commit layout");

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

        let setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::setup_prover(
            NV,
            total_claims,
            num_polys_per_point.len())
        .unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let verifier_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::setup_verifier(&setup);

        // Public `batched_commit` derives the shared root layout from the
        // full multipoint incidence, so the produced commitments are
        // compatible with the batched prove root by construction.
        let commit_outputs = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::batched_commit(
            &setup, &CpuBackend, &prepared, &polys_per_point_refs
        )
        .expect("dense batched commit");
        let (commitments, hints): (Vec<_>, Vec<_>) = commit_outputs.into_iter().unzip();

        let mut prover_transcript = AkitaTranscript::<F>::new(b"multipoint_batched_e2e/dense");
        let proof = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::batched_prove(
            &setup, &CpuBackend,
            &prepared,
            prove_inputs_from_groups(&opening_points, &polys_per_point_refs, &commitments, hints),
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
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

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"multipoint_batched_e2e/dense");
        let result = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentVerifier<F, DENSE_D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_inputs_from_groups(&opening_points, &openings_per_point_refs, &commitments),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
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
        let layout = OneHotCfg::get_params_for_batched_commitment(&incidence)
            .expect("onehot batched commit layout");

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

        let setup = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::setup_prover(
            NV,
            total_claims,
            num_polys_per_point.len())
        .unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let verifier_setup = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::setup_verifier(&setup);

        let commit_outputs = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::batched_commit(
            &setup, &CpuBackend, &prepared, &polys_per_point_refs
        )
        .expect("onehot batched commit");
        let (commitments, hints): (Vec<_>, Vec<_>) = commit_outputs.into_iter().unzip();

        let mut prover_transcript = AkitaTranscript::<F>::new(b"multipoint_batched_e2e/onehot");
        let proof =
            <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::batched_prove(&setup, &CpuBackend, &prepared, prove_inputs_from_groups(
                    &opening_points,
                    &polys_per_point_refs,
                    &commitments,
                    hints,
                ), &mut prover_transcript, BasisMode::Lagrange, akita_types::SetupContributionMode::Direct)
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

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"multipoint_batched_e2e/onehot");
        let result = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentVerifier<
            F,
            ONEHOT_D,
        >>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_inputs_from_groups(&opening_points, &openings_per_point_refs, &commitments),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        );
        assert!(
            result.is_ok(),
            "onehot multipoint verification failed: {:?}",
            result.err()
        );
    });
}

#[test]
fn multipoint_dense_shared_commitment_round_trip() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 10;
        const BUNDLE: usize = 2;
        const NUM_POINTS: usize = 2;
        // One commitment bundle is reused at every opening point, exercising
        // the `same commitment may be referenced by multiple points` contract
        // documented on `ClaimIncidence`.
        let num_polys_per_point = [BUNDLE; NUM_POINTS];
        let total_claims: usize = num_polys_per_point.iter().sum();

        // The commit-time layout must agree with the multipoint root layout
        // the prover will pick, otherwise per-point re-commit checks reject.
        let incidence = ClaimIncidenceSummary::from_point_polys(NV, num_polys_per_point.to_vec())
            .expect("valid dense incidence");
        let layout = DenseCfg::get_params_for_batched_commitment(&incidence)
            .expect("dense batched commit layout");

        let polys: Vec<DensePoly<F, DENSE_D>> = (0..BUNDLE)
            .map(|poly_idx| make_dense_poly(NV, 0xd3e5_7000 + poly_idx as u64))
            .collect();
        let opening_points_owned: Vec<Vec<F>> = (0..NUM_POINTS)
            .map(|point_idx| random_point(NV, 0xaaaa_7000 + point_idx as u64))
            .collect();
        let openings_per_point: Vec<Vec<F>> = opening_points_owned
            .iter()
            .map(|point| {
                polys
                    .iter()
                    .map(|poly| opening_from_poly(poly, point, &layout))
                    .collect()
            })
            .collect();

        let setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::setup_prover(
            NV,
            total_claims,
            NUM_POINTS)
        .unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let verifier_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::setup_verifier(&setup);

        // Commit the bundle exactly once with the shared multipoint layout.
        let (commitment, hint) =
            commit_with_params::<F, DENSE_D, DensePoly<F, DENSE_D>, CpuBackend>(
                &polys,
                setup.expanded.as_ref(),
                &CpuBackend,
                &prepared,
                &layout,
            )
            .expect("dense single commit");

        let opening_points: Vec<&[F]> = opening_points_owned.iter().map(Vec::as_slice).collect();
        let polys_slice = polys.as_slice();
        let prover_claims: ProverClaims<F, DensePoly<F, DENSE_D>, _, _> = opening_points
            .iter()
            .map(|point| {
                (
                    *point,
                    CommittedPolynomials {
                        polynomials: polys_slice,
                        commitment: &commitment,
                        hint: hint.clone(),
                    },
                )
            })
            .collect();

        let mut prover_transcript =
            AkitaTranscript::<F>::new(b"multipoint_batched_e2e/dense_shared");
        let proof = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::batched_prove(
            &setup, &CpuBackend,
            &prepared,
            prover_claims,
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("shared-commitment multipoint batched prove");

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

        let openings_refs: Vec<&[F]> = openings_per_point.iter().map(Vec::as_slice).collect();
        let verifier_claims: VerifierClaims<F, _> = opening_points
            .iter()
            .zip(openings_refs.iter())
            .map(|(point, openings)| {
                (
                    *point,
                    CommittedOpenings {
                        commitment: &commitment,
                        openings,
                    },
                )
            })
            .collect();

        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"multipoint_batched_e2e/dense_shared");
        let result = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentVerifier<F, DENSE_D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verifier_claims,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        );
        assert!(
            result.is_ok(),
            "shared-commitment multipoint verification failed: {:?}",
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
            let layout = DenseCfg::get_params_for_batched_commitment(&incidence)
                .expect("dense batched commit layout");

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
            >>::setup_prover(NV, total_claims, num_polys_per_point.len())
            .unwrap();
            let prepared = CpuBackend.prepare_setup(&setup).unwrap();
            let verifier_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::setup_verifier(&setup);

            let commit_outputs = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::batched_commit(
                &setup, &CpuBackend, &prepared, &polys_per_point_refs
            )
            .expect("dense batched commit");
            let (commitments, hints): (Vec<_>, Vec<_>) = commit_outputs.into_iter().unzip();

            let mut prover_transcript =
                AkitaTranscript::<F>::new(b"multipoint_batched_e2e/dense_wrong_point");
            let proof = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::batched_prove(
                &setup,
                &CpuBackend,
                &prepared,
                prove_inputs_from_groups(
                    &opening_points,
                    &polys_per_point_refs,
                    &commitments,
                    hints,
                ),
                &mut prover_transcript,
                BasisMode::Lagrange,
                akita_types::SetupContributionMode::Direct,
            )
            .expect("multipoint batched prove");

            let swapped_points = vec![opening_points[1], opening_points[0]];
            let mut verifier_transcript =
                AkitaTranscript::<F>::new(b"multipoint_batched_e2e/dense_wrong_point");
            let result = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentVerifier<
                F,
                DENSE_D,
            >>::batched_verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                verify_inputs_from_groups(&swapped_points, &openings_per_point_refs, &commitments),
                BasisMode::Lagrange,
                akita_types::SetupContributionMode::Direct,
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
            )
            .unwrap();
            let commit_prepared = CpuBackend.prepare_setup(&commit_setup).unwrap();
            // Prove setup with strictly smaller capacity than total_claims.
            let prove_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::setup_prover(
                NV, total_claims - 1, num_polys_per_point.len()
            )
            .unwrap();
            let prove_prepared = CpuBackend.prepare_setup(&prove_setup).unwrap();
            // Use the over-capacity setup so that commit succeeds; the
            // intent is to drive `batched_prove` against `prove_setup` that
            // cannot fit `total_claims` and observe the rejection there.
            let commit_outputs = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::batched_commit(
                &commit_setup,
                &CpuBackend,
                &commit_prepared,
                &polys_per_point_refs,
            )
            .expect("dense batched commit");
            let (commitments, hints): (Vec<_>, Vec<_>) = commit_outputs.into_iter().unzip();
            let mut transcript =
                AkitaTranscript::<F>::new(b"multipoint_batched_e2e/capacity-overflow");
            let result = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::batched_prove(
                &prove_setup,
                &CpuBackend,
                &prove_prepared,
                prove_inputs_from_groups(
                    &opening_points,
                    &polys_per_point_refs,
                    &commitments,
                    hints,
                ),
                &mut transcript,
                BasisMode::Lagrange,
                akita_types::SetupContributionMode::Direct,
            );
            assert!(result.is_err(), "capacity overflow must be rejected");
        });
    }

    #[test]
    fn multipoint_dense_verify_rejects_opening_count_mismatch() {
        init_rayon_pool();
        let _guard = E2E_TEST_LOCK.lock().unwrap();
        run_on_large_stack(|| {
            // Build a valid multipoint proof, then ask the verifier to check
            // it against an opening vector with one fewer entry at the first
            // point. The bundled commitment claims 2 polynomials there, so
            // the verifier must reject the shape before any cryptographic
            // work begins.
            const NV: usize = 10;
            let num_polys_per_point = [2usize, 2];
            let total_claims: usize = num_polys_per_point.iter().sum();
            let incidence =
                ClaimIncidenceSummary::from_point_polys(NV, num_polys_per_point.to_vec())
                    .expect("valid dense incidence");
            let layout = DenseCfg::get_params_for_batched_commitment(&incidence)
                .expect("dense batched commit layout");

            let polys_per_point: Vec<Vec<DensePoly<F, DENSE_D>>> = num_polys_per_point
                .iter()
                .enumerate()
                .map(|(point_idx, &count)| {
                    (0..count)
                        .map(|poly_idx| {
                            make_dense_poly(
                                NV,
                                0xd3e5_8000 + (point_idx as u64) * 100 + poly_idx as u64,
                            )
                        })
                        .collect()
                })
                .collect();
            let opening_points_owned: Vec<Vec<F>> = (0..num_polys_per_point.len())
                .map(|i| random_point(NV, 0xaaaa_8000 + i as u64))
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
            let opening_points: Vec<&[F]> =
                opening_points_owned.iter().map(Vec::as_slice).collect();

            let setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::setup_prover(NV, total_claims, num_polys_per_point.len())
            .unwrap();
            let prepared = CpuBackend.prepare_setup(&setup).unwrap();
            let verifier_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::setup_verifier(&setup);

            let commit_outputs = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::batched_commit(
                &setup, &CpuBackend, &prepared, &polys_per_point_refs
            )
            .expect("dense batched commit");
            let (commitments, hints): (Vec<_>, Vec<_>) = commit_outputs.into_iter().unzip();

            let mut prover_transcript =
                AkitaTranscript::<F>::new(b"multipoint_batched_e2e/dense_opening_count");
            let proof = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::batched_prove(
                &setup,
                &CpuBackend,
                &prepared,
                prove_inputs_from_groups(
                    &opening_points,
                    &polys_per_point_refs,
                    &commitments,
                    hints,
                ),
                &mut prover_transcript,
                BasisMode::Lagrange,
                akita_types::SetupContributionMode::Direct,
            )
            .expect("multipoint batched prove");

            // Drop one opening at point 0 — verifier now sees 1 opening
            // where the bundle claims 2 polynomials.
            let truncated_p0: Vec<F> =
                openings_per_point[0][..openings_per_point[0].len() - 1].to_vec();
            let truncated_refs: Vec<&[F]> =
                vec![truncated_p0.as_slice(), openings_per_point[1].as_slice()];

            let mut verifier_transcript =
                AkitaTranscript::<F>::new(b"multipoint_batched_e2e/dense_opening_count");
            let result = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentVerifier<
                F,
                DENSE_D,
            >>::batched_verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                verify_inputs_from_groups(&opening_points, &truncated_refs, &commitments),
                BasisMode::Lagrange,
                akita_types::SetupContributionMode::Direct,
            );
            assert!(
                result.is_err(),
                "verifier must reject mismatched opening counts"
            );
        });
    }
}
