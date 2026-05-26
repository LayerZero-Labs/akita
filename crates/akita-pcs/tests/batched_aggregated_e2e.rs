//! End-to-end tests for **batched aggregated** commitments.
//!
//! All polynomials in a batch are placed into a single commitment group, so
//! `batched_commit` produces exactly one commitment that aggregates every
//! polynomial.  The test exercises `batched_commit` → `batched_prove` →
//! serialize/deserialize → `batched_verify`.
//!
//! This file intentionally keeps a much smaller matrix than the grouped and
//! multipoint suites, because those tests already cover most batching-shape
//! permutations. The aggregated suite now focuses on the unique
//! single-commitment path with a few representative cases:
//!
//! * **One-hot** — singleton baseline, irregular larger batch, and a
//!   max-`nv` schedule.
//! * **Dense** — singleton baseline and irregular larger batch.
//! * **Mixed dense + one-hot under the dense config** — heterogeneous
//!   aggregated commitment/proof/verify.
//!
//! This keeps good coverage of the aggregated path while avoiding the old
//! near-cartesian-product runtime blowup.

#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

mod common;

use akita_pcs::AkitaCommitmentScheme;
use akita_prover::CommitmentProver;
use akita_prover::MultilinearPolynomial;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{AkitaBatchedProof, ClaimIncidenceSummary};
use akita_verifier::CommitmentVerifier;
use common::*;

const DENSE_ONEHOT_K: usize = DENSE_D;

fn make_dense_cfg_onehot_poly(layout: &LevelParams, seed: u64) -> OneHotPoly<F, DENSE_D, u8> {
    let total_ring = layout.num_blocks * layout.block_len;
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_ring)
        .map(|_| Some(rng.gen_range(0..DENSE_ONEHOT_K) as u8))
        .collect();
    OneHotPoly::<F, DENSE_D, u8>::new(DENSE_ONEHOT_K, indices)
        .expect("onehot poly under dense config")
}

#[cfg(not(feature = "zk"))]
mod non_zk_aggregated_cases {
    use super::*;
    use akita_prover::{ComputeBackendSetup, CpuBackend};

    /// All one-hot polynomials are aggregated into a single commitment group.
    fn run_aggregated_onehot(nv: usize, batch_size: usize) {
        init_rayon_pool();
        run_on_large_stack(move || {
            let incidence = ClaimIncidenceSummary::same_point(nv, batch_size).expect("incidence");
            let layout = OneHotCfg::get_params_for_batched_commitment(&incidence).expect("layout");

            let polys: Vec<OneHotPoly<F, ONEHOT_D, u8>> = (0..batch_size)
                .map(|idx| make_onehot_poly(&layout, 0xa66e_0000 + (nv as u64) * 100 + idx as u64))
                .collect();

            let pt = random_point(nv, 0xf00d_0000 + nv as u64);
            let openings: Vec<F> = polys
                .iter()
                .map(|poly| opening_from_poly(poly, &pt, &layout))
                .collect();

            let setup = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
                F,
                ONEHOT_D,
            >>::setup_prover(nv, batch_size, 1)
            .unwrap();
            let prepared = CpuBackend.prepare_setup(&setup).unwrap();
            let verifier_setup = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
                F,
                ONEHOT_D,
            >>::setup_verifier(&setup);

            let (commitment, hint) = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
                F,
                ONEHOT_D,
            >>::commit(&setup, &CpuBackend, &prepared, &polys)
            .expect("grouped commit");
            let commitments = [commitment];
            let hints = vec![hint];

            assert_eq!(
                commitments.len(),
                1,
                "single group should yield exactly one commitment"
            );

            let mut prover_transcript = AkitaTranscript::<F>::new(b"batched_aggregated_e2e/onehot");
            let proof = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
                F,
                ONEHOT_D,
            >>::batched_prove(
                &setup,
                &CpuBackend,
                &prepared,
                prove_input(
                    &pt[..],
                    &polys[..],
                    &commitments[0],
                    hints.into_iter().next().unwrap(),
                ),
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("batched prove");

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

            let opening_groups: [&[F]; 1] = [&openings];
            let mut verifier_transcript =
                AkitaTranscript::<F>::new(b"batched_aggregated_e2e/onehot");
            let result = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentVerifier<
                F,
                ONEHOT_D,
            >>::batched_verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input(&pt[..], opening_groups[0], &commitments[0]),
                BasisMode::Lagrange,
            );
            assert!(
                result.is_ok(),
                "aggregated onehot nv={nv} batch={batch_size} verification failed: {:?}",
                result.err()
            );
        });
    }

    /// All dense polynomials are aggregated into a single commitment group.
    fn run_aggregated_dense(nv: usize, batch_size: usize) {
        init_rayon_pool();
        run_on_large_stack(move || {
            let incidence = ClaimIncidenceSummary::same_point(nv, batch_size).expect("incidence");
            let layout = DenseCfg::get_params_for_batched_commitment(&incidence).expect("layout");

            let polys: Vec<DensePoly<F, DENSE_D>> = (0..batch_size)
                .map(|idx| make_dense_poly(nv, 0xd3e5_0000 + (nv as u64) * 100 + idx as u64))
                .collect();

            let pt = random_point(nv, 0xaaaa_0000 + nv as u64);
            let openings: Vec<F> = polys
                .iter()
                .map(|poly| opening_from_poly(poly, &pt, &layout))
                .collect();

            let setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::setup_prover(nv, batch_size, 1)
            .unwrap();
            let prepared = CpuBackend.prepare_setup(&setup).unwrap();
            let verifier_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::setup_verifier(&setup);

            let (commitments, hints) =
                <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::commit(
                    &setup,
                    &CpuBackend,
                    &prepared,
                    &polys,
                )
                .map(|(commitment, hint)| (vec![commitment], vec![hint]))
                .expect("grouped commit");

            assert_eq!(
                commitments.len(),
                1,
                "single group should yield exactly one commitment"
            );

            let mut prover_transcript = AkitaTranscript::<F>::new(b"batched_aggregated_e2e/dense");
            let proof = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
                F,
                DENSE_D,
            >>::batched_prove(
                &setup,
                &CpuBackend,
                &prepared,
                prove_input(
                    &pt[..],
                    &polys[..],
                    &commitments[0],
                    hints.into_iter().next().unwrap(),
                ),
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("batched prove");

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

            let opening_groups: [&[F]; 1] = [&openings];
            let mut verifier_transcript =
                AkitaTranscript::<F>::new(b"batched_aggregated_e2e/dense");
            let result = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentVerifier<
                F,
                DENSE_D,
            >>::batched_verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input(&pt[..], opening_groups[0], &commitments[0]),
                BasisMode::Lagrange,
            );
            assert!(
                result.is_ok(),
                "aggregated dense nv={nv} batch={batch_size} verification failed: {:?}",
                result.err()
            );
        });
    }

    macro_rules! aggregated_onehot_case {
        ($name:ident, $nv:expr, $batch:expr) => {
            #[test]
            fn $name() {
                run_aggregated_onehot($nv, $batch);
            }
        };
    }

    macro_rules! aggregated_dense_case {
        ($name:ident, $nv:expr, $batch:expr) => {
            #[test]
            fn $name() {
                run_aggregated_dense($nv, $batch);
            }
        };
    }

    aggregated_onehot_case!(aggregated_onehot_nv10_batch1, 10, 1);
    aggregated_onehot_case!(aggregated_onehot_nv20_batch7, 20, 7);
    aggregated_onehot_case!(aggregated_onehot_nv25_batch4, 25, 4);

    aggregated_dense_case!(aggregated_dense_nv10_batch1, 10, 1);
    aggregated_dense_case!(aggregated_dense_nv20_batch7, 20, 7);
}

#[test]
fn aggregated_mixed_dense_and_onehot_under_dense_cfg() {
    init_rayon_pool();
    run_on_large_stack(|| {
        const NV: usize = 20;
        const BATCH_SIZE: usize = 4;

        let incidence = ClaimIncidenceSummary::same_point(NV, BATCH_SIZE).expect("incidence");
        let layout = DenseCfg::get_params_for_batched_commitment(&incidence).expect("layout");
        let dense_a = make_dense_poly(NV, 0x4d10_0001);
        let dense_b = make_dense_poly(NV, 0x4d10_0002);
        let onehot_a = make_dense_cfg_onehot_poly(&layout, 0x4d10_1001);
        let onehot_b = make_dense_cfg_onehot_poly(&layout, 0x4d10_1002);

        let polys = [
            MultilinearPolynomial::dense(&dense_a),
            MultilinearPolynomial::onehot(&onehot_a),
            MultilinearPolynomial::dense(&dense_b),
            MultilinearPolynomial::onehot(&onehot_b),
        ];
        let pt = random_point(NV, 0x4d10_ffff);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly(poly, &pt, &layout))
            .collect();

        let setup =
            <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::setup_prover(
                NV,
                BATCH_SIZE,
                1,
            ).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let verifier_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::setup_verifier(&setup);

        let (commitment, hint) = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::commit(&setup, &CpuBackend, &prepared, &polys)
        .expect("mixed aggregated commit");
        let commitments = [commitment];
        let hints = vec![hint];

        let mut prover_transcript =
            AkitaTranscript::<F>::new(b"batched_aggregated_e2e/mixed_dense_onehot");
        let proof = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::batched_prove(
            &setup, &CpuBackend,
            &prepared,
            prove_input(
                &pt[..],
                &polys[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("mixed batched prove");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize mixed batched proof");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize mixed batched proof");

        let opening_groups: [&[F]; 1] = [&openings];
        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"batched_aggregated_e2e/mixed_dense_onehot");
        let result =
            <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentVerifier<F, DENSE_D>>::batched_verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input(&pt[..], opening_groups[0], &commitments[0]),
                BasisMode::Lagrange,
            );
        assert!(
            result.is_ok(),
            "aggregated mixed dense/onehot verification failed: {:?}",
            result.err()
        );
    });
}
