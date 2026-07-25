//! Per-role commitment-compression E2E acceptance test.
//!
//! The root fold uses non-uniform commitment-role ring dimensions
//! `d_a/d_b/d_d = 128/64/64` (A-role at the envelope dimension, B/D compressed);
//! later folds retain the shipped `D128Dense` schedule. The proof is produced
//! and checked exclusively through the public PCS API
//! (`AkitaCommitmentScheme::{commit, batched_prove, batched_verify}`).
//!
//! This is the correctness oracle for the verifier's non-uniform-role relation
//! evaluation: verifying the honest proof exercises whichever relation path the
//! verifier selects for `role_dims = {128, 64, 64}` (mixed scan or the succinct
//! fast path), and the tamper cases confirm soundness.

#![allow(missing_docs)]

mod common;

use akita_config::proof_optimized::fp128;
use akita_field::AkitaError;
use akita_pcs::test_support::CompressedRoleRootConfig;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_transcript::AkitaTranscript;
use akita_types::{validate_schedule_ring_dims, CommitmentRingDims, OpeningClaimsLayout, RingVec};
use common::*;

/// Envelope preset: uniform `D = 128`, generation ring dimension 128.
type Envelope = fp128::D128Dense;
/// Root commits at A=128 while B and D are compressed to 64.
type Cfg = CompressedRoleRootConfig<Envelope, 64, 64>;
type Scheme = AkitaCommitmentScheme<Cfg>;

const NUM_VARS: usize = 16;
const ENVELOPE_D: usize = 128;
const ROOT_ROLE_DIMS: CommitmentRingDims = CommitmentRingDims {
    inner: 128,
    outer: 64,
    opening: 64,
};
const LABEL: &[u8] = b"test/compressed_role_e2e";

fn dense_poly(seed: u64) -> DensePoly<F> {
    DensePoly::<F>::from_field_evals(NUM_VARS, ENVELOPE_D, &dense_field_evals(NUM_VARS, seed))
        .expect("dense poly")
}

fn verify_with(
    verifier_setup: &akita_types::AkitaVerifierSetup<F>,
    proof: &akita_types::AkitaBatchedProof<F, F>,
    point: &[F],
    openings: &[F],
    commitment: &Commitment<F>,
) -> Result<(), AkitaError> {
    let mut transcript = AkitaTranscript::<F>::new(LABEL);
    Scheme::batched_verify(
        proof,
        verifier_setup,
        &mut transcript,
        verify_input(point, openings, commitment),
        BasisMode::Lagrange,
    )
}

#[test]
fn compressed_role_root_proves_verifies_and_rejects_tamper() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let opening_batch = OpeningClaimsLayout::new(NUM_VARS, 1).expect("opening batch");
        let layout = <Cfg as akita_config::CommitmentConfig>::get_params_for_batched_commitment(
            &opening_batch,
        )
        .expect("commit layout");

        // The whole point of this fixture: the root fold is genuinely per-role.
        let schedule =
            <Cfg as akita_config::CommitmentConfig>::get_params_for_prove(&opening_batch)
                .expect("schedule");
        assert_eq!(
            schedule.root.params.final_group.commitment.role_dims(),
            ROOT_ROLE_DIMS,
            "root must commit at d_a=128, d_b=d_d=64"
        );

        let poly = dense_poly(0xc0de_5501);
        let point = random_point(NUM_VARS, 0xc0de_5502);
        let opening = opening_from_poly::<ENVELOPE_D, _>(&poly, &point, &layout);

        let setup = Scheme::setup_prover(NUM_VARS, 1).expect("setup");
        validate_schedule_ring_dims(&schedule, setup.expanded.seed()).expect("valid role dims");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("prover stack");
        let verifier_setup = Scheme::setup_verifier(&setup).expect("verifier setup");
        let (commitment, hint) =
            Scheme::commit(&setup, std::slice::from_ref(&poly), &stack).expect("commit");

        let poly_refs = [&poly];
        let mut prover_transcript = AkitaTranscript::<F>::new(LABEL);
        let proof = Scheme::batched_prove(
            &setup,
            prove_input(&point, &poly_refs, &commitment, hint),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("per-role prove");

        // Completeness: the honest proof must verify (this exercises the
        // verifier's non-uniform-role relation evaluation).
        verify_with(&verifier_setup, &proof, &point, &[opening], &commitment)
            .expect("honest per-role proof must verify");

        // Soundness (claimed value): a wrong opening must be rejected.
        verify_with(
            &verifier_setup,
            &proof,
            &point,
            &[opening + F::one()],
            &commitment,
        )
        .expect_err("tampered opening must be rejected");

        // Soundness (commitment): a tampered commitment row must be rejected.
        let mut tampered_commitment = commitment.clone();
        let mut coeffs = tampered_commitment.0.coeffs().to_vec();
        coeffs[0] += F::one();
        tampered_commitment.0 = RingVec::from_coeffs(coeffs);
        verify_with(
            &verifier_setup,
            &proof,
            &point,
            &[opening],
            &tampered_commitment,
        )
        .expect_err("tampered commitment must be rejected");
    });
}
