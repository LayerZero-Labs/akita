//! Multi-level ring-dimension transition E2E acceptance test.
//!
//! - L0 (root): `d_a/d_b/d_d = 128/128/64`.
//! - L1: `128/64/64`.
//! - L2+: uniform `64`.
//!
//! Exercises distinct per-matrix ring dimensions at both the root and a
//! recursive fold, through the public PCS API, with tamper rejection.

#![allow(missing_docs)]

mod common;

use akita_config::proof_optimized::fp128;
use akita_field::AkitaError;
use akita_pcs::test_support::RingDimensionTransitionConfig;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_transcript::AkitaTranscript;
use akita_types::{
    validate_schedule_ring_dims, CommitmentRingDims, CommittedGroup, OpeningClaimsLayout, RingVec,
};
use common::*;

type Envelope = fp128::D128Dense;
type Suffix = fp128::D64Dense;
/// Root D compressed to 64 (L0 = 128/128/64), L1 = 128/64/64, then 64.
type Cfg = RingDimensionTransitionConfig<Envelope, Suffix, 64, 64>;
type Scheme = AkitaCommitmentScheme<Cfg>;

const NUM_VARS: usize = 16;
const ENVELOPE_D: usize = 128;
const LABEL: &[u8] = b"test/ring_dimension_transition_e2e";

fn dense_poly(seed: u64) -> DensePoly<F> {
    DensePoly::<F>::from_field_evals(NUM_VARS, ENVELOPE_D, &dense_field_evals(NUM_VARS, seed))
        .expect("dense poly")
}

fn verify_with(
    verifier_setup: &akita_types::AkitaVerifierSetup<F>,
    proof: &akita_types::AkitaBatchedProof<F, F>,
    point: &[F],
    openings: &[F],
    commitment: &CommittedGroup<F>,
) -> Result<(), AkitaError> {
    let mut transcript = AkitaTranscript::<F>::new(LABEL);
    Scheme::batched_verify(
        proof,
        verifier_setup,
        &mut transcript,
        verify_input::<Cfg>(point, openings, commitment),
        BasisMode::Lagrange,
    )
}

#[test]
fn ring_dimension_transition_proves_verifies_and_rejects_tamper() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let opening_batch = OpeningClaimsLayout::new(NUM_VARS, 1).expect("opening batch");
        let layout = <Cfg as akita_config::CommitmentConfig>::get_params_for_batched_commitment(
            &opening_batch,
        )
        .expect("commit layout");

        let schedule =
            <Cfg as akita_config::CommitmentConfig>::get_params_for_prove(&opening_batch)
                .expect("schedule");
        // L0 root: A=B=128, D=64.
        assert_eq!(
            schedule.root.params.final_group.commitment.role_dims(),
            CommitmentRingDims {
                inner: 128,
                outer: 128,
                opening: 64
            },
            "L0 must be 128/128/64"
        );
        // L1: A=128, B=D=64.
        assert_eq!(
            schedule.recursive_folds[0].params.witness.role_dims(),
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 64
            },
            "L1 must be 128/64/64"
        );
        // L2: uniform 64.
        assert_eq!(
            schedule.recursive_folds[1].params.witness.role_dims(),
            CommitmentRingDims {
                inner: 64,
                outer: 64,
                opening: 64
            },
            "L2 must be 64/64/64"
        );

        let poly = dense_poly(0x5717_c401);
        let point = random_point(NUM_VARS, 0x5717_c402);
        let opening = opening_from_poly::<ENVELOPE_D, _>(&poly, &point, &layout);

        let setup = Scheme::setup_prover(NUM_VARS, 1).expect("setup");
        validate_schedule_ring_dims(&schedule, setup.expanded.seed())
            .expect("valid per-matrix ring dimensions");
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
            prove_input::<Cfg, _>(&point, &poly_refs, &commitment, hint),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("ring-dimension transition prove");

        verify_with(&verifier_setup, &proof, &point, &[opening], &commitment)
            .expect("honest ring-dimension transition proof must verify");

        verify_with(
            &verifier_setup,
            &proof,
            &point,
            &[opening + F::one()],
            &commitment,
        )
        .expect_err("tampered opening must be rejected");

        let mut tampered_commitment = commitment.clone();
        let mut coeffs = tampered_commitment.commitment.0.coeffs().to_vec();
        coeffs[0] += F::one();
        tampered_commitment.commitment.0 = RingVec::from_coeffs(coeffs);
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
