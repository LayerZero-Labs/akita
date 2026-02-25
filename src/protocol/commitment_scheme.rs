//! Commitment scheme trait and stub implementation.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::commitment::field_to_ring_reduction::{
    reduce_coeffs_to_ring_elements, ring_opening_point_from_field,
};
use crate::protocol::commitment::{
    CommitmentConfig, CommitmentScheme, DefaultCommitmentConfig, HachiCommitmentCore,
    RingCommitment, RingCommitmentScheme, RingCommitmentSetup,
};
use crate::protocol::iteration_prover::HachiProver;
use crate::protocol::proof::HachiProof;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling, Polynomial};

/// Prover-side hint produced at commitment time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiCommitmentHint<F: FieldCore, const D: usize> {
    /// Decomposed `s_i` blocks from the commitment phase.
    pub s: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Decomposed `t̂_i` blocks from the commitment phase.
    pub t_hat: Vec<Vec<CyclotomicRing<F, D>>>,
}

/// Placeholder for the end-to-end PCS wrapper.
#[derive(Clone, Copy, Debug, Default)]
pub struct HachiCommitmentScheme;

impl<F> CommitmentScheme<F> for HachiCommitmentScheme
where
    F: FieldCore + CanonicalField + FieldSampling,
{
    type ProverSetup = RingCommitmentSetup<F, { DefaultCommitmentConfig::D }>;
    type VerifierSetup = RingCommitmentSetup<F, { DefaultCommitmentConfig::D }>;
    type Commitment = RingCommitment<F, { DefaultCommitmentConfig::D }>;
    type Proof = HachiProof<F, { DefaultCommitmentConfig::D }>;
    type OpeningProofHint = HachiCommitmentHint<F, { DefaultCommitmentConfig::D }>;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let (setup, _) = <HachiCommitmentCore as RingCommitmentScheme<
            F,
            { DefaultCommitmentConfig::D },
            DefaultCommitmentConfig,
        >>::setup(max_num_vars)
        .expect("commitment setup failed");
        setup
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        setup.clone()
    }

    fn commit<P: Polynomial<F>>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Commitment, Self::OpeningProofHint), HachiError> {
        let num_vars = poly.num_vars();
        let coeffs = poly.coeffs();
        // Section 3.1 (Reducing to multilinear evaluation over Rq)
        let ring_coeffs = reduce_coeffs_to_ring_elements::<F, { DefaultCommitmentConfig::D }>(
            num_vars, &coeffs, 1,
        )?;
        let (commitment, s, t_hat) = <HachiCommitmentCore as RingCommitmentScheme<
            F,
            { DefaultCommitmentConfig::D },
            DefaultCommitmentConfig,
        >>::commit_coeffs(&ring_coeffs, setup)?;
        let hint = HachiCommitmentHint { s, t_hat };
        Ok((commitment, hint))
    }

    fn prove<T: Transcript<F>, P: Polynomial<F>>(
        setup: &Self::ProverSetup,
        _poly: &P,
        opening_point: &[F],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut T,
    ) -> Result<Self::Proof, HachiError> {
        let hint = hint.ok_or_else(|| {
            HachiError::InvalidInput("missing commitment hint for proving".to_string())
        })?;
        let ring_opening_point = ring_opening_point_from_field::<F, { DefaultCommitmentConfig::D }>(
            opening_point,
            DefaultCommitmentConfig::R,
            DefaultCommitmentConfig::M,
        )?;
        let proof = HachiProver::<F, { DefaultCommitmentConfig::D }>::prove::<
            T,
            DefaultCommitmentConfig,
        >(setup, &ring_opening_point, transcript, &hint)?;
        Ok(proof)
    }

    fn verify<T: Transcript<F>>(
        _proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut T,
        _opening_point: &[F],
        _opening: &F,
        _commitment: &Self::Commitment,
    ) -> Result<(), HachiError> {
        unimplemented!()
    }

    fn combine_commitments(_commitments: &[Self::Commitment], _coeffs: &[F]) -> Self::Commitment {
        unimplemented!()
    }

    fn combine_hints(_hints: Vec<Self::OpeningProofHint>, _coeffs: &[F]) -> Self::OpeningProofHint {
        unimplemented!()
    }

    fn protocol_name() -> &'static [u8] {
        unimplemented!()
    }
}
