//! Commitment scheme trait implementation.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::commitment::field_to_ring_reduction::{
    evaluate_packed_ring_poly, reduce_coeffs_to_ring_elements,
    reduce_inner_openings_to_ring_elements, ring_opening_point_from_field, trace,
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
    /// Ring coefficients from the §3.1 reduction (evaluation table).
    pub ring_coeffs: Vec<CyclotomicRing<F, D>>,
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
        let ring_coeffs =
            reduce_coeffs_to_ring_elements::<F, { DefaultCommitmentConfig::D }>(num_vars, &coeffs)?;
        let (commitment, s, t_hat) = <HachiCommitmentCore as RingCommitmentScheme<
            F,
            { DefaultCommitmentConfig::D },
            DefaultCommitmentConfig,
        >>::commit_coeffs(&ring_coeffs, setup)?;
        let hint = HachiCommitmentHint {
            s,
            t_hat,
            ring_coeffs,
        };
        Ok((commitment, hint))
    }

    fn prove<T: Transcript<F>, P: Polynomial<F>>(
        setup: &Self::ProverSetup,
        poly: &P,
        opening_point: &[F],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut T,
    ) -> Result<Self::Proof, HachiError> {
        let hint = hint.ok_or_else(|| {
            HachiError::InvalidInput("missing commitment hint for proving".to_string())
        })?;
        let num_vars = poly.num_vars();
        let alpha = DefaultCommitmentConfig::D.trailing_zeros() as usize;
        let reduced_len = num_vars
            .checked_sub(alpha)
            .ok_or_else(|| HachiError::InvalidSetup("reduction length underflow".to_string()))?;
        if opening_point.len() < reduced_len {
            return Err(HachiError::InvalidPointDimension {
                expected: reduced_len,
                actual: opening_point.len(),
            });
        }

        let ring_opening_point = ring_opening_point_from_field::<F, { DefaultCommitmentConfig::D }>(
            &opening_point[..reduced_len],
            DefaultCommitmentConfig::R,
            DefaultCommitmentConfig::M,
        )?;

        let y = evaluate_packed_ring_poly::<F, { DefaultCommitmentConfig::D }>(
            &hint.ring_coeffs,
            &opening_point[..reduced_len],
        );

        let proof = HachiProver::<F, { DefaultCommitmentConfig::D }>::prove::<
            T,
            DefaultCommitmentConfig,
        >(setup, &ring_opening_point, transcript, &hint)?;
        Ok(HachiProof {
            v: proof.v,
            y_ring: y,
        })
    }

    fn verify<T: Transcript<F>>(
        proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut T,
        opening_point: &[F],
        opening: &F,
        _commitment: &Self::Commitment,
    ) -> Result<(), HachiError> {
        let alpha = DefaultCommitmentConfig::D.trailing_zeros() as usize;
        let reduced_len = opening_point.len().checked_sub(alpha).ok_or_else(|| {
            HachiError::InvalidSetup("opening point length underflow".to_string())
        })?;
        let inner_point = &opening_point[reduced_len..];

        let v = reduce_inner_openings_to_ring_elements::<F, { DefaultCommitmentConfig::D }>(
            inner_point,
        )?;
        let d = F::from_u64(DefaultCommitmentConfig::D as u64);
        let trace_lhs = trace::<F, { DefaultCommitmentConfig::D }>(&(proof.y_ring * v.sigma_m1()));
        let trace_rhs = d * *opening;
        if trace_lhs != trace_rhs {
            return Err(HachiError::InvalidProof);
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::multilinear_evals::DenseMultilinearEvals;
    use crate::protocol::commitment::CommitmentConfig;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::test_utils::F;
    use crate::{CommitmentScheme, Polynomial};

    #[test]
    fn verify_passes_for_consistent_opening() {
        let alpha = DefaultCommitmentConfig::D.trailing_zeros() as usize;
        let num_vars = DefaultCommitmentConfig::R + DefaultCommitmentConfig::M + alpha;
        let len = 1usize << num_vars;

        let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DenseMultilinearEvals::new_padded(evals);

        let setup = <HachiCommitmentScheme as CommitmentScheme<F>>::setup_prover(num_vars);
        let verifier_setup = <HachiCommitmentScheme as CommitmentScheme<F>>::setup_verifier(&setup);

        let (commitment, hint) =
            <HachiCommitmentScheme as CommitmentScheme<F>>::commit(&poly, &setup).unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let opening = poly.evaluate(&opening_point);

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let proof = <HachiCommitmentScheme as CommitmentScheme<F>>::prove(
            &setup,
            &poly,
            &opening_point,
            Some(hint),
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/verify");
        let result = <HachiCommitmentScheme as CommitmentScheme<F>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &opening,
            &commitment,
        );

        assert!(result.is_ok());
    }
}
