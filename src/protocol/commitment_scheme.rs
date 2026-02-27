//! Commitment scheme trait implementation.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::primitives::poly::multilinear_lagrange_basis;
use crate::protocol::commitment::{
    CommitmentConfig, CommitmentScheme, DefaultCommitmentConfig, HachiCommitmentCore,
    RingCommitmentScheme, RingCommitmentSetup,
};
use crate::protocol::proof::{HachiCommitmentHint, HachiProof, SumcheckAux};
use crate::protocol::quadratic_equation::QuadraticEquation;
use crate::protocol::ring_switch::{ring_switch_prover, ring_switch_verifier};
use crate::protocol::sumcheck::hachi_sumcheck::{
    F0Prover, F0Verifier, FAlphaProver, FAlphaVerifier,
};
use crate::protocol::sumcheck::{prove_sumcheck, verify_sumcheck};
use crate::protocol::transcript::labels::CHALLENGE_SUMCHECK_ROUND;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling, Polynomial};

/// Placeholder for the end-to-end PCS wrapper.
#[derive(Clone, Copy, Debug, Default)]
pub struct HachiCommitmentScheme;

impl<F> CommitmentScheme<F> for HachiCommitmentScheme
where
    F: FieldCore + CanonicalField + FieldSampling,
{
    type ProverSetup = RingCommitmentSetup<F, { DefaultCommitmentConfig::D }>;
    type VerifierSetup = RingCommitmentSetup<F, { DefaultCommitmentConfig::D }>;
    type Commitment =
        crate::protocol::commitment::RingCommitment<F, { DefaultCommitmentConfig::D }>;
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
        commitment: &Self::Commitment,
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

        let y_ring = evaluate_packed_ring_poly::<F, { DefaultCommitmentConfig::D }>(
            &hint.ring_coeffs,
            &opening_point[..reduced_len],
        );

        // §4.2 Quadratic equation
        let quad_eq = QuadraticEquation::<
            F,
            { DefaultCommitmentConfig::D },
            DefaultCommitmentConfig,
        >::new_prover(
            setup,
            &ring_opening_point,
            &hint,
            transcript,
            commitment,
            &y_ring,
        )?;

        // §4.3 Ring switch
        let rs = ring_switch_prover::<F, T, { DefaultCommitmentConfig::D }, DefaultCommitmentConfig>(
            &quad_eq, transcript,
        )?;

        // F_0 sumcheck (range check)
        let mut f0_prover = F0Prover::new(&rs.tau0, rs.w_evals.clone(), rs.b);
        let (f0_proof, ..) = prove_sumcheck::<F, _, F, _, _>(&mut f0_prover, transcript, |tr| {
            tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
        })?;

        // F_alpha sumcheck (evaluation relation)
        let mut f_alpha_prover = FAlphaProver::new(
            rs.w_evals,
            &rs.alpha_evals_y,
            &rs.m_evals_x,
            rs.num_u,
            rs.num_l,
        );
        let (f_alpha_proof, ..) =
            prove_sumcheck::<F, _, F, _, _>(&mut f_alpha_prover, transcript, |tr| {
                tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
            })?;

        Ok(HachiProof {
            v: quad_eq.v,
            y_ring,
            f0_proof,
            f_alpha_proof,
            sumcheck_aux: SumcheckAux { w: rs.w },
            w_commitment: rs.w_commitment,
        })
    }

    fn verify<T: Transcript<F>>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        opening_point: &[F],
        opening: &F,
        commitment: &Self::Commitment,
    ) -> Result<(), HachiError> {
        let alpha_bits = DefaultCommitmentConfig::D.trailing_zeros() as usize;
        let reduced_len = opening_point.len().checked_sub(alpha_bits).ok_or_else(|| {
            HachiError::InvalidSetup("opening point length underflow".to_string())
        })?;
        let reduced_opening_point = &opening_point[..reduced_len];
        let inner_point = &opening_point[reduced_len..];

        // §3.1 trace check
        let v = reduce_inner_openings_to_ring_elements::<F, { DefaultCommitmentConfig::D }>(
            inner_point,
        )?;
        let d = F::from_u64(DefaultCommitmentConfig::D as u64);
        let trace_lhs = trace::<F, { DefaultCommitmentConfig::D }>(&(proof.y_ring * v.sigma_m1()));
        let trace_rhs = d * *opening;
        if trace_lhs != trace_rhs {
            return Err(HachiError::InvalidProof);
        }

        // §4.2 Quadratic equation
        let ring_opening_point = ring_opening_point_from_field::<F, { DefaultCommitmentConfig::D }>(
            reduced_opening_point,
            DefaultCommitmentConfig::R,
            DefaultCommitmentConfig::M,
        )?;
        let quad_eq = QuadraticEquation::<
            F,
            { DefaultCommitmentConfig::D },
            DefaultCommitmentConfig,
        >::new_verifier(
            setup,
            &ring_opening_point,
            &proof.v,
            transcript,
            commitment,
            &proof.y_ring,
        )?;

        // §4.3 Ring switch (verifier side)
        let rs =
            ring_switch_verifier::<F, T, { DefaultCommitmentConfig::D }, DefaultCommitmentConfig>(
                &quad_eq,
                &proof.sumcheck_aux.w,
                &proof.w_commitment,
                transcript,
            )?;

        // F_0 sumcheck verification (range check)
        let f0_verifier = F0Verifier::new(rs.tau0, rs.w_evals.clone(), rs.b);
        verify_sumcheck::<F, _, F, _, _>(&proof.f0_proof, &f0_verifier, transcript, |tr| {
            tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
        })?;

        // F_alpha sumcheck verification (evaluation relation)
        let f_alpha_verifier = FAlphaVerifier::new(
            rs.w_evals,
            rs.alpha_evals_y,
            rs.m_evals_x,
            rs.tau1,
            proof.v.clone(),
            commitment.u.clone(),
            proof.y_ring,
            rs.alpha,
            rs.num_u,
            rs.num_l,
        );
        verify_sumcheck::<F, _, F, _, _>(
            &proof.f_alpha_proof,
            &f_alpha_verifier,
            transcript,
            |tr| tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND),
        )?;

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

/// Re-derive the ring-switch challenge `alpha` and the expanded `M_a` vector
/// by replaying the transcript from the proof data and setup, exactly as the
/// verifier does.
#[cfg(test)]
pub(crate) fn rederive_alpha_and_m_a<F>(
    proof: &HachiProof<F, { DefaultCommitmentConfig::D }>,
    setup: &<HachiCommitmentScheme as CommitmentScheme<F>>::ProverSetup,
    opening_point: &[F],
    commitment: &<HachiCommitmentScheme as CommitmentScheme<F>>::Commitment,
) -> Result<(F, Vec<F>), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + 'static,
{
    let alpha_bits = DefaultCommitmentConfig::D.trailing_zeros() as usize;
    let reduced_len = opening_point
        .len()
        .checked_sub(alpha_bits)
        .ok_or_else(|| HachiError::InvalidSetup("opening point length underflow".to_string()))?;
    let ring_opening_point = ring_opening_point_from_field::<F, { DefaultCommitmentConfig::D }>(
        &opening_point[..reduced_len],
        DefaultCommitmentConfig::R,
        DefaultCommitmentConfig::M,
    )?;
    let mut transcript = crate::protocol::transcript::Blake2bTranscript::<F>::new(
        crate::protocol::transcript::labels::DOMAIN_HACHI_PROTOCOL,
    );
    let quad_eq = QuadraticEquation::<F, { DefaultCommitmentConfig::D }, DefaultCommitmentConfig>::new_verifier(
        setup,
        &ring_opening_point,
        &proof.v,
        &mut transcript,
        commitment,
        &proof.y_ring,
    )?;
    transcript.append_serde(
        crate::protocol::transcript::labels::ABSORB_SUMCHECK_W,
        &proof.w_commitment,
    );
    let alpha: F =
        transcript.challenge_scalar(crate::protocol::transcript::labels::CHALLENGE_RING_SWITCH);
    let m_a = crate::protocol::ring_switch::eval_ring_matrix_at::<F, { DefaultCommitmentConfig::D }>(
        quad_eq.m(),
        &alpha,
    );
    let m_a_vec = crate::protocol::ring_switch::expand_m_a::<
        F,
        { DefaultCommitmentConfig::D },
        DefaultCommitmentConfig,
    >(&m_a, alpha)?;
    Ok((alpha, m_a_vec))
}

// ---------------------------------------------------------------------------
// §3.1 reduction helpers
// ---------------------------------------------------------------------------

fn constant_ring<F: FieldCore, const D: usize>(value: F) -> CyclotomicRing<F, D> {
    let mut coeffs = [F::zero(); D];
    coeffs[0] = value;
    CyclotomicRing::from_coefficients(coeffs)
}

fn lagrange_weights<F: FieldCore>(point: &[F]) -> Vec<F> {
    let len = 1usize << point.len();
    let mut weights = vec![F::zero(); len];
    multilinear_lagrange_basis(&mut weights, point);
    weights
}

fn ring_opening_point_from_field<F: FieldCore, const D: usize>(
    opening_point: &[F],
    r_vars: usize,
    m_vars: usize,
) -> Result<crate::protocol::opening_point::RingOpeningPoint<F, D>, HachiError> {
    let expected_len = r_vars
        .checked_add(m_vars)
        .ok_or_else(|| HachiError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() != expected_len {
        return Err(HachiError::InvalidPointDimension {
            expected: expected_len,
            actual: opening_point.len(),
        });
    }

    let b = lagrange_vector_from_field::<F, D>(&opening_point[..r_vars]);
    let a = lagrange_vector_from_field::<F, D>(&opening_point[r_vars..]);
    Ok(crate::protocol::opening_point::RingOpeningPoint { a, b })
}

fn lagrange_vector_from_field<F: FieldCore, const D: usize>(
    point: &[F],
) -> Vec<CyclotomicRing<F, D>> {
    lagrange_weights(point)
        .into_iter()
        .map(constant_ring::<F, D>)
        .collect()
}

fn reduce_coeffs_to_ring_elements<F: FieldCore, const D: usize>(
    num_vars: usize,
    coeffs: &[F],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    if D == 0 || !D.is_power_of_two() {
        return Err(HachiError::InvalidInput(format!(
            "ring degree D={D} is not a power of two"
        )));
    }
    let alpha = D.trailing_zeros() as usize;
    if num_vars < alpha {
        return Err(HachiError::InvalidInput(format!(
            "num_vars {num_vars} is smaller than alpha {alpha}"
        )));
    }

    let expected_len = 1usize
        .checked_shl(num_vars as u32)
        .ok_or_else(|| HachiError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
    if coeffs.len() != expected_len {
        return Err(HachiError::InvalidSize {
            expected: expected_len,
            actual: coeffs.len(),
        });
    }

    let outer_vars = num_vars - alpha;
    let outer_len = 1usize
        .checked_shl(outer_vars as u32)
        .ok_or_else(|| HachiError::InvalidInput(format!("2^{outer_vars} does not fit usize")))?;

    let mut out = Vec::with_capacity(outer_len);
    for i in 0..outer_len {
        let coeffs = std::array::from_fn(|j| {
            let idx = i + (j << outer_vars);
            coeffs[idx]
        });
        out.push(CyclotomicRing::from_coefficients(coeffs));
    }
    Ok(out)
}

fn reduce_inner_openings_to_ring_elements<F: FieldCore, const D: usize>(
    inner_point: &[F],
) -> Result<CyclotomicRing<F, D>, HachiError> {
    let weights = lagrange_weights(inner_point);
    if weights.len() != D {
        return Err(HachiError::InvalidInput(format!(
            "inner basis length {} does not match D={D}",
            weights.len()
        )));
    }
    let coeffs = std::array::from_fn(|i| weights[i]);
    Ok(CyclotomicRing::from_coefficients(coeffs))
}

fn evaluate_packed_ring_poly<F: FieldCore, const D: usize>(
    packed_coeffs: &[CyclotomicRing<F, D>],
    outer_point: &[F],
) -> CyclotomicRing<F, D> {
    let weights = lagrange_weights(outer_point);
    debug_assert_eq!(weights.len(), packed_coeffs.len());
    packed_coeffs
        .iter()
        .zip(weights.iter())
        .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, w_i)| {
            acc + f_i.scale(w_i)
        })
}

fn trace<F: CanonicalField, const D: usize>(u: &CyclotomicRing<F, D>) -> F {
    let d = F::from_u64(D as u64);
    u.coefficients()[0] * d
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
            &commitment,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/prove");
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
