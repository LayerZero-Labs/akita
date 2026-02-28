//! Commitment scheme trait implementation.

use crate::algebra::ring::CyclotomicRing;
use crate::cfg_into_iter;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::primitives::poly::multilinear_lagrange_basis;
use crate::protocol::commitment::{
    CommitmentConfig, CommitmentScheme, HachiCommitmentCore, RingCommitment, RingCommitmentScheme,
    RingCommitmentSetup,
};
use crate::protocol::opening_point::RingOpeningPoint;
use crate::protocol::proof::{HachiCommitmentHint, HachiProof, SumcheckAux};
use crate::protocol::quadratic_equation::QuadraticEquation;
use crate::protocol::ring_switch::{ring_switch_prover, ring_switch_verifier};
use crate::protocol::sumcheck::batched_sumcheck::{
    prove_batched_sumcheck, verify_batched_sumcheck,
};
use crate::protocol::sumcheck::norm_sumcheck::{NormSumcheckProver, NormSumcheckVerifier};
use crate::protocol::sumcheck::relation_sumcheck::{
    RelationSumcheckProver, RelationSumcheckVerifier,
};
use crate::protocol::sumcheck::{SumcheckInstanceProver, SumcheckInstanceVerifier};
use crate::protocol::transcript::labels::CHALLENGE_SUMCHECK_ROUND;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling, FromSmallInt, Polynomial};

#[cfg(test)]
use crate::protocol::ring_switch::{eval_ring_matrix_at, expand_m_a};
#[cfg(test)]
use crate::protocol::transcript::labels::{
    ABSORB_SUMCHECK_W, CHALLENGE_RING_SWITCH, DOMAIN_HACHI_PROTOCOL,
};
#[cfg(test)]
use crate::protocol::transcript::Blake2bTranscript;
#[cfg(test)]
use crate::protocol::SmallTestCommitmentConfig;

/// End-to-end PCS wrapper, generic over ring degree `D` and config `Cfg`.
#[derive(Clone, Copy, Debug, Default)]
pub struct HachiCommitmentScheme<const D: usize, Cfg: CommitmentConfig> {
    _cfg: std::marker::PhantomData<Cfg>,
}

impl<F, const D: usize, Cfg> CommitmentScheme<F> for HachiCommitmentScheme<D, Cfg>
where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig,
{
    type ProverSetup = RingCommitmentSetup<F, D>;
    type VerifierSetup = RingCommitmentSetup<F, D>;
    type Commitment = RingCommitment<F, D>;
    type Proof = HachiProof<F, D>;
    type OpeningProofHint = HachiCommitmentHint<F, D>;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let (setup, _) =
            <HachiCommitmentCore as RingCommitmentScheme<F, { D }, Cfg>>::setup(max_num_vars)
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
        let ring_coeffs =
            reduce_coeffs_to_ring_elements::<F, { D }>(poly.num_vars(), &poly.coeffs())?;
        let w = <HachiCommitmentCore as RingCommitmentScheme<F, { D }, Cfg>>::commit_coeffs(
            &ring_coeffs,
            setup,
        )?;
        let hint = HachiCommitmentHint {
            s: w.s,
            t_hat: w.t_hat,
            ring_coeffs,
        };
        Ok((w.commitment, hint))
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
        let alpha = Cfg::D.trailing_zeros() as usize;
        let reduced_len = num_vars
            .checked_sub(alpha)
            .ok_or_else(|| HachiError::InvalidSetup("reduction length underflow".to_string()))?;
        if opening_point.len() < reduced_len {
            return Err(HachiError::InvalidPointDimension {
                expected: reduced_len,
                actual: opening_point.len(),
            });
        }

        let ring_opening_point =
            ring_opening_point_from_field::<F>(&opening_point[..reduced_len], Cfg::R, Cfg::M)?;

        let y_ring =
            evaluate_packed_ring_poly::<F, { D }>(&hint.ring_coeffs, &opening_point[..reduced_len]);

        // §4.2 Quadratic equation
        let quad_eq = QuadraticEquation::<F, { D }, Cfg>::new_prover(
            setup,
            &ring_opening_point,
            &hint,
            transcript,
            commitment,
            &y_ring,
        )?;

        // §4.3 Ring switch
        let rs = ring_switch_prover::<F, T, { D }, Cfg>(&quad_eq, transcript)?;

        // Batched sumcheck: norm + relation
        let mut norm_prover = NormSumcheckProver::new(&rs.tau0, rs.w_evals.clone(), rs.b);
        let mut relation_prover = RelationSumcheckProver::new(
            rs.w_evals,
            &rs.alpha_evals_y,
            &rs.m_evals_x,
            rs.num_u,
            rs.num_l,
        );

        let instances: Vec<&mut dyn SumcheckInstanceProver<F>> =
            vec![&mut norm_prover, &mut relation_prover];
        let (sumcheck_proof, ..) =
            prove_batched_sumcheck::<F, _, F, _>(instances, transcript, |tr| {
                tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
            })?;

        Ok(HachiProof {
            v: quad_eq.v,
            y_ring,
            sumcheck_proof,
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
        let alpha_bits = Cfg::D.trailing_zeros() as usize;
        let reduced_len = opening_point.len().checked_sub(alpha_bits).ok_or_else(|| {
            HachiError::InvalidSetup("opening point length underflow".to_string())
        })?;
        let reduced_opening_point = &opening_point[..reduced_len];
        let inner_point = &opening_point[reduced_len..];

        // §3.1 trace check
        let v = reduce_inner_openings_to_ring_elements::<F, { D }>(inner_point)?;
        let d = F::from_u64(Cfg::D as u64);
        let trace_lhs = trace::<F, { D }>(&(proof.y_ring * v.sigma_m1()));
        let trace_rhs = d * *opening;
        if trace_lhs != trace_rhs {
            return Err(HachiError::InvalidProof);
        }

        // §4.2 Quadratic equation
        let ring_opening_point =
            ring_opening_point_from_field::<F>(reduced_opening_point, Cfg::R, Cfg::M)?;
        let quad_eq = QuadraticEquation::<F, { D }, Cfg>::new_verifier(
            setup,
            &ring_opening_point,
            &proof.v,
            transcript,
            commitment,
            &proof.y_ring,
        )?;

        // §4.3 Ring switch (verifier side)
        let rs = ring_switch_verifier::<F, T, { D }, Cfg>(
            &quad_eq,
            &proof.sumcheck_aux.w,
            &proof.w_commitment,
            transcript,
        )?;

        // Batched sumcheck verification: norm (F_0) + relation (F_α)
        let norm_verifier = NormSumcheckVerifier::new(rs.tau0, rs.w_evals.clone(), rs.b);
        let relation_verifier = RelationSumcheckVerifier::new(
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

        let verifiers: Vec<&dyn SumcheckInstanceVerifier<F>> =
            vec![&norm_verifier, &relation_verifier];
        verify_batched_sumcheck::<F, _, F, _>(
            &proof.sumcheck_proof,
            verifiers,
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
pub(crate) fn rederive_alpha_and_m_a<F, const D: usize, Cfg>(
    proof: &HachiProof<F, D>,
    setup: &RingCommitmentSetup<F, D>,
    opening_point: &[F],
    commitment: &RingCommitment<F, D>,
) -> Result<(F, Vec<F>), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + 'static,
    Cfg: CommitmentConfig,
{
    let alpha_bits = Cfg::D.trailing_zeros() as usize;
    let reduced_len = opening_point
        .len()
        .checked_sub(alpha_bits)
        .ok_or_else(|| HachiError::InvalidSetup("opening point length underflow".to_string()))?;
    let ring_opening_point =
        ring_opening_point_from_field::<F>(&opening_point[..reduced_len], Cfg::R, Cfg::M)?;
    let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_HACHI_PROTOCOL);
    let quad_eq = QuadraticEquation::<F, D, Cfg>::new_verifier(
        setup,
        &ring_opening_point,
        &proof.v,
        &mut transcript,
        commitment,
        &proof.y_ring,
    )?;
    transcript.append_serde(ABSORB_SUMCHECK_W, &proof.w_commitment);
    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);
    let m_a = eval_ring_matrix_at::<F, D>(quad_eq.m(), &alpha);
    let m_a_vec = expand_m_a::<F, D, Cfg>(&m_a, alpha)?;
    Ok((alpha, m_a_vec))
}

fn lagrange_weights<F: FieldCore>(point: &[F]) -> Vec<F> {
    let len = 1usize << point.len();
    let mut weights = vec![F::zero(); len];
    multilinear_lagrange_basis(&mut weights, point);
    weights
}

fn ring_opening_point_from_field<F: FieldCore>(
    opening_point: &[F],
    r_vars: usize,
    m_vars: usize,
) -> Result<RingOpeningPoint<F>, HachiError> {
    let expected_len = r_vars
        .checked_add(m_vars)
        .ok_or_else(|| HachiError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() != expected_len {
        return Err(HachiError::InvalidPointDimension {
            expected: expected_len,
            actual: opening_point.len(),
        });
    }

    let b = lagrange_weights(&opening_point[..r_vars]);
    let a = lagrange_weights(&opening_point[r_vars..]);
    Ok(RingOpeningPoint { a, b })
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

    let out: Vec<CyclotomicRing<F, D>> = cfg_into_iter!(0..outer_len)
        .map(|i| {
            let ring_coeffs = std::array::from_fn(|j| {
                let idx = i + (j << outer_vars);
                coeffs[idx]
            });
            CyclotomicRing::from_coefficients(ring_coeffs)
        })
        .collect();
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
    #[cfg(feature = "parallel")]
    {
        packed_coeffs
            .par_iter()
            .zip(weights.par_iter())
            .fold(
                || CyclotomicRing::<F, D>::zero(),
                |acc, (f_i, w_i)| acc + f_i.scale(w_i),
            )
            .reduce(|| CyclotomicRing::<F, D>::zero(), |a, b| a + b)
    }
    #[cfg(not(feature = "parallel"))]
    {
        packed_coeffs
            .iter()
            .zip(weights.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, w_i)| {
                acc + f_i.scale(w_i)
            })
    }
}

fn trace<F: FieldCore + FromSmallInt, const D: usize>(u: &CyclotomicRing<F, D>) -> F {
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
    use crate::{CommitmentScheme, FromSmallInt, Polynomial};

    type Cfg = SmallTestCommitmentConfig;
    type Scheme = HachiCommitmentScheme<{ Cfg::D }, Cfg>;

    #[test]
    fn verify_passes_for_consistent_opening() {
        let alpha = Cfg::D.trailing_zeros() as usize;
        let num_vars = Cfg::R + Cfg::M + alpha;
        let len = 1usize << num_vars;

        let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DenseMultilinearEvals::new_padded(evals);

        let setup = <Scheme as CommitmentScheme<F>>::setup_prover(num_vars);
        let verifier_setup = <Scheme as CommitmentScheme<F>>::setup_verifier(&setup);

        let (commitment, hint) = <Scheme as CommitmentScheme<F>>::commit(&poly, &setup).unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let opening = poly.evaluate(&opening_point);

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let proof = <Scheme as CommitmentScheme<F>>::prove(
            &setup,
            &poly,
            &opening_point,
            Some(hint),
            &mut prover_transcript,
            &commitment,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let result = <Scheme as CommitmentScheme<F>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &opening,
            &commitment,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn verify_rejects_wrong_opening() {
        let alpha = Cfg::D.trailing_zeros() as usize;
        let num_vars = Cfg::R + Cfg::M + alpha;
        let len = 1usize << num_vars;

        let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DenseMultilinearEvals::new_padded(evals);

        let setup = <Scheme as CommitmentScheme<F>>::setup_prover(num_vars);
        let verifier_setup = <Scheme as CommitmentScheme<F>>::setup_verifier(&setup);

        let (commitment, hint) = <Scheme as CommitmentScheme<F>>::commit(&poly, &setup).unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let opening = poly.evaluate(&opening_point);

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let proof = <Scheme as CommitmentScheme<F>>::prove(
            &setup,
            &poly,
            &opening_point,
            Some(hint),
            &mut prover_transcript,
            &commitment,
        )
        .unwrap();

        let wrong_opening = opening + F::one();
        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let result = <Scheme as CommitmentScheme<F>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &wrong_opening,
            &commitment,
        );

        assert!(
            result.is_err(),
            "verify must reject an incorrect opening value"
        );
    }
}
