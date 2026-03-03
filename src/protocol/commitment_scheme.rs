//! Commitment scheme trait implementation.

use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
use crate::primitives::poly::multilinear_lagrange_basis;
use crate::protocol::commitment::utils::linear::{mat_vec_mul_ntt_cached, MatrixSlot};
use crate::protocol::commitment::{
    AppendToTranscript, CommitmentConfig, CommitmentScheme, HachiCommitmentCore, HachiProverSetup,
    HachiVerifierSetup, RingCommitment, RingCommitmentScheme,
};
use crate::protocol::hachi_poly_ops::HachiPolyOps;
use crate::protocol::opening_point::{BasisMode, RingOpeningPoint};
use crate::protocol::proof::{HachiCommitmentHint, HachiProof, SumcheckAux};
use crate::protocol::quadratic_equation::QuadraticEquation;
use crate::protocol::ring_switch::{build_w_evals, ring_switch_prover, ring_switch_verifier};
use crate::protocol::sumcheck::hachi_sumcheck::{HachiSumcheckProver, HachiSumcheckVerifier};
use crate::protocol::sumcheck::{prove_sumcheck, verify_sumcheck};
use crate::protocol::transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, CHALLENGE_SUMCHECK_BATCH, CHALLENGE_SUMCHECK_ROUND,
};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling, FromSmallInt};
use std::marker::PhantomData;
use std::time::Instant;

#[cfg(test)]
use crate::protocol::quadratic_equation::compute_m_a_streaming;
#[cfg(test)]
use crate::protocol::ring_switch::expand_m_a;
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
    _cfg: PhantomData<Cfg>,
}

impl<F, const D: usize, Cfg> CommitmentScheme<F, D> for HachiCommitmentScheme<D, Cfg>
where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig,
{
    type ProverSetup = HachiProverSetup<F, D>;
    type VerifierSetup = HachiVerifierSetup<F, D>;
    type Commitment = RingCommitment<F, D>;
    type Proof = HachiProof<F, D>;
    type CommitHint = HachiCommitmentHint<F, D>;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let (setup, _) =
            <HachiCommitmentCore as RingCommitmentScheme<F, { D }, Cfg>>::setup(max_num_vars)
                .expect("commitment setup failed");
        setup
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        HachiVerifierSetup {
            expanded: setup.expanded.clone(),
        }
    }

    fn commit<P: HachiPolyOps<F, D>>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Commitment, Self::CommitHint), HachiError> {
        let layout = setup.layout();
        let cache = setup.ntt_cache()?;
        let t_hat_all = poly.commit_inner(
            &setup.expanded.A,
            cache,
            layout.block_len,
            layout.delta,
            layout.log_basis,
        )?;
        let t_hat_flat: Vec<CyclotomicRing<F, D>> =
            t_hat_all.iter().flat_map(|v| v.iter().copied()).collect();
        let u = mat_vec_mul_ntt_cached(cache, MatrixSlot::B, &t_hat_flat)?;
        let hint = HachiCommitmentHint { t_hat: t_hat_all };
        Ok((RingCommitment { u }, hint))
    }

    fn prove<T: Transcript<F>, P: HachiPolyOps<F, D>>(
        setup: &Self::ProverSetup,
        poly: &P,
        opening_point: &[F],
        hint: Self::CommitHint,
        transcript: &mut T,
        commitment: &Self::Commitment,
        basis: BasisMode,
    ) -> Result<Self::Proof, HachiError> {
        let t_prove_total = Instant::now();
        let alpha = Cfg::D.trailing_zeros() as usize;
        if opening_point.len() < alpha {
            return Err(HachiError::InvalidPointDimension {
                expected: alpha,
                actual: opening_point.len(),
            });
        }

        let layout = <HachiCommitmentCore as RingCommitmentScheme<F, { D }, Cfg>>::layout(setup)?;
        let target_num_vars = layout.m_vars + layout.r_vars + alpha;
        if opening_point.len() > target_num_vars {
            return Err(HachiError::InvalidPointDimension {
                expected: target_num_vars,
                actual: opening_point.len(),
            });
        }
        let mut padded_point = opening_point.to_vec();
        padded_point.resize(target_num_vars, F::zero());
        let outer_point = &padded_point[alpha..];

        let ring_opening_point =
            ring_opening_point_from_field::<F>(outer_point, layout.r_vars, layout.m_vars, basis)?;

        let t0 = Instant::now();
        let outer_weights = basis_weights(outer_point, basis);
        let y_ring = poly.evaluate_ring(&outer_weights);
        eprintln!(
            "  [hachi prove] eval ring poly: {:.2}s (num_ring_elems={})",
            t0.elapsed().as_secs_f64(),
            poly.num_ring_elems()
        );

        commitment.append_to_transcript(ABSORB_COMMITMENT, transcript);
        for pt in &padded_point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

        let t1 = Instant::now();
        let mut quad_eq = QuadraticEquation::<F, { D }, Cfg>::new_prover(
            setup,
            ring_opening_point,
            poly,
            hint,
            transcript,
            commitment,
            &y_ring,
        )?;
        eprintln!(
            "  [hachi prove] quad_eq new_prover: {:.2}s",
            t1.elapsed().as_secs_f64()
        );

        let t2 = Instant::now();
        let rs = ring_switch_prover::<F, T, { D }, Cfg>(&mut quad_eq, &setup.expanded, transcript)?;
        eprintln!(
            "  [hachi prove] ring_switch_prover: {:.2}s",
            t2.elapsed().as_secs_f64()
        );

        let batching_coeff: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);

        let t3 = Instant::now();
        let mut fused_prover = HachiSumcheckProver::new(
            batching_coeff,
            rs.w_evals,
            &rs.tau0,
            rs.b,
            &rs.alpha_evals_y,
            &rs.m_evals_x,
            rs.num_u,
            rs.num_l,
        );

        let (sumcheck_proof, ..) =
            prove_sumcheck::<F, _, F, _, _>(&mut fused_prover, transcript, |tr| {
                tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
            })?;
        eprintln!(
            "  [hachi prove] fused sumcheck: {:.2}s",
            t3.elapsed().as_secs_f64()
        );
        eprintln!(
            "  [hachi prove] total: {:.2}s",
            t_prove_total.elapsed().as_secs_f64()
        );

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
        basis: BasisMode,
    ) -> Result<(), HachiError> {
        let alpha_bits = Cfg::D.trailing_zeros() as usize;
        if opening_point.len() < alpha_bits {
            return Err(HachiError::InvalidSetup(
                "opening point length underflow".to_string(),
            ));
        }
        let layout = setup.expanded.seed.layout;
        let target_num_vars = layout.m_vars + layout.r_vars + alpha_bits;
        let mut padded_point = opening_point.to_vec();
        padded_point.resize(target_num_vars, F::zero());
        let inner_point = &padded_point[..alpha_bits];
        let reduced_opening_point = &padded_point[alpha_bits..];

        commitment.append_to_transcript(ABSORB_COMMITMENT, transcript);
        for pt in &padded_point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &proof.y_ring);

        let v = reduce_inner_openings_to_ring_elements::<F, { D }>(inner_point, basis)?;
        let d = F::from_u64(Cfg::D as u64);
        let trace_lhs = trace::<F, { D }>(&(proof.y_ring * v.sigma_m1()));
        let trace_rhs = d * *opening;
        if trace_lhs != trace_rhs {
            return Err(HachiError::InvalidProof);
        }

        let ring_opening_point = ring_opening_point_from_field::<F>(
            reduced_opening_point,
            layout.r_vars,
            layout.m_vars,
            basis,
        )?;
        let quad_eq = QuadraticEquation::<F, { D }, Cfg>::new_verifier(
            setup,
            ring_opening_point,
            proof.v.clone(),
            transcript,
            commitment,
            &proof.y_ring,
        )?;

        let rs = ring_switch_verifier::<F, T, { D }, Cfg>(
            &quad_eq,
            &setup.expanded,
            &proof.sumcheck_aux.w,
            &proof.w_commitment,
            transcript,
        )?;

        let batching_coeff: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);
        let (w_evals_full, _, _) = build_w_evals(&proof.sumcheck_aux.w, Cfg::D)?;

        let fused_verifier = HachiSumcheckVerifier::new(
            batching_coeff,
            w_evals_full,
            rs.tau0,
            rs.b,
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
            &proof.sumcheck_proof,
            &fused_verifier,
            transcript,
            |tr| tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND),
        )?;

        Ok(())
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
    setup: &HachiVerifierSetup<F, D>,
    opening_point: &[F],
    commitment: &RingCommitment<F, D>,
) -> Result<(F, Vec<F>), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + 'static,
    Cfg: CommitmentConfig,
{
    let alpha_bits = Cfg::D.trailing_zeros() as usize;
    if opening_point.len() < alpha_bits {
        return Err(HachiError::InvalidSetup(
            "opening point length underflow".to_string(),
        ));
    }
    let layout = setup.expanded.seed.layout;
    let ring_opening_point = ring_opening_point_from_field::<F>(
        &opening_point[alpha_bits..],
        layout.r_vars,
        layout.m_vars,
        BasisMode::Lagrange,
    )?;
    let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_HACHI_PROTOCOL);

    // Replay the same Fiat-Shamir absorptions the real verifier performs.
    commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
    for pt in opening_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &proof.y_ring);

    let quad_eq = QuadraticEquation::<F, D, Cfg>::new_verifier(
        setup,
        ring_opening_point,
        proof.v.clone(),
        &mut transcript,
        commitment,
        &proof.y_ring,
    )?;
    transcript.append_serde(ABSORB_SUMCHECK_W, &proof.w_commitment);
    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);
    let m_a = compute_m_a_streaming::<F, D, Cfg>(
        &setup.expanded,
        quad_eq.opening_point(),
        &quad_eq.challenges,
        &alpha,
    )?;
    let m_a_vec = expand_m_a::<F, D>(&m_a, alpha, setup.expanded.seed.layout.log_basis)?;
    Ok((alpha, m_a_vec))
}

fn lagrange_weights<F: FieldCore>(point: &[F]) -> Vec<F> {
    let len = 1usize << point.len();
    let mut weights = vec![F::zero(); len];
    multilinear_lagrange_basis(&mut weights, point);
    weights
}

/// Multilinear monomial weights: `⊗ᵢ (1, xᵢ)`.
///
/// The j-th entry is `∏_{i ∈ bits(j)} point[i]`.
fn monomial_weights<F: FieldCore>(point: &[F]) -> Vec<F> {
    let len = 1usize << point.len();
    let mut weights = vec![F::zero(); len];
    weights[0] = F::one();
    for (level, &p) in point.iter().enumerate() {
        let k = 1usize << level;
        for i in (0..k).rev() {
            weights[i + k] = weights[i] * p;
        }
    }
    weights
}

fn basis_weights<F: FieldCore>(point: &[F], mode: BasisMode) -> Vec<F> {
    match mode {
        BasisMode::Lagrange => lagrange_weights(point),
        BasisMode::Monomial => monomial_weights(point),
    }
}

fn ring_opening_point_from_field<F: FieldCore>(
    opening_point: &[F],
    r_vars: usize,
    m_vars: usize,
    basis: BasisMode,
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

    // Sequential ordering: M variables (position in block) come first,
    // R variables (block selection) come second.
    let a = basis_weights(&opening_point[..m_vars], basis);
    let b = basis_weights(&opening_point[m_vars..], basis);
    Ok(RingOpeningPoint { a, b })
}

fn reduce_inner_openings_to_ring_elements<F: FieldCore, const D: usize>(
    inner_point: &[F],
    basis: BasisMode,
) -> Result<CyclotomicRing<F, D>, HachiError> {
    let weights = basis_weights(inner_point, basis);
    if weights.len() != D {
        return Err(HachiError::InvalidInput(format!(
            "inner basis length {} does not match D={D}",
            weights.len()
        )));
    }
    Ok(CyclotomicRing::from_slice(&weights))
}

fn trace<F: FieldCore + FromSmallInt, const D: usize>(u: &CyclotomicRing<F, D>) -> F {
    let d = F::from_u64(D as u64);
    u.coefficients()[0] * d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::commitment::CommitmentConfig;
    use crate::protocol::hachi_poly_ops::DensePoly;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::test_utils::F;
    use crate::{CommitmentScheme, FromSmallInt};

    type Cfg = SmallTestCommitmentConfig;
    const D: usize = Cfg::D;
    type Scheme = HachiCommitmentScheme<D, Cfg>;

    fn make_dense_poly(num_vars: usize) -> (DensePoly<F, D>, Vec<F>) {
        let len = 1usize << num_vars;
        let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();
        (poly, evals)
    }

    #[test]
    fn verify_passes_for_consistent_opening() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;

        let (poly, evals) = make_dense_poly(num_vars);

        let setup = <Scheme as CommitmentScheme<F, D>>::setup_prover(num_vars);
        let verifier_setup = <Scheme as CommitmentScheme<F, D>>::setup_verifier(&setup);

        let (commitment, hint) = <Scheme as CommitmentScheme<F, D>>::commit(&poly, &setup).unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let lw = lagrange_weights(&opening_point);
        let opening: F = evals
            .iter()
            .zip(lw.iter())
            .fold(F::zero(), |a, (&c, &w)| a + c * w);

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let proof = <Scheme as CommitmentScheme<F, D>>::prove(
            &setup,
            &poly,
            &opening_point,
            hint,
            &mut prover_transcript,
            &commitment,
            BasisMode::Lagrange,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let result = <Scheme as CommitmentScheme<F, D>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &opening,
            &commitment,
            BasisMode::Lagrange,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn verify_rejects_wrong_opening() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;

        let (poly, evals) = make_dense_poly(num_vars);

        let setup = <Scheme as CommitmentScheme<F, D>>::setup_prover(num_vars);
        let verifier_setup = <Scheme as CommitmentScheme<F, D>>::setup_verifier(&setup);

        let (commitment, hint) = <Scheme as CommitmentScheme<F, D>>::commit(&poly, &setup).unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let lw = lagrange_weights(&opening_point);
        let opening: F = evals
            .iter()
            .zip(lw.iter())
            .fold(F::zero(), |a, (&c, &w)| a + c * w);

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let proof = <Scheme as CommitmentScheme<F, D>>::prove(
            &setup,
            &poly,
            &opening_point,
            hint,
            &mut prover_transcript,
            &commitment,
            BasisMode::Lagrange,
        )
        .unwrap();

        let wrong_opening = opening + F::one();
        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let result = <Scheme as CommitmentScheme<F, D>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &wrong_opening,
            &commitment,
            BasisMode::Lagrange,
        );

        assert!(
            result.is_err(),
            "verify must reject an incorrect opening value"
        );
    }

    #[test]
    fn monomial_basis_prove_verify_round_trip() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;
        let len = 1usize << num_vars;

        let coeffs: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &coeffs).unwrap();

        let setup = <Scheme as CommitmentScheme<F, D>>::setup_prover(num_vars);
        let verifier_setup = <Scheme as CommitmentScheme<F, D>>::setup_verifier(&setup);

        let (commitment, hint) = <Scheme as CommitmentScheme<F, D>>::commit(&poly, &setup).unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();

        let mw = monomial_weights(&opening_point);
        let opening: F = coeffs
            .iter()
            .zip(mw.iter())
            .fold(F::zero(), |acc, (&c, &w)| acc + c * w);

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/monomial");
        let proof = <Scheme as CommitmentScheme<F, D>>::prove(
            &setup,
            &poly,
            &opening_point,
            hint,
            &mut prover_transcript,
            &commitment,
            BasisMode::Monomial,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/monomial");
        let result = <Scheme as CommitmentScheme<F, D>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &opening,
            &commitment,
            BasisMode::Monomial,
        );

        assert!(
            result.is_ok(),
            "monomial-basis proof should verify: {result:?}"
        );
    }
}
