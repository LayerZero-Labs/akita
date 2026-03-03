//! Commitment scheme trait implementation.

use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
use crate::primitives::poly::multilinear_lagrange_basis;
use crate::protocol::commitment::utils::linear::{mat_vec_mul_ntt_cached, MatrixSlot};
use crate::protocol::commitment::{
    AppendToTranscript, CommitmentConfig, CommitmentScheme, HachiCommitmentCore, HachiProverSetup,
    HachiVerifierSetup, RingCommitment, RingCommitmentScheme,
};
use crate::protocol::hachi_poly_ops::{DensePoly, HachiPolyOps};
use crate::protocol::opening_point::{BasisMode, RingOpeningPoint};
use crate::protocol::proof::{HachiCommitmentHint, HachiLevelProof, HachiProof};
use crate::protocol::quadratic_equation::QuadraticEquation;
use crate::protocol::ring_switch::{
    build_w_evals, ring_switch_prover, ring_switch_verifier, w_ring_element_count,
    WCommitmentConfig,
};
use crate::protocol::sumcheck::hachi_sumcheck::{HachiSumcheckProver, HachiSumcheckVerifier};
use crate::protocol::sumcheck::{multilinear_eval, prove_sumcheck, verify_sumcheck};
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

/// Minimum w vector length (in field elements) below which further folding
/// is not beneficial.  When `w.len() <= MIN_W_LEN_FOR_FOLDING`, the prover
/// sends `w` directly instead of recursing.
const MIN_W_LEN_FOR_FOLDING: usize = 4096;

/// Minimum shrink ratio (next_w / prev_w) below which further folding
/// stops being worthwhile.  If the w vector doesn't shrink by at least
/// this factor, the overhead of another fold level outweighs the saving.
const MIN_SHRINK_RATIO: f64 = 0.5;

/// End-to-end PCS wrapper, generic over ring degree `D` and config `Cfg`.
#[derive(Clone, Copy, Debug, Default)]
pub struct HachiCommitmentScheme<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

/// Output from a single prove level, needed to chain into the next level.
struct ProveLevelOutput<F: FieldCore, const D: usize> {
    level_proof: HachiLevelProof<F, D>,
    w: Vec<F>,
    w_hint: HachiCommitmentHint<F, D>,
    sumcheck_challenges: Vec<F>,
    num_u: usize,
    num_l: usize,
}

/// Prove one fold level: quad_eq -> ring_switch -> sumcheck.
///
/// Generic over the commitment config so it works for both the original
/// polynomial (using `Cfg`) and recursive w-openings (using `WCommitmentConfig`).
#[allow(clippy::too_many_arguments)]
fn prove_one_level<F, T, const D: usize, Cfg, P>(
    setup: &HachiProverSetup<F, D>,
    poly: &P,
    opening_point: &[F],
    hint: HachiCommitmentHint<F, D>,
    transcript: &mut T,
    commitment: &RingCommitment<F, D>,
    basis: BasisMode,
    level: usize,
) -> Result<ProveLevelOutput<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
    P: HachiPolyOps<F, D>,
{
    let alpha = Cfg::D.trailing_zeros() as usize;
    if opening_point.len() < alpha {
        return Err(HachiError::InvalidPointDimension {
            expected: alpha,
            actual: opening_point.len(),
        });
    }

    let layout = Cfg::commitment_layout(opening_point.len())?;
    let target_num_vars = layout.m_vars + layout.r_vars + alpha;
    let mut padded_point = opening_point.to_vec();
    padded_point.resize(target_num_vars, F::zero());
    let outer_point = &padded_point[alpha..];

    let ring_opening_point =
        ring_opening_point_from_field::<F>(outer_point, layout.r_vars, layout.m_vars, basis)?;

    let t0 = Instant::now();
    let outer_weights = basis_weights(outer_point, basis);
    let y_ring = {
        let _span = tracing::info_span!("evaluate_ring", level).entered();
        poly.evaluate_ring(&outer_weights)
    };
    eprintln!(
        "  [hachi prove L{level}] eval ring poly: {:.2}s (num_ring_elems={})",
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
        layout,
    )?;
    eprintln!(
        "  [hachi prove L{level}] quad_eq new_prover: {:.2}s",
        t1.elapsed().as_secs_f64()
    );

    let t2 = Instant::now();
    let ntt_cache = setup.ntt_cache()?;
    let rs = ring_switch_prover::<F, T, { D }, Cfg>(
        &mut quad_eq,
        &setup.expanded,
        transcript,
        ntt_cache,
        layout,
    )?;
    eprintln!(
        "  [hachi prove L{level}] ring_switch_prover: {:.2}s",
        t2.elapsed().as_secs_f64()
    );

    let batching_coeff: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);

    let t3 = Instant::now();
    let num_u = rs.num_u;
    let num_l = rs.num_l;
    let mut fused_prover = HachiSumcheckProver::new(
        batching_coeff,
        rs.w_evals,
        &rs.tau0,
        rs.b,
        &rs.alpha_evals_y,
        &rs.m_evals_x,
        num_u,
        num_l,
    );

    let (sumcheck_proof, sumcheck_challenges, _final_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut fused_prover, transcript, |tr| {
            tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
        })?;
    eprintln!(
        "  [hachi prove L{level}] fused sumcheck: {:.2}s",
        t3.elapsed().as_secs_f64()
    );

    let (w_evals_for_eval, _, _) = build_w_evals(&rs.w, D)?;
    let w_eval = multilinear_eval(&w_evals_for_eval, &sumcheck_challenges)?;

    Ok(ProveLevelOutput {
        level_proof: HachiLevelProof {
            v: quad_eq.v,
            y_ring,
            sumcheck_proof,
            w_commitment: rs.w_commitment,
            w_eval,
        },
        w: rs.w,
        w_hint: rs.w_hint,
        sumcheck_challenges,
        num_u,
        num_l,
    })
}

/// Whether the prover should stop folding and send `w` directly.
///
/// `prev_w_len` is the polynomial length at the previous level (or the
/// original polynomial's field-element count for level 0).
fn should_stop_folding(w_len: usize, prev_w_len: usize) -> bool {
    if w_len <= MIN_W_LEN_FOR_FOLDING {
        return true;
    }
    // Stop if w didn't shrink enough relative to the input.
    let ratio = w_len as f64 / prev_w_len as f64;
    ratio > MIN_SHRINK_RATIO
}

/// Derive the opening point for the next fold level from the sumcheck
/// challenges of the current level.
///
/// Sumcheck challenges are ordered `[x_0..x_{num_u-1}, y_0..y_{num_l-1}]`
/// where x selects ring elements and y selects coefficients.
/// The PCS opening point is `[inner, outer]` = `[y, x]`.
fn next_level_opening_point<F: FieldCore>(
    sumcheck_challenges: &[F],
    num_u: usize,
    num_l: usize,
) -> Vec<F> {
    let (x, y) = sumcheck_challenges.split_at(num_u);
    debug_assert_eq!(y.len(), num_l);
    let mut point = Vec::with_capacity(num_u + num_l);
    point.extend_from_slice(y);
    point.extend_from_slice(x);
    point
}

/// Build a `DensePoly` from the flat w coefficient vector, padding to
/// the next power of two in total field elements.
fn dense_poly_from_w<F: FieldCore, const D: usize>(w: &[F]) -> Result<DensePoly<F, D>, HachiError> {
    let total_coeffs = w.len().next_power_of_two().max(D);
    let num_vars = total_coeffs.trailing_zeros() as usize;
    let mut padded = w.to_vec();
    padded.resize(total_coeffs, F::zero());
    DensePoly::from_field_evals(num_vars, &padded)
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

    #[tracing::instrument(skip_all, name = "HachiCommitmentScheme::setup_prover")]
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

    #[tracing::instrument(skip_all, name = "HachiCommitmentScheme::commit")]
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
            layout.num_digits_commit,
            layout.log_basis,
        )?;
        let t_hat_flat: Vec<CyclotomicRing<F, D>> =
            t_hat_all.iter().flat_map(|v| v.iter().copied()).collect();
        let u = mat_vec_mul_ntt_cached(cache, MatrixSlot::B, &t_hat_flat)?;
        let hint = HachiCommitmentHint { t_hat: t_hat_all };
        Ok((RingCommitment { u }, hint))
    }

    #[tracing::instrument(skip_all, name = "HachiCommitmentScheme::prove")]
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
        let mut levels = Vec::new();

        // --- Level 0: original polynomial with Cfg ---
        let out = prove_one_level::<F, T, D, Cfg, P>(
            setup,
            poly,
            opening_point,
            hint,
            transcript,
            commitment,
            basis,
            0,
        )?;
        levels.push(out.level_proof);

        let mut prev_poly_len = poly.num_ring_elems() * D;
        let mut current_w = out.w;
        let mut current_hint = out.w_hint;
        let mut current_challenges = out.sumcheck_challenges;
        let mut current_num_u = out.num_u;
        let mut current_num_l = out.num_l;
        let mut level = 1usize;

        // --- Subsequent levels: recursive w-opening with WCommitmentConfig ---
        while !should_stop_folding(current_w.len(), prev_poly_len) {
            let w_poly = dense_poly_from_w::<F, D>(&current_w)?;
            let opening_point =
                next_level_opening_point(&current_challenges, current_num_u, current_num_l);
            let w_commitment = &levels.last().unwrap().w_commitment;

            let out = prove_one_level::<F, T, D, WCommitmentConfig<D, Cfg>, DensePoly<F, D>>(
                setup,
                &w_poly,
                &opening_point,
                current_hint,
                transcript,
                w_commitment,
                BasisMode::Lagrange,
                level,
            )?;
            levels.push(out.level_proof);

            prev_poly_len = current_w.len();
            current_w = out.w;
            current_hint = out.w_hint;
            current_challenges = out.sumcheck_challenges;
            current_num_u = out.num_u;
            current_num_l = out.num_l;
            level += 1;
        }

        eprintln!(
            "  [hachi prove] total ({level} levels): {:.2}s",
            t_prove_total.elapsed().as_secs_f64()
        );

        Ok(HachiProof {
            levels,
            final_w: current_w,
        })
    }

    #[tracing::instrument(skip_all, name = "HachiCommitmentScheme::verify")]
    fn verify<T: Transcript<F>>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        opening_point: &[F],
        opening: &F,
        commitment: &Self::Commitment,
        basis: BasisMode,
    ) -> Result<(), HachiError> {
        if proof.levels.is_empty() {
            return Err(HachiError::InvalidProof);
        }

        let num_levels = proof.levels.len();

        // State carried between levels.
        let mut current_point = opening_point.to_vec();
        let mut current_opening = *opening;
        let mut current_commitment = commitment.clone();
        let mut current_basis = basis;

        for (i, level_proof) in proof.levels.iter().enumerate() {
            let is_last = i == num_levels - 1;

            // --- Verify one level ---
            let challenges = if i == 0 {
                verify_one_level::<F, T, D, Cfg>(
                    level_proof,
                    setup,
                    transcript,
                    &current_point,
                    &current_opening,
                    &current_commitment,
                    current_basis,
                    is_last,
                    if is_last { Some(&proof.final_w) } else { None },
                )?
            } else {
                verify_one_level::<F, T, D, WCommitmentConfig<D, Cfg>>(
                    level_proof,
                    setup,
                    transcript,
                    &current_point,
                    &current_opening,
                    &current_commitment,
                    current_basis,
                    is_last,
                    if is_last { Some(&proof.final_w) } else { None },
                )?
            };

            if !is_last {
                let alpha_bits = D.trailing_zeros() as usize;
                let num_l = alpha_bits;
                let num_u = challenges.len() - num_l;

                current_point = next_level_opening_point(&challenges, num_u, num_l);
                current_opening = level_proof.w_eval;
                current_commitment = level_proof.w_commitment.clone();
                current_basis = BasisMode::Lagrange;
            }
        }

        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        unimplemented!()
    }
}

/// Verify one fold level.
///
/// At the final level, `final_w` is provided and the verifier checks w_val
/// from it directly. At intermediate levels, `level_proof.w_eval` is used.
///
/// Returns the sumcheck challenges for chaining into the next level.
#[allow(clippy::too_many_arguments)]
fn verify_one_level<F, T, const D: usize, Cfg>(
    level_proof: &HachiLevelProof<F, D>,
    setup: &HachiVerifierSetup<F, D>,
    transcript: &mut T,
    opening_point: &[F],
    opening: &F,
    commitment: &RingCommitment<F, D>,
    basis: BasisMode,
    is_last: bool,
    final_w: Option<&[F]>,
) -> Result<Vec<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    let alpha_bits = Cfg::D.trailing_zeros() as usize;
    if opening_point.len() < alpha_bits {
        return Err(HachiError::InvalidSetup(
            "opening point length underflow".to_string(),
        ));
    }
    let layout = Cfg::commitment_layout(opening_point.len())?;
    let target_num_vars = layout.m_vars + layout.r_vars + alpha_bits;
    let mut padded_point = opening_point.to_vec();
    padded_point.resize(target_num_vars, F::zero());
    let inner_point = &padded_point[..alpha_bits];
    let reduced_opening_point = &padded_point[alpha_bits..];

    commitment.append_to_transcript(ABSORB_COMMITMENT, transcript);
    for pt in &padded_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &level_proof.y_ring);

    // Trace check
    let v = reduce_inner_openings_to_ring_elements::<F, { D }>(inner_point, basis)?;
    let d = F::from_u64(Cfg::D as u64);
    let trace_lhs = trace::<F, { D }>(&(level_proof.y_ring * v.sigma_m1()));
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
        ring_opening_point,
        level_proof.v.clone(),
        transcript,
        commitment,
        &level_proof.y_ring,
        layout,
    )?;

    let w_len = if is_last {
        final_w.map_or(0, |fw| fw.len())
    } else {
        w_ring_element_count::<F, Cfg>(layout) * D
    };

    let rs = ring_switch_verifier::<F, T, { D }, Cfg>(
        &quad_eq,
        &setup.expanded,
        w_len,
        &level_proof.w_commitment,
        transcript,
        layout,
    )?;

    let batching_coeff: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);

    let fused_verifier = if is_last {
        let fw = final_w.ok_or(HachiError::InvalidProof)?;
        let (w_evals_full, _, _) = build_w_evals(fw, Cfg::D)?;
        HachiSumcheckVerifier::new(
            batching_coeff,
            w_evals_full,
            rs.tau0,
            rs.b,
            rs.alpha_evals_y,
            rs.m_evals_x,
            rs.tau1,
            level_proof.v.clone(),
            commitment.u.clone(),
            level_proof.y_ring,
            rs.alpha,
            rs.num_u,
            rs.num_l,
        )
    } else {
        HachiSumcheckVerifier::new(
            batching_coeff,
            Vec::new(),
            rs.tau0,
            rs.b,
            rs.alpha_evals_y,
            rs.m_evals_x,
            rs.tau1,
            level_proof.v.clone(),
            commitment.u.clone(),
            level_proof.y_ring,
            rs.alpha,
            rs.num_u,
            rs.num_l,
        )
        .with_w_val_override(level_proof.w_eval)
    };

    let challenges = verify_sumcheck::<F, _, F, _, _>(
        &level_proof.sumcheck_proof,
        &fused_verifier,
        transcript,
        |tr| tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND),
    )?;

    Ok(challenges)
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
    let level0 = proof.levels.first().ok_or(HachiError::InvalidProof)?;
    let alpha_bits = Cfg::D.trailing_zeros() as usize;
    if opening_point.len() < alpha_bits {
        return Err(HachiError::InvalidSetup(
            "opening point length underflow".to_string(),
        ));
    }
    let layout = Cfg::commitment_layout(opening_point.len())?;
    let ring_opening_point = ring_opening_point_from_field::<F>(
        &opening_point[alpha_bits..],
        layout.r_vars,
        layout.m_vars,
        BasisMode::Lagrange,
    )?;
    let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_HACHI_PROTOCOL);

    commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
    for pt in opening_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &level0.y_ring);

    let quad_eq = QuadraticEquation::<F, D, Cfg>::new_verifier(
        ring_opening_point,
        level0.v.clone(),
        &mut transcript,
        commitment,
        &level0.y_ring,
        layout,
    )?;
    transcript.append_serde(ABSORB_SUMCHECK_W, &level0.w_commitment);
    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);
    let m_a = compute_m_a_streaming::<F, D, Cfg>(
        &setup.expanded,
        quad_eq.opening_point(),
        &quad_eq.challenges,
        &alpha,
        layout,
    )?;
    let m_a_vec = expand_m_a::<F, D>(&m_a, alpha, layout.log_basis)?;
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
