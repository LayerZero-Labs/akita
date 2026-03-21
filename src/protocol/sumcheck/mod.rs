//! Sumcheck protocol: traits, proof driver, and concrete instances.
//!
//! Types (`UniPoly`, `CompressedUniPoly`, `SumcheckProof`) live in the
//! [`types`] submodule. Polynomial evaluation utilities (`multilinear_eval`,
//! `fold_evals_in_place`, `range_check_eval`) live in [`crate::algebra::poly`].
//!
//! ## Temporary duplication notice (Jolt integration)
//!
//! Jolt already has a mature, streaming-aware sumcheck implementation. Long-term, we
//! expect to extract the common sumcheck machinery into a dedicated crate and depend
//! on it from both Hachi and Jolt. Until that exists, this module intentionally
//! duplicates the essential sumcheck data types and transcript-driving logic as a
//! pragmatic workaround.

pub mod batched_sumcheck;
pub mod eq_poly;
pub mod hachi_stage1;
pub mod hachi_stage2;
pub mod split_eq;
pub mod two_round_prefix;
pub mod types;

use crate::algebra::fields::HasUnreducedOps;
use crate::error::HachiError;
use crate::protocol::transcript::labels;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FromSmallInt};

pub use crate::algebra::poly::{
    fold_evals_in_place, multilinear_eval, multilinear_eval_small, range_check_eval,
};
pub use types::{CompressedUniPoly, SumcheckProof, UniPoly};

#[inline]
pub(crate) fn trim_trailing_zeros<E: FieldCore>(coeffs: &mut Vec<E>) {
    while coeffs.len() > 1 && coeffs.last().is_some_and(|c| c.is_zero()) {
        coeffs.pop();
    }
}

/// Precomputed lookup table for folding pairs of small integer values at a
/// fixed challenge `r`.
///
/// This is useful for the round-0 compact tables in Hachi's stage-1 and
/// stage-2 sumchecks: the table entries are small integers, the fold formula is
/// always `left + r * (right - left)`, and the set of possible `(left, right)`
/// pairs is tiny.
pub(crate) struct CompactPairFoldLut<E: FieldCore> {
    min_value: i16,
    value_to_index: Vec<usize>,
    pair_values: Vec<E>,
    num_values: usize,
}

impl<E: FieldCore + FromSmallInt + HasUnreducedOps> CompactPairFoldLut<E> {
    pub(crate) fn from_allowed_values(allowed_values: &[i16], r: E) -> Self {
        assert!(
            !allowed_values.is_empty(),
            "allowed_values must be non-empty"
        );
        let min_value = *allowed_values.iter().min().expect("non-empty");
        let max_value = *allowed_values.iter().max().expect("non-empty");
        let mut value_to_index = vec![usize::MAX; (max_value - min_value + 1) as usize];
        for (idx, &value) in allowed_values.iter().enumerate() {
            let offset = (value - min_value) as usize;
            debug_assert_eq!(
                value_to_index[offset],
                usize::MAX,
                "allowed_values must be unique"
            );
            value_to_index[offset] = idx;
        }

        let num_values = allowed_values.len();
        let mut pair_values = Vec::with_capacity(num_values * num_values);
        for &left in allowed_values {
            let left_field = E::from_i64(left as i64);
            for &right in allowed_values {
                let delta = i64::from(right) - i64::from(left);
                let delta_abs = delta.unsigned_abs();
                let r_delta = E::reduce_mul_u64_accum(r.mul_u64_unreduced(delta_abs));
                pair_values.push(if delta < 0 {
                    left_field - r_delta
                } else {
                    left_field + r_delta
                });
            }
        }

        Self {
            min_value,
            value_to_index,
            pair_values,
            num_values,
        }
    }

    pub(crate) fn from_contiguous_range(min_value: i16, max_value: i16, r: E) -> Self {
        assert!(min_value <= max_value, "invalid compact fold range");
        let allowed_values: Vec<i16> = (min_value..=max_value).collect();
        Self::from_allowed_values(&allowed_values, r)
    }
}

impl<E: FieldCore> CompactPairFoldLut<E> {
    #[inline]
    fn index_of(&self, value: i16) -> usize {
        let offset = (value - self.min_value) as usize;
        let idx = self.value_to_index[offset];
        debug_assert_ne!(idx, usize::MAX, "value missing from compact fold LUT");
        idx
    }

    #[inline]
    pub(crate) fn fold(&self, left: i16, right: i16) -> E {
        let left_idx = self.index_of(left);
        let right_idx = self.index_of(right);
        self.pair_values[left_idx * self.num_values + right_idx]
    }
}

/// Prover-side sumcheck instance interface.
///
/// This trait encapsulates the protocol-specific logic required to compute each
/// per-round univariate polynomial `g_j(X)` and to update (fold) internal state
/// after receiving the verifier challenge `r_j`.
///
/// Hachi §4.3 will implement concrete instances for `H_0` and `H_α`.
pub trait SumcheckInstanceProver<E: FieldCore>: Send + Sync {
    /// Number of rounds (i.e. number of variables bound by sumcheck).
    fn num_rounds(&self) -> usize;

    /// Maximum allowed degree for any round univariate polynomial.
    fn degree_bound(&self) -> usize;

    /// The initial claimed sum that this sumcheck instance is proving.
    fn input_claim(&self) -> E;

    /// Compute the prover message `g_round(X)` given the previous running claim.
    ///
    /// In standard sumcheck, `previous_claim` is the expected value of the
    /// remaining sum after binding previous challenges, and must satisfy:
    ///
    /// `g_round(0) + g_round(1) == previous_claim`.
    fn compute_round_univariate(&mut self, round: usize, previous_claim: E) -> UniPoly<E>;

    /// Ingest the verifier challenge `r_round` to fold/bind the current variable.
    fn ingest_challenge(&mut self, round: usize, r_round: E);

    /// Optional end-of-protocol hook after the last challenge has been ingested.
    fn finalize(&mut self) {}
}

/// Verifier-side sumcheck instance interface.
///
/// Implementations provide the initial claim and the oracle evaluation at the
/// challenge point, enabling the verifier to perform the final consistency check.
pub trait SumcheckInstanceVerifier<E: FieldCore>: Send + Sync {
    /// Number of rounds (i.e. number of variables bound by sumcheck).
    fn num_rounds(&self) -> usize;

    /// Maximum allowed degree for any round univariate polynomial.
    fn degree_bound(&self) -> usize;

    /// The initial claimed sum that this sumcheck instance is proving.
    fn input_claim(&self) -> E;

    /// Compute the expected final evaluation `f(r_0, ..., r_{n-1})` at the
    /// challenge point derived during the protocol.
    ///
    /// # Errors
    ///
    /// May return an error if internal evaluations fail (e.g., malformed
    /// evaluation tables from untrusted proof data).
    fn expected_output_claim(&self, challenges: &[E]) -> Result<E, HachiError>;
}

/// Produce a sumcheck proof while omitting the first `omitted_prefix_rounds`
/// transcript rounds from the stored proof.
///
/// This still drives the prover in the ordinary strict pipeline
/// `compute message -> absorb challenge -> ingest challenge -> ...`; it only
/// changes which compressed univariates are retained in the returned
/// [`SumcheckProof`]. Callers can use this to serialize early rounds via a
/// stage-local bivariate-skip proof instead of directly in the sumcheck proof.
///
/// # Errors
///
/// Returns an error if `omitted_prefix_rounds` exceeds the instance round
/// count, or if any per-round polynomial exceeds the instance's degree bound.
#[tracing::instrument(skip_all, name = "prove_sumcheck")]
#[inline(never)]
pub(crate) fn prove_sumcheck_with_omitted_prefix_rounds<F, T, E, S, Inst, A>(
    instance: &mut Inst,
    transcript: &mut T,
    mut sample_challenge: S,
    omitted_prefix_rounds: usize,
    mut absorb_after_compute: A,
) -> Result<(SumcheckProof<E>, Vec<E>, E), HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore,
    S: FnMut(&mut T) -> E,
    Inst: SumcheckInstanceProver<E>,
    A: FnMut(usize, &Inst, &mut T) -> Result<(), HachiError>,
{
    let num_rounds = instance.num_rounds();
    if omitted_prefix_rounds > num_rounds {
        return Err(HachiError::InvalidInput(format!(
            "sumcheck omitted_prefix_rounds {omitted_prefix_rounds} exceeds num_rounds {num_rounds}"
        )));
    }

    let mut claim = instance.input_claim();
    tracing::debug!(
        is_zero = claim.is_zero(),
        num_rounds,
        omitted_prefix_rounds,
        "prove_sumcheck input_claim"
    );
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

    let degree_bound = instance.degree_bound();
    let mut round_polys = Vec::with_capacity(num_rounds - omitted_prefix_rounds);
    let mut r = Vec::with_capacity(num_rounds);

    for round in 0..num_rounds {
        let g = instance.compute_round_univariate(round, claim);
        let round_sum = g.evaluate(&E::zero()) + g.evaluate(&E::one());
        debug_assert!(
            round_sum == claim,
            "sumcheck round {round} univariate does not match previous claim hint"
        );

        let compressed = g.compress();
        if compressed.degree() > degree_bound {
            return Err(HachiError::InvalidInput(format!(
                "sumcheck round poly degree {} exceeds bound {}",
                compressed.degree(),
                degree_bound
            )));
        }

        absorb_after_compute(round, instance, transcript)?;
        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &compressed);
        let r_i = sample_challenge(transcript);
        r.push(r_i);

        claim = compressed.eval_from_hint(&claim, &r_i);
        instance.ingest_challenge(round, r_i);
        if round >= omitted_prefix_rounds {
            round_polys.push(compressed);
        }
    }

    instance.finalize();
    Ok((SumcheckProof { round_polys }, r, claim))
}

/// Verify a sumcheck proof whose first `prefix_rounds` rounds are reconstructed by
/// a caller-supplied generator instead of being stored in `proof`.
///
/// The verifier still follows the ordinary transcript pipeline, sampling each
/// challenge only after absorbing that round's compressed univariate. For
/// rounds `round < prefix_rounds`, the compressed univariate is provided by
/// `prefix_round_poly`; later rounds are read from `proof`.
///
/// Returns the full challenge point `r` on success.
///
/// # Errors
///
/// Returns an error if `prefix_rounds` exceeds the verifier round count, if the
/// suffix proof length is inconsistent, if a generated/stored round polynomial
/// exceeds the degree bound, or if the final oracle check fails.
#[tracing::instrument(skip_all, name = "verify_sumcheck")]
#[inline(never)]
pub(crate) fn verify_sumcheck_with_prefix_rounds<F, T, E, S, V, A, P>(
    proof: &SumcheckProof<E>,
    verifier: &V,
    transcript: &mut T,
    mut sample_challenge: S,
    prefix_rounds: usize,
    mut absorb_before_round: A,
    mut prefix_round_poly: P,
) -> Result<Vec<E>, HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore,
    S: FnMut(&mut T) -> E,
    V: SumcheckInstanceVerifier<E>,
    A: FnMut(usize, &mut T) -> Result<(), HachiError>,
    P: FnMut(usize, E, &[E]) -> CompressedUniPoly<E>,
{
    let num_rounds = verifier.num_rounds();
    if prefix_rounds > num_rounds {
        return Err(HachiError::InvalidInput(format!(
            "sumcheck prefix_rounds {prefix_rounds} exceeds num_rounds {num_rounds}"
        )));
    }
    let expected_suffix_rounds = num_rounds - prefix_rounds;
    if proof.round_polys.len() != expected_suffix_rounds {
        return Err(HachiError::InvalidSize {
            expected: expected_suffix_rounds,
            actual: proof.round_polys.len(),
        });
    }

    let mut claim = verifier.input_claim();
    tracing::debug!(
        is_zero = claim.is_zero(),
        num_rounds,
        prefix_rounds,
        "verify_sumcheck input_claim"
    );
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

    let degree_bound = verifier.degree_bound();
    let mut challenges = Vec::with_capacity(num_rounds);
    let mut suffix_iter = proof.round_polys.iter();

    for round in 0..num_rounds {
        absorb_before_round(round, transcript)?;
        let poly = if round < prefix_rounds {
            prefix_round_poly(round, claim, &challenges)
        } else {
            suffix_iter
                .next()
                .cloned()
                .expect("suffix proof length checked above")
        };
        if poly.degree() > degree_bound {
            return Err(HachiError::InvalidInput(format!(
                "sumcheck round poly degree {} exceeds bound {}",
                poly.degree(),
                degree_bound
            )));
        }

        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &poly);
        let r_i = sample_challenge(transcript);
        challenges.push(r_i);
        claim = poly.eval_from_hint(&claim, &r_i);
    }
    debug_assert!(suffix_iter.next().is_none());

    check_sumcheck_output_claim(claim, verifier, &challenges)?;
    Ok(challenges)
}

/// Run the sumcheck round loop and return the challenges and final accumulated
/// claim, **without** performing the oracle check.  The caller is responsible
/// for verifying the oracle equality afterwards (e.g. via
/// [`check_sumcheck_output_claim`]).
///
/// This is useful when the oracle check requires external data that is only
/// available after the sumcheck challenges have been determined (e.g. deferred
/// `m_val` computation in stage 2).
///
/// # Errors
///
/// Returns [`HachiError::InvalidSize`] if the proof round count does not match
/// the verifier, or [`HachiError::InvalidInput`] if any round polynomial
/// exceeds the degree bound.
#[tracing::instrument(skip_all, name = "verify_sumcheck_rounds_only")]
pub fn verify_sumcheck_rounds_only<F, T, E, S, V>(
    proof: &SumcheckProof<E>,
    verifier: &V,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<(Vec<E>, E), HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore,
    S: FnMut(&mut T) -> E,
    V: SumcheckInstanceVerifier<E>,
{
    let num_rounds = verifier.num_rounds();
    if proof.round_polys.len() != num_rounds {
        return Err(HachiError::InvalidSize {
            expected: num_rounds,
            actual: proof.round_polys.len(),
        });
    }

    let mut claim = verifier.input_claim();
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

    let degree_bound = verifier.degree_bound();
    let mut challenges = Vec::with_capacity(num_rounds);

    for poly in proof.round_polys.iter() {
        if poly.degree() > degree_bound {
            return Err(HachiError::InvalidInput(format!(
                "sumcheck round poly degree {} exceeds bound {}",
                poly.degree(),
                degree_bound
            )));
        }
        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, poly);
        let r_i = sample_challenge(transcript);
        challenges.push(r_i);
        claim = poly.eval_from_hint(&claim, &r_i);
    }

    Ok((challenges, claim))
}

/// Enforce the final sumcheck oracle equality for the provided challenge point.
///
/// This is useful when some prefix rounds are reconstructed outside the generic
/// verifier driver and the caller needs to check the final oracle value against
/// the full concatenated challenge vector.
///
/// # Errors
///
/// Returns any error produced by `verifier.expected_output_claim`, or
/// [`HachiError::InvalidProof`] if the final claim does not match the oracle
/// evaluation at `challenges`.
pub fn check_sumcheck_output_claim<E, V>(
    final_claim: E,
    verifier: &V,
    challenges: &[E],
) -> Result<(), HachiError>
where
    E: FieldCore,
    V: SumcheckInstanceVerifier<E>,
{
    let expected = verifier.expected_output_claim(challenges)?;
    if final_claim != expected {
        tracing::error!(
            rounds = verifier.num_rounds(),
            degree_bound = verifier.degree_bound(),
            diff_is_zero = (final_claim - expected).is_zero(),
            "verify_sumcheck MISMATCH"
        );
        return Err(HachiError::InvalidProof);
    }
    Ok(())
}

/// Produce a sumcheck proof for a single instance, driving the Fiat-Shamir transcript.
///
/// This method:
/// - does **not** absorb the initial claim into the transcript (callers should do so),
/// - appends each round message under `labels::ABSORB_SUMCHECK_ROUND`,
/// - samples one challenge per round via `sample_challenge`,
/// - updates the running claim using the per-round hint (`g(0)+g(1)`).
///
/// It returns the proof, the derived point `r`, and the final claimed value at `r`.
///
/// # Errors
///
/// Returns an error if any per-round polynomial exceeds the instance's degree bound.
pub fn prove_sumcheck<F, T, E, S, Inst>(
    instance: &mut Inst,
    transcript: &mut T,
    sample_challenge: S,
) -> Result<(SumcheckProof<E>, Vec<E>, E), HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore,
    S: FnMut(&mut T) -> E,
    Inst: SumcheckInstanceProver<E>,
{
    prove_sumcheck_with_omitted_prefix_rounds::<F, T, E, S, Inst, _>(
        instance,
        transcript,
        sample_challenge,
        0,
        |_, _, _| Ok(()),
    )
}

/// Verify a single-instance sumcheck proof.
///
/// This function:
/// - absorbs the initial claim into the transcript,
/// - delegates round-by-round verification to [`SumcheckProof::verify`],
/// - performs the final oracle check: `final_claim == verifier.expected_output_claim(r)`.
///
/// Returns the challenge point `r` on success.
///
/// # Errors
///
/// Returns [`HachiError::InvalidProof`] if the final sumcheck claim does not
/// match the oracle evaluation, or propagates any error from the per-round
/// verification (e.g. degree-bound violation, round-count mismatch).
pub fn verify_sumcheck<F, T, E, S, V>(
    proof: &SumcheckProof<E>,
    verifier: &V,
    transcript: &mut T,
    sample_challenge: S,
) -> Result<Vec<E>, HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore,
    S: FnMut(&mut T) -> E,
    V: SumcheckInstanceVerifier<E>,
{
    verify_sumcheck_with_prefix_rounds::<F, T, E, S, V, _, _>(
        proof,
        verifier,
        transcript,
        sample_challenge,
        0,
        |_, _| Ok(()),
        |_, _, _| unreachable!("no prefix rounds requested"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Prime128M8M4M1M0;
    use crate::protocol::transcript::labels as tr_labels;
    use crate::protocol::transcript::Blake2bTranscript;

    type F = Prime128M8M4M1M0;

    #[derive(Clone)]
    struct ToyMlInstance {
        original: Vec<F>,
        current: Vec<F>,
        num_rounds: usize,
    }

    impl ToyMlInstance {
        fn new(evals: Vec<F>) -> Self {
            let len = evals.len();
            let num_rounds = len.trailing_zeros() as usize;
            debug_assert_eq!(1usize << num_rounds, len);
            Self {
                original: evals.clone(),
                current: evals,
                num_rounds,
            }
        }
    }

    impl SumcheckInstanceProver<F> for ToyMlInstance {
        fn num_rounds(&self) -> usize {
            self.num_rounds
        }

        fn degree_bound(&self) -> usize {
            1
        }

        fn input_claim(&self) -> F {
            self.original
                .iter()
                .copied()
                .fold(F::zero(), |acc, x| acc + x)
        }

        fn compute_round_univariate(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
            debug_assert_eq!(self.current.len(), 1usize << (self.num_rounds - round));
            let half = self.current.len() / 2;
            let mut at_zero = F::zero();
            let mut slope = F::zero();
            for j in 0..half {
                let left = self.current[2 * j];
                let right = self.current[2 * j + 1];
                at_zero += left;
                slope += right - left;
            }
            let poly = UniPoly::from_coeffs(vec![at_zero, slope]);
            debug_assert_eq!(
                poly.evaluate(&F::zero()) + poly.evaluate(&F::one()),
                previous_claim
            );
            poly
        }

        fn ingest_challenge(&mut self, _round: usize, r_round: F) {
            fold_evals_in_place(&mut self.current, r_round);
        }
    }

    impl SumcheckInstanceVerifier<F> for ToyMlInstance {
        fn num_rounds(&self) -> usize {
            self.num_rounds
        }

        fn degree_bound(&self) -> usize {
            1
        }

        fn input_claim(&self) -> F {
            self.original
                .iter()
                .copied()
                .fold(F::zero(), |acc, x| acc + x)
        }

        fn expected_output_claim(&self, challenges: &[F]) -> Result<F, HachiError> {
            multilinear_eval(&self.original, challenges)
        }
    }

    fn new_transcript() -> Blake2bTranscript<F> {
        <Blake2bTranscript<F> as Transcript<F>>::new(tr_labels::DOMAIN_HACHI_PROTOCOL)
    }

    fn sample_round(tr: &mut Blake2bTranscript<F>) -> F {
        tr.challenge_scalar(tr_labels::CHALLENGE_SUMCHECK_ROUND)
    }

    #[test]
    fn prove_sumcheck_with_omitted_prefix_rounds_matches_full_proof_tail() {
        let evals: Vec<F> = (0..16).map(|i| F::from_u64((7 * i as u64) + 3)).collect();
        let mut full = ToyMlInstance::new(evals.clone());
        let mut full_tr = new_transcript();
        let (full_proof, full_challenges, full_final_claim) =
            prove_sumcheck::<F, _, F, _, _>(&mut full, &mut full_tr, sample_round).unwrap();

        let mut omitted = ToyMlInstance::new(evals);
        let mut omitted_tr = new_transcript();
        let (suffix_proof, challenges, suffix_final_claim) =
            prove_sumcheck_with_omitted_prefix_rounds::<F, _, F, _, _, _>(
                &mut omitted,
                &mut omitted_tr,
                sample_round,
                2,
                |_, _, _| Ok(()),
            )
            .unwrap();

        assert_eq!(challenges, full_challenges);
        assert_eq!(
            suffix_proof.round_polys.as_slice(),
            &full_proof.round_polys[2..]
        );
        assert_eq!(suffix_final_claim, full_final_claim);
    }

    #[test]
    fn verify_sumcheck_with_prefix_rounds_matches_full_verification_tail() {
        let evals: Vec<F> = (0..16).map(|i| F::from_u64((11 * i as u64) + 5)).collect();
        let mut prover = ToyMlInstance::new(evals.clone());
        let mut proof_tr = new_transcript();
        let (full_proof, full_challenges, full_final_claim) =
            prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut proof_tr, sample_round).unwrap();

        let verifier = ToyMlInstance::new(evals);
        let suffix_proof = SumcheckProof {
            round_polys: full_proof.round_polys[2..].to_vec(),
        };
        let prefix_rounds = full_proof.round_polys[..2].to_vec();
        let mut verify_tr = new_transcript();
        let challenges = verify_sumcheck_with_prefix_rounds::<F, _, F, _, _, _, _>(
            &suffix_proof,
            &verifier,
            &mut verify_tr,
            sample_round,
            2,
            |_, _| Ok(()),
            |round, _, _| prefix_rounds[round].clone(),
        )
        .unwrap();

        assert_eq!(challenges, full_challenges);
        assert_eq!(
            verifier.expected_output_claim(&challenges).unwrap(),
            full_final_claim
        );
    }
}
