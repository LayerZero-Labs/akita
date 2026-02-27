//! Sumcheck core building blocks (univariate messages + proof transcript driver).
//!
//! This module provides:
//! - Univariate polynomials over a field `E` (`UniPoly<E>`).
//! - A compressed univariate representation (`CompressedUniPoly<E>`) that omits the
//!   linear term and reconstructs it from the per-round sumcheck hint `g(0)+g(1)`.
//! - A minimal sumcheck proof container (`SumcheckProof<E>`) with a verifier-side
//!   transcript driver that returns the derived point and final claimed value.
//!
//! It intentionally does **not** implement a concrete sumcheck prover for any
//! particular oracle/table representation. Higher-level protocols (e.g. Hachi §4.3)
//! should implement the prover logic that constructs each round's univariate `g_i(X)`.
//!
//! ## Temporary duplication notice (Jolt integration)
//!
//! Jolt already has a mature, streaming-aware sumcheck implementation. Long-term, we
//! expect to extract the common sumcheck machinery into a dedicated crate and depend
//! on it from both Hachi and Jolt. Until that exists, this module intentionally
//! duplicates the essential sumcheck data types and transcript-driving logic as a
//! pragmatic workaround.

pub mod norm_sumcheck;
pub mod relation_sumcheck;

use crate::error::HachiError;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::protocol::transcript::labels;
use crate::protocol::transcript::Transcript;
use crate::CanonicalField;
use crate::FieldCore;
use std::io::{Read, Write};

/// Univariate polynomial in coefficient form: `p(X) = Σ_{i=0}^d coeffs[i] * X^i`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniPoly<E: FieldCore> {
    /// Coefficients from low degree to high degree.
    pub coeffs: Vec<E>,
}

impl<E: FieldCore> UniPoly<E> {
    /// Construct from coefficients in increasing-degree order.
    pub fn from_coeffs(coeffs: Vec<E>) -> Self {
        Self { coeffs }
    }

    /// Degree of the polynomial.
    ///
    /// # Panics
    ///
    /// Panics if the polynomial has no coefficients.
    pub fn degree(&self) -> usize {
        self.coeffs
            .len()
            .checked_sub(1)
            .expect("UniPoly must have at least one coefficient")
    }

    /// Evaluate at `x` via Horner's method.
    pub fn evaluate(&self, x: &E) -> E {
        // Horner from highest degree.
        let mut acc = E::zero();
        for c in self.coeffs.iter().rev() {
            acc = acc * *x + *c;
        }
        acc
    }

    /// Compress this polynomial by omitting the linear coefficient.
    ///
    /// The verifier can reconstruct/evaluate the missing linear coefficient using
    /// the per-round hint `g(0)+g(1)` from the sumcheck protocol.
    ///
    /// This matches the technique used by Jolt's sumcheck (`CompressedUniPoly`).
    pub fn compress(&self) -> CompressedUniPoly<E> {
        let coeffs = &self.coeffs;
        if coeffs.is_empty() {
            return CompressedUniPoly {
                coeffs_except_linear_term: Vec::new(),
            };
        }
        if coeffs.len() == 1 {
            return CompressedUniPoly {
                coeffs_except_linear_term: vec![coeffs[0]],
            };
        }
        // Keep coeff[0], drop coeff[1], keep coeff[2..].
        let mut out = Vec::with_capacity(coeffs.len().saturating_sub(1));
        out.push(coeffs[0]);
        out.extend_from_slice(&coeffs[2..]);
        CompressedUniPoly {
            coeffs_except_linear_term: out,
        }
    }
}

impl<E: FieldCore + crate::CanonicalField> UniPoly<E> {
    /// Interpolate from evaluations at equispaced integer points `x = 0, 1, ..., d`.
    ///
    /// Uses Newton forward-difference interpolation: compute divided differences,
    /// then expand via Horner on the nested Newton form.
    ///
    /// # Panics
    ///
    /// Panics if any required factorial inverse does not exist (field characteristic
    /// must exceed the number of evaluation points).
    pub fn from_evals(evals: &[E]) -> Self {
        let n = evals.len();
        if n == 0 {
            return Self::from_coeffs(vec![]);
        }
        if n == 1 {
            return Self::from_coeffs(vec![evals[0]]);
        }

        // Forward differences: delta^k[0]
        let mut table = evals.to_vec();
        let mut deltas = vec![table[0]];
        for _ in 1..n {
            for j in 0..table.len() - 1 {
                table[j] = table[j + 1] - table[j];
            }
            table.pop();
            deltas.push(table[0]);
        }

        // Divided differences: dd[k] = delta^k / k!
        let mut factorial = E::one();
        let mut divided_diffs = vec![deltas[0]];
        for (k, delta_k) in deltas.iter().enumerate().skip(1) {
            factorial = factorial * E::from_u64(k as u64);
            divided_diffs.push(
                *delta_k
                    * factorial
                        .inv()
                        .expect("field characteristic too small for interpolation"),
            );
        }

        // Horner expansion of the Newton nested form:
        //   p(x) = dd[0] + (x-0)(dd[1] + (x-1)(dd[2] + ...))
        let mut coeffs = vec![divided_diffs[n - 1]];

        for k in (0..n - 1).rev() {
            let shift = E::from_u64(k as u64);
            let old_len = coeffs.len();
            let mut new_coeffs = vec![E::zero(); old_len + 1];

            new_coeffs[0] = divided_diffs[k];
            for i in 0..old_len {
                new_coeffs[i + 1] = new_coeffs[i + 1] + coeffs[i];
                new_coeffs[i] = new_coeffs[i] - shift * coeffs[i];
            }

            coeffs = new_coeffs;
        }

        while coeffs.len() > 1 && coeffs.last().is_some_and(|c| c.is_zero()) {
            coeffs.pop();
        }

        Self::from_coeffs(coeffs)
    }
}

impl<E: Valid + FieldCore> Valid for UniPoly<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.coeffs.check()
    }
}

impl<E: FieldCore> HachiSerialize for UniPoly<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.coeffs.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coeffs.serialized_size(compress)
    }
}

impl<E: FieldCore + Valid> HachiDeserialize for UniPoly<E> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let coeffs = Vec::<E>::deserialize_with_mode(&mut reader, compress, validate)?;
        let out = Self { coeffs };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

/// Compressed univariate polynomial representation omitting the linear term.
///
/// We store `[c0, c2, c3, ..., cd]`. Given the sumcheck hint `hint = g(0)+g(1)`,
/// the missing linear coefficient is:
///
/// `c1 = hint - 2*c0 - Σ_{i=2..d} ci`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedUniPoly<E: FieldCore> {
    /// Coefficients excluding the linear term: `[c0, c2, c3, ..., cd]`.
    pub coeffs_except_linear_term: Vec<E>,
}

impl<E: FieldCore> CompressedUniPoly<E> {
    /// Degree of the underlying uncompressed polynomial.
    ///
    /// For degree `d`, this stores `d` coefficients (all except the linear term).
    pub fn degree(&self) -> usize {
        self.coeffs_except_linear_term.len().saturating_sub(1)
    }

    fn recover_linear_term(&self, hint: &E) -> E {
        if self.coeffs_except_linear_term.is_empty() {
            // Treat empty as the zero polynomial.
            return E::zero();
        }

        // linear = hint - 2*c0 - Σ_{i>=2} ci
        let c0 = self.coeffs_except_linear_term[0];
        let mut linear = *hint - c0 - c0;
        for c in self.coeffs_except_linear_term.iter().skip(1) {
            linear = linear - *c;
        }
        linear
    }

    /// Decompress using `hint = g(0)+g(1)`.
    pub fn decompress(&self, hint: &E) -> UniPoly<E> {
        if self.coeffs_except_linear_term.is_empty() {
            return UniPoly::from_coeffs(Vec::new());
        }
        let linear = self.recover_linear_term(hint);
        // Always materialize the missing linear coefficient.
        // For truly-constant polynomials, the hint forces `linear = 0`, so this is harmless.
        let mut coeffs = Vec::with_capacity(self.coeffs_except_linear_term.len() + 1);
        coeffs.push(self.coeffs_except_linear_term[0]);
        coeffs.push(linear);
        coeffs.extend_from_slice(&self.coeffs_except_linear_term[1..]);
        UniPoly::from_coeffs(coeffs)
    }

    /// Evaluate the uncompressed polynomial at `x`, using `hint = g(0)+g(1)`.
    ///
    /// This avoids materializing the full coefficient list.
    pub fn eval_from_hint(&self, hint: &E, x: &E) -> E {
        if self.coeffs_except_linear_term.is_empty() {
            return E::zero();
        }

        let linear = self.recover_linear_term(hint);
        let mut acc = self.coeffs_except_linear_term[0] + (*x * linear);

        // Add Σ_{i=2..d} c_i * x^i, where stored slice is [c2, c3, ..., cd]
        let mut pow = *x * *x; // x^2
        for c in self.coeffs_except_linear_term.iter().skip(1) {
            acc = acc + (*c * pow);
            pow = pow * *x;
        }
        acc
    }
}

impl<E: Valid + FieldCore> Valid for CompressedUniPoly<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.coeffs_except_linear_term.check()
    }
}

impl<E: FieldCore> HachiSerialize for CompressedUniPoly<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.coeffs_except_linear_term
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coeffs_except_linear_term.serialized_size(compress)
    }
}

impl<E: FieldCore + Valid> HachiDeserialize for CompressedUniPoly<E> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let coeffs_except_linear_term =
            Vec::<E>::deserialize_with_mode(&mut reader, compress, validate)?;
        let out = Self {
            coeffs_except_linear_term,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

/// Sumcheck proof containing one compressed univariate polynomial per round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumcheckProof<E: FieldCore> {
    /// One compressed univariate polynomial per sumcheck round.
    pub round_polys: Vec<CompressedUniPoly<E>>,
}

impl<E: Valid + FieldCore> Valid for SumcheckProof<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.round_polys.check()
    }
}

impl<E: FieldCore> HachiSerialize for SumcheckProof<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.round_polys.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.round_polys.serialized_size(compress)
    }
}

impl<E: FieldCore + Valid> HachiDeserialize for SumcheckProof<E> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let round_polys =
            Vec::<CompressedUniPoly<E>>::deserialize_with_mode(&mut reader, compress, validate)?;
        let out = Self { round_polys };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<E: FieldCore> SumcheckProof<E> {
    /// Verifier-side sumcheck transcript driver.
    ///
    /// This method:
    /// - absorbs the per-round prover message (compressed univariate),
    /// - samples one challenge per round via `sample_challenge`,
    /// - updates the running claim using `eval_from_hint`.
    ///
    /// It does **not** perform the final oracle check `final_claim == f(r*)`.
    /// Callers (e.g. ring-switching) must compute `f(r*)` themselves and compare.
    ///
    /// # Errors
    ///
    /// Returns an error if the proof length does not match `num_rounds` or if any
    /// per-round polynomial exceeds `degree_bound`.
    pub fn verify<F, T, S>(
        &self,
        mut claim: E,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut T,
        mut sample_challenge: S,
    ) -> Result<(E, Vec<E>), HachiError>
    where
        F: crate::FieldCore + crate::CanonicalField,
        T: Transcript<F>,
        S: FnMut(&mut T) -> E,
    {
        if self.round_polys.len() != num_rounds {
            return Err(HachiError::InvalidSize {
                expected: num_rounds,
                actual: self.round_polys.len(),
            });
        }

        let mut r = Vec::with_capacity(num_rounds);
        for poly in &self.round_polys {
            if poly.degree() > degree_bound {
                return Err(HachiError::InvalidInput(format!(
                    "sumcheck round poly degree {} exceeds bound {}",
                    poly.degree(),
                    degree_bound
                )));
            }

            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, poly);
            let r_i = sample_challenge(transcript);
            r.push(r_i);

            claim = poly.eval_from_hint(&claim, &r_i);
        }

        Ok((claim, r))
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
    fn expected_output_claim(&self, challenges: &[E]) -> E;
}

/// Produce a sumcheck proof for a single instance, driving the Fiat–Shamir transcript.
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
    mut sample_challenge: S,
) -> Result<(SumcheckProof<E>, Vec<E>, E), HachiError>
where
    F: crate::FieldCore + crate::CanonicalField,
    T: Transcript<F>,
    E: FieldCore,
    S: FnMut(&mut T) -> E,
    Inst: SumcheckInstanceProver<E>,
{
    let mut claim = instance.input_claim();
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

    let num_rounds = instance.num_rounds();
    let degree_bound = instance.degree_bound();

    let mut round_polys = Vec::with_capacity(num_rounds);
    let mut r = Vec::with_capacity(num_rounds);

    for round in 0..num_rounds {
        let g = instance.compute_round_univariate(round, claim);
        // Optional sanity: enforce hint relation in debug builds.
        debug_assert!(
            g.evaluate(&E::zero()) + g.evaluate(&E::one()) == claim,
            "sumcheck round univariate does not match previous claim hint"
        );

        let compressed = g.compress();
        if compressed.degree() > degree_bound {
            return Err(HachiError::InvalidInput(format!(
                "sumcheck round poly degree {} exceeds bound {}",
                compressed.degree(),
                degree_bound
            )));
        }

        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &compressed);
        let r_i = sample_challenge(transcript);
        r.push(r_i);

        // Update running claim (this is g(r_i)).
        claim = compressed.eval_from_hint(&claim, &r_i);

        instance.ingest_challenge(round, r_i);
        round_polys.push(compressed);
    }

    instance.finalize();

    let proof = SumcheckProof { round_polys };
    Ok((proof, r, claim))
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
    F: crate::FieldCore + crate::CanonicalField,
    T: Transcript<F>,
    E: FieldCore,
    S: FnMut(&mut T) -> E,
    V: SumcheckInstanceVerifier<E>,
{
    let claim = verifier.input_claim();
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);
    let (final_claim, challenges) = proof.verify::<F, T, S>(
        claim,
        verifier.num_rounds(),
        verifier.degree_bound(),
        transcript,
        sample_challenge,
    )?;

    let expected = verifier.expected_output_claim(&challenges);
    if final_claim != expected {
        return Err(HachiError::InvalidProof);
    }

    Ok(challenges)
}

// ---------------------------------------------------------------------------
// Shared helpers used by both sumcheck instances
// ---------------------------------------------------------------------------

/// Build the full eq polynomial evaluation table.
///
/// Returns a vector of size `2^{tau.len()}` where entry `b` (interpreted as a
/// little-endian bit string, i.e. bit `k` of `b` corresponds to `τ[k]`) equals
/// `ẽq(τ, b) = Π_i (τ_i·b_i + (1−τ_i)(1−b_i))`.
pub fn eq_evals<E: FieldCore>(tau: &[E]) -> Vec<E> {
    let size = 1usize << tau.len();
    let mut evals = vec![E::zero(); size];
    evals[0] = E::one();
    let mut len = 1usize;
    for &t in tau.iter().rev() {
        let one_minus_t = E::one() - t;
        for j in (0..len).rev() {
            evals[2 * j + 1] = evals[j] * t;
            evals[2 * j] = evals[j] * one_minus_t;
        }
        len *= 2;
    }
    evals
}

/// Evaluate ẽq(τ, r) at a single point.
pub fn eq_eval<E: FieldCore>(tau: &[E], point: &[E]) -> E {
    debug_assert_eq!(tau.len(), point.len());
    tau.iter().zip(point).fold(E::one(), |acc, (&t, &r)| {
        acc * (t * r + (E::one() - t) * (E::one() - r))
    })
}

/// Evaluate the range-check polynomial `w · Π_{k=1}^{b−1} (w − k)(w + k)`.
///
/// This polynomial vanishes exactly when `w ∈ {−(b−1), …, b−1}`.
/// Total degree in `w` is `2b − 1`.
pub fn range_check_eval<E: FieldCore + CanonicalField>(w: E, b: usize) -> E {
    let mut acc = w;
    for k in 1..b {
        let k_e = E::from_u64(k as u64);
        acc = acc * (w - k_e) * (w + k_e);
    }
    acc
}

/// Evaluate a multilinear polynomial (given by boolean-hypercube evaluations in
/// little-endian bit order) at an arbitrary point via iterated folding.
pub fn multilinear_eval<E: FieldCore>(evals: &[E], point: &[E]) -> E {
    let mut current = evals.to_vec();
    for &r in point {
        let half = current.len() / 2;
        let mut next = Vec::with_capacity(half);
        for i in 0..half {
            next.push(current[2 * i] + r * (current[2 * i + 1] - current[2 * i]));
        }
        current = next;
    }
    current[0]
}

/// Fold an evaluation table by binding its first variable to `r`, halving the
/// table size.
pub(crate) fn fold_evals<E: FieldCore>(evals: &[E], r: E) -> Vec<E> {
    let half = evals.len() / 2;
    let mut out = Vec::with_capacity(half);
    for i in 0..half {
        out.push(evals[2 * i] + r * (evals[2 * i + 1] - evals[2 * i]));
    }
    out
}
