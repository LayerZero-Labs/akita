//! Sumcheck proof containers and round-message types.

use akita_algebra::uni_poly::CompressedUniPoly;
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use akita_transcript::labels;
use akita_transcript::Transcript;
use std::io::{Read, Write};

/// Eq-factored round message storing `q(X)` without its linear coefficient.
///
/// The wire encoding is headerless, just like [`CompressedUniPoly`]. We store
/// `[q_0, q_2, q_3, ..., q_d]` for an inner polynomial
/// `q(X) = q_0 + q_1 X + ... + q_d X^d`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EqFactoredUniPoly<E: FieldCore> {
    /// Coefficients excluding the linear term: `[q_0, q_2, q_3, ..., q_d]`.
    pub coeffs_except_linear_term: Vec<E>,
}

impl<E: FieldCore> EqFactoredUniPoly<E> {
    /// Construct from the full coefficient list of `q(X)`.
    pub fn from_q_coeffs(q_coeffs: Vec<E>) -> Self {
        if q_coeffs.is_empty() {
            return Self {
                coeffs_except_linear_term: Vec::new(),
            };
        }
        if q_coeffs.len() == 1 {
            return Self {
                coeffs_except_linear_term: vec![q_coeffs[0]],
            };
        }

        let mut coeffs_except_linear_term = Vec::with_capacity(q_coeffs.len() - 1);
        coeffs_except_linear_term.push(q_coeffs[0]);
        coeffs_except_linear_term.extend_from_slice(&q_coeffs[2..]);
        Self {
            coeffs_except_linear_term,
        }
    }

    /// Number of stored coefficients for a degree-`degree` inner polynomial.
    pub fn stored_coeff_count_for_degree(degree: usize) -> usize {
        degree.max(1)
    }

    /// Degree of the underlying inner polynomial, conservatively estimated.
    pub fn degree(&self) -> usize {
        let len = self.coeffs_except_linear_term.len();
        if len <= 1 {
            0
        } else {
            len
        }
    }

    /// Constant term `q(0)`.
    pub fn constant_term(&self) -> E {
        self.coeffs_except_linear_term
            .first()
            .copied()
            .unwrap_or_else(E::zero)
    }

    /// Sum of all stored coefficients of degree at least 2, evaluated at `X = 1`.
    pub fn higher_term_sum_at_one(&self) -> E {
        self.coeffs_except_linear_term
            .iter()
            .skip(1)
            .copied()
            .fold(E::zero(), |acc, coeff| acc + coeff)
    }

    /// Evaluate the stored part of `q(X)`, omitting the linear term.
    pub fn eval_known_terms(&self, x: &E) -> E {
        if self.coeffs_except_linear_term.is_empty() {
            return E::zero();
        }

        let mut acc = self.coeffs_except_linear_term[0];
        let mut pow = *x * *x;
        for coeff in self.coeffs_except_linear_term.iter().skip(1) {
            acc += *coeff * pow;
            pow *= *x;
        }
        acc
    }
}

impl<E: Valid + FieldCore> Valid for EqFactoredUniPoly<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.coeffs_except_linear_term.check()
    }
}

impl<E: FieldCore + AkitaSerialize> AkitaSerialize for EqFactoredUniPoly<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for coeff in &self.coeffs_except_linear_term {
            coeff.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coeffs_except_linear_term
            .iter()
            .map(|coeff| coeff.serialized_size(compress))
            .sum()
    }
}

impl<E: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for EqFactoredUniPoly<E>
{
    /// Degree of the inner polynomial `q(X)`.
    type Context = usize;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        degree: &usize,
    ) -> Result<Self, SerializationError> {
        let stored_coeffs = Self::stored_coeff_count_for_degree(*degree);
        let mut coeffs_except_linear_term = Vec::with_capacity(stored_coeffs);
        for _ in 0..stored_coeffs {
            coeffs_except_linear_term.push(E::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
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

/// Full univariate polynomial in fixed-width proof form.
///
/// Unlike [`UniPoly`]'s generic serialization, this type is encoded without a
/// length prefix. The surrounding proof shape supplies the expected degree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullUniPoly<E: FieldCore> {
    /// Coefficients from low degree to high degree.
    pub coeffs: Vec<E>,
}

impl<E: FieldCore> FullUniPoly<E> {
    /// Construct from a dense coefficient list.
    pub fn from_coeffs(coeffs: Vec<E>) -> Self {
        Self { coeffs }
    }

    /// Degree of the polynomial, conservatively treating empty as degree 0.
    pub fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    /// Evaluate at `x` via Horner's method.
    pub fn evaluate(&self, x: &E) -> E {
        let mut acc = E::zero();
        for coeff in self.coeffs.iter().rev() {
            acc = acc * *x + *coeff;
        }
        acc
    }

    /// Borrow the full coefficient vector.
    pub fn coeffs(&self) -> &[E] {
        &self.coeffs
    }
}

impl<E: Valid + FieldCore> Valid for FullUniPoly<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.coeffs.check()
    }
}

impl<E: FieldCore + AkitaSerialize> AkitaSerialize for FullUniPoly<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for coeff in &self.coeffs {
            coeff.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coeffs
            .iter()
            .map(|coeff| coeff.serialized_size(compress))
            .sum()
    }
}

impl<E: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize for FullUniPoly<E> {
    /// Degree of the full polynomial.
    type Context = usize;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        degree: &usize,
    ) -> Result<Self, SerializationError> {
        let mut coeffs = Vec::with_capacity(degree.saturating_add(1));
        for _ in 0..degree.saturating_add(1) {
            coeffs.push(E::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
        let out = Self { coeffs };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

/// ZK plain-opening mask payload for standard sumcheck rounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumcheckProofMasked<E: FieldCore> {
    /// Transcript-visible masked round polynomials.
    pub masked_round_polys: Vec<FullUniPoly<E>>,
}

impl<E: Valid + FieldCore> Valid for SumcheckProofMasked<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.masked_round_polys.check()
    }
}

impl<E: FieldCore + AkitaSerialize> AkitaSerialize for SumcheckProofMasked<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for poly in &self.masked_round_polys {
            poly.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.masked_round_polys
            .iter()
            .map(|poly| poly.serialized_size(compress))
            .sum()
    }
}

impl<E: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for SumcheckProofMasked<E>
{
    /// `(num_rounds, degree)` — number of round polynomials and their degree.
    type Context = SumcheckProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &SumcheckProofShape,
    ) -> Result<Self, SerializationError> {
        let (num_rounds, degree) = *ctx;
        let mut masked_round_polys = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            masked_round_polys.push(FullUniPoly::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &degree,
            )?);
        }
        let out = Self { masked_round_polys };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<E: Valid + FieldCore> Valid for SumcheckProof<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.round_polys.check()
    }
}

/// Shape context for deserializing a [`SumcheckProof`]: `(num_rounds, degree)`.
pub type SumcheckProofShape = (usize, usize);

impl<E: FieldCore + AkitaSerialize> AkitaSerialize for SumcheckProof<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for poly in &self.round_polys {
            poly.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.round_polys
            .iter()
            .map(|p| p.serialized_size(compress))
            .sum()
    }
}

impl<E: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize for SumcheckProof<E> {
    /// `(num_rounds, degree)` — number of round polynomials and their degree.
    type Context = SumcheckProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &SumcheckProofShape,
    ) -> Result<Self, SerializationError> {
        let (num_rounds, degree) = *ctx;
        let mut round_polys = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            round_polys.push(CompressedUniPoly::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &degree,
            )?);
        }
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
    ) -> Result<(E, Vec<E>), AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        if self.round_polys.len() != num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: num_rounds,
                actual: self.round_polys.len(),
            });
        }

        let mut r = Vec::with_capacity(num_rounds);
        for poly in &self.round_polys {
            if poly.degree() > degree_bound {
                return Err(AkitaError::InvalidInput(format!(
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

/// Eq-factored sumcheck proof containing one compressed inner polynomial per round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EqFactoredSumcheckProof<E: FieldCore> {
    /// One eq-factored inner polynomial per sumcheck round.
    pub round_polys: Vec<EqFactoredUniPoly<E>>,
}

/// ZK plain-opening mask payload for eq-factored sumcheck rounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EqFactoredSumcheckProofMasked<E: FieldCore> {
    /// Transcript-visible masked eq-factored round polynomials.
    pub masked_round_polys: Vec<EqFactoredUniPoly<E>>,
}

impl<E: Valid + FieldCore> Valid for EqFactoredSumcheckProofMasked<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.masked_round_polys.check()
    }
}

impl<E: FieldCore + AkitaSerialize> AkitaSerialize for EqFactoredSumcheckProofMasked<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for poly in &self.masked_round_polys {
            poly.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.masked_round_polys
            .iter()
            .map(|poly| poly.serialized_size(compress))
            .sum()
    }
}

impl<E: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for EqFactoredSumcheckProofMasked<E>
{
    /// `(num_rounds, q_degree)` — number of round polynomials and the degree of `q`.
    type Context = EqFactoredSumcheckProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &EqFactoredSumcheckProofShape,
    ) -> Result<Self, SerializationError> {
        let (num_rounds, degree) = *ctx;
        let mut masked_round_polys = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            masked_round_polys.push(EqFactoredUniPoly::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &degree,
            )?);
        }
        let out = Self { masked_round_polys };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<E: Valid + FieldCore> Valid for EqFactoredSumcheckProof<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.round_polys.check()
    }
}

/// Shape context for deserializing an [`EqFactoredSumcheckProof`]:
/// `(num_rounds, q_degree)`.
pub type EqFactoredSumcheckProofShape = (usize, usize);

impl<E: FieldCore + AkitaSerialize> AkitaSerialize for EqFactoredSumcheckProof<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for poly in &self.round_polys {
            poly.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.round_polys
            .iter()
            .map(|poly| poly.serialized_size(compress))
            .sum()
    }
}

impl<E: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for EqFactoredSumcheckProof<E>
{
    /// `(num_rounds, q_degree)` — number of round polynomials and the degree of `q`.
    type Context = EqFactoredSumcheckProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &EqFactoredSumcheckProofShape,
    ) -> Result<Self, SerializationError> {
        let (num_rounds, degree) = *ctx;
        let mut round_polys = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            round_polys.push(EqFactoredUniPoly::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &degree,
            )?);
        }
        let out = Self { round_polys };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}
