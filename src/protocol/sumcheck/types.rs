//! Sumcheck proof containers and round-message types.

use crate::algebra::uni_poly::CompressedUniPoly;
use crate::error::HachiError;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::protocol::transcript::labels;
use crate::protocol::transcript::Transcript;
use crate::FieldCore;
use std::io::{Read, Write};

/// Eq-compressed round message storing `q(X)` without its linear coefficient.
///
/// We store `[q_0, q_2, q_3, ..., q_d]` for an inner polynomial
/// `q(X) = q_0 + q_1 X + ... + q_d X^d`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EqCompressedUniPoly<E: FieldCore> {
    /// Coefficients excluding the linear term: `[q_0, q_2, q_3, ..., q_d]`.
    pub coeffs_except_linear_term: Vec<E>,
}

impl<E: FieldCore> EqCompressedUniPoly<E> {
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
            pow = pow * *x;
        }
        acc
    }
}

impl<E: Valid + FieldCore> Valid for EqCompressedUniPoly<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.coeffs_except_linear_term.check()
    }
}

impl<E: FieldCore> HachiSerialize for EqCompressedUniPoly<E> {
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

impl<E: FieldCore + Valid> HachiDeserialize for EqCompressedUniPoly<E> {
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

impl<E: Valid + FieldCore> Valid for SumcheckProof<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.round_polys.check()
    }
}

/// Shape context for deserializing a [`SumcheckProof`]: `(num_rounds, degree)`.
pub type SumcheckProofShape = (usize, usize);

impl<E: FieldCore> HachiSerialize for SumcheckProof<E> {
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

impl<E: FieldCore + Valid> HachiDeserialize for SumcheckProof<E> {
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

/// Eq-compressed sumcheck proof containing one compressed inner polynomial per round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EqCompressedSumcheckProof<E: FieldCore> {
    /// One eq-compressed inner polynomial per sumcheck round.
    pub round_polys: Vec<EqCompressedUniPoly<E>>,
}

impl<E: Valid + FieldCore> Valid for EqCompressedSumcheckProof<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.round_polys.check()
    }
}

/// Shape context for deserializing an [`EqCompressedSumcheckProof`]:
/// `(num_rounds, q_degree)`.
pub type EqCompressedSumcheckProofShape = (usize, usize);

impl<E: FieldCore> HachiSerialize for EqCompressedSumcheckProof<E> {
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

impl<E: FieldCore + Valid> HachiDeserialize for EqCompressedSumcheckProof<E> {
    /// `(num_rounds, q_degree)` — number of round polynomials and the degree of `q`.
    type Context = EqCompressedSumcheckProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &EqCompressedSumcheckProofShape,
    ) -> Result<Self, SerializationError> {
        let (num_rounds, degree) = *ctx;
        let mut round_polys = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            round_polys.push(EqCompressedUniPoly::deserialize_with_mode(
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
