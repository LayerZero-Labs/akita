//! Sumcheck proof containers and round-message types.

use akita_algebra::uni_poly::CompressedUniPoly;
use akita_error::AkitaError;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use akita_transcript::labels;
use akita_transcript::Transcript;
use jolt_field::{CanonicalEncoding, Field};
use std::io::{Read, Write};

/// Eq-factored round message storing `q(X)` without its constant coefficient.
///
/// The wire encoding is headerless, just like [`CompressedUniPoly`]. We store
/// `[q_1, q_2, ..., q_d]` for an inner polynomial
/// `q(X) = q_0 + q_1 X + ... + q_d X^d`.
/// This convention is specific to the normalized equality-factored identity;
/// ordinary compressed sum-check polynomials omit their linear coefficient.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EqFactoredUniPoly<E: Field> {
    /// Coefficients excluding the constant term: `[q_1, q_2, ..., q_d]`.
    pub coeffs_except_constant_term: Vec<E>,
}

impl<E: Field> EqFactoredUniPoly<E> {
    /// Construct from the full coefficient list of `q(X)`.
    pub fn from_q_coeffs(q_coeffs: Vec<E>) -> Self {
        Self {
            coeffs_except_constant_term: q_coeffs.into_iter().skip(1).collect(),
        }
    }

    /// Degree of the underlying inner polynomial, conservatively estimated.
    pub fn degree(&self) -> usize {
        self.coeffs_except_constant_term.len()
    }

    /// Sum of the nonconstant coefficients, evaluated at `X = 1`.
    pub fn nonconstant_term_sum_at_one(&self) -> E {
        self.coeffs_except_constant_term
            .iter()
            .copied()
            .fold(E::zero(), |acc, coeff| acc + coeff)
    }

    /// Evaluate the nonconstant part of `q(X)` at `x`.
    pub fn eval_nonconstant_terms(&self, x: &E) -> E {
        let mut acc = E::zero();
        let mut pow = *x;
        for coeff in &self.coeffs_except_constant_term {
            acc += *coeff * pow;
            pow *= *x;
        }
        acc
    }
}

impl<E: Valid + Field> Valid for EqFactoredUniPoly<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.coeffs_except_constant_term.check()
    }
}

impl<E: Field + AkitaSerialize> AkitaSerialize for EqFactoredUniPoly<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for coeff in &self.coeffs_except_constant_term {
            coeff.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coeffs_except_constant_term
            .iter()
            .map(|coeff| coeff.serialized_size(compress))
            .sum()
    }
}

impl<E: Field + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize for EqFactoredUniPoly<E> {
    /// Degree of the inner polynomial `q(X)`.
    type Context = usize;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        degree: &usize,
    ) -> Result<Self, SerializationError> {
        // Every nonconstant coefficient is transmitted, so a degree-`d`
        // polynomial has exactly `d` encoded field elements. In particular, a
        // degree-zero round has an empty message.
        let stored_coeffs = *degree;
        let mut coeffs_except_constant_term = Vec::new();
        coeffs_except_constant_term
            .try_reserve_exact(stored_coeffs)
            .map_err(|_| {
                SerializationError::InvalidData(
                    "eq-factored polynomial allocation failed".to_string(),
                )
            })?;
        for _ in 0..stored_coeffs {
            coeffs_except_constant_term.push(E::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
        let out = Self {
            coeffs_except_constant_term,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

/// Sumcheck proof containing one compressed univariate polynomial per round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumcheckProof<E: Field> {
    /// One compressed univariate polynomial per sumcheck round.
    pub round_polys: Vec<CompressedUniPoly<E>>,
}

impl<E: Valid + Field> Valid for SumcheckProof<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.round_polys.check()
    }
}

/// Shape context for deserializing a [`SumcheckProof`].
///
/// Each entry is the number of serialized coefficients in the corresponding
/// compressed round polynomial. Round polynomials are headerless and need not
/// all have the same compact degree, though the prover currently emits a
/// uniform degree per sumcheck.
pub type SumcheckProofShape = Vec<usize>;

/// Construct a sumcheck shape for proofs whose rounds all use the same compact
/// coefficient count.
pub fn uniform_sumcheck_shape(num_rounds: usize, degree: usize) -> SumcheckProofShape {
    vec![degree; num_rounds]
}

impl<E: Field + AkitaSerialize> AkitaSerialize for SumcheckProof<E> {
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

impl<E: Field + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize for SumcheckProof<E> {
    /// Per-round compact coefficient counts.
    type Context = SumcheckProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &SumcheckProofShape,
    ) -> Result<Self, SerializationError> {
        let mut round_polys = Vec::new();
        round_polys.try_reserve_exact(ctx.len()).map_err(|_| {
            SerializationError::InvalidData("sumcheck proof allocation failed".to_string())
        })?;
        for degree in ctx {
            round_polys.push(CompressedUniPoly::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                degree,
            )?);
        }
        let out = Self { round_polys };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<E: Field> SumcheckProof<E> {
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
        F: Field + CanonicalEncoding,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> Result<E, AkitaError>,
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
            let r_i = sample_challenge(transcript)?;
            r.push(r_i);

            claim = poly.eval_from_hint(&claim, &r_i);
        }

        Ok((claim, r))
    }
}

/// Eq-factored sumcheck proof containing one compressed inner polynomial per round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EqFactoredSumcheckProof<E: Field> {
    /// One eq-factored inner polynomial per sumcheck round.
    pub round_polys: Vec<EqFactoredUniPoly<E>>,
}

impl<E: Valid + Field> Valid for EqFactoredSumcheckProof<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.round_polys.check()
    }
}

/// Shape context for deserializing an [`EqFactoredSumcheckProof`]:
/// `(num_rounds, q_degree)`.
pub type EqFactoredSumcheckProofShape = (usize, usize);

impl<E: Field + AkitaSerialize> AkitaSerialize for EqFactoredSumcheckProof<E> {
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

impl<E: Field + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
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
        let mut round_polys = Vec::new();
        round_polys.try_reserve_exact(num_rounds).map_err(|_| {
            SerializationError::InvalidData(
                "eq-factored sumcheck proof allocation failed".to_string(),
            )
        })?;
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
