//! Sumcheck data types: univariate polynomials, compressed representation, and proof container.

use crate::error::HachiError;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::protocol::transcript::labels;
use crate::protocol::transcript::Transcript;
use crate::FieldCore;
use crate::FromSmallInt;
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
        let mut out = Vec::with_capacity(coeffs.len().saturating_sub(1));
        out.push(coeffs[0]);
        out.extend_from_slice(&coeffs[2..]);
        CompressedUniPoly {
            coeffs_except_linear_term: out,
        }
    }
}

impl<E: FieldCore + FromSmallInt> UniPoly<E> {
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

        let mut table = evals.to_vec();
        let mut deltas = vec![table[0]];
        for _ in 1..n {
            for j in 0..table.len() - 1 {
                table[j] = table[j + 1] - table[j];
            }
            table.pop();
            deltas.push(table[0]);
        }

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
    /// `compress()` stores `[c0, c2, ..., cd]` — exactly `d` entries for
    /// degree `d >= 2`.  For `len <= 1` (degree 0 or 1, which are ambiguous
    /// in compressed form) we report 0; this is conservative for the
    /// verifier's degree-bound check since `degree_bound >= 2` in practice.
    pub fn degree(&self) -> usize {
        let len = self.coeffs_except_linear_term.len();
        if len <= 1 {
            0
        } else {
            len
        }
    }

    fn recover_linear_term(&self, hint: &E) -> E {
        if self.coeffs_except_linear_term.is_empty() {
            return E::zero();
        }

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

        let mut pow = *x * *x;
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
