//! Univariate polynomial types: dense coefficient form and compressed representation.

use crate::FieldCore;
use crate::FromPrimitiveInt;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
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

    /// Degree of the polynomial (0 for empty or constant).
    pub fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
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

impl<E: FieldCore + FromPrimitiveInt> UniPoly<E> {
    /// Interpolate from evaluations at equispaced integer points `x = 0, 1, ..., d`.
    ///
    /// Uses Newton forward-difference interpolation: compute divided differences,
    /// then expand via Horner on the nested Newton form.
    ///
    /// # Panics
    ///
    /// Panics if any required factorial inverse does not exist (field characteristic
    /// must exceed the number of evaluation points). This is a prover-only
    /// function and the condition always holds for Akita's fields.
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
            factorial *= E::from_u64(k as u64);
            divided_diffs.push(
                *delta_k
                    * factorial
                        .inverse()
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
                new_coeffs[i + 1] += coeffs[i];
                new_coeffs[i] -= shift * coeffs[i];
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

impl<E: FieldCore + AkitaSerialize> AkitaSerialize for UniPoly<E> {
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

impl<E: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize for UniPoly<E> {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let coeffs = Vec::<E>::deserialize_with_mode(&mut reader, compress, validate, &())?;
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
            linear -= *c;
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
            acc += *c * pow;
            pow *= *x;
        }
        acc
    }
}

impl<E: Valid + FieldCore> Valid for CompressedUniPoly<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.coeffs_except_linear_term.check()
    }
}

impl<E: FieldCore + AkitaSerialize> AkitaSerialize for CompressedUniPoly<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for c in &self.coeffs_except_linear_term {
            c.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coeffs_except_linear_term
            .iter()
            .map(|c| c.serialized_size(compress))
            .sum()
    }
}

impl<E: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for CompressedUniPoly<E>
{
    /// Degree of the polynomial (= number of coefficients to read).
    type Context = usize;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        degree: &usize,
    ) -> Result<Self, SerializationError> {
        let mut coeffs = Vec::new();
        coeffs.try_reserve_exact(*degree).map_err(|_| {
            SerializationError::InvalidData("compressed polynomial allocation failed".to_string())
        })?;
        for _ in 0..*degree {
            coeffs.push(E::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
        let out = Self {
            coeffs_except_linear_term: coeffs,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}
