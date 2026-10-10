//! The claims one root reduction proves about a commitment.
//!
//! A commitment is opened to a binary claim, a prime claim, or both. The mode
//! is part of the statement: it fixes the grinding plan, whose canonical bytes
//! are absorbed before any claim or proof message.

use akita_algebra::{offset_eq::eq_eval_at_index, EqPolynomial};
use akita_error::{checked, AkitaError};
use jolt_field::{CanonicalEncoding, ExtField, Field};

use crate::{
    codec,
    lowered::{zero_vec, LoweredRootLayout},
};

/// Which claims a root reduction proves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootOpeningMode {
    /// A host-field evaluation of the committed words.
    Binary,
    /// A linear functional of the committed bits over the challenge field.
    Prime,
    /// Both claims about the same commitment, under the same fold.
    Both,
}

impl RootOpeningMode {
    pub const fn has_binary(self) -> bool {
        matches!(self, Self::Binary | Self::Both)
    }

    pub const fn has_prime(self) -> bool {
        matches!(self, Self::Prime | Self::Both)
    }
}

/// A prime claim: a linear functional of the committed bits, read as zero and
/// one in the challenge field `E`.
///
/// Write `src[j][col]` for packed source ring element `j` of column `col`,
/// with integer coefficients (`pack_source_column`): coefficient `k * s + c`
/// is bit `s` of cell `c` of that ring element, negated for odd `s` when
/// `k > 1`. The claim is
///
/// `value = sum_col eq(column_point, col) * sum_j eq(ring_point, j)
///          * sum_t weights[t] * src[j][col][t]`.
///
/// The weights range over all `D` coefficients of a ring element, including
/// the positions no source bit occupies, so a consumer can test every
/// position. [`Self::cell_evaluation`] builds the weights of an evaluation
/// over cells.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeClaim<E> {
    /// `r_col`: one coordinate per bit of the column index, low bit first.
    pub column_point: Vec<E>,
    /// `r_ring`: one coordinate per bit of the ring-element index, low bit
    /// first.
    pub ring_point: Vec<E>,
    /// `omega`: one weight per coefficient of a ring element.
    pub weights: Vec<E>,
    pub value: E,
}

impl<E: Field> PrimeClaim<E> {
    /// The claim `value = sum_cell eq(point, cell) * sum_s bit_weights[s] *
    /// bit_s(cell)` over the cells of the commitment.
    ///
    /// `point` has one coordinate per bit of the cell index
    /// `col * scalar_rows + j * k + c`, low bit first: the split the binary
    /// opening uses for its rows and columns. `bit_weights` has one entry per
    /// coordinate of an embedded cell. The weights are
    /// `omega[k * s + c] = sigma(s) * bit_weights[s] * eq(r_c, c)`, which
    /// cancels the packing sign `sigma(s)`.
    pub fn cell_evaluation(
        layout: &LoweredRootLayout,
        point: &[E],
        bit_weights: &[E],
        value: E,
    ) -> Result<Self, AkitaError> {
        let invalid = || AkitaError::InvalidInput("prime cell claim geometry mismatch".into());
        let k = layout.k();
        if !k.is_power_of_two()
            || !layout.scalar_rows().is_power_of_two()
            || checked::exact_div(layout.degree(), k) != Some(bit_weights.len())
        {
            return Err(invalid());
        }
        let (rows, column_point) = point
            .split_at_checked(layout.scalar_rows().trailing_zeros() as usize)
            .ok_or_else(invalid)?;
        let (cell_point, ring_point) = rows
            .split_at_checked(k.trailing_zeros() as usize)
            .ok_or_else(invalid)?;
        let cells = EqPolynomial::evals_serial(cell_point, None)?;
        let mut weights = zero_vec(layout.degree())?;
        for (s, (slots, &bit)) in weights.chunks_exact_mut(k).zip(bit_weights).enumerate() {
            let coefficient = checked::product([s, k]).ok_or_else(invalid)?;
            let signed = if layout.sigma(coefficient)? == 1 {
                bit
            } else {
                -bit
            };
            for (slot, &cell) in slots.iter_mut().zip(&cells) {
                *slot = signed * cell;
            }
        }
        let claim = Self {
            column_point: column_point.to_vec(),
            ring_point: ring_point.to_vec(),
            weights,
            value,
        };
        claim.validate(layout)?;
        Ok(claim)
    }

    /// Reject a claim whose points or weights do not have the layout's
    /// dimensions.
    pub fn validate(&self, layout: &LoweredRootLayout) -> Result<(), AkitaError> {
        if checked::pow2(self.column_point.len()) != Some(layout.columns())
            || checked::pow2(self.ring_point.len()) != Some(layout.m())
            || self.weights.len() != layout.degree()
        {
            return Err(AkitaError::InvalidInput(
                "prime claim geometry mismatch".into(),
            ));
        }
        Ok(())
    }

    /// Weights of the value claim on the prime left opening, coefficient-low
    /// and column-high: `eq(column_point, col) * weights[t]` on the first `D`
    /// coefficients of column `col`, zero on the coefficient tail.
    pub fn value_weights(&self, layout: &LoweredRootLayout) -> Result<Vec<E>, AkitaError> {
        self.validate(layout)?;
        let columns = EqPolynomial::evals_serial(&self.column_point, None)?;
        let mut table = zero_vec(layout.prime_len())?;
        for (column, &weight) in table
            .chunks_exact_mut(layout.padded_coefficients())
            .zip(&columns)
        {
            for (slot, &omega) in column.iter_mut().zip(&self.weights) {
                *slot = weight * omega;
            }
        }
        Ok(table)
    }

    /// Multilinear extension of [`Self::value_weights`] at `rho`:
    /// `eq(rho_col, column_point) * sum_t eq(rho_t, t) * weights[t]`.
    pub fn value_weight_mle(&self, layout: &LoweredRootLayout, rho: &[E]) -> Result<E, AkitaError> {
        self.validate(layout)?;
        if rho.len() != layout.prime_log_len() {
            return Err(AkitaError::InvalidInput(
                "prime opening MLE point dimension mismatch".into(),
            ));
        }
        let (rho_t, rho_col) = rho
            .split_at_checked(layout.padded_coefficients().trailing_zeros() as usize)
            .ok_or(AkitaError::InvalidProof)?;
        let coefficients = self
            .weights
            .iter()
            .enumerate()
            .fold(E::zero(), |sum, (t, &omega)| {
                sum + eq_eval_at_index(rho_t, t) * omega
            });
        Ok(EqPolynomial::mle(rho_col, &self.column_point)? * coefficients)
    }

    /// The statement bytes of the claim: the length-prefixed domain
    /// `akita/labinius/root-prime-claim/v1`, then the column point, the ring
    /// point, the weights and the value, every element as its canonical base
    /// coordinates. The layout fixes every count.
    pub(crate) fn statement_bytes<F>(&self) -> Result<Vec<u8>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        let mut bytes = Vec::new();
        codec::length_prefixed(&mut bytes, b"akita/labinius/root-prime-claim/v1")?;
        for values in [
            self.column_point.as_slice(),
            &self.ring_point,
            &self.weights,
            core::slice::from_ref(&self.value),
        ] {
            codec::encode_extensions::<F, E>(&mut bytes, values)?;
        }
        Ok(bytes)
    }
}

/// The statement of one root reduction about an admitted setup and its bound
/// image table: at least one of the two claims.
#[derive(Clone, Copy, Debug)]
pub struct RootStatement<'a, H, E> {
    /// Host point and value: the multilinear evaluation of the committed
    /// words over the host field `H`.
    pub binary: Option<(&'a [H], H)>,
    pub prime: Option<&'a PrimeClaim<E>>,
}

impl<H, E> RootStatement<'_, H, E> {
    /// The opening mode; a statement without a claim is an error.
    pub fn mode(&self) -> Result<RootOpeningMode, AkitaError> {
        match (self.binary.is_some(), self.prime.is_some()) {
            (true, false) => Ok(RootOpeningMode::Binary),
            (false, true) => Ok(RootOpeningMode::Prime),
            (true, true) => Ok(RootOpeningMode::Both),
            (false, false) => Err(AkitaError::InvalidInput(
                "a root statement needs a binary or a prime claim".into(),
            )),
        }
    }
}
