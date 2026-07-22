use super::utils::{
    accumulate_left_round, accumulate_right_round, fold_left_round, fold_right_round, product_claim,
};
use akita_algebra::uni_poly::UniPoly;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};

/// One dense factored setup-product term
/// `sum_{left,right} table[left,right] * left_factor[left] * right_factor[right]`.
pub(super) struct FactoredProductTerm<E: FieldCore> {
    table: Vec<E>,
    left_factor: Vec<E>,
    right_factor: Vec<E>,
    input_claim: E,
    right_rounds: usize,
    total_rounds: usize,
}

impl<E: FieldCore + FromPrimitiveInt> FactoredProductTerm<E> {
    /// Construct a dense factored product-sumcheck term.
    ///
    /// Returns an error if factor lengths are not powers of two, are empty, or
    /// if `table.len() != left_factor.len() * right_factor.len()`.
    pub(super) fn new_dense(
        table: Vec<E>,
        left_factor: Vec<E>,
        right_factor: Vec<E>,
    ) -> Result<Self, AkitaError> {
        if left_factor.is_empty()
            || right_factor.is_empty()
            || !left_factor.len().is_power_of_two()
            || !right_factor.len().is_power_of_two()
        {
            return Err(AkitaError::InvalidInput(
                "factored product dimensions must be non-empty powers of two".into(),
            ));
        }
        let expected_len = left_factor
            .len()
            .checked_mul(right_factor.len())
            .ok_or_else(|| AkitaError::InvalidInput("factored product size overflow".into()))?;
        if table.len() != expected_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_len,
                actual: table.len(),
            });
        }

        let input_claim = product_claim(&table, &left_factor, &right_factor);
        let right_rounds = right_factor.len().trailing_zeros() as usize;
        let total_rounds = right_rounds + left_factor.len().trailing_zeros() as usize;
        Ok(Self {
            table,
            left_factor,
            right_factor,
            input_claim,
            right_rounds,
            total_rounds,
        })
    }

    pub(super) const fn num_rounds(&self) -> usize {
        self.total_rounds
    }

    pub(super) const fn input_claim(&self) -> E {
        self.input_claim
    }

    pub(super) fn compute_round_univariate(&self, round: usize, _previous_claim: E) -> UniPoly<E> {
        let (constant, linear, quadratic) = if round < self.right_rounds {
            accumulate_right_round(&self.table, &self.left_factor, &self.right_factor)
        } else {
            accumulate_left_round(&self.table, &self.left_factor, self.right_factor[0])
        };
        UniPoly::from_coeffs(vec![constant, linear, quadratic])
    }

    pub(super) fn ingest_challenge(&mut self, round: usize, challenge: E) {
        if round < self.right_rounds {
            fold_right_round(&mut self.table, &mut self.right_factor, challenge);
        } else {
            fold_left_round(&mut self.table, &mut self.left_factor, challenge);
        }
    }

    pub(super) fn folded_table_value(&self) -> Result<E, AkitaError> {
        if self.table.len() != 1 {
            return Err(AkitaError::InvalidSize {
                expected: 1,
                actual: self.table.len(),
            });
        }
        Ok(self.table[0])
    }
}
