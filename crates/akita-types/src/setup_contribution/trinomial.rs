//! Checked setup-table views for a trinomial relation bridge.
//!
//! The view keeps natural polynomial degree separate from padded response-table
//! strides. It prepares one canonical paired-tensor description that drives
//! both prover-side weight materialization and verifier-side MLE evaluation.

use std::sync::Arc;

use akita_algebra::offset_eq::{
    eval_boolean_pair_tensor_families, materialize_eq_tensor_left, EqPairTensorAxis,
    EqPairTensorFamily, OffsetEqWindow, MAX_COMPACT_STRIDE_TERMS,
};
use akita_error::{checked, AkitaError};
use jolt_field::Field;

use crate::{
    RelationCoefficientLayout, RelationCoefficientRole, RelationPolynomial, RelationPolynomialKind,
};

/// Padded response-table layout for trinomial setup contributions.
///
/// Digits are innermost, followed by padded coefficients and then
/// polynomials:
///
/// ```text
/// offset + polynomial * polynomial_stride
///        + coefficient * coefficient_stride
///        + digit * digit_stride
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TrinomialResponseLayout {
    polynomial: RelationPolynomial,
    coefficients: RelationCoefficientLayout,
    polynomial_count: usize,
    digit_depth: usize,
    offset: usize,
    domain_len: usize,
    coefficient_stride: usize,
    polynomial_stride: usize,
}

impl TrinomialResponseLayout {
    /// Construct a checked digit-innermost response layout.
    pub fn new(
        polynomial: RelationPolynomial,
        padded_coefficient_len: usize,
        polynomial_count: usize,
        digit_depth: usize,
        offset: usize,
        domain_len: usize,
    ) -> Result<Self, AkitaError> {
        if !matches!(polynomial.kind(), RelationPolynomialKind::Trinomial(_)) {
            return Err(AkitaError::InvalidSetup(
                "trinomial response layout requires a trinomial polynomial".into(),
            ));
        }
        if polynomial_count == 0 || digit_depth == 0 {
            return Err(AkitaError::InvalidSetup(
                "trinomial response dimensions must be nonzero".into(),
            ));
        }
        if !domain_len.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "trinomial response domain must be a nonzero power of two".into(),
            ));
        }
        let coefficients = RelationCoefficientLayout::new(
            polynomial,
            RelationCoefficientRole::Element,
            padded_coefficient_len,
        )?;
        let coefficient_stride = digit_depth;
        let polynomial_stride = padded_coefficient_len
            .checked_mul(coefficient_stride)
            .ok_or_else(|| AkitaError::InvalidSetup("trinomial response stride overflow".into()))?;
        let table_len = polynomial_count
            .checked_mul(polynomial_stride)
            .ok_or_else(|| AkitaError::InvalidSetup("trinomial response length overflow".into()))?;
        validate_work(table_len)?;
        let end = offset
            .checked_add(table_len)
            .ok_or_else(|| AkitaError::InvalidSetup("trinomial response extent overflow".into()))?;
        if end > domain_len {
            return Err(AkitaError::InvalidSetup(
                "trinomial response extent exceeds its domain".into(),
            ));
        }
        Ok(Self {
            polynomial,
            coefficients,
            polynomial_count,
            digit_depth,
            offset,
            domain_len,
            coefficient_stride,
            polynomial_stride,
        })
    }

    #[must_use]
    pub const fn polynomial(self) -> RelationPolynomial {
        self.polynomial
    }

    #[must_use]
    pub const fn coefficient_layout(self) -> RelationCoefficientLayout {
        self.coefficients
    }

    #[must_use]
    pub const fn polynomial_count(self) -> usize {
        self.polynomial_count
    }

    #[must_use]
    pub const fn digit_depth(self) -> usize {
        self.digit_depth
    }

    #[must_use]
    pub const fn offset(self) -> usize {
        self.offset
    }

    #[must_use]
    pub const fn domain_len(self) -> usize {
        self.domain_len
    }

    #[must_use]
    pub const fn digit_stride(self) -> usize {
        1
    }

    #[must_use]
    pub const fn coefficient_stride(self) -> usize {
        self.coefficient_stride
    }

    #[must_use]
    pub const fn polynomial_stride(self) -> usize {
        self.polynomial_stride
    }

    /// Return the checked address of one natural response coefficient digit.
    pub fn address(
        self,
        polynomial: usize,
        digit: usize,
        coefficient: usize,
    ) -> Result<usize, AkitaError> {
        if polynomial >= self.polynomial_count
            || digit >= self.digit_depth
            || coefficient >= self.coefficients.natural_len()
        {
            return Err(AkitaError::InvalidInput(
                "trinomial response coordinate out of range".into(),
            ));
        }
        self.padded_address(polynomial, digit, coefficient)
    }

    /// Validate a complete response domain and require every coefficient tail
    /// in every polynomial and digit lane to be zero.
    pub fn validate_zero_tails<F: Field>(self, table: &[F]) -> Result<(), AkitaError> {
        if table.len() != self.domain_len {
            return Err(AkitaError::InvalidSize {
                expected: self.domain_len,
                actual: table.len(),
            });
        }
        for polynomial in 0..self.polynomial_count {
            for coefficient in self.coefficients.natural_len()..self.coefficients.padded_len() {
                for digit in 0..self.digit_depth {
                    let address = self.padded_address(polynomial, digit, coefficient)?;
                    if !table
                        .get(address)
                        .ok_or(AkitaError::InvalidProof)?
                        .is_zero()
                    {
                        return Err(AkitaError::InvalidInput(
                            "trinomial response coefficient padding must be zero".into(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn padded_address(
        self,
        polynomial: usize,
        digit: usize,
        coefficient: usize,
    ) -> Result<usize, AkitaError> {
        let polynomial_offset =
            polynomial
                .checked_mul(self.polynomial_stride)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("trinomial response address overflow".into())
                })?;
        let coefficient_offset = coefficient
            .checked_mul(self.coefficient_stride)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("trinomial response address overflow".into())
            })?;
        checked::sum([self.offset, polynomial_offset, coefficient_offset, digit])
            .filter(|&address| address < self.domain_len)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("trinomial response address out of range".into())
            })
    }
}

/// Tight row-major view of an `n`-by-`m` matrix of degree-`D` ring elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TrinomialASetupView {
    polynomial: RelationPolynomial,
    rows: usize,
    columns: usize,
    setup_offset: usize,
    setup_domain_len: usize,
    response: TrinomialResponseLayout,
}

impl TrinomialASetupView {
    /// Construct a checked tight setup view.
    pub fn new(
        polynomial: RelationPolynomial,
        rows: usize,
        columns: usize,
        setup_offset: usize,
        setup_domain_len: usize,
        response: TrinomialResponseLayout,
    ) -> Result<Self, AkitaError> {
        if rows == 0 || columns == 0 {
            return Err(AkitaError::InvalidSetup(
                "trinomial setup matrix dimensions must be nonzero".into(),
            ));
        }
        if !setup_domain_len.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "trinomial setup domain must be a nonzero power of two".into(),
            ));
        }
        if response.polynomial() != polynomial || response.polynomial_count() != columns {
            return Err(AkitaError::InvalidSetup(
                "trinomial response layout does not match the setup view".into(),
            ));
        }
        let setup_len = checked::product([rows, columns, polynomial.degree()])
            .ok_or_else(|| AkitaError::InvalidSetup("trinomial setup length overflow".into()))?;
        validate_work(setup_len)?;
        validate_work(setup_domain_len)?;
        let setup_end = setup_offset
            .checked_add(setup_len)
            .ok_or_else(|| AkitaError::InvalidSetup("trinomial setup extent overflow".into()))?;
        if setup_end > setup_domain_len {
            return Err(AkitaError::InvalidSetup(
                "trinomial setup extent exceeds its domain".into(),
            ));
        }
        Ok(Self {
            polynomial,
            rows,
            columns,
            setup_offset,
            setup_domain_len,
            response,
        })
    }

    #[must_use]
    pub const fn polynomial(self) -> RelationPolynomial {
        self.polynomial
    }

    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn columns(self) -> usize {
        self.columns
    }

    #[must_use]
    pub const fn setup_offset(self) -> usize {
        self.setup_offset
    }

    #[must_use]
    pub const fn setup_domain_len(self) -> usize {
        self.setup_domain_len
    }

    #[must_use]
    pub const fn response_layout(self) -> TrinomialResponseLayout {
        self.response
    }

    /// Address `A[i,j][coefficient]` in the tight setup table.
    pub fn setup_address(
        self,
        row: usize,
        column: usize,
        coefficient: usize,
    ) -> Result<usize, AkitaError> {
        if row >= self.rows || column >= self.columns || coefficient >= self.polynomial.degree() {
            return Err(AkitaError::InvalidInput(
                "trinomial setup coordinate out of range".into(),
            ));
        }
        let element = checked::mul_add(row, self.columns, column)
            .ok_or_else(|| AkitaError::InvalidSetup("trinomial setup address overflow".into()))?;
        let address = checked::mul_add(element, self.polynomial.degree(), coefficient)
            .and_then(|relative| self.setup_offset.checked_add(relative))
            .ok_or_else(|| AkitaError::InvalidSetup("trinomial setup address overflow".into()))?;
        if address >= self.setup_domain_len {
            return Err(AkitaError::InvalidSetup(
                "trinomial setup address out of range".into(),
            ));
        }
        Ok(address)
    }

    /// Prepare `gadget * eq(column_point,j) * dense_rows[i*D+s]`.
    /// The dense row factors are multiplication-adjoint contractions of the
    /// coefficient point, supplied by the relation owner. No alpha powers or
    /// unreduced polynomial weights are implied by this view.
    pub fn prepare<E: Field>(
        self,
        column_point: &[E],
        dense_rows: &[E],
        gadget: E,
    ) -> Result<PreparedTrinomialASetupWeights<E>, AkitaError> {
        let column_domain = checked::ceil_log2(self.columns)
            .and_then(checked::pow2)
            .ok_or_else(|| AkitaError::InvalidSetup("trinomial column domain overflow".into()))?;
        validate_point_domain(column_point, column_domain, "column")?;
        let degree = self.polynomial.degree();
        let row_len = checked::product([self.rows, degree])
            .ok_or_else(|| AkitaError::InvalidSetup("trinomial row factor overflow".into()))?;
        if dense_rows.len() != row_len {
            return Err(AkitaError::InvalidSize {
                expected: row_len,
                actual: dense_rows.len(),
            });
        }
        let mut column_weights = Vec::new();
        column_weights
            .try_reserve_exact(self.columns)
            .map_err(|_| {
                AkitaError::InvalidInput("trinomial column weight allocation failed".into())
            })?;
        column_weights.extend(
            (0..self.columns).map(|j| akita_algebra::offset_eq::eq_eval_at_index(column_point, j)),
        );
        let column_weights = Arc::<[E]>::from(column_weights);
        let mut setup_families = Vec::new();
        setup_families.try_reserve_exact(self.rows).map_err(|_| {
            AkitaError::InvalidInput("trinomial row weight allocation failed".into())
        })?;
        for (row, coefficients) in dense_rows.chunks_exact(degree).enumerate() {
            setup_families.push(EqPairTensorFamily::new(
                self.setup_address(row, 0, 0)?,
                0,
                gadget,
                vec![
                    EqPairTensorAxis::dense(1, 0, Arc::<[E]>::from(coefficients)),
                    EqPairTensorAxis::dense(degree, 0, Arc::clone(&column_weights)),
                ],
            )?);
        }
        let mut rows = Vec::new();
        rows.try_reserve_exact(row_len).map_err(|_| {
            AkitaError::InvalidInput("trinomial dense row allocation failed".into())
        })?;
        rows.extend_from_slice(dense_rows);
        Ok(PreparedTrinomialASetupWeights {
            view: self,
            gadget,
            dense_rows: rows,
            column_weights,
            setup_families,
        })
    }
}

/// Prepared setup weights for the remainder relation, in one canonical owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedTrinomialASetupWeights<E: Field> {
    view: TrinomialASetupView,
    gadget: E,
    dense_rows: Vec<E>,
    column_weights: Arc<[E]>,
    setup_families: Vec<EqPairTensorFamily<E>>,
}

impl<E: Field> PreparedTrinomialASetupWeights<E> {
    #[must_use]
    pub const fn view(&self) -> TrinomialASetupView {
        self.view
    }

    /// Materialize one scalar weight per setup-domain address, including zeros
    /// before and after the tight matrix extent.
    pub fn materialize_setup_weights(&self) -> Result<Vec<E>, AkitaError> {
        validate_work(self.view.setup_domain_len())?;
        let equality = OffsetEqWindow::new(&[])?;
        materialize_eq_tensor_left(
            &equality,
            &self.setup_families,
            self.view.setup_domain_len(),
        )
    }

    /// Evaluate the setup-weight MLE without scanning the matrix or allocating
    /// its dense setup-domain weight vector.
    pub fn evaluate_setup_weight_at(&self, setup_point: &[E]) -> Result<E, AkitaError> {
        validate_point_domain(setup_point, self.view.setup_domain_len(), "setup")?;
        eval_boolean_pair_tensor_families::<E, false, false>(setup_point, &[], &self.setup_families)
    }

    /// Contract a tight row-major matrix in exactly one pass. Accumulate
    /// `Abar_i = sum_j eq(column_point,j) A_ij` in `rows*D` field slots, then
    /// contract with the prepared dense row factors and digit gadget.
    /// Each matrix entry must contain exactly D coefficients.
    pub fn contract_matrix<'a>(
        &self,
        matrix: impl IntoIterator<Item = &'a [E]>,
    ) -> Result<E, AkitaError>
    where
        E: 'a,
    {
        let degree = self.view.polynomial.degree();
        let expected = checked::product([self.view.rows, self.view.columns])
            .ok_or(AkitaError::InvalidProof)?;
        let mut accumulators = Vec::new();
        accumulators
            .try_reserve_exact(self.dense_rows.len())
            .map_err(|_| {
                AkitaError::InvalidInput("trinomial contraction allocation failed".into())
            })?;
        accumulators.resize(self.dense_rows.len(), E::zero());
        let mut count = 0;
        for (entry, coefficients) in matrix.into_iter().enumerate() {
            if entry >= expected || coefficients.len() != degree {
                return Err(AkitaError::InvalidInput(
                    "trinomial matrix contraction geometry mismatch".into(),
                ));
            }
            let weight = *self
                .column_weights
                .get(entry % self.view.columns)
                .ok_or(AkitaError::InvalidProof)?;
            let start = checked::product([entry / self.view.columns, degree])
                .ok_or(AkitaError::InvalidProof)?;
            let range = checked::range(start, degree).ok_or(AkitaError::InvalidProof)?;
            for (destination, &coefficient) in accumulators
                .get_mut(range)
                .ok_or(AkitaError::InvalidProof)?
                .iter_mut()
                .zip(coefficients)
            {
                *destination += weight * coefficient;
            }
            count += 1;
        }
        if count != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: count,
            });
        }
        Ok(self.gadget
            * accumulators
                .iter()
                .zip(&self.dense_rows)
                .fold(E::zero(), |sum, (&a, &z)| sum + a * z))
    }
}

fn validate_point_domain<F: Field>(
    point: &[F],
    domain_len: usize,
    name: &'static str,
) -> Result<(), AkitaError> {
    let actual = checked::pow2(point.len())
        .ok_or_else(|| AkitaError::InvalidInput(format!("trinomial {name} point is too wide")))?;
    if actual != domain_len {
        return Err(AkitaError::InvalidSize {
            expected: domain_len.trailing_zeros() as usize,
            actual: point.len(),
        });
    }
    Ok(())
}

fn validate_work(actual: usize) -> Result<(), AkitaError> {
    if actual > MAX_COMPACT_STRIDE_TERMS {
        return Err(AkitaError::InvalidSize {
            expected: MAX_COMPACT_STRIDE_TERMS,
            actual,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::offset_eq::eq_eval_at_index;
    use jolt_field::{One, Prime128OffsetA7F7, Ring, Zero};

    type F = Prime128OffsetA7F7;

    fn point(len: usize, start: u64) -> Vec<F> {
        (0..len)
            .map(|index| F::from_u64(start + u64::try_from(index).unwrap()))
            .collect()
    }

    #[test]
    fn degree_648_weights_match_direct_coefficient_oracle() {
        let polynomial = RelationPolynomial::minus_trinomial(648).unwrap();
        let response = TrinomialResponseLayout::new(polynomial, 1024, 2, 2, 29, 8192).unwrap();
        let view = TrinomialASetupView::new(polynomial, 2, 2, 37, 4096, response).unwrap();
        assert_eq!(view.setup_address(1, 1, 647).unwrap(), 2628);
        assert_eq!(response.address(1, 1, 647).unwrap(), 3372);
        let column_point = point(1, 3);
        let setup_point = point(12, 19);
        let rows = point(1296, 7);
        let gadget = F::from_u64(17);
        let prepared = view.prepare(&column_point, &rows, gadget).unwrap();
        let mut expected = vec![F::zero(); view.setup_domain_len()];
        let matrix: Vec<Vec<F>> = (0..4).map(|j| point(648, j * 37 + 1)).collect();
        let mut direct_contraction = F::zero();
        for row in 0..2 {
            for column in 0..2 {
                for coefficient in 0..648 {
                    let address = view.setup_address(row, column, coefficient).unwrap();
                    let weight = gadget
                        * rows[row * 648 + coefficient]
                        * eq_eval_at_index(&column_point, column);
                    expected[address] = weight;
                    direct_contraction += weight * matrix[row * 2 + column][coefficient];
                }
            }
        }
        assert_eq!(prepared.materialize_setup_weights().unwrap(), expected);
        let direct_evaluation = expected
            .iter()
            .enumerate()
            .fold(F::zero(), |sum, (address, &weight)| {
                sum + weight * eq_eval_at_index(&setup_point, address)
            });
        assert_eq!(
            prepared.evaluate_setup_weight_at(&setup_point).unwrap(),
            direct_evaluation
        );
        assert_eq!(
            prepared
                .contract_matrix(matrix.iter().map(Vec::as_slice))
                .unwrap(),
            direct_contraction
        );
        assert!(prepared
            .contract_matrix(matrix[..3].iter().map(Vec::as_slice))
            .is_err());
        assert!(view.prepare(&column_point, &rows[..1295], gadget).is_err());
    }

    #[test]
    fn response_layout_rejects_padding_and_nonzero_tails() {
        let polynomial = RelationPolynomial::plus_trinomial(162).unwrap();
        assert!(TrinomialResponseLayout::new(
            RelationPolynomial::negacyclic(256).unwrap(),
            256,
            1,
            1,
            0,
            256
        )
        .is_err());
        assert!(TrinomialResponseLayout::new(polynomial, 256, 2, 3, 17, 1024).is_err());
        let response = TrinomialResponseLayout::new(polynomial, 256, 2, 3, 17, 2048).unwrap();
        assert_eq!(response.digit_stride(), 1);
        assert_eq!(response.coefficient_stride(), 3);
        assert_eq!(response.polynomial_stride(), 768);
        assert!(response.address(2, 0, 0).is_err());
        assert!(response.address(0, 3, 0).is_err());
        assert!(response.address(0, 0, 162).is_err());

        let mut table = vec![F::zero(); response.domain_len()];
        response.validate_zero_tails(&table).unwrap();
        let first_tail = response.offset() + 162 * response.coefficient_stride();
        table[first_tail] = F::one();
        assert!(response.validate_zero_tails(&table).is_err());
        assert!(response.validate_zero_tails(&table[..1024]).is_err());
    }

    #[test]
    fn setup_view_checks_matching_layout_and_point_domains() {
        let polynomial = RelationPolynomial::plus_trinomial(324).unwrap();
        let response = TrinomialResponseLayout::new(polynomial, 512, 2, 1, 0, 1024).unwrap();
        assert!(TrinomialASetupView::new(polynomial, 2, 2, 0, 1024, response).is_err());
        let view = TrinomialASetupView::new(polynomial, 1, 2, 0, 1024, response).unwrap();
        assert!(view.setup_address(1, 0, 0).is_err());
        assert!(view
            .prepare(&point(9, 1), &[F::one(); 324], F::one())
            .is_err());
        let wrong_polynomial = RelationPolynomial::minus_trinomial(324).unwrap();
        assert!(TrinomialASetupView::new(wrong_polynomial, 1, 2, 0, 1024, response).is_err());
        assert!(TrinomialASetupView::new(polynomial, 1, 2, 0, 1 << 29, response).is_err());
    }
}
