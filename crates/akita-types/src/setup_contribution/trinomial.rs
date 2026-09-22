//! Checked setup-table views for a trinomial relation bridge.
//!
//! The view keeps natural polynomial degree separate from padded response-table
//! strides. It prepares one canonical paired-tensor description that drives
//! both prover-side weight materialization and verifier-side MLE evaluation.

use std::sync::Arc;

use akita_algebra::{
    fft::field_pow,
    offset_eq::{
        eval_boolean_pair_tensor_families, materialize_eq_tensor_left, EqPairTensorAxis,
        EqPairTensorFamily, OffsetEqWindow, MAX_COMPACT_STRIDE_TERMS,
    },
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

    /// Contract the response point and prepare the one canonical family set
    /// used for both dense setup weights and verifier-side MLE evaluation.
    pub fn prepare<E: Field>(
        self,
        response_point: &[E],
        alpha: E,
        row_weights: &[E],
        gadget: &[E],
    ) -> Result<PreparedTrinomialASetupWeights<E>, AkitaError> {
        validate_point_domain(response_point, self.response.domain_len(), "response")?;
        if row_weights.len() != self.rows {
            return Err(AkitaError::InvalidSize {
                expected: self.rows,
                actual: row_weights.len(),
            });
        }
        if gadget.len() != self.response.digit_depth() {
            return Err(AkitaError::InvalidSize {
                expected: self.response.digit_depth(),
                actual: gadget.len(),
            });
        }

        let gadget = Arc::<[E]>::from(gadget);
        let column_weights = (0..self.columns)
            .map(|column| {
                let response_families =
                    response_tensor_families(self.response, column, alpha, Arc::clone(&gadget))?;
                eval_boolean_pair_tensor_families::<E, false, false>(
                    &[],
                    response_point,
                    &response_families,
                )
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let setup_families = setup_tensor_families(self, alpha, row_weights, &column_weights)?;
        let modulus_evaluation = self.polynomial.evaluate_modulus_at(alpha)?;
        Ok(PreparedTrinomialASetupWeights {
            view: self,
            modulus_evaluation,
            setup_families,
        })
    }
}

/// Prepared setup weights for a trinomial relation row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedTrinomialASetupWeights<E: Field> {
    view: TrinomialASetupView,
    modulus_evaluation: E,
    setup_families: Vec<EqPairTensorFamily<E>>,
}

impl<E: Field> PreparedTrinomialASetupWeights<E> {
    #[must_use]
    pub const fn view(&self) -> TrinomialASetupView {
        self.view
    }

    /// `Phi(alpha)` for the exact polynomial named by this view.
    #[must_use]
    pub const fn modulus_evaluation(&self) -> E {
        self.modulus_evaluation
    }

    /// Materialize one scalar weight per setup-domain address.
    pub fn materialize_setup_weights(&self) -> Result<Vec<E>, AkitaError> {
        validate_work(self.view.setup_domain_len())?;
        let equality = OffsetEqWindow::new(&[])?;
        materialize_eq_tensor_left(
            &equality,
            &self.setup_families,
            self.view.setup_domain_len(),
        )
    }

    /// Evaluate the setup-weight multilinear polynomial without scanning the
    /// setup matrix or materializing its dense weight vector.
    pub fn evaluate_setup_weight_at(&self, setup_point: &[E]) -> Result<E, AkitaError> {
        validate_point_domain(setup_point, self.view.setup_domain_len(), "setup")?;
        eval_boolean_pair_tensor_families::<E, false, false>(setup_point, &[], &self.setup_families)
    }
}

fn response_tensor_families<E: Field>(
    layout: TrinomialResponseLayout,
    column: usize,
    alpha: E,
    gadget: Arc<[E]>,
) -> Result<Vec<EqPairTensorFamily<E>>, AkitaError> {
    let coefficient_families = alpha_dyadic_axes(
        alpha,
        layout.polynomial().degree(),
        0,
        layout.coefficient_stride(),
    )?;
    coefficient_families
        .into_iter()
        .map(|(start, scalar, coefficient_axis)| {
            EqPairTensorFamily::new(
                0,
                layout.address(column, 0, start)?,
                scalar,
                vec![
                    coefficient_axis,
                    EqPairTensorAxis::dense(0, layout.digit_stride(), Arc::clone(&gadget)),
                ],
            )
        })
        .collect()
}

fn setup_tensor_families<E: Field>(
    view: TrinomialASetupView,
    alpha: E,
    row_weights: &[E],
    column_weights: &[E],
) -> Result<Vec<EqPairTensorFamily<E>>, AkitaError> {
    let degree = view.polynomial().degree();
    let row_stride = view
        .columns()
        .checked_mul(degree)
        .ok_or_else(|| AkitaError::InvalidSetup("trinomial setup row stride overflow".into()))?;
    let row_weights = Arc::<[E]>::from(row_weights);
    let column_weights = Arc::<[E]>::from(column_weights);
    alpha_dyadic_axes(alpha, degree, 1, 0)?
        .into_iter()
        .map(|(start, scalar, coefficient_axis)| {
            EqPairTensorFamily::new(
                view.setup_address(0, 0, start)?,
                0,
                scalar,
                vec![
                    coefficient_axis,
                    EqPairTensorAxis::dense(degree, 0, Arc::clone(&column_weights)),
                    EqPairTensorAxis::dense(row_stride, 0, Arc::clone(&row_weights)),
                ],
            )
        })
        .collect()
}

fn alpha_dyadic_axes<E: Field>(
    alpha: E,
    len: usize,
    left_stride: usize,
    right_stride: usize,
) -> Result<Vec<(usize, E, EqPairTensorAxis<E>)>, AkitaError> {
    let mut factors = Vec::new();
    let mut power = alpha;
    let max_bits = usize::BITS as usize - len.leading_zeros() as usize;
    for _ in 0..max_bits {
        factors.push([E::one(), power]);
        power *= power;
    }

    let mut start = 0usize;
    let mut remaining = len;
    let mut output = Vec::with_capacity(len.count_ones() as usize);
    while remaining != 0 {
        let log_len = usize::BITS as usize - 1 - remaining.leading_zeros() as usize;
        let block_len = 1usize
            .checked_shl(log_len as u32)
            .ok_or_else(|| AkitaError::InvalidSetup("trinomial dyadic length overflow".into()))?;
        let exponent = u64::try_from(start)
            .map_err(|_| AkitaError::InvalidSetup("trinomial degree exceeds u64".into()))?;
        output.push((
            start,
            field_pow(alpha, exponent),
            EqPairTensorAxis::bit_product(
                left_stride,
                right_stride,
                Arc::<[[E; 2]]>::from(&factors[..log_len]),
            )?,
        ));
        debug_assert_eq!(output.last().map(|(_, _, axis)| axis.len), Some(block_len));
        start = start
            .checked_add(block_len)
            .ok_or_else(|| AkitaError::InvalidSetup("trinomial dyadic offset overflow".into()))?;
        remaining -= block_len;
    }
    Ok(output)
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

        let response_point = point(13, 3);
        let setup_point = point(12, 19);
        let alpha = F::from_u64(7);
        let row_weights = [F::from_u64(11), F::from_u64(13)];
        let gadget = [F::from_u64(5), F::from_u64(17)];
        let prepared = view
            .prepare(&response_point, alpha, &row_weights, &gadget)
            .unwrap();

        let mut alpha_powers = Vec::with_capacity(polynomial.degree());
        let mut power = F::one();
        for _ in 0..polynomial.degree() {
            alpha_powers.push(power);
            power *= alpha;
        }
        let column_weights = (0..view.columns())
            .map(|column| {
                let mut weight = F::zero();
                for (coefficient, &coefficient_weight) in alpha_powers.iter().enumerate() {
                    for (digit, &digit_weight) in gadget.iter().enumerate() {
                        let address = response.address(column, digit, coefficient).unwrap();
                        weight += coefficient_weight
                            * digit_weight
                            * eq_eval_at_index(&response_point, address);
                    }
                }
                weight
            })
            .collect::<Vec<_>>();

        let materialized = prepared.materialize_setup_weights().unwrap();
        let mut expected = vec![F::zero(); view.setup_domain_len()];
        for (row, &row_weight) in row_weights.iter().enumerate() {
            for (column, &column_weight) in column_weights.iter().enumerate() {
                for (coefficient, &coefficient_weight) in alpha_powers.iter().enumerate() {
                    let address = view.setup_address(row, column, coefficient).unwrap();
                    expected[address] = row_weight * column_weight * coefficient_weight;
                }
            }
        }
        assert_eq!(materialized, expected);
        assert!(materialized[..view.setup_offset()].iter().all(F::is_zero));
        assert!(materialized[2629..].iter().all(F::is_zero));

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
            prepared.modulus_evaluation(),
            polynomial.evaluate_modulus_at(alpha).unwrap()
        );
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
            .prepare(&point(9, 1), F::from_u64(3), &[F::one()], &[F::one()])
            .is_err());
        let wrong_polynomial = RelationPolynomial::minus_trinomial(324).unwrap();
        assert!(TrinomialASetupView::new(wrong_polynomial, 1, 2, 0, 1024, response).is_err());
        assert!(TrinomialASetupView::new(polynomial, 1, 2, 0, 1 << 29, response).is_err());
    }
}
