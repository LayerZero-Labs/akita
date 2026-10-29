use super::utils::{accumulate_left_round, fold_dense_left_round, fold_factor_in_place};
#[cfg(test)]
use super::utils::{accumulate_right_round, fold_left_round, fold_right_round, product_claim};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::eval_flat_ring_at_pows_fast;
use jolt_poly::UnivariatePoly;

use akita_error::{checked, AkitaError};
use akita_sumcheck::SumcheckKernel;
use jolt_field::solinas::parallel::*;
use jolt_field::{Field, MulBaseUnreduced, Ring, Zero};
use std::sync::Arc;

pub(in crate::opaque::sumcheck) enum SetupProductSource<F: Field> {
    Expanded(Arc<akita_types::AkitaExpandedSetup<F>>),
    #[cfg(test)]
    Table(Vec<F>),
}

impl<F: Field> SetupProductSource<F> {
    fn coefficients(&self) -> &[F] {
        match self {
            Self::Expanded(setup) => setup.shared_matrix().as_field_slice(),
            #[cfg(test)]
            Self::Table(table) => table,
        }
    }
}

/// One dense factored setup-product term
/// `sum_{left,right} table[left,right] * left_factor[left] * right_factor[right]`.
#[cfg(test)]
pub(in crate::opaque::sumcheck) struct FactoredProductTerm<E: Field> {
    table: Vec<E>,
    left_factor: Vec<E>,
    right_factor: Vec<E>,
    input_claim: E,
    right_rounds: usize,
    total_rounds: usize,
}

/// One coefficient contraction and its factor, folded at the common challenges.
pub(in crate::opaque::sumcheck) struct CoefficientTerm<E> {
    pub(in crate::opaque::sumcheck) table: Vec<E>,
    pub(in crate::opaque::sumcheck) factor: Vec<E>,
}

/// Index factors retained until all coefficient factors have become scalars.
pub(in crate::opaque::sumcheck) enum SetupProductWeights<E: Field> {
    Dense(Vec<E>),
    #[cfg(feature = "external-setup-accumulation")]
    Retained(Vec<akita_types::setup_claims::SetupWeightTerm<E>>),
    #[cfg(feature = "external-setup-accumulation")]
    Regenerate {
        polynomial: akita_types::setup_claims::SetupPolynomialId,
        query: akita_types::setup_claims::SetupQuery<E>,
    },
}

/// Degree-two product sumcheck over one shared, base-field setup source.
///
/// Ordinary Stage 3 supplies one term. Accumulation supplies several weighted
/// terms, sharing coefficient challenges and merging into one index product.
/// Both cases retain only contracted coefficient and index tables.
#[cfg_attr(not(feature = "external-setup-accumulation"), allow(unreachable_pub))]
pub struct SetupProductSumcheck<F: Field, E: Field> {
    setup: SetupProductSource<F>,
    row_capacity: usize,
    coefficient_len: usize,
    total_rounds: usize,
    coefficient_challenges: Vec<E>,
    terms: Vec<CoefficientTerm<E>>,
    weights: Vec<SetupProductWeights<E>>,
    index_table: Vec<E>,
    index_factor: Vec<E>,
    input_claim: E,
    claim: E,
    round: usize,
    pending: Option<UnivariatePoly<E>>,
}

impl<F, E> SetupProductSumcheck<F, E>
where
    F: Field,
    E: Field + Ring + MulBaseUnreduced<F>,
{
    pub(in crate::opaque::sumcheck) fn new(
        source: SetupProductSource<F>,
        required_rows: usize,
        index_factor: Vec<E>,
        coefficient_factor: Vec<E>,
    ) -> Result<Self, AkitaError> {
        let row_capacity = index_factor.len();
        let coefficient_len = coefficient_factor.len();
        let source_len = checked::product([row_capacity, coefficient_len])
            .ok_or_else(|| AkitaError::Internal("setup source length overflow".into()))?;
        if required_rows == 0
            || !row_capacity.is_power_of_two()
            || !coefficient_len.is_power_of_two()
            || required_rows > row_capacity
            || source.coefficients().len() < source_len
        {
            return Err(AkitaError::Internal(
                "rectangular setup-product dimensions are invalid".into(),
            ));
        }
        let table = {
            let _span = tracing::info_span!(
                "stage3_setup_coefficient_pass",
                kernel = "rectangular_base_field",
                source_pass = 1u64,
                source_rows = required_rows as u64,
                coefficient_len = coefficient_len as u64,
                base_to_extension_lifts = 0u64,
                setup_table_state_elements = (coefficient_len + row_capacity) as u64,
            )
            .entered();
            contract_setup_coefficients(
                source.coefficients(),
                coefficient_len,
                required_rows,
                |index| index_factor[index],
            )
        };
        Self::from_prepared(
            source,
            row_capacity,
            coefficient_len,
            vec![CoefficientTerm {
                table,
                factor: coefficient_factor,
            }],
            vec![SetupProductWeights::Dense(index_factor)],
        )
    }

    pub(in crate::opaque::sumcheck) fn from_prepared(
        setup: SetupProductSource<F>,
        row_capacity: usize,
        coefficient_len: usize,
        terms: Vec<CoefficientTerm<E>>,
        weights: Vec<SetupProductWeights<E>>,
    ) -> Result<Self, AkitaError> {
        let source_len =
            checked::product([row_capacity, coefficient_len]).ok_or(AkitaError::InvalidProof)?;
        if !row_capacity.is_power_of_two()
            || !coefficient_len.is_power_of_two()
            || setup.coefficients().len() < source_len
            || terms.iter().any(|term| {
                term.table.len() != coefficient_len || term.factor.len() != coefficient_len
            })
        {
            return Err(AkitaError::InvalidInput(
                "invalid prepared setup product".into(),
            ));
        }
        let input_claim = terms
            .iter()
            .flat_map(|term| term.table.iter().zip(&term.factor))
            .fold(E::zero(), |sum, (&value, &factor)| sum + value * factor);
        let mut product = Self {
            setup,
            row_capacity,
            coefficient_len,
            total_rounds: coefficient_len.trailing_zeros() as usize
                + row_capacity.trailing_zeros() as usize,
            coefficient_challenges: Vec::new(),
            terms,
            weights,
            index_table: Vec::new(),
            index_factor: Vec::new(),
            input_claim,
            claim: input_claim,
            round: 0,
            pending: None,
        };
        if coefficient_len == 1 {
            product.materialize_index_table()?;
        }
        Ok(product)
    }

    fn compute_round_univariate(&self) -> UnivariatePoly<E> {
        let (constant, linear, quadratic) = if self.index_table.is_empty() {
            self.terms
                .iter()
                .fold((E::zero(), E::zero(), E::zero()), |sum, term| {
                    let p = accumulate_left_round(&term.table, &term.factor, E::one());
                    (sum.0 + p.0, sum.1 + p.1, sum.2 + p.2)
                })
        } else {
            accumulate_left_round(&self.index_table, &self.index_factor, E::one())
        };
        UnivariatePoly::new(vec![constant, linear, quadratic])
    }

    fn ingest_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        if round < self.coefficient_len.trailing_zeros() as usize {
            self.coefficient_challenges.push(challenge);
            for term in &mut self.terms {
                fold_dense_left_round(&mut term.table, challenge);
                fold_factor_in_place(&mut term.factor, challenge);
            }
            if round + 1 == self.coefficient_len.trailing_zeros() as usize {
                self.materialize_index_table()?;
            }
        } else {
            fold_dense_left_round(&mut self.index_table, challenge);
            fold_factor_in_place(&mut self.index_factor, challenge);
        }
        Ok(())
    }

    fn materialize_index_table(&mut self) -> Result<(), AkitaError> {
        let mut combined = vec![E::zero(); self.row_capacity];
        let mut terms = self.terms.iter();
        for weights in std::mem::take(&mut self.weights) {
            match weights {
                SetupProductWeights::Dense(index) => {
                    let scale = *terms
                        .next()
                        .and_then(|term| term.factor.first())
                        .ok_or(AkitaError::InvalidProof)?;
                    if index.len() != combined.len() {
                        return Err(AkitaError::InvalidProof);
                    }
                    for (output, weight) in combined.iter_mut().zip(index) {
                        *output += scale * weight;
                    }
                }
                #[cfg(feature = "external-setup-accumulation")]
                weights => {
                    let factors = match weights {
                        SetupProductWeights::Retained(factors) => factors,
                        SetupProductWeights::Regenerate { polynomial, query } => {
                            query.compile(&polynomial, self.coefficient_len)?
                        }
                        SetupProductWeights::Dense(_) => return Err(AkitaError::InvalidProof),
                    };
                    for factor in factors {
                        let scale = *terms
                            .next()
                            .and_then(|term| term.factor.first())
                            .ok_or(AkitaError::InvalidProof)?;
                        if !scale.is_zero() {
                            factor.add_scaled_index(scale, &mut combined)?;
                        }
                    }
                }
            }
        }
        if terms.next().is_some() {
            return Err(AkitaError::InvalidProof);
        }
        self.terms.clear();
        let equality = EqPolynomial::evals(&self.coefficient_challenges)?;
        let _span = tracing::info_span!(
            "stage3_setup_index_pass",
            kernel = "rectangular_base_field",
            source_pass = 2u64,
            source_rows = self.row_capacity as u64,
            coefficient_len = self.coefficient_len as u64,
            base_to_extension_lifts = 0u64,
            setup_table_state_elements = (self.row_capacity + self.coefficient_len) as u64,
        )
        .entered();
        self.index_table = contract_setup_indices(
            self.setup.coefficients(),
            self.coefficient_len,
            self.row_capacity,
            &equality,
        );
        self.index_factor = combined;
        Ok(())
    }

    pub(in crate::opaque::sumcheck) fn folded_table_value(&self) -> Result<E, AkitaError> {
        if self.round != self.total_rounds
            || self.pending.is_some()
            || self.index_table.len() != 1
            || self.index_factor.len() != 1
        {
            return Err(AkitaError::InvalidInput("incomplete setup product".into()));
        }
        Ok(self.index_table[0])
    }
}

impl<F, E> SumcheckKernel<E> for SetupProductSumcheck<F, E>
where
    F: Field,
    E: Field + Ring + MulBaseUnreduced<F>,
{
    fn num_rounds(&self) -> usize {
        self.total_rounds
    }
    fn degree_bound(&self) -> usize {
        2
    }
    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn round_polynomial(
        &mut self,
        round: usize,
        claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError> {
        if round != self.round
            || round >= self.total_rounds
            || self.pending.is_some()
            || claim != self.claim
        {
            return Err(AkitaError::InvalidInput(
                "invalid setup product round progression".into(),
            ));
        }
        let polynomial = self.compute_round_univariate();
        self.pending = Some(polynomial.clone());
        Ok(polynomial)
    }

    fn bind_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        if round != self.round || round >= self.total_rounds {
            return Err(AkitaError::InvalidInput(
                "invalid setup product challenge progression".into(),
            ));
        }
        let polynomial = self.pending.take().ok_or_else(|| {
            AkitaError::InvalidInput("setup challenge without round polynomial".into())
        })?;
        self.claim = polynomial.evaluate(challenge);
        self.ingest_challenge(round, challenge)?;
        self.round += 1;
        Ok(())
    }

    fn finish(&mut self) -> Result<(), AkitaError> {
        self.folded_table_value().map(|_| ())
    }
}

#[cfg(test)]
impl<E: Field + Ring> FactoredProductTerm<E> {
    /// Construct a dense factored product-sumcheck term.
    ///
    /// Returns an error if factor lengths are not powers of two, are empty, or
    /// if `table.len() != left_factor.len() * right_factor.len()`.
    pub(in crate::opaque::sumcheck) fn new_dense(
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

    pub(in crate::opaque::sumcheck) const fn num_rounds(&self) -> usize {
        self.total_rounds
    }

    pub(in crate::opaque::sumcheck) const fn input_claim(&self) -> E {
        self.input_claim
    }

    pub(in crate::opaque::sumcheck) fn compute_round_univariate(
        &self,
        round: usize,
        _previous_claim: E,
    ) -> UnivariatePoly<E> {
        let (constant, linear, quadratic) = if round < self.right_rounds {
            accumulate_right_round(&self.table, &self.left_factor, &self.right_factor)
        } else {
            accumulate_left_round(&self.table, &self.left_factor, self.right_factor[0])
        };
        UnivariatePoly::new(vec![constant, linear, quadratic])
    }

    pub(in crate::opaque::sumcheck) fn ingest_challenge(&mut self, round: usize, challenge: E) {
        if round < self.right_rounds {
            fold_right_round(&mut self.table, &mut self.right_factor, challenge);
        } else {
            fold_left_round(&mut self.table, &mut self.left_factor, challenge);
        }
    }

    pub(in crate::opaque::sumcheck) fn folded_table_value(&self) -> Result<E, AkitaError> {
        if self.table.len() != 1 {
            return Err(AkitaError::InvalidSize {
                expected: 1,
                actual: self.table.len(),
            });
        }
        Ok(self.table[0])
    }
}

/// First rectangular pass, preserving base-by-extension products and delayed reduction.
pub(in crate::opaque::sumcheck) fn contract_setup_coefficients<F, E>(
    setup: &[F],
    coefficient_len: usize,
    required_rows: usize,
    index_weight: impl Fn(usize) -> E + Sync,
) -> Vec<E>
where
    F: Field,
    E: Field + Ring + MulBaseUnreduced<F>,
{
    let accumulators = cfg_fold_reduce!(
        0..required_rows,
        || (0..coefficient_len)
            .map(|_| { <E as jolt_field::Unreduced>::Product::zero() })
            .collect::<Vec<_>>(),
        |mut accumulators, setup_index| {
            let row_start = setup_index * coefficient_len;
            let factor = index_weight(setup_index);
            if factor.is_zero() {
                return accumulators;
            }
            for (accumulator, &coefficient) in accumulators
                .iter_mut()
                .zip(&setup[row_start..row_start + coefficient_len])
            {
                *accumulator += factor.mul_base_unreduced(coefficient);
            }
            accumulators
        },
        |mut left, right| {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
            left
        }
    );
    accumulators
        .into_iter()
        .map(E::reduce_product)
        .collect::<Vec<_>>()
}

/// Shared second pass: read the actual full prefix once at the common coefficient point.
pub(in crate::opaque::sumcheck) fn contract_setup_indices<F, E>(
    setup: &[F],
    coefficient_len: usize,
    rows: usize,
    coefficient_eq: &[E],
) -> Vec<E>
where
    F: Field,
    E: Field + Ring + MulBaseUnreduced<F>,
{
    cfg_into_iter!(0..rows)
        .map(|index| {
            let start = index * coefficient_len;
            eval_flat_ring_at_pows_fast(&setup[start..start + coefficient_len], coefficient_eq)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::ring::scalar_powers;
    use jolt_field::Prime128Offset275 as F;

    fn scalar(value: u64) -> F {
        F::from_u64(value)
    }

    fn setup_source(rows: usize, coefficient_len: usize) -> Vec<F> {
        (0..rows * coefficient_len)
            .map(|index| scalar(((index * 17 + index / coefficient_len * 5) % 251 + 1) as u64))
            .collect()
    }

    fn dense_term(
        setup: &[F],
        _required_rows: usize,
        row_capacity: usize,
        coefficient_len: usize,
        index_factor: Vec<F>,
        coefficient_factor: Vec<F>,
    ) -> FactoredProductTerm<F> {
        let mut table = vec![F::zero(); row_capacity * coefficient_len];
        table.copy_from_slice(&setup[..row_capacity * coefficient_len]);
        FactoredProductTerm::new_dense(table, index_factor, coefficient_factor)
            .expect("dense setup product")
    }

    fn assert_round_parity(required_rows: usize, row_capacity: usize, coefficient_len: usize) {
        let setup = setup_source(row_capacity, coefficient_len);
        let mut index_factor = (0..row_capacity)
            .map(|index| scalar((index * 13 + 3) as u64))
            .collect::<Vec<_>>();
        index_factor[required_rows..].fill(F::zero());
        let coefficient_factor = scalar_powers(scalar(7), coefficient_len).to_vec();
        let mut dense = dense_term(
            &setup,
            required_rows,
            row_capacity,
            coefficient_len,
            index_factor.clone(),
            coefficient_factor.clone(),
        );
        let mut rectangular = SetupProductSumcheck::new(
            SetupProductSource::Table(setup),
            required_rows,
            index_factor,
            coefficient_factor,
        )
        .expect("rectangular setup product");
        assert_eq!(dense.input_claim(), rectangular.input_claim());
        assert_eq!(dense.num_rounds(), rectangular.num_rounds());

        let mut claim = rectangular.input_claim();
        for round in 0..dense.num_rounds() {
            let dense_poly = dense.compute_round_univariate(round, dense.input_claim());
            let rectangular_poly = rectangular.round_polynomial(round, claim).unwrap();
            assert_eq!(dense_poly, rectangular_poly, "round {round}");
            let challenge = scalar((round * 19 + 11) as u64);
            dense.ingest_challenge(round, challenge);
            rectangular.bind_challenge(round, challenge).unwrap();
            claim = rectangular_poly.evaluate(challenge);
        }
        assert_eq!(
            dense.folded_table_value().expect("dense folded setup"),
            rectangular
                .folded_table_value()
                .expect("rectangular folded setup")
        );
    }

    #[test]
    fn rectangular_setup_product_matches_dense_rounds_with_padding() {
        assert_round_parity(5, 8, 8);
    }

    #[test]
    fn rectangular_setup_product_matches_dense_rounds_without_padding() {
        assert_round_parity(16, 16, 64);
    }

    #[test]
    fn rectangular_setup_product_materializes_index_state_without_coefficient_rounds() {
        assert_round_parity(3, 4, 1);
        assert_round_parity(1, 1, 1);
    }
}
