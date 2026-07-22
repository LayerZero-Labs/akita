use super::utils::{accumulate_left_round, fold_dense_left_round, fold_factor_in_place};
#[cfg(test)]
use super::utils::{accumulate_right_round, fold_left_round, fold_right_round, product_claim};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::eval_flat_ring_at_pows_fast;
use akita_algebra::uni_poly::UniPoly;
use akita_field::parallel::*;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, MulBaseUnreduced, Zero};

/// One dense factored setup-product term
/// `sum_{left,right} table[left,right] * left_factor[left] * right_factor[right]`.
#[cfg(test)]
pub(super) struct FactoredProductTerm<E: FieldCore> {
    table: Vec<E>,
    left_factor: Vec<E>,
    right_factor: Vec<E>,
    input_claim: E,
    right_rounds: usize,
    total_rounds: usize,
}

/// Two-pass setup-product term over the canonical flat setup layout.
///
/// The setup source stays in the base field with row-major layout
/// `setup[setup_index * coefficient_len + coefficient]`. Akita's committed
/// setup MLE binds coefficient variables first, so this term first proves the
/// common-coefficient rounds and then the setup-index rounds. This preserves
/// the Stage-3 suffix-opening projection while storing only one contracted
/// coefficient vector and one contracted setup-index vector in the extension
/// field.
pub(super) struct RectangularSetupProductTerm<'a, F: FieldCore, E: FieldCore> {
    setup: &'a [F],
    required_rows: usize,
    row_capacity: usize,
    coefficient_len: usize,
    coefficient_rounds: usize,
    total_rounds: usize,
    coefficient_challenges: Vec<E>,
    coefficient_table: Vec<E>,
    coefficient_factor: Vec<E>,
    index_table: Option<Vec<E>>,
    index_factor: Vec<E>,
    input_claim: E,
}

impl<'a, F, E> RectangularSetupProductTerm<'a, F, E>
where
    F: FieldCore,
    E: FieldCore + FromPrimitiveInt + MulBaseUnreduced<F>,
{
    pub(super) fn new(
        setup: &'a [F],
        required_rows: usize,
        index_factor: Vec<E>,
        coefficient_factor: Vec<E>,
    ) -> Result<Self, AkitaError> {
        if required_rows == 0
            || index_factor.is_empty()
            || coefficient_factor.is_empty()
            || !index_factor.len().is_power_of_two()
            || !coefficient_factor.len().is_power_of_two()
            || required_rows > index_factor.len()
        {
            return Err(AkitaError::InvalidInput(
                "rectangular setup-product dimensions are invalid".into(),
            ));
        }
        let required_source_len = required_rows
            .checked_mul(coefficient_factor.len())
            .ok_or_else(|| AkitaError::InvalidSetup("setup source length overflow".into()))?;
        if setup.len() < required_source_len {
            return Err(AkitaError::InvalidSize {
                expected: required_source_len,
                actual: setup.len(),
            });
        }

        let coefficient_len = coefficient_factor.len();
        let coefficient_table = {
            let _span = tracing::info_span!(
                "stage3_setup_coefficient_pass",
                kernel = "rectangular_base_field",
                source_pass = 1u64,
                source_rows = required_rows as u64,
                coefficient_len = coefficient_len as u64,
                base_to_extension_lifts = 0u64,
                setup_table_state_elements = (coefficient_len + index_factor.len()) as u64,
            )
            .entered();
            let accumulators = cfg_fold_reduce!(
                0..required_rows,
                || (0..coefficient_len)
                    .map(|_| {
                        <E as akita_field::unreduced::HasUnreducedOps>::ProductAccum::zero()
                    })
                    .collect::<Vec<_>>(),
                |mut accumulators, setup_index| {
                    let row_start = setup_index * coefficient_len;
                    let factor = index_factor[setup_index];
                    for (accumulator, &coefficient) in accumulators
                        .iter_mut()
                        .zip(&setup[row_start..row_start + coefficient_len])
                    {
                        *accumulator += factor.mul_base_to_product_accum(coefficient);
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
                .map(E::reduce_product_accum)
                .collect::<Vec<_>>()
        };
        let input_claim = coefficient_table
            .iter()
            .zip(&coefficient_factor)
            .fold(E::zero(), |acc, (&value, &factor)| acc + value * factor);
        let coefficient_rounds = coefficient_len.trailing_zeros() as usize;
        let total_rounds = coefficient_rounds + index_factor.len().trailing_zeros() as usize;

        Ok(Self {
            setup,
            required_rows,
            row_capacity: index_factor.len(),
            coefficient_len,
            coefficient_rounds,
            total_rounds,
            coefficient_challenges: Vec::with_capacity(coefficient_rounds),
            coefficient_table,
            coefficient_factor,
            index_table: None,
            index_factor,
            input_claim,
        })
    }

    pub(super) const fn num_rounds(&self) -> usize {
        self.total_rounds
    }

    pub(super) const fn input_claim(&self) -> E {
        self.input_claim
    }

    pub(super) fn compute_round_univariate(&self, round: usize) -> UniPoly<E> {
        let (constant, linear, quadratic) = if round < self.coefficient_rounds {
            accumulate_left_round(&self.coefficient_table, &self.coefficient_factor, E::one())
        } else {
            accumulate_left_round(
                self.index_table
                    .as_deref()
                    .expect("setup index table exists after coefficient rounds"),
                &self.index_factor,
                self.coefficient_factor[0],
            )
        };
        UniPoly::from_coeffs(vec![constant, linear, quadratic])
    }

    pub(super) fn ingest_challenge(&mut self, round: usize, challenge: E) {
        if round < self.coefficient_rounds {
            self.coefficient_challenges.push(challenge);
            fold_dense_left_round(&mut self.coefficient_table, challenge);
            fold_factor_in_place(&mut self.coefficient_factor, challenge);
            if round + 1 == self.coefficient_rounds {
                self.materialize_index_table();
            }
        } else {
            fold_dense_left_round(
                self.index_table
                    .as_mut()
                    .expect("setup index table exists after coefficient rounds"),
                challenge,
            );
            fold_factor_in_place(&mut self.index_factor, challenge);
        }
    }

    fn materialize_index_table(&mut self) {
        let coefficient_eq = EqPolynomial::evals(&self.coefficient_challenges)
            .expect("validated power-of-two setup coefficient domain");
        debug_assert_eq!(coefficient_eq.len(), self.coefficient_len);
        let _span = tracing::info_span!(
            "stage3_setup_index_pass",
            kernel = "rectangular_base_field",
            source_pass = 2u64,
            source_rows = self.required_rows as u64,
            coefficient_len = self.coefficient_len as u64,
            base_to_extension_lifts = 0u64,
            setup_table_state_elements = (self.row_capacity + self.coefficient_len) as u64,
        )
        .entered();
        let mut index_table = cfg_into_iter!(0..self.required_rows)
            .map(|setup_index| {
                let start = setup_index * self.coefficient_len;
                eval_flat_ring_at_pows_fast(
                    &self.setup[start..start + self.coefficient_len],
                    &coefficient_eq,
                )
            })
            .collect::<Vec<_>>();
        index_table.resize(self.row_capacity, E::zero());
        self.index_table = Some(index_table);
    }

    pub(super) fn folded_table_value(&self) -> Result<E, AkitaError> {
        let table = self
            .index_table
            .as_deref()
            .ok_or(AkitaError::InvalidProof)?;
        if table.len() != 1 {
            return Err(AkitaError::InvalidSize {
                expected: 1,
                actual: table.len(),
            });
        }
        Ok(table[0])
    }
}

#[cfg(test)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::ring::scalar_powers;
    use akita_field::Prime128Offset275 as F;
    use std::time::{Duration, Instant};

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
        required_rows: usize,
        row_capacity: usize,
        coefficient_len: usize,
        index_factor: Vec<F>,
        coefficient_factor: Vec<F>,
    ) -> FactoredProductTerm<F> {
        let mut table = vec![F::zero(); row_capacity * coefficient_len];
        table[..required_rows * coefficient_len]
            .copy_from_slice(&setup[..required_rows * coefficient_len]);
        FactoredProductTerm::new_dense(table, index_factor, coefficient_factor)
            .expect("dense setup product")
    }

    fn assert_round_parity(required_rows: usize, row_capacity: usize, coefficient_len: usize) {
        let setup = setup_source(required_rows, coefficient_len);
        let index_factor = (0..row_capacity)
            .map(|index| scalar((index * 13 + 3) as u64))
            .collect::<Vec<_>>();
        let coefficient_factor = scalar_powers(scalar(7), coefficient_len).to_vec();
        let mut dense = dense_term(
            &setup,
            required_rows,
            row_capacity,
            coefficient_len,
            index_factor.clone(),
            coefficient_factor.clone(),
        );
        let mut rectangular = RectangularSetupProductTerm::new(
            &setup,
            required_rows,
            index_factor,
            coefficient_factor,
        )
        .expect("rectangular setup product");
        assert_eq!(dense.input_claim(), rectangular.input_claim());
        assert_eq!(dense.num_rounds(), rectangular.num_rounds());

        for round in 0..dense.num_rounds() {
            let dense_poly = dense.compute_round_univariate(round, dense.input_claim());
            let rectangular_poly = rectangular.compute_round_univariate(round);
            assert_eq!(dense_poly, rectangular_poly, "round {round}");
            let challenge = scalar((round * 19 + 11) as u64);
            dense.ingest_challenge(round, challenge);
            rectangular.ingest_challenge(round, challenge);
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

    fn elapsed(mut run: impl FnMut(), samples: usize) -> Duration {
        let start = Instant::now();
        for _ in 0..samples {
            run();
        }
        start.elapsed()
    }

    /// Explicit microbenchmark for the old dense setup table (S0) and the
    /// canonical two-pass rectangular prover (S2).
    ///
    /// `cargo test -p akita-prover --release stage3_setup_product_ab -- --ignored --nocapture`
    #[test]
    #[ignore = "release-only Stage 3 setup-product A/B benchmark"]
    fn stage3_setup_product_ab() {
        const REQUIRED_ROWS: usize = 1 << 15;
        const ROW_CAPACITY: usize = 1 << 15;
        const COEFFICIENT_LEN: usize = 64;
        const SAMPLES: usize = 5;

        let setup = setup_source(REQUIRED_ROWS, COEFFICIENT_LEN);
        let index_factor = (0..ROW_CAPACITY)
            .map(|index| scalar((index * 13 + 3) as u64))
            .collect::<Vec<_>>();
        let coefficient_factor = scalar_powers(scalar(7), COEFFICIENT_LEN).to_vec();
        let challenges = (0..(ROW_CAPACITY * COEFFICIENT_LEN).trailing_zeros() as usize)
            .map(|round| scalar((round * 19 + 11) as u64))
            .collect::<Vec<_>>();

        let dense_elapsed = elapsed(
            || {
                let mut term = dense_term(
                    &setup,
                    REQUIRED_ROWS,
                    ROW_CAPACITY,
                    COEFFICIENT_LEN,
                    index_factor.clone(),
                    coefficient_factor.clone(),
                );
                for (round, &challenge) in challenges.iter().enumerate() {
                    let _ = term.compute_round_univariate(round, term.input_claim());
                    term.ingest_challenge(round, challenge);
                }
                std::hint::black_box(term.folded_table_value().expect("dense folded setup"));
            },
            SAMPLES,
        );
        let rectangular_elapsed = elapsed(
            || {
                let mut term = RectangularSetupProductTerm::new(
                    &setup,
                    REQUIRED_ROWS,
                    index_factor.clone(),
                    coefficient_factor.clone(),
                )
                .expect("rectangular setup product");
                for (round, &challenge) in challenges.iter().enumerate() {
                    let _ = term.compute_round_univariate(round);
                    term.ingest_challenge(round, challenge);
                }
                std::hint::black_box(term.folded_table_value().expect("rectangular folded setup"));
            },
            SAMPLES,
        );

        eprintln!(
            "stage3 setup product A/B: S0_dense={dense_elapsed:?}, S2_rectangular={rectangular_elapsed:?}, ratio={:.4}, dense_table_state_elements={}, rectangular_table_state_elements={}",
            rectangular_elapsed.as_secs_f64() / dense_elapsed.as_secs_f64(),
            ROW_CAPACITY * COEFFICIENT_LEN,
            ROW_CAPACITY + COEFFICIENT_LEN,
        );
    }
}
