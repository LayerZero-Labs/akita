//! Eq-weighted multilinear table sumcheck instances.
//!
//! These instances prove claims of the form:
//!
//! ```text
//! scale * sum_z eq(target, z) * table(z)
//! ```
//!
//! They are the protocol-independent core needed by setup-side claim reduction:
//! the verifier can reduce a weighted setup-table claim to a final point claim
//! on the committed setup polynomial.

use crate::{SumcheckInstanceProver, SumcheckInstanceVerifier, UniPoly};
use akita_algebra::poly::{fold_evals_in_place, multilinear_eval};
use akita_field::{AkitaError, FieldCore};

/// Prover instance for `scale * sum_z eq(target, z) * table(z)`.
pub struct EqWeightedTableProver<E: FieldCore> {
    table: Vec<E>,
    weights: Vec<E>,
    input_claim: E,
    scale: E,
    num_rounds: usize,
}

impl<E: FieldCore> EqWeightedTableProver<E> {
    /// Construct a prover from table evaluations and the eq target point.
    ///
    /// # Errors
    ///
    /// Returns an error if `table` does not have length `2^target_point.len()`.
    pub fn new(table: Vec<E>, target_point: &[E], scale: E) -> Result<Self, AkitaError> {
        validate_table_shape(table.len(), target_point.len())?;
        let weights = eq_table(target_point);
        let input_claim = table
            .iter()
            .zip(weights.iter())
            .fold(E::zero(), |acc, (&value, &weight)| {
                acc + scale * value * weight
            });
        Ok(Self {
            table,
            weights,
            input_claim,
            scale,
            num_rounds: target_point.len(),
        })
    }

    /// Current folded table value after all rounds have been ingested.
    ///
    /// # Panics
    ///
    /// Panics if called before the instance is fully folded.
    pub fn final_table_eval(&self) -> E {
        assert_eq!(self.table.len(), 1, "table is not fully folded");
        self.table[0]
    }
}

impl<E: FieldCore> SumcheckInstanceProver<E> for EqWeightedTableProver<E> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        let mut poly = product_round_poly(&self.table, &self.weights);
        for coeff in &mut poly.coeffs {
            *coeff *= self.scale;
        }
        poly
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: E) {
        fold_evals_in_place(&mut self.table, r_round);
        fold_evals_in_place(&mut self.weights, r_round);
    }
}

/// Prover instance for `sum_z weight(z) * table(z)`.
pub struct WeightedTableProver<E: FieldCore> {
    table: Vec<E>,
    weights: Vec<E>,
    input_claim: E,
    num_rounds: usize,
}

impl<E: FieldCore> WeightedTableProver<E> {
    /// Construct a prover from table and weight evaluations over the same
    /// Boolean hypercube.
    ///
    /// # Errors
    ///
    /// Returns an error if the tables have different non-power-of-two lengths.
    pub fn new(table: Vec<E>, weights: Vec<E>) -> Result<Self, AkitaError> {
        validate_matching_tables(table.len(), weights.len())?;
        let input_claim = table
            .iter()
            .zip(weights.iter())
            .fold(E::zero(), |acc, (&value, &weight)| acc + value * weight);
        let num_rounds = table.len().trailing_zeros() as usize;
        Ok(Self {
            table,
            weights,
            input_claim,
            num_rounds,
        })
    }
}

impl<E: FieldCore> SumcheckInstanceProver<E> for WeightedTableProver<E> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        product_round_poly(&self.table, &self.weights)
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: E) {
        fold_evals_in_place(&mut self.table, r_round);
        fold_evals_in_place(&mut self.weights, r_round);
    }
}

/// Verifier instance for `sum_z weight(z) * table(z)`.
pub struct WeightedTableVerifier<E: FieldCore> {
    table: Vec<E>,
    weights: Vec<E>,
    input_claim: E,
}

impl<E: FieldCore> WeightedTableVerifier<E> {
    /// Construct a verifier from table and weight evaluations.
    ///
    /// # Errors
    ///
    /// Returns an error if the tables have different non-power-of-two lengths.
    pub fn new(table: Vec<E>, weights: Vec<E>, input_claim: E) -> Result<Self, AkitaError> {
        validate_matching_tables(table.len(), weights.len())?;
        Ok(Self {
            table,
            weights,
            input_claim,
        })
    }
}

impl<E: FieldCore> SumcheckInstanceVerifier<E> for WeightedTableVerifier<E> {
    fn num_rounds(&self) -> usize {
        self.table.len().trailing_zeros() as usize
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn expected_output_claim(&self, challenges: &[E]) -> Result<E, AkitaError> {
        Ok(multilinear_eval(&self.table, challenges)?
            * multilinear_eval(&self.weights, challenges)?)
    }
}

/// Verifier instance for `scale * sum_z eq(target, z) * table(z)`.
pub struct EqWeightedTableVerifier<E: FieldCore> {
    table: Vec<E>,
    target_point: Vec<E>,
    input_claim: E,
    scale: E,
}

impl<E: FieldCore> EqWeightedTableVerifier<E> {
    /// Construct a verifier from table evaluations, target point, and claim.
    ///
    /// # Errors
    ///
    /// Returns an error if `table` does not have length `2^target_point.len()`.
    pub fn new(
        table: Vec<E>,
        target_point: Vec<E>,
        input_claim: E,
        scale: E,
    ) -> Result<Self, AkitaError> {
        validate_table_shape(table.len(), target_point.len())?;
        Ok(Self {
            table,
            target_point,
            input_claim,
            scale,
        })
    }
}

impl<E: FieldCore> SumcheckInstanceVerifier<E> for EqWeightedTableVerifier<E> {
    fn num_rounds(&self) -> usize {
        self.target_point.len()
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn expected_output_claim(&self, challenges: &[E]) -> Result<E, AkitaError> {
        let setup_eval = multilinear_eval(&self.table, challenges)?;
        let weight_eval = eq_eval(&self.target_point, challenges)?;
        Ok(self.scale * setup_eval * weight_eval)
    }
}

/// Evaluate `eq(target, point)`.
///
/// # Errors
///
/// Returns an error if the points have different lengths.
fn eq_eval<E: FieldCore>(target: &[E], point: &[E]) -> Result<E, AkitaError> {
    if target.len() != point.len() {
        return Err(AkitaError::InvalidSize {
            expected: target.len(),
            actual: point.len(),
        });
    }
    Ok(target
        .iter()
        .zip(point.iter())
        .fold(E::one(), |acc, (&target_i, &point_i)| {
            acc * ((E::one() - target_i) * (E::one() - point_i) + target_i * point_i)
        }))
}

fn validate_table_shape(table_len: usize, num_vars: usize) -> Result<(), AkitaError> {
    let expected = 1usize
        .checked_shl(num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("eq-weighted table too large".to_string()))?;
    if table_len != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: table_len,
        });
    }
    Ok(())
}

fn validate_matching_tables(table_len: usize, weight_len: usize) -> Result<(), AkitaError> {
    if table_len != weight_len || !table_len.is_power_of_two() {
        return Err(AkitaError::InvalidSize {
            expected: table_len.next_power_of_two(),
            actual: weight_len,
        });
    }
    Ok(())
}

fn product_round_poly<E: FieldCore>(table: &[E], weights: &[E]) -> UniPoly<E> {
    debug_assert_eq!(table.len(), weights.len());
    debug_assert!(table.len().is_power_of_two());
    debug_assert!(table.len() >= 2);

    let mut coeffs = [E::zero(); 3];
    for (table_pair, weight_pair) in table.chunks_exact(2).zip(weights.chunks_exact(2)) {
        let value_0 = table_pair[0];
        let value_delta = table_pair[1] - value_0;
        let weight_0 = weight_pair[0];
        let weight_delta = weight_pair[1] - weight_0;
        coeffs[0] += value_0 * weight_0;
        coeffs[1] += value_0 * weight_delta + value_delta * weight_0;
        coeffs[2] += value_delta * weight_delta;
    }
    UniPoly::from_coeffs(coeffs.to_vec())
}

fn eq_table<E: FieldCore>(point: &[E]) -> Vec<E> {
    let len = 1usize << point.len();
    (0..len)
        .map(|idx| {
            point
                .iter()
                .enumerate()
                .fold(E::one(), |acc, (bit_idx, &point_i)| {
                    if (idx >> bit_idx) & 1 == 1 {
                        acc * point_i
                    } else {
                        acc * (E::one() - point_i)
                    }
                })
        })
        .collect()
}
