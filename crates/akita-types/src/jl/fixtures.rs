//! Shared JL consistency harness helpers for downstream crate tests.

use akita_algebra::EqPolynomial;
use akita_field::{AkitaError, CanonicalField, FieldCore};

use super::{embed_signed_i32, field_modulus};

/// Embed a balanced witness digit for sumcheck table tests.
pub fn signed_witness_digit<F: FieldCore + CanonicalField>(value: i32) -> F {
    embed_signed_i32::<F>(value, field_modulus::<F>() / 2).expect("test digit in signed window")
}

/// Map witness digits to embedded field evals for padded consistency tables.
pub fn witness_evals_from_digits<F: FieldCore + CanonicalField>(digits: &[i32]) -> Vec<F> {
    digits.iter().copied().map(signed_witness_digit).collect()
}

/// Evaluate a padded witness table at a multilinear point via `eq(·)`.
pub fn eval_padded_table_at<F: FieldCore>(table: &[F], point: &[F]) -> Result<F, AkitaError> {
    let eq = EqPolynomial::evals(point)?;
    if eq.len() != table.len() {
        return Err(AkitaError::InvalidSize {
            expected: table.len(),
            actual: eq.len(),
        });
    }
    Ok(table
        .iter()
        .zip(eq.iter())
        .fold(F::zero(), |acc, (&value, &weight)| acc + value * weight))
}
