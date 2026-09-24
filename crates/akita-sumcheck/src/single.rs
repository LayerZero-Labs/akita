//! Direct single-instance sumcheck protocol functions.

use crate::EqFactoredUniPoly;
use jolt_field::Field;

/// Advance the normalized claim for one equality-factored round.
pub fn advance_eq_factored_claim<E: Field>(
    claim: E,
    tau: E,
    poly: &EqFactoredUniPoly<E>,
    challenge: E,
) -> E {
    let constant = claim - tau * poly.nonconstant_term_sum_at_one();
    constant + poly.eval_nonconstant_terms(&challenge)
}
