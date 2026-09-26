//! Direct single-instance sumcheck protocol functions.

use jolt_field::Field;
use jolt_poly::OmittedConstantPoly;

/// Advance the normalized claim for one equality-factored round.
pub fn advance_eq_factored_claim<E: Field>(
    claim: E,
    tau: E,
    poly: &OmittedConstantPoly<E>,
    challenge: E,
) -> E {
    let constant = claim - tau * poly.nonconstant_term_sum_at_one();
    constant + poly.evaluate_nonconstant_terms(challenge)
}
