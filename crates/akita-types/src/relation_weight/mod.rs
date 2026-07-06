//! Materialized and prepared relation-weight polynomials for Stage-2 sumcheck.

mod materialized;
mod prepared;

pub use materialized::{RelationWeightPolynomial, RelationWeightPolynomialError};
pub use prepared::PreparedRelationWeightPolynomial;
