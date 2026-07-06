//! Materialized and prepared relation-weight polynomials for Stage-2 sumcheck.

mod materialized;
mod prepared;

pub use materialized::{
    bridge_relation_weight_from_split, RelationWeightPolynomial, RelationWeightPolynomialError,
};
pub use prepared::PreparedRelationWeightPolynomial;
