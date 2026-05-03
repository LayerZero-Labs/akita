//! Signed accumulator reduction helpers.
//!
//! These live in a dedicated module so that both [`super::hachi_stage2`] and
//! [`super::two_round_prefix`] can import them without creating a circular
//! dependency.

use crate::FieldCore;
use akita_algebra::fields::HasUnreducedOps;

#[inline]
pub(crate) fn reduce_signed_accum<E: FieldCore + HasUnreducedOps>(
    pos: E::MulU64Accum,
    neg: E::MulU64Accum,
) -> E {
    E::reduce_mul_u64_accum(pos) - E::reduce_mul_u64_accum(neg)
}
