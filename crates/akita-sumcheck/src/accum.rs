//! Signed accumulator reduction helpers.
//!
//! These live in a dedicated module so Akita-specific stage implementations can
//! share the same arithmetic helper without coupling to one another.

use akita_field::unreduced::HasUnreducedOps;
use akita_field::FieldCore;

#[inline]
/// Reduce separated positive and negative unreduced accumulators into one field
/// element.
pub fn reduce_signed_accum<E: FieldCore + HasUnreducedOps>(
    pos: E::SmallMulAccum,
    neg: E::SmallMulAccum,
) -> E {
    E::reduce_small_accum(pos) - E::reduce_small_accum(neg)
}
