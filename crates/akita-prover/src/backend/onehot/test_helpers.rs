/// Test-only helpers for this module that need access to private invariants
/// (`FlatBlocks`' monotonic `offsets` / contiguous `entries`, and the
/// non-wide reference path for `inner_ajtai_wide_onehot`).
///
/// Gated on `#[cfg(test)]` so the production binary never sees them.
#[cfg(test)]
use super::{CyclotomicRing, FlatBlocks, OneHotIndex, OneHotPoly, SparseRingBlockEntry};
use akita_field::parallel::*;
use akita_field::{CanonicalField, FieldCore};

/// Reference ring-space evaluation for [`OneHotPoly`].
///
/// Computes the global weighted sum `y = Σᵢ scalars[i] · self[i]`.
/// `scalars` has length >= `num_ring_elems`; excess entries are ignored.
///
/// Only used by tests to cross-check fused prover paths
/// (e.g. `evaluate_and_fold`) against a straight-line implementation,
/// so it lives in `test_helpers` rather than on the production trait.
pub(crate) fn evaluate_ring_onehot<F, const D: usize, I>(
    poly: &OneHotPoly<F, I>,
    scalars: &[F],
) -> CyclotomicRing<F, D>
where
    F: FieldCore + CanonicalField,
    I: OneHotIndex,
{
    let onehot_k = poly.onehot_k;
    cfg_fold_reduce!(
        0..poly.indices.len(),
        || CyclotomicRing::<F, D>::zero(),
        |mut acc: CyclotomicRing<F, D>, chunk_idx: usize| {
            if let Some(raw) = poly.indices[chunk_idx] {
                let field_pos = chunk_idx * onehot_k + raw.as_usize();
                let ring_idx = field_pos / D;
                let coeff_idx = field_pos % D;
                if ring_idx < scalars.len() {
                    acc.coeffs[coeff_idx] += scalars[ring_idx];
                }
            }
            acc
        },
        |a, b| a + b
    )
}

pub(crate) fn from_buckets<E>(buckets: Vec<Vec<E>>) -> FlatBlocks<E> {
    FlatBlocks::from_buckets(buckets).expect("test block offsets fit in u32")
}

/// Reference (non-wide) multi-chunk inner Ajtai used to cross-check
/// [`super::inner_ajtai_wide_onehot`].
///
/// Production code always uses the wide accumulator; this simpler
/// variant only exists so tests can assert the two paths agree.
#[allow(non_snake_case)]
pub(crate) fn inner_ajtai_reference<F: FieldCore + CanonicalField, const D: usize>(
    A: &[Vec<CyclotomicRing<F, D>>],
    entries: &[SparseRingBlockEntry],
    num_digits: usize,
) -> Vec<CyclotomicRing<F, D>> {
    let n_a = A.len();
    let mut t = vec![CyclotomicRing::<F, D>::zero(); n_a];
    for entry in entries {
        let pos_in_block = entry.pos_in_block();
        let coeff_idx = entry.coeff_idx();
        let col = pos_in_block * num_digits;
        for a in 0..n_a {
            A[a][col].shift_accumulate_into(&mut t[a], coeff_idx);
        }
    }
    t
}
