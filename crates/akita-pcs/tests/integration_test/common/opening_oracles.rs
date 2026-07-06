//! Independent opening oracles for PCS correctness tests.
//!
//! These deliberately share no code with the prover. The prover reaches
//! `OpeningFoldKernel::evaluate_and_fold`, so using it for expected openings
//! would let a shared point-layout or fold-order bug move both values together.
//! These functions instead evaluate raw coefficients directly.

use akita_field::FieldCore;
use akita_prover::OneHotPoly;

/// Lagrange weight at one Boolean index, in little-endian variable order.
fn lagrange_weight_at<E: FieldCore>(point: &[E], index: usize) -> E {
    point
        .iter()
        .enumerate()
        .fold(E::one(), |acc, (bit, &coordinate)| {
            if (index >> bit) & 1 == 1 {
                acc * coordinate
            } else {
                acc * (E::one() - coordinate)
            }
        })
}

/// Dense multilinear opening `Σ_x eq(point, x) · evals[x]`.
///
/// The recursion borrows evaluation halves and reuses one small leaf buffer.
/// It therefore avoids the materialized Lagrange table limit and a full copy
/// of production-sized evaluation vectors.
pub(crate) fn dense_opening_lagrange<E: FieldCore>(evals: &[E], point: &[E]) -> E {
    debug_assert_eq!(
        evals.len(),
        1usize << point.len(),
        "dense evaluation count must be 2^|point|"
    );
    const LEAF: usize = 1 << 12;

    let mut scratch = vec![E::zero(); LEAF.min(evals.len())];
    dense_opening_lagrange_rec(evals, point, &mut scratch)
}

fn dense_opening_lagrange_rec<E: FieldCore>(evals: &[E], point: &[E], scratch: &mut [E]) -> E {
    if evals.len() <= scratch.len() {
        let mut len = evals.len();
        scratch[..len].copy_from_slice(evals);
        for &coordinate in point {
            let half = len / 2;
            for j in 0..half {
                let low = scratch[2 * j];
                let high = scratch[2 * j + 1];
                scratch[j] = low + (high - low) * coordinate;
            }
            len = half;
        }
        return scratch[0];
    }

    let (&high_coordinate, rest) = point.split_last().expect("non-leaf slice has coordinates");
    let (low_half, high_half) = evals.split_at(evals.len() / 2);
    let low = dense_opening_lagrange_rec(low_half, rest, scratch);
    let high = dense_opening_lagrange_rec(high_half, rest, scratch);
    low + (high - low) * high_coordinate
}

/// Dense opening in the monomial basis.
pub(crate) fn dense_opening_monomial<E: FieldCore>(evals: &[E], point: &[E]) -> E {
    assert_eq!(
        evals.len(),
        1usize << point.len(),
        "dense evaluation count must be 2^|point|"
    );
    evals
        .iter()
        .enumerate()
        .fold(E::zero(), |acc, (index, &eval)| {
            let weight = point
                .iter()
                .enumerate()
                .filter(|(bit, _)| (index >> bit) & 1 == 1)
                .fold(E::one(), |product, (_, &coordinate)| product * coordinate);
            acc + weight * eval
        })
}

/// One-hot multilinear opening over an arbitrary extension field.
///
/// Each selected weight is computed on demand, so production arities above
/// the materialized Lagrange-table limit remain executable.
pub(crate) fn onehot_opening_lagrange<Base: FieldCore, E: FieldCore>(
    poly: &OneHotPoly<Base, u8>,
    point: &[E],
) -> E {
    let k = poly.onehot_k();
    poly.indices()
        .iter()
        .enumerate()
        .filter_map(|(chunk, hot)| {
            hot.map(|idx| lagrange_weight_at(point, chunk * k + usize::from(idx)))
        })
        .fold(E::zero(), |acc, weight| acc + weight)
}
