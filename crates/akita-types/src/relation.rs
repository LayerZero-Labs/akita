//! Shared protocol relation helpers.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::eval_ring_at;
use akita_algebra::CyclotomicRing;
use akita_field::{CanonicalField, FieldCore};

/// Compute the stage-2 relation claim from the public M-row data.
///
/// This evaluates `sum_i eq(tau1, i) * y_alpha[i]` where `y_alpha` follows
/// the M row layout: consistency zero row, public `y_rings`, D rows `v`, B
/// rows `u`, then A zero rows.
///
/// # Panics
///
/// Panics if `D` is zero because cyclotomic rings require a nonzero const
/// dimension.
#[tracing::instrument(skip_all, name = "relation_claim_from_rows")]
pub fn relation_claim_from_rows<F: FieldCore + CanonicalField, const D: usize>(
    tau1: &[F],
    alpha: F,
    v: &[CyclotomicRing<F, D>],
    u: &[CyclotomicRing<F, D>],
    y_rings: &[CyclotomicRing<F, D>],
) -> F {
    let eq_tau1 = EqPolynomial::evals(tau1);
    let mut acc = F::zero();
    let mut row_idx = 1usize;

    for y_ring in y_rings {
        if row_idx >= eq_tau1.len() {
            return acc;
        }
        acc += eq_tau1[row_idx] * eval_ring_at(y_ring, &alpha);
        row_idx += 1;
    }
    for r in v {
        if row_idx >= eq_tau1.len() {
            return acc;
        }
        acc += eq_tau1[row_idx] * eval_ring_at(r, &alpha);
        row_idx += 1;
    }
    for r in u {
        if row_idx >= eq_tau1.len() {
            return acc;
        }
        acc += eq_tau1[row_idx] * eval_ring_at(r, &alpha);
        row_idx += 1;
    }
    acc
}
