//! AArch64 PMULL packed-buffer kernels.

use std::arch::aarch64::{uint64x2_t, vdupq_n_u64, veorq_u64, vmull_p64};

use super::{fold_with, PackedBinary162};
use crate::binary::{product, BinaryField162 as F};

// Keep the six Karatsuba products in vector registers until the entire round
// has accumulated. Extracting and recombining each pair defeats deferred work.
#[inline]
#[target_feature(enable = "aes")]
unsafe fn accumulate(sums: &mut [uint64x2_t; 6], a: F, b: F) {
    let [a0, a1, a2] = a.to_words();
    let [b0, b1, b2] = b.to_words();
    let terms = [
        (a0, b0),
        (a1, b1),
        (a2, b2),
        (a0 ^ a1, b0 ^ b1),
        (a0 ^ a2, b0 ^ b2),
        (a1 ^ a2, b1 ^ b2),
    ];
    for (sum, (left, right)) in sums.iter_mut().zip(terms) {
        // SAFETY: PMULL is enabled by the caller. Every 128-bit product bit
        // pattern is valid in the vector representation, with no padding.
        let term = unsafe { std::mem::transmute::<u128, uint64x2_t>(vmull_p64(left, right)) };
        *sum = veorq_u64(*sum, term);
    }
}

#[inline]
fn finish(sums: [uint64x2_t; 6]) -> F {
    // SAFETY: both representations contain the same six 128-bit polynomials;
    // all bit patterns are valid and no alignment-dependent access is used.
    let [d0, d1, d2, m01, m02, m12] =
        unsafe { std::mem::transmute::<[uint64x2_t; 6], [u128; 6]>(sums) };
    product::reduce(product::karatsuba_product(d0, d1, d2, m01, m02, m12))
}

#[target_feature(enable = "aes")]
pub(super) unsafe fn round_product(
    lhs: &PackedBinary162,
    rhs: &PackedBinary162,
    current_claim: F,
) -> [F; 3] {
    let mut constant = [vdupq_n_u64(0); 6];
    let mut quadratic = constant;
    let mut index = 0;
    while index + 1 < lhs.len() {
        let a0 = lhs.element(index);
        let b0 = rhs.element(index);
        let da = a0 + lhs.element(index + 1);
        let db = b0 + rhs.element(index + 1);
        // SAFETY: this function requires PMULL; the public boundary validated
        // equal lengths, and both entries are within each private limb array.
        unsafe {
            accumulate(&mut constant, a0, b0);
            accumulate(&mut quadratic, da, db);
        }
        index += 2;
    }
    if index < lhs.len() {
        let a0 = lhs.element(index);
        let b0 = rhs.element(index);
        // SAFETY: the remaining odd entry is paired with zero; the same
        // target features and validated length invariant hold here.
        unsafe {
            accumulate(&mut constant, a0, b0);
            accumulate(&mut quadratic, a0, b0);
        }
    }
    let c0 = finish(constant);
    let c2 = finish(quadratic);
    [c0, current_claim + c2, c2]
}

#[target_feature(enable = "aes")]
pub(super) unsafe fn fold_in_place(values: &mut PackedBinary162, r: F) {
    fold_with(values, r, |a, b| {
        // SAFETY: the enclosing function carries the same target feature.
        unsafe { product::arm_multiply(a, b) }
    });
}
