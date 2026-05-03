//! Multilinear polynomial utility functions.

use akita_field::FieldCore;

/// Compute multilinear Lagrange basis evaluations at a point
///
/// For variables (r₀, r₁, ..., r_{n-1}), computes all 2^n basis polynomial evaluations.
/// The i-th basis polynomial evaluates to 1 at the i-th hypercube vertex and 0 elsewhere.
///
/// Uses an iterative doubling approach:
/// - Start with [1-r₀, r₀]
/// - For each variable rᵢ, split each value v into [v*(1-rᵢ), v*rᵢ]
pub(crate) fn multilinear_lagrange_basis<F: FieldCore>(output: &mut [F], point: &[F]) {
    assert!(
        output.len() <= (1 << point.len()),
        "Output length must be at most 2^point.len()"
    );

    if point.is_empty() || output.is_empty() {
        output.fill(F::one());
        return;
    }

    // Initialize for first variable: [1-r₀, r₀]
    let one_minus_p0 = F::one() - point[0];
    output[0] = one_minus_p0;
    if output.len() > 1 {
        output[1] = point[0];
    }

    // For each subsequent variable, double the active portion
    for (level, p) in point[1..].iter().enumerate() {
        let mid = 1 << (level + 1);
        let one_minus_p = F::one() - p;

        if mid >= output.len() {
            // No split possible, just multiply all by (1-p)
            for val in output.iter_mut() {
                *val = val.mul(&one_minus_p);
            }
        } else {
            // Split: left *= (1-p), right = left * p
            let (left, right) = output.split_at_mut(mid);
            let k = left.len().min(right.len());

            for (l, r) in left[..k].iter_mut().zip(right[..k].iter_mut()) {
                let l_val = *l;
                *r = l_val.mul(p);
                *l = l_val.mul(&one_minus_p);
            }

            // Handle remaining left elements if any
            for l in left[k..].iter_mut() {
                *l = l.mul(&one_minus_p);
            }
        }
    }
}
