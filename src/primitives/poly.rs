//! Multilinear polynomial utility functions.

use super::arithmetic::FieldCore;
use std::marker::PhantomData;

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

/// Utilities for the equality polynomial `eq(x, y)`.
pub struct EqPolynomial<E: FieldCore>(PhantomData<E>);

impl<E: FieldCore> EqPolynomial<E> {
    /// Compute the MLE of the equality polynomial at two points.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != y.len()`.
    pub fn mle(x: &[E], y: &[E]) -> E {
        assert_eq!(x.len(), y.len());
        x.iter()
            .zip(y.iter())
            .map(|(&x_i, &y_i)| x_i * y_i + (E::one() - x_i) * (E::one() - y_i))
            .fold(E::one(), |acc, v| acc * v)
    }

    /// Compute the zero selector: `eq(r, 0) = Πᵢ (1 − rᵢ)`.
    pub fn zero_selector(r: &[E]) -> E {
        r.iter().fold(E::one(), |acc, &r_i| acc * (E::one() - r_i))
    }

    /// Compute the full evaluation table `{ eq(r, x) : x ∈ {0,1}^n }`.
    pub fn evals(r: &[E]) -> Vec<E> {
        Self::evals_with_scaling(r, None)
    }

    /// Compute the full evaluation table with optional scaling.
    pub fn evals_with_scaling(r: &[E], scaling_factor: Option<E>) -> Vec<E> {
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 16;
            if r.len() > PARALLEL_THRESHOLD {
                return Self::evals_parallel(r, scaling_factor);
            }
        }
        Self::evals_serial(r, scaling_factor)
    }

    /// Serial version of [`Self::evals_with_scaling`].
    pub fn evals_serial(r: &[E], scaling_factor: Option<E>) -> Vec<E> {
        let size = 1usize << r.len();
        let mut evals = vec![E::zero(); size];
        evals[0] = scaling_factor.unwrap_or(E::one());
        let mut len = 1usize;
        for &t in r.iter().rev() {
            let one_minus_t = E::one() - t;
            for j in (0..len).rev() {
                evals[2 * j + 1] = evals[j] * t;
                evals[2 * j] = evals[j] * one_minus_t;
            }
            len *= 2;
        }
        evals
    }

    /// Compute eq evaluations and cache intermediate tables.
    pub fn evals_cached(r: &[E]) -> Vec<Vec<E>> {
        Self::evals_cached_with_scaling(r, None)
    }

    /// Like [`Self::evals_cached`], but with optional scaling.
    pub fn evals_cached_with_scaling(r: &[E], scaling_factor: Option<E>) -> Vec<Vec<E>> {
        let mut result: Vec<Vec<E>> = (0..r.len() + 1).map(|i| vec![E::zero(); 1 << i]).collect();
        result[0][0] = scaling_factor.unwrap_or(E::one());
        for j in 0..r.len() {
            let idx = r.len() - 1 - j;
            let t = r[idx];
            let one_minus_t = E::one() - t;
            let prev_len = 1 << j;
            for i in (0..prev_len).rev() {
                result[j + 1][2 * i + 1] = result[j][i] * t;
                result[j + 1][2 * i] = result[j][i] * one_minus_t;
            }
        }
        result
    }

    /// Parallel version of [`Self::evals_with_scaling`].
    #[cfg(feature = "parallel")]
    pub fn evals_parallel(r: &[E], scaling_factor: Option<E>) -> Vec<E> {
        use rayon::prelude::*;

        let final_size = 1usize << r.len();
        let mut evals = vec![E::zero(); final_size];
        evals[0] = scaling_factor.unwrap_or(E::one());
        let mut size = 1;

        for &r_i in r.iter() {
            let (evals_left, evals_right) = evals.split_at_mut(size);
            let (evals_right, _) = evals_right.split_at_mut(size);

            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r_i;
                    *x -= *y;
                });

            size *= 2;
        }

        evals
    }
}

/// Compute left and right vectors from evaluation point
///
/// Given a point arranged for matrix evaluation, computes L and R such that:
/// polynomial_evaluation(point) = L^T × M × R
///
/// Splits variables between rows and columns based on sigma and nu.
pub fn compute_left_right_vectors<F: FieldCore>(
    point: &[F],
    nu: usize,
    sigma: usize,
) -> (Vec<F>, Vec<F>) {
    let mut left_vec = vec![F::zero(); 1 << nu];
    let mut right_vec = vec![F::zero(); 1 << sigma];
    let point_dim = point.len();

    match point_dim {
        // Case 1: Constant polynomial (0 variables)
        0 => {
            left_vec[0] = F::one();
            right_vec[0] = F::one();
        }

        // Case 2: All variables fit in columns (single row)
        n if n <= sigma => {
            multilinear_lagrange_basis(&mut right_vec[..1 << point_dim], point);
            left_vec[0] = F::one();
        }

        // Case 3: Variables split between rows and columns (no padding)
        n if n <= nu + sigma => {
            multilinear_lagrange_basis(&mut right_vec, &point[..sigma]);
            multilinear_lagrange_basis(&mut left_vec[..1 << (point_dim - sigma)], &point[sigma..]);
        }

        // Case 4: Too many variables - need column padding
        _ => {
            multilinear_lagrange_basis(&mut right_vec[..1 << sigma], &point[..sigma]);
            multilinear_lagrange_basis(&mut left_vec, &point[sigma..]);
        }
    }

    (left_vec, right_vec)
}
