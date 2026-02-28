//! Polynomial trait for multilinear polynomials.

use super::arithmetic::FieldCore;

/// Trait for multilinear Lagrange polynomial operations
pub trait MultilinearLagrange<F: FieldCore>: Polynomial<F> {
    /// Compute multilinear Lagrange basis evaluations at a point
    ///
    /// For variables (r₀, r₁, ..., r_{n-1}), computes all 2^n basis polynomial evaluations.
    /// The i-th basis polynomial evaluates to 1 at the i-th hypercube vertex and 0 elsewhere.
    fn lagrange_basis(&self, output: &mut [F], point: &[F]) {
        multilinear_lagrange_basis(output, point)
    }

    /// Compute vector-matrix product: v = L^T * M
    ///
    /// Treats coefficients as a 2^nu × 2^sigma matrix.
    /// For each column j: v\[j\] = Σ_i left_vec\[i\] * coefficients\[i\]\[j\]
    fn vector_matrix_product(&self, left_vec: &[F], nu: usize, sigma: usize) -> Vec<F>;

    /// Compute left and right vectors from evaluation point
    ///
    /// Given a point arranged for matrix evaluation, computes L and R such that:
    /// polynomial_evaluation(point) = L^T × M × R
    fn compute_evaluation_vectors(&self, point: &[F], nu: usize, sigma: usize) -> (Vec<F>, Vec<F>) {
        compute_left_right_vectors(point, nu, sigma)
    }
}

/// Trait for multilinear polynomials
///
/// Represents a polynomial in evaluation form (coefficients at hypercube points).
pub trait Polynomial<F: FieldCore> {
    /// Number of variables
    fn num_vars(&self) -> usize;

    /// Total number of coefficients (2^num_vars)
    fn len(&self) -> usize {
        1 << self.num_vars()
    }

    /// Check if polynomial is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Evaluate polynomial at a point
    ///
    /// # Parameters
    /// - `point`: Evaluation point (length must equal num_vars)
    ///
    /// # Returns
    /// Polynomial evaluation result
    fn evaluate(&self, point: &[F]) -> F;

    /// Return the coefficient/evaluation table on `{0,1}^n` in LSB-first order.
    fn coeffs(&self) -> Vec<F>;
}

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
