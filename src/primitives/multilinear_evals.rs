//! Dense multilinear polynomials in evaluation form.
//!
//! This module intentionally follows the same high-level representation style as
//! Jolt's `DensePolynomial` for multilinear extensions (MLEs): store the values
//! of the multilinear polynomial on the Boolean hypercube `{0,1}^n` and provide
//! binding/evaluation by iterative folding.
//!
//! The key convention for this repo (used by the ring-switch witness table) is:
//!
//! - An evaluation index `idx` is interpreted in binary.
//! - The **lowest** index bit is the **first** variable bound under
//!   [`BindingOrder::LowToHigh`].
//!
//! This matches the row-major flattening `idx = row * d + col` when `d` is a
//! power of two: the low `log2(d)` bits correspond to the `col` coordinate.

use crate::primitives::arithmetic::FieldCore;
use crate::primitives::poly::Polynomial;

/// The order in which variables are bound/evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BindingOrder {
    /// Bind the lowest index bit first (LSB → MSB).
    #[default]
    LowToHigh,
    /// Bind the highest index bit first (MSB → LSB).
    HighToLow,
}

/// Dense multilinear polynomial in evaluation form over `{0,1}^num_vars`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseMultilinearEvals<F: FieldCore> {
    /// Number of variables in the multilinear extension.
    pub num_vars: usize,
    /// Active length (decreases as variables are bound).
    pub len: usize,
    /// Evaluations on the hypercube, length is a power of two.
    pub evals: Vec<F>,
}

impl<F: FieldCore> Default for DenseMultilinearEvals<F> {
    fn default() -> Self {
        Self {
            num_vars: 0,
            len: 1,
            evals: vec![F::zero()],
        }
    }
}

impl<F: FieldCore> DenseMultilinearEvals<F> {
    /// Construct from evaluations, padding with zeros to a power of two.
    ///
    /// The variable count is derived from the padded length.
    pub fn new_padded(mut evals: Vec<F>) -> Self {
        if evals.is_empty() {
            evals.push(F::zero());
        }
        while !evals.len().is_power_of_two() {
            evals.push(F::zero());
        }
        let num_vars = evals.len().trailing_zeros() as usize;
        let len = evals.len();
        Self {
            num_vars,
            len,
            evals,
        }
    }

    /// Return the original (backing) evaluation length.
    pub fn original_len(&self) -> usize {
        self.evals.len()
    }

    /// Bind one variable in-place, reducing `len` by a factor of 2.
    ///
    /// After binding, the polynomial has one fewer variable.
    pub fn bind_in_place(&mut self, r: F, order: BindingOrder) {
        assert!(self.len.is_power_of_two());
        assert!(self.len >= 2, "cannot bind variable of constant polynomial");
        match order {
            BindingOrder::LowToHigh => self.bind_lsb_in_place(r),
            BindingOrder::HighToLow => self.bind_msb_in_place(r),
        }
    }

    #[inline]
    fn bind_lsb_in_place(&mut self, r: F) {
        let next_len = self.len / 2;
        for i in 0..next_len {
            let v0 = self.evals[(i << 1) | 0];
            let v1 = self.evals[(i << 1) | 1];
            // (1-r)*v0 + r*v1 = v0 + r*(v1-v0)
            self.evals[i] = v0 + r * (v1 - v0);
        }
        self.len = next_len;
        self.num_vars = self.num_vars.saturating_sub(1);
    }

    #[inline]
    fn bind_msb_in_place(&mut self, r: F) {
        let next_len = self.len / 2;
        let (left, right) = self.evals.split_at_mut(next_len);
        for i in 0..next_len {
            let v0 = left[i];
            let v1 = right[i];
            left[i] = v0 + r * (v1 - v0);
        }
        self.len = next_len;
        self.num_vars = self.num_vars.saturating_sub(1);
    }

    /// Evaluate without mutating `self`.
    pub fn evaluate_with_order(&self, point: &[F], order: BindingOrder) -> F {
        if point.is_empty() {
            return self.evals[0];
        }
        assert_eq!(
            point.len(),
            self.num_vars,
            "point dimension mismatch: expected {}, got {}",
            self.num_vars,
            point.len()
        );
        let mut tmp = self.clone();
        for r in point.iter().copied() {
            tmp.bind_in_place(r, order);
        }
        tmp.evals[0]
    }
}

impl<F: FieldCore> Polynomial<F> for DenseMultilinearEvals<F> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn evaluate(&self, point: &[F]) -> F {
        self.evaluate_with_order(point, BindingOrder::LowToHigh)
    }

    fn coeffs(&self) -> Vec<F> {
        self.evals[..self.len].to_vec()
    }
}
