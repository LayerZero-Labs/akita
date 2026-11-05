//! Concrete algebra backends and arithmetic building blocks.
//!
//! This module includes multilinear polynomial, eq-polynomial, and offset-eq
//! utilities.
//!
//! Concrete fields, field packing, the cyclotomic ring, NTT/CRT arithmetic,
//! and the smooth FFT live in `jolt-algebra`; [`fft`], [`ntt`], and [`ring`]
//! re-export its modules under their former paths. Sparse Fiat–Shamir
//! challenge representations and samplers live in `akita-challenges`.

#![warn(missing_docs)]
#![warn(unreachable_pub)]

pub mod eq_poly;
pub mod offset_eq;
pub mod poly;
pub mod row_batch;
pub mod split_eq;

pub use jolt_algebra::{fft, ntt, ring};

// The Jolt RISC-V guest inlines serve jolt-algebra's 64-point `i32` NTT and
// pointwise dot (the `ntt-inline` feature). jolt-algebra cannot depend on
// `jolt-inlines-ntt` itself: that crate's host registration reaches
// jolt-algebra again through `jolt-inlines-sdk` and `tracer`, a package
// cycle Cargo rejects, so the backend is supplied from here.
#[cfg(feature = "ntt-inline")]
jolt_algebra::ntt64_backend!(
    jolt_inlines_ntt::forward_ntt64,
    jolt_inlines_ntt::pointwise_dot64
);

// Flat re-exports for convenience.
pub use eq_poly::{EqPolynomial, SplitEqEvals};
pub use fft::SmoothFftField;
pub use jolt_algebra::{
    cfg_chunks, cfg_chunks_mut, cfg_fold_reduce, cfg_into_iter, cfg_iter, cfg_iter_mut, cfg_join,
};
pub use jolt_algebra::{AdditiveGroup, CanonicalEncoding, Field, One, PseudoMersenne, Ring, Zero};
pub use ntt::tables;
pub use ntt::{
    CrtCapacity, GarnerData, LimbQ, MontCoeff, NttKernelPlan, NttPrime, PrimeWidth, RADIX_BITS,
};
pub use ring::{
    balanced_decompose_coefficients_pow2_i8_into, cyclic_ntt_with_i16_tail_to_ring,
    mat_vec_i16_with_tail, ntt_with_i16_tail_to_ring, residue_kernel, terminal_residue_kernel,
    CenteredMontLut, CrtNttConvertibleField, CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing,
    DigitMontLut, I16TailParams, Ifma52NttMatrix, Ifma52Params, ResidueKernelPoint,
};
pub use split_eq::GruenSplitEq;

/// Fallible parallel fold-reduce over a range.
///
/// The identity argument is a zero-argument factory, matching Rayon's
/// `try_fold` and `try_reduce` contracts. The sequential path calls the same
/// factory once to obtain its initial accumulator.
#[macro_export]
macro_rules! cfg_try_fold_reduce {
    ($range:expr, $identity_factory:expr, $fold_op:expr, $reduce_op:expr) => {{
        let identity_factory = $identity_factory;
        #[cfg(feature = "parallel")]
        let result = $range
            .into_par_iter()
            .try_fold(&identity_factory, $fold_op)
            .try_reduce(&identity_factory, $reduce_op);
        #[cfg(not(feature = "parallel"))]
        let result = $range.into_iter().try_fold(identity_factory(), $fold_op);
        result
    }};
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    #[test]
    fn try_fold_reduce_uses_identity_factory() {
        let sum = crate::cfg_try_fold_reduce!(
            0usize..100,
            usize::default,
            |acc, value| acc.checked_add(value),
            |lhs, rhs| lhs.checked_add(rhs)
        );
        assert_eq!(sum, Some((0usize..100).sum()));

        let empty = crate::cfg_try_fold_reduce!(
            0usize..0,
            usize::default,
            |acc, value| acc.checked_add(value),
            |lhs, rhs| lhs.checked_add(rhs)
        );
        assert_eq!(empty, Some(0));
    }
}
