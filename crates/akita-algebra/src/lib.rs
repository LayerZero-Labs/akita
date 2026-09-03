//! Concrete algebra backends and arithmetic building blocks.
//!
//! This module includes:
//! - Module and polynomial containers (`module`, `poly`)
//! - Low-level NTT and CRT+NTT arithmetic scaffolding (`ntt`)
//! - Cyclotomic ring and backend arithmetic structure
//!
//! Concrete fields and field packing live in `jolt-field`. Sparse
//! Fiat–Shamir challenge representations and samplers live in
//! `akita-challenges`.

#![warn(missing_docs)]
#![warn(unreachable_pub)]

pub mod eq_poly;
pub mod fft;
pub mod module;
pub mod ntt;
pub mod offset_eq;
pub mod poly;
pub mod ring;
pub mod split_eq;
pub mod uni_poly;

// Flat re-exports for convenience.
pub use eq_poly::{EqPolynomial, SplitEqEvals};
pub use fft::SmoothFftField;
pub use jolt_field::{
    cfg_chunks, cfg_chunks_mut, cfg_fold_reduce, cfg_into_iter, cfg_iter, cfg_iter_mut, cfg_join,
};
pub use jolt_field::{AdditiveGroup, CanonicalEncoding, Field, One, PseudoMersenne, Ring, Zero};
pub use module::{Module, VectorModule};
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
pub use uni_poly::{CompressedUniPoly, UniPoly};

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
