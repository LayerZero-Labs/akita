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

pub mod backend;
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
pub use backend::{CrtReconstruct, NttPrimeOps, NttTransform, RingBackend, ScalarBackend};
pub use eq_poly::{EqPolynomial, SplitEqEvals};
pub use fft::SmoothFftField;
pub use jolt_field::{
    cfg_chunks, cfg_chunks_mut, cfg_fold_reduce, cfg_into_iter, cfg_iter, cfg_iter_mut, cfg_join,
};
pub use jolt_field::{AdditiveGroup, CanonicalEncoding, Field, One, PseudoMersenne, Ring, Zero};
pub use module::{Module, VectorModule};
pub use ntt::tables;
pub use ntt::{GarnerData, LimbQ, MontCoeff, NttPrime, PrimeWidth, RADIX_BITS};
pub use ring::{
    balanced_decompose_coefficients_pow2_i8_into, mat_vec_i16_with_tail, CenteredMontLut,
    CrtNttConvertibleField, CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing, DigitMontLut,
    I16TailParams,
};
pub use split_eq::GruenSplitEq;
pub use uni_poly::{CompressedUniPoly, UniPoly};

/// Fallible parallel fold-reduce over a range.
///
/// With `parallel`: `range.into_par_iter().try_fold(identity, fold_op).try_reduce(identity, reduce_op)`.
/// Without: `range.into_iter().try_fold(identity(), fold_op)`.
///
/// Companion to the `cfg_*` macros re-exported from `jolt-field`, which does
/// not provide a fallible fold-reduce.
#[macro_export]
macro_rules! cfg_try_fold_reduce {
    ($range:expr, $identity:expr, $fold_op:expr, $reduce_op:expr) => {{
        #[cfg(feature = "parallel")]
        let result = $range
            .into_par_iter()
            .try_fold($identity, $fold_op)
            .try_reduce($identity, $reduce_op);
        #[cfg(not(feature = "parallel"))]
        let result = $range.into_iter().try_fold(($identity)(), $fold_op);
        result
    }};
}
