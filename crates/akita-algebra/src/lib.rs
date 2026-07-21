//! Concrete algebra backends and arithmetic building blocks.
//!
//! This module includes:
//! - Module and polynomial containers (`module`, `poly`)
//! - Low-level NTT and CRT+NTT arithmetic scaffolding (`ntt`)
//! - Cyclotomic ring and backend arithmetic structure
//!
//! Concrete fields and field packing live in `akita-field`. Sparse
//! Fiat–Shamir challenge representations and samplers live in
//! `akita-challenges`.

#![warn(missing_docs)]
#![warn(unreachable_pub)]

pub mod backend;
pub mod eq_poly;
pub mod module;
pub mod ntt;
pub mod offset_eq;
pub mod poly;
pub mod ring;
pub mod split_eq;
pub mod uni_poly;

// Flat re-exports for convenience.
pub use akita_field::{
    cfg_chunks, cfg_chunks_mut, cfg_fold_reduce, cfg_into_iter, cfg_iter, cfg_iter_mut, cfg_join,
    cfg_try_fold_reduce,
};
pub use akita_field::{
    AdditiveGroup, AkitaError, BalancedDigitLookup, CanonicalField, FieldCore, FromPrimitiveInt,
    HalvingField, Invertible, One, PseudoMersenneField, RandomSampling, RingCore, SmoothFftField,
    Zero,
};
pub use backend::{CrtReconstruct, NttPrimeOps, NttTransform, RingBackend, ScalarBackend};
pub use eq_poly::{EqPolynomial, SplitEqEvals};
pub use module::{Module, VectorModule};
pub use ntt::tables;
pub use ntt::{GarnerData, LimbQ, MontCoeff, NttPrime, PrimeWidth, RADIX_BITS};
pub use ring::{
    CenteredMontLut, CrtNttConvertibleField, CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing,
    DigitMontLut, PackedPartialSplitEval16, PackedPartialSplitNtt16, PartialSplitEval16,
    PartialSplitNtt16,
};
pub use split_eq::GruenSplitEq;
pub use uni_poly::{CompressedUniPoly, UniPoly};
