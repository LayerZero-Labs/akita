//! Concrete algebra backends and arithmetic building blocks.
//!
//! This module includes:
//! - Generic prime fields and extensions (`fields`)
//! - Module and polynomial containers (`module`, `poly`)
//! - Low-level NTT and CRT+NTT arithmetic scaffolding (`ntt`)

#![cfg_attr(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    feature(stdarch_x86_avx512)
)]
#![warn(missing_docs)]
#![warn(unreachable_pub)]

pub mod backend;
pub mod eq_poly;
pub mod fields;
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
};
pub use akita_field::{
    AdditiveGroup, AkitaError, CanonicalField, FieldCore, FieldSampling, FromSmallInt, Invertible,
    Module, PseudoMersenneField, SmoothFftField,
};
pub use backend::{CrtReconstruct, NttPrimeOps, NttTransform, RingBackend, ScalarBackend};
pub use eq_poly::EqPolynomial;
pub use fields::{
    is_pow2_offset, pow2_offset, pseudo_mersenne_modulus, ExtField, Fp128, Fp128Packing, Fp2,
    Fp2Config, Fp32, Fp32Packing, Fp4, Fp4Config, Fp64, Fp64Packing, HasPacking, LiftBase,
    NoPacking, PackedField, PackedValue, Pow2Offset128Field, Pow2Offset24Field, Pow2Offset30Field,
    Pow2Offset31Field, Pow2Offset32Field, Pow2Offset40Field, Pow2Offset48Field, Pow2Offset56Field,
    Pow2Offset64Field, Pow2OffsetPrimeSpec, Prime128Offset159, Prime128Offset2355,
    Prime128Offset275, Prime128OffsetA7F7, POW2_OFFSET_IMPLEMENTED_MAX_BITS, POW2_OFFSET_MAX,
    POW2_OFFSET_PRIMES, POW2_OFFSET_TABLE,
};
pub use module::VectorModule;
pub use ntt::tables;
pub use ntt::{GarnerData, LimbQ, MontCoeff, NttPrime, PrimeWidth, RADIX_BITS};
pub use ring::{
    CenteredMontLut, CrtNttConvertibleField, CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing,
    DigitMontLut, PackedPartialSplitEval16, PackedPartialSplitNtt16, PartialSplitEval16,
    PartialSplitNtt16, SparseChallenge, SparseChallengeConfig,
};
pub use split_eq::GruenSplitEq;
pub use uni_poly::{CompressedUniPoly, UniPoly};
