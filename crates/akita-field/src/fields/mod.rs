//! Prime fields and extension field towers.

pub mod ext;
pub mod fft;
pub mod fp128;
pub mod fp32;
pub mod fp64;
pub mod lift;
pub mod packed;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512dq"))
))]
pub mod packed_avx2;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq"
))]
pub mod packed_avx512;
pub mod packed_ext;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod packed_neon;
pub mod pseudo_mersenne;
pub(crate) mod util;
pub mod wide;

pub use ext::{
    Ext2, FpExt2, FpExt2Config, NegOneNr, PowerBasisFpExt4, PowerBasisFpExt4Config,
    PowerBasisFpExt4MulBackend, RingSubfieldFpExt4, RingSubfieldFpExt4MulBackend,
    RingSubfieldFpExt8, RingSubfieldFpExt8MulBackend, TowerBasisFpExt4, TowerBasisFpExt4Config,
    TwoNr, UnitNr,
};
pub use fp128::{
    Fp128, Prime128Offset159, Prime128Offset2355, Prime128Offset275, Prime128OffsetA7F7,
};
pub use fp32::Fp32;
pub use fp64::Fp64;
pub use lift::{
    canonical_frobenius_thetas, solve_frobenius_moore, validate_canonical_frobenius_thetas,
    ExtField, FrobeniusExtField, LiftBase, MulBase, MulBaseUnreduced,
};
pub use packed::{
    Fp128Packing, Fp32Packing, Fp64Packing, HasPacking, NoPacking, PackedField, PackedValue,
};
pub use pseudo_mersenne::{
    is_registered_prime_offset, pseudo_mersenne_modulus, registered_prime_offset_spec,
    Prime24Offset3, Prime30Offset35, Prime31Offset19, Prime32Offset99, Prime40Offset195,
    Prime48Offset59, Prime56Offset27, Prime64Offset59, PrimeOffsetSpec,
    PRIME_OFFSET_IMPLEMENTED_MAX_BITS, PRIME_OFFSET_MAX, PRIME_OFFSET_SPECS,
};
pub use wide::{
    AccumPair, FoldMatrixFp32, Fp128MulU64Accum, Fp128ProductAccum, Fp128x8i32,
    Fp2Fp64ProductAccum, Fp32ProductAccum, Fp32x2i32, Fp64ProductAccum, Fp64x4i32,
    HasOptimizedFold, HasUnreducedOps, HasWide, ReduceTo, RingSubfieldFp4Fp32ProductAccum,
};
