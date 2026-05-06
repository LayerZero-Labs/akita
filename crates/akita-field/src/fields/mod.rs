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

pub use ext::{Ext2, Ext4, Fp2, Fp2Config, Fp4, Fp4Config, NegOneNr, TwoNr, UnitNr};
pub use fp128::{
    Fp128, Prime128Offset159, Prime128Offset2355, Prime128Offset275, Prime128OffsetA7F7,
};
pub use fp32::Fp32;
pub use fp64::Fp64;
pub use lift::{ExtField, LiftBase, MulBase};
pub use packed::{
    Fp128Packing, Fp32Packing, Fp64Packing, HasPacking, NoPacking, PackedField, PackedValue,
};
pub use pseudo_mersenne::{
    is_pow2_offset, pow2_offset, pseudo_mersenne_modulus, Pow2Offset128Field, Pow2Offset24Field,
    Pow2Offset30Field, Pow2Offset31Field, Pow2Offset32Field, Pow2Offset40Field, Pow2Offset48Field,
    Pow2Offset56Field, Pow2Offset64Field, Pow2OffsetPrimeSpec, POW2_OFFSET_IMPLEMENTED_MAX_BITS,
    POW2_OFFSET_MAX, POW2_OFFSET_PRIMES, POW2_OFFSET_TABLE,
};
pub use wide::{
    AccumPair, Fp128MulU64Accum, Fp128ProductAccum, Fp128x8i32, Fp32ProductAccum, Fp32x2i32,
    Fp64ProductAccum, Fp64x4i32, HasUnreducedOps, HasWide, ReduceTo,
};
