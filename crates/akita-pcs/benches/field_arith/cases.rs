use akita_field::fields::fp32::Fp32;
use akita_field::fields::pseudo_mersenne::*;
use akita_field::{Fp128Packing, Fp32Packing, Fp64Packing, Prime128Offset275};

pub(crate) type Mersenne31 = Fp32<{ (1u32 << 31) - 1 }>;
pub(crate) type PackedMersenne31 = Fp32Packing<{ (1u32 << 31) - 1 }>;
pub(crate) type P31O19 = Fp32Packing<{ PRIME31_OFFSET19_MODULUS }>;
pub(crate) type P32O99 = Fp32Packing<{ PRIME32_OFFSET99_MODULUS }>;
pub(crate) type P40O195 = Fp64Packing<{ PRIME40_OFFSET195_MODULUS }>;
pub(crate) type P48O59 = Fp64Packing<{ PRIME48_OFFSET59_MODULUS }>;
pub(crate) type P56O27 = Fp64Packing<{ PRIME56_OFFSET27_MODULUS }>;
pub(crate) type P64O59 = Fp64Packing<{ PRIME64_OFFSET59_MODULUS }>;
pub(crate) type P128O275 = Fp128Packing<{ PRIME128_OFFSET275_MODULUS }>;
pub(crate) type F128 = Prime128Offset275;

pub(crate) const PRIME31_OFFSET19: &str = "prime31_offset19";
pub(crate) const MERSENNE31: &str = "mersenne31";
pub(crate) const PRIME32_OFFSET99: &str = "prime32_offset99";
pub(crate) const PRIME40_OFFSET195: &str = "prime40_offset195";
pub(crate) const PRIME48_OFFSET59: &str = "prime48_offset59";
pub(crate) const PRIME56_OFFSET27: &str = "prime56_offset27";
pub(crate) const PRIME64_OFFSET59: &str = "prime64_offset59";
pub(crate) const PRIME128_OFFSET275: &str = "prime128_offset275";
