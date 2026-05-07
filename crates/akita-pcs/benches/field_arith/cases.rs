use akita_field::fields::fp32::Fp32;
use akita_field::fields::pseudo_mersenne::*;
use akita_field::{Fp128Packing, Fp32Packing, Fp64Packing, Prime128Offset275};

pub(crate) type M31 = Fp32<{ (1u32 << 31) - 1 }>;
pub(crate) type PM31 = Fp32Packing<{ (1u32 << 31) - 1 }>;
pub(crate) type P31 = Fp32Packing<{ POW2_OFFSET_MODULUS_31 }>;
pub(crate) type P32 = Fp32Packing<{ POW2_OFFSET_MODULUS_32 }>;
pub(crate) type P40 = Fp64Packing<{ POW2_OFFSET_MODULUS_40 }>;
pub(crate) type P48 = Fp64Packing<{ POW2_OFFSET_MODULUS_48 }>;
pub(crate) type P56 = Fp64Packing<{ POW2_OFFSET_MODULUS_56 }>;
pub(crate) type P64 = Fp64Packing<{ POW2_OFFSET_MODULUS_64 }>;
pub(crate) type P128 = Fp128Packing<{ POW2_OFFSET_MODULUS_128 }>;
pub(crate) type F128 = Prime128Offset275;

pub(crate) const FP32_31B: &str = "fp32_31b";
pub(crate) const FP32_M31: &str = "fp32_m31";
pub(crate) const FP32_32B: &str = "fp32_32b";
pub(crate) const FP64_40B: &str = "fp64_40b";
pub(crate) const FP64_48B: &str = "fp64_48b";
pub(crate) const FP64_56B: &str = "fp64_56b";
pub(crate) const FP64_64B: &str = "fp64_64b";
pub(crate) const FP128: &str = "fp128";
