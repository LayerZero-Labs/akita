//! Quadratic, quartic, and ring-subfield extension fields.

mod fp_ext2;
mod power_fp_ext4;
mod ring_subfield_fp_ext4;
mod ring_subfield_fp_ext8;
#[cfg(all(test, not(feature = "zk")))]
mod tests;
mod tower_fp_ext4;

use super::wide::{
    AccumPair, FoldMatrixFp32, FoldMatrixFp64, Fp2Fp64ProductAccum, HasOptimizedFold,
    HasUnreducedOps, RingSubfieldFp4Fp32ProductAccum,
};
use super::{fp128::Fp128, fp32::Fp32, fp64::Fp64};
use crate::{BalancedDigitLookup, CanonicalField, FieldCore, HalvingField, MulBaseUnreduced};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_field::{FromPrimitiveInt, Invertible, RandomSampling, RingCore};
use rand_core::RngCore;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub use fp_ext2::{Ext2, FpExt2, FpExt2Config, NegOneNr, TwoNr};
pub(crate) use power_fp_ext4::power_basis_fp_ext4_mul_coeffs;
pub use power_fp_ext4::{PowerBasisFpExt4, PowerBasisFpExt4Config, PowerBasisFpExt4MulBackend};
pub use ring_subfield_fp_ext4::{RingSubfieldFpExt4, RingSubfieldFpExt4MulBackend};
pub(crate) use ring_subfield_fp_ext8::{
    ring_subfield_fp_ext8_mul_schedule, ring_subfield_fp_ext8_square_schedule,
};
pub use ring_subfield_fp_ext8::{RingSubfieldFpExt8, RingSubfieldFpExt8MulBackend};
pub use tower_fp_ext4::{TowerBasisFpExt4, TowerBasisFpExt4Config, UnitNr};

/// Arithmetic shape shared by scalar and packed extension coefficients.
pub trait ExtensionCoeff<F: FieldCore>:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
{
}

impl<F, A> ExtensionCoeff<F> for A
where
    F: FieldCore,
    A: Copy + Add<Output = A> + Sub<Output = A> + Mul<Output = A>,
{
}
