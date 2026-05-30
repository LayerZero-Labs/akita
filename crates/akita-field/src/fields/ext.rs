//! Quadratic, quartic, and ring-subfield extension fields.

mod fp2;
mod power_fp4;
mod ring_subfield_fp4;
mod ring_subfield_fp8;
#[cfg(all(test, not(feature = "zk")))]
mod tests;
mod tower_fp4;

use super::wide::{
    AccumPair, FoldMatrixFp16, FoldMatrixFp32, Fp2Fp64ProductAccum, HasOptimizedFold,
    HasUnreducedOps, RingSubfieldFp4Fp32ProductAccum, RingSubfieldFp8Fp16ProductAccum,
};
use super::{fp128::Fp128, fp16::Fp16, fp32::Fp32, fp64::Fp64};
use crate::{BalancedDigitLookup, CanonicalField, FieldCore, HalvingField};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_field::{FromPrimitiveInt, Invertible, RandomSampling, RingCore};
use rand_core::RngCore;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub use fp2::{Ext2, Fp2, Fp2Config, NegOneNr, TwoNr};
pub(crate) use power_fp4::power_basis_fp4_mul_coeffs;
pub use power_fp4::{PowerBasisFp4, PowerBasisFp4Config, PowerBasisFp4MulBackend};
pub use ring_subfield_fp4::{RingSubfieldFp4, RingSubfieldFp4MulBackend};
pub use ring_subfield_fp8::{RingSubfieldFp8, RingSubfieldFp8MulBackend};
pub use tower_fp4::{TowerBasisFp4, TowerBasisFp4Config, UnitNr};

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
