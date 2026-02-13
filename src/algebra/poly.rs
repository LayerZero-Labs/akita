//! Fixed-size polynomial container.

use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::FieldCore;
use std::io::{Read, Write};

/// A degree-<D polynomial over `F`, stored as coefficients `[a0, a1, ..., a_{D-1}]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Poly<F: FieldCore, const D: usize>(pub [F; D]);

impl<F: FieldCore, const D: usize> Poly<F, D> {
    /// Construct the zero polynomial.
    pub fn zero() -> Self {
        Self([F::zero(); D])
    }
}

impl<F: FieldCore, const D: usize> std::ops::Add for Poly<F, D> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = self.0;
        for (dst, src) in out.iter_mut().zip(rhs.0.iter()) {
            *dst = *dst + *src;
        }
        Self(out)
    }
}

impl<F: FieldCore, const D: usize> std::ops::Sub for Poly<F, D> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = self.0;
        for (dst, src) in out.iter_mut().zip(rhs.0.iter()) {
            *dst = *dst - *src;
        }
        Self(out)
    }
}

impl<F: FieldCore, const D: usize> std::ops::Neg for Poly<F, D> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let mut out = self.0;
        for coeff in &mut out {
            *coeff = -*coeff;
        }
        Self(out)
    }
}

impl<F: FieldCore + Valid, const D: usize> Valid for Poly<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        for x in self.0.iter() {
            x.check()?;
        }
        Ok(())
    }
}

impl<F: FieldCore, const D: usize> HachiSerialize for Poly<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for x in self.0.iter() {
            x.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.iter().map(|x| x.serialized_size(compress)).sum()
    }
}

impl<F: FieldCore + Valid, const D: usize> HachiDeserialize for Poly<F, D> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let mut arr = [F::zero(); D];
        for coeff in &mut arr {
            *coeff = F::deserialize_with_mode(&mut reader, compress, validate)?;
        }
        let out = Self(arr);
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}
