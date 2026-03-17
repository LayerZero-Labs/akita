//! Protocol commitment types.

use crate::algebra::ring::CyclotomicRing;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::FieldCore;
use std::io::{Read, Write};

/// Ring-native commitment object `u in R_q^{n_B}` used by §4.1.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RingCommitment<F: FieldCore, const D: usize> {
    /// Outer commitment vector.
    pub u: Vec<CyclotomicRing<F, D>>,
}

impl<F: FieldCore + Valid, const D: usize> Valid for RingCommitment<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        self.u.check()
    }
}

impl<F: FieldCore, const D: usize> HachiSerialize for RingCommitment<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.u.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.u.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, const D: usize> HachiDeserialize for RingCommitment<F, D> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let u =
            Vec::<CyclotomicRing<F, D>>::deserialize_with_mode(&mut reader, compress, validate)?;
        let out = Self { u };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}
