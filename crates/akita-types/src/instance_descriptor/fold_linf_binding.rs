//! Shared fold-nonce wire contract bound into every transcript preamble.

use crate::sis::MAX_FOLD_GRIND_ATTEMPTS;
use akita_field::AkitaError;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::io::{Read, Write};

/// One protocol-wide sequential `u32` fold-nonce contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldLinfProtocolBinding {
    /// Exclusive upper bound for sequential probes `0, 1, ...`.
    pub max_grind_attempts: u32,
    /// Wire width of every fold nonce.
    pub grind_nonce_wire_bytes: u8,
    /// Challenge entropy charged for the nonce range.
    pub grind_entropy_bits_per_level: u8,
}

impl FoldLinfProtocolBinding {
    pub const CURRENT: Self = Self {
        max_grind_attempts: MAX_FOLD_GRIND_ATTEMPTS,
        grind_nonce_wire_bytes: 4,
        grind_entropy_bits_per_level: 12,
    };

    /// Validate a Fiat–Shamir grind nonce against this protocol binding.
    pub fn validate_grind_nonce(self, grind_nonce: u32) -> Result<(), AkitaError> {
        if grind_nonce >= self.max_grind_attempts {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }
}

impl Valid for FoldLinfProtocolBinding {
    fn check(&self) -> Result<(), SerializationError> {
        if *self != Self::CURRENT {
            return Err(SerializationError::InvalidData(
                "descriptor fold nonce binding does not match the active protocol".to_string(),
            ));
        }
        Ok(())
    }
}

impl AkitaSerialize for FoldLinfProtocolBinding {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.max_grind_attempts
            .serialize_with_mode(&mut writer, compress)?;
        self.grind_nonce_wire_bytes
            .serialize_with_mode(&mut writer, compress)?;
        self.grind_entropy_bits_per_level
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.max_grind_attempts.serialized_size(compress)
            + self.grind_nonce_wire_bytes.serialized_size(compress)
            + self.grind_entropy_bits_per_level.serialized_size(compress)
    }
}

impl AkitaDeserialize for FoldLinfProtocolBinding {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &Self::Context,
    ) -> Result<Self, SerializationError> {
        let binding = Self {
            max_grind_attempts: u32::deserialize_with_mode(&mut reader, compress, validate, &())?,
            grind_nonce_wire_bytes: u8::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
            grind_entropy_bits_per_level: u8::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
        };
        if matches!(validate, Validate::Yes) {
            binding.check()?;
        }
        Ok(binding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_binding_is_one_u32_sequential_nonce() {
        let binding = FoldLinfProtocolBinding::CURRENT;
        assert_eq!(binding.grind_nonce_wire_bytes, 4);
        assert_eq!(binding.max_grind_attempts, 4096);
        binding.check().unwrap();
    }

    #[test]
    fn current_binding_rejects_nonce_at_attempt_limit() {
        let binding = FoldLinfProtocolBinding::CURRENT;
        binding
            .validate_grind_nonce(binding.max_grind_attempts - 1)
            .expect("last in-range nonce");
        assert_eq!(
            binding.validate_grind_nonce(binding.max_grind_attempts),
            Err(AkitaError::InvalidProof)
        );
    }
}
