//! Fold-l∞ rejection protocol identity bound into every transcript preamble.

use crate::sis::MAX_FOLD_GRIND_ATTEMPTS;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::io::{Read, Write};

/// Fold-l∞ rejection protocol identity bound into every transcript preamble.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldLinfProtocolBinding {
    /// Tail-bound formula tag (`1` = integer `t*` from fold-linf-rejection spec).
    pub formula_tag: u8,
    /// Fiat-Shamir reroll cap per fold level.
    pub max_grind_attempts: u32,
    /// Wire width of `fold_grind_nonce` on every fold level proof.
    pub grind_nonce_wire_bytes: u8,
    /// Challenge-entropy budget per fold level: `log2(max_grind_attempts)`.
    pub grind_entropy_bits_per_level: u8,
}

impl FoldLinfProtocolBinding {
    /// Active fold-l∞ rejection cutover parameters.
    pub const CURRENT: Self = Self {
        formula_tag: 1,
        max_grind_attempts: MAX_FOLD_GRIND_ATTEMPTS,
        grind_nonce_wire_bytes: 4,
        grind_entropy_bits_per_level: 12,
    };
}

impl Valid for FoldLinfProtocolBinding {
    fn check(&self) -> Result<(), SerializationError> {
        if *self != Self::CURRENT {
            return Err(SerializationError::InvalidData(
                "descriptor fold_linf binding does not match active protocol cutover".to_string(),
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
        self.formula_tag
            .serialize_with_mode(&mut writer, compress)?;
        self.max_grind_attempts
            .serialize_with_mode(&mut writer, compress)?;
        self.grind_nonce_wire_bytes
            .serialize_with_mode(&mut writer, compress)?;
        self.grind_entropy_bits_per_level
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.formula_tag.serialized_size(compress)
            + self.max_grind_attempts.serialized_size(compress)
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
        let out = Self {
            formula_tag: u8::deserialize_with_mode(&mut reader, compress, validate, &())?,
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
            out.check()?;
        }
        Ok(out)
    }
}
