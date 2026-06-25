//! Fold-l∞ rejection protocol identity bound into every transcript preamble.

use crate::golomb_rice::TAIL_Z_PLANNER_MODEL_K_PLUS_TWO;
use crate::sis::{
    FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_DEN, FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_NUM,
    MAX_FOLD_GRIND_ATTEMPTS,
};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::io::{Read, Write};

/// Probe `nonce = 0, 1, …` and publish the minimum accepting index (plain presets).
pub const FOLD_GRIND_PROBE_ORDER_SEQUENTIAL_MIN: u8 = 0;

/// Probe a transcript-seeded uniform permutation of `[0, cap)` (ZK presets).
pub const FOLD_GRIND_PROBE_ORDER_TRANSCRIPT_SHUFFLE: u8 = 1;

/// Fold-l∞ rejection protocol identity bound into every transcript preamble.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldLinfProtocolBinding {
    /// Tail-bound formula tag (`2` = integer `t*` with explicit grind accept target).
    pub formula_tag: u8,
    /// Per-challenge grind acceptance target `p_grind = NUM / DEN` in the union bound.
    pub grind_target_accept_prob_num: u32,
    pub grind_target_accept_prob_den: u32,
    /// Fiat-Shamir reroll cap per fold level.
    pub max_grind_attempts: u32,
    /// Wire width of `fold_grind_nonce` on every fold level proof.
    pub grind_nonce_wire_bytes: u8,
    /// Challenge-entropy budget per fold level: `log2(max_grind_attempts)`.
    pub grind_entropy_bits_per_level: u8,
    /// Prover grind search order (`FOLD_GRIND_PROBE_ORDER_*`).
    pub grind_probe_order: u8,
    /// Terminal `z` Golomb average-case planner model (e.g. [`TAIL_Z_PLANNER_MODEL_K_PLUS_TWO`]).
    pub tail_z_planner_model_id: u8,
}

impl FoldLinfProtocolBinding {
    /// Active fold-l∞ rejection cutover parameters.
    pub const CURRENT: Self = Self {
        formula_tag: 2,
        grind_target_accept_prob_num: FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_NUM,
        grind_target_accept_prob_den: FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_DEN,
        max_grind_attempts: MAX_FOLD_GRIND_ATTEMPTS,
        grind_nonce_wire_bytes: 4,
        grind_entropy_bits_per_level: 12,
        grind_probe_order: {
            #[cfg(feature = "zk")]
            {
                FOLD_GRIND_PROBE_ORDER_TRANSCRIPT_SHUFFLE
            }
            #[cfg(not(feature = "zk"))]
            {
                FOLD_GRIND_PROBE_ORDER_SEQUENTIAL_MIN
            }
        },
        tail_z_planner_model_id: TAIL_Z_PLANNER_MODEL_K_PLUS_TWO,
    };

    /// Rational grind acceptance target `(NUM, DEN)` for tail-bound sizing.
    #[inline]
    pub const fn grind_target_accept_prob(&self) -> (u128, u128) {
        (
            self.grind_target_accept_prob_num as u128,
            self.grind_target_accept_prob_den as u128,
        )
    }
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
        self.grind_target_accept_prob_num
            .serialize_with_mode(&mut writer, compress)?;
        self.grind_target_accept_prob_den
            .serialize_with_mode(&mut writer, compress)?;
        self.max_grind_attempts
            .serialize_with_mode(&mut writer, compress)?;
        self.grind_nonce_wire_bytes
            .serialize_with_mode(&mut writer, compress)?;
        self.grind_entropy_bits_per_level
            .serialize_with_mode(&mut writer, compress)?;
        self.grind_probe_order
            .serialize_with_mode(&mut writer, compress)?;
        self.tail_z_planner_model_id
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.formula_tag.serialized_size(compress)
            + self.grind_target_accept_prob_num.serialized_size(compress)
            + self.grind_target_accept_prob_den.serialized_size(compress)
            + self.max_grind_attempts.serialized_size(compress)
            + self.grind_nonce_wire_bytes.serialized_size(compress)
            + self.grind_entropy_bits_per_level.serialized_size(compress)
            + self.grind_probe_order.serialized_size(compress)
            + self.tail_z_planner_model_id.serialized_size(compress)
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
            grind_target_accept_prob_num: u32::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
            grind_target_accept_prob_den: u32::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
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
            grind_probe_order: u8::deserialize_with_mode(&mut reader, compress, validate, &())?,
            tail_z_planner_model_id: u8::deserialize_with_mode(
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
