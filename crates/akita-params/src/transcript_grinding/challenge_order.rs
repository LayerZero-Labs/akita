//! The challenge set a grinding plan is priced against, and how a plan
//! descriptor records it.
//!
//! The pricing arithmetic is Jolt's [`jolt_transcript::grind_bits_for_loss`].

use super::{push_u32, MAX_GRINDING_BITS, TRANSCRIPT_SECURITY_BITS};
use akita_error::AkitaError;
use jolt_transcript::{grind_bits_for_loss, ChallengeSetSize};

/// Exact cardinality of a challenge set used when pricing transcript grinding.
///
/// A field-backed order records the base modulus and extension degree, so its
/// cardinality is exactly `modulus^extension_degree`. A full-capacity order is
/// available for protocols whose challenge set has exactly a power-of-two
/// cardinality.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ChallengeFieldOrder {
    nominal_capacity_bits: u32,
    field_modulus: u128,
    extension_degree: u32,
    set_size: ChallengeSetSize,
}

impl ChallengeFieldOrder {
    /// Build the exact extension-field order from a prime-field modulus.
    pub fn from_field(
        modulus_bits: u32,
        extension_degree: usize,
        field_modulus: u128,
    ) -> Result<Self, AkitaError> {
        let actual_modulus_bits = u128::BITS - field_modulus.leading_zeros();
        if field_modulus <= 1 || actual_modulus_bits != modulus_bits {
            return Err(AkitaError::InvalidSetup(
                "challenge modulus does not match its declared bit width".into(),
            ));
        }
        let extension_degree = u32::try_from(extension_degree).map_err(|_| {
            AkitaError::InvalidSetup("challenge extension degree exceeds u32".into())
        })?;
        if extension_degree == 0 || !extension_degree.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "challenge extension degree must be a nonzero power of two".into(),
            ));
        }
        let nominal_capacity_bits = modulus_bits
            .checked_mul(extension_degree)
            .ok_or_else(|| AkitaError::InvalidSetup("challenge capacity overflow".into()))?;
        let set_size = ChallengeSetSize::field_order(field_modulus, extension_degree)
            .map_err(|_| AkitaError::InvalidSetup("challenge field order is invalid".into()))?;
        Ok(Self {
            nominal_capacity_bits,
            field_modulus,
            extension_degree,
            set_size,
        })
    }

    /// Build a challenge set whose cardinality is exactly `2^capacity_bits`.
    pub fn from_full_capacity(capacity_bits: u32) -> Result<Self, AkitaError> {
        if capacity_bits == 0 {
            return Err(AkitaError::InvalidSetup(
                "grinding nominal capacity must be nonzero".into(),
            ));
        }
        Ok(Self {
            nominal_capacity_bits: capacity_bits,
            field_modulus: 0,
            extension_degree: 0,
            set_size: ChallengeSetSize::power_of_two(capacity_bits),
        })
    }

    /// Nominal bit width for diagnostics only; grinding uses exact cardinality.
    #[must_use]
    pub const fn nominal_capacity_bits(self) -> u32 {
        self.nominal_capacity_bits
    }

    pub(super) fn append_canonical_bytes(self, out: &mut Vec<u8>) {
        if let Some(base_modulus_bits) = self
            .nominal_capacity_bits
            .checked_div(self.extension_degree)
        {
            out.push(1);
            push_u32(out, base_modulus_bits);
            push_u32(out, self.extension_degree);
            out.extend_from_slice(&self.field_modulus.to_le_bytes());
        } else {
            out.push(0);
            push_u32(out, self.nominal_capacity_bits);
        }
    }

    /// The least proof-of-work target that gives a challenge with
    /// `loss_factor` at most `2^-TRANSCRIPT_SECURITY_BITS` per hash call.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when `loss_factor` is zero or
    /// [`MAX_GRINDING_BITS`] bits do not suffice.
    pub fn grind_bits(self, loss_factor: u64) -> Result<u8, AkitaError> {
        let bits = grind_bits_for_loss(
            loss_factor,
            self.set_size,
            u32::from(TRANSCRIPT_SECURITY_BITS),
        )
        .map_err(|error| AkitaError::InvalidSetup(format!("grinding target: {error}")))?;
        // Jolt caps the target at its MAX_POW_BITS, which equals MAX_GRINDING_BITS.
        u8::try_from(bits).map_err(|_| {
            AkitaError::InvalidSetup(format!(
                "grinding target exceeds supported maximum {MAX_GRINDING_BITS}"
            ))
        })
    }
}
