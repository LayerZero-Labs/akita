//! Fixed-width generated-schedule selection identities.
//!
//! These values are public protocol inputs. They intentionally contain no
//! prover representation metadata: a row is identified only by its ordered
//! exact committed profiles and the expanded verifier schedule.

use crate::descriptor_bytes::push_usize;
use crate::instance_descriptor::digest_descriptor_bytes;
use crate::{CommittedGroupBatchProfile, FoldSchedule};
use akita_field::AkitaError;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::io::{Read, Write};

const SCHEDULE_ROW_DOMAIN_V2: &[u8] = b"akita/schedule-row/v2";

/// Cryptographic identity of one complete expanded schedule row.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ScheduleRowDigest([u8; 32]);

impl ScheduleRowDigest {
    /// Build a row identity from its fixed-width digest bytes.
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Borrow the fixed-width digest bytes.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Public batch-level selection of one verifier-approved generated row.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct OpeningScheduleSelection {
    /// Complete expanded row identity.
    pub row_digest: ScheduleRowDigest,
}

/// Hash one complete schedule row under the development-v1 row namespace.
///
/// The ordered profile bytes are encoded before the expanded schedule so the
/// final group layout and group ordering cannot alias schedules with identical
/// aggregate widths. Provider/source identity is absent from the profile
/// encoding and must remain absent from `FoldSchedule` descriptor bytes.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] if the ordered group count overflows.
pub fn schedule_row_digest(
    profiles: &CommittedGroupBatchProfile,
    schedule: &FoldSchedule,
) -> Result<ScheduleRowDigest, AkitaError> {
    let num_groups = profiles
        .prior_group_profiles
        .len()
        .checked_add(1)
        .ok_or_else(|| AkitaError::InvalidSetup("schedule row group count overflow".to_string()))?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(SCHEDULE_ROW_DOMAIN_V2);
    bytes.push(1);
    push_usize(&mut bytes, num_groups);
    for profile in &profiles.prior_group_profiles {
        let encoded = profile.canonical_descriptor_bytes();
        push_usize(&mut bytes, encoded.len());
        bytes.extend_from_slice(&encoded);
    }
    let final_profile = profiles.final_group.canonical_descriptor_bytes();
    push_usize(&mut bytes, final_profile.len());
    bytes.extend_from_slice(&final_profile);
    let schedule_bytes = schedule.canonical_descriptor_bytes();
    push_usize(&mut bytes, schedule_bytes.len());
    bytes.extend_from_slice(&schedule_bytes);
    Ok(ScheduleRowDigest::from_bytes(digest_descriptor_bytes(
        &bytes,
    )))
}

macro_rules! impl_fixed_digest_wire {
    ($type:ty) => {
        impl Valid for $type {
            fn check(&self) -> Result<(), SerializationError> {
                Ok(())
            }
        }

        impl AkitaSerialize for $type {
            fn serialize_with_mode<W: Write>(
                &self,
                mut writer: W,
                _compress: Compress,
            ) -> Result<(), SerializationError> {
                writer.write_all(self.as_bytes())?;
                Ok(())
            }

            fn serialized_size(&self, _compress: Compress) -> usize {
                32
            }
        }

        impl AkitaDeserialize for $type {
            type Context = ();

            fn deserialize_with_mode<R: Read>(
                mut reader: R,
                _compress: Compress,
                validate: Validate,
                _ctx: &Self::Context,
            ) -> Result<Self, SerializationError> {
                let mut bytes = [0u8; 32];
                reader.read_exact(&mut bytes)?;
                let value = Self::from_bytes(bytes);
                if matches!(validate, Validate::Yes) {
                    value.check()?;
                }
                Ok(value)
            }
        }
    };
}

impl_fixed_digest_wire!(ScheduleRowDigest);

impl Valid for OpeningScheduleSelection {
    fn check(&self) -> Result<(), SerializationError> {
        self.row_digest.check()
    }
}

impl AkitaSerialize for OpeningScheduleSelection {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.row_digest.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.row_digest.serialized_size(compress)
    }
}

impl AkitaDeserialize for OpeningScheduleSelection {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &Self::Context,
    ) -> Result<Self, SerializationError> {
        let value = Self {
            row_digest: ScheduleRowDigest::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
        };
        if matches!(validate, Validate::Yes) {
            value.check()?;
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_selection_round_trips_as_exactly_thirty_two_bytes() {
        let selection = OpeningScheduleSelection {
            row_digest: ScheduleRowDigest::from_bytes([0x22; 32]),
        };
        let mut bytes = Vec::new();
        selection
            .serialize_uncompressed(&mut bytes)
            .expect("serialize selection");
        assert_eq!(bytes.len(), 32);
        let decoded = OpeningScheduleSelection::deserialize_uncompressed(bytes.as_slice(), &())
            .expect("deserialize selection");
        assert_eq!(decoded, selection);
    }
}
