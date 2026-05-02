use super::proof::QuadraticMask;
use super::statement::ZkSigmaStatement;
use crate::primitives::serialization::{Compress, HachiSerialize, SerializationError};
use crate::protocol::transcript::labels;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};
use std::io::Write;

struct ZkSigmaFirstMessage<'a, F: FieldCore> {
    attempt: u32,
    mask_commitment: &'a Vec<F>,
    linear_masks: &'a Vec<F>,
    quadratic_masks: &'a Vec<QuadraticMask<F>>,
}

impl<F: FieldCore> HachiSerialize for ZkSigmaFirstMessage<'_, F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.attempt.serialize_with_mode(&mut writer, compress)?;
        self.mask_commitment
            .serialize_with_mode(&mut writer, compress)?;
        self.linear_masks
            .serialize_with_mode(&mut writer, compress)?;
        self.quadratic_masks
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.attempt.serialized_size(compress)
            + self.mask_commitment.serialized_size(compress)
            + self.linear_masks.serialized_size(compress)
            + self.quadratic_masks.serialized_size(compress)
    }
}

pub(super) fn append_statement<F, T>(transcript: &mut T, statement: &ZkSigmaStatement<F>)
where
    F: CanonicalField + 'static,
    T: Transcript<F>,
{
    transcript.append_serde(labels::ABSORB_ZK_SIGMA_STATEMENT, statement);
}

pub(super) fn append_first_message<F, T>(
    transcript: &mut T,
    attempt: u32,
    mask_commitment: &Vec<F>,
    linear_masks: &Vec<F>,
    quadratic_masks: &Vec<QuadraticMask<F>>,
) where
    F: CanonicalField + 'static,
    T: Transcript<F>,
{
    let first = ZkSigmaFirstMessage {
        attempt,
        mask_commitment,
        linear_masks,
        quadratic_masks,
    };
    transcript.append_serde(labels::ABSORB_ZK_SIGMA_FIRST_MESSAGE, &first);
}

pub(super) fn append_response<F, T>(transcript: &mut T, response: &Vec<F>)
where
    F: CanonicalField + 'static,
    T: Transcript<F>,
{
    transcript.append_serde(labels::ABSORB_ZK_SIGMA_RESPONSE, response);
}
