//! Protocol-wide transcript-grinding contract bound into every preamble.

use crate::{
    GrindingPlan, FOLD_COORDINATE_ORACLE_REVISION, FOLD_RESPONSE_ATTEMPTS,
    FOLD_RESPONSE_NONCE_BITS, GRINDING_ENCODING_VERSION, GRINDING_LITTLE_ENDIAN_BIT_ORDER,
    GRINDING_NONCE_SLACK_BITS, GRINDING_PREDICATE_BYTES, GRINDING_QUERY_POLICY_REVISION,
    MAX_GRINDING_BITS, TRANSCRIPT_SECURITY_BITS,
};
use akita_error::AkitaError;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::io::{Read, Write};

/// Active transcript proof-of-work, fold-response, and plan-digest binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscriptGrindingBinding {
    pub encoding_version: u16,
    pub target_security_bits: u16,
    pub proof_of_work_slack_bits: u8,
    pub maximum_proof_of_work_bits: u8,
    pub predicate_bytes: u8,
    pub predicate_bit_order_tag: u8,
    pub fold_response_attempt_bits: u8,
    pub fold_response_attempts: u32,
    pub query_policy_revision: u16,
    pub fold_coordinate_oracle_revision: u16,
    pub plan_digest: [u8; 32],
}

impl TranscriptGrindingBinding {
    /// Bind the active protocol policy to one validated public plan.
    pub fn for_plan(plan: &GrindingPlan) -> Result<Self, AkitaError> {
        Ok(Self {
            encoding_version: GRINDING_ENCODING_VERSION,
            target_security_bits: TRANSCRIPT_SECURITY_BITS,
            proof_of_work_slack_bits: GRINDING_NONCE_SLACK_BITS,
            maximum_proof_of_work_bits: MAX_GRINDING_BITS,
            predicate_bytes: GRINDING_PREDICATE_BYTES,
            predicate_bit_order_tag: GRINDING_LITTLE_ENDIAN_BIT_ORDER,
            fold_response_attempt_bits: FOLD_RESPONSE_NONCE_BITS,
            fold_response_attempts: FOLD_RESPONSE_ATTEMPTS,
            query_policy_revision: GRINDING_QUERY_POLICY_REVISION,
            fold_coordinate_oracle_revision: FOLD_COORDINATE_ORACLE_REVISION,
            plan_digest: plan.digest()?,
        })
    }

    /// Validate an existing fold-response nonce against the shared policy.
    pub fn validate_fold_response_nonce(nonce: u32) -> Result<(), AkitaError> {
        if nonce >= FOLD_RESPONSE_ATTEMPTS {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }

    fn has_active_policy(&self) -> bool {
        self.encoding_version == GRINDING_ENCODING_VERSION
            && self.target_security_bits == TRANSCRIPT_SECURITY_BITS
            && self.proof_of_work_slack_bits == GRINDING_NONCE_SLACK_BITS
            && self.maximum_proof_of_work_bits == MAX_GRINDING_BITS
            && self.predicate_bytes == GRINDING_PREDICATE_BYTES
            && self.predicate_bit_order_tag == GRINDING_LITTLE_ENDIAN_BIT_ORDER
            && self.fold_response_attempt_bits == FOLD_RESPONSE_NONCE_BITS
            && self.fold_response_attempts == FOLD_RESPONSE_ATTEMPTS
            && self.query_policy_revision == GRINDING_QUERY_POLICY_REVISION
            && self.fold_coordinate_oracle_revision == FOLD_COORDINATE_ORACLE_REVISION
    }
}

impl Valid for TranscriptGrindingBinding {
    fn check(&self) -> Result<(), SerializationError> {
        if !self.has_active_policy() {
            return Err(SerializationError::InvalidData(
                "descriptor grinding binding does not match the active protocol".to_string(),
            ));
        }
        Ok(())
    }
}

impl AkitaSerialize for TranscriptGrindingBinding {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.encoding_version
            .serialize_with_mode(&mut writer, compress)?;
        self.target_security_bits
            .serialize_with_mode(&mut writer, compress)?;
        self.proof_of_work_slack_bits
            .serialize_with_mode(&mut writer, compress)?;
        self.maximum_proof_of_work_bits
            .serialize_with_mode(&mut writer, compress)?;
        self.predicate_bytes
            .serialize_with_mode(&mut writer, compress)?;
        self.predicate_bit_order_tag
            .serialize_with_mode(&mut writer, compress)?;
        self.fold_response_attempt_bits
            .serialize_with_mode(&mut writer, compress)?;
        self.fold_response_attempts
            .serialize_with_mode(&mut writer, compress)?;
        self.query_policy_revision
            .serialize_with_mode(&mut writer, compress)?;
        self.fold_coordinate_oracle_revision
            .serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.plan_digest)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.encoding_version.serialized_size(compress)
            + self.target_security_bits.serialized_size(compress)
            + self.proof_of_work_slack_bits.serialized_size(compress)
            + self.maximum_proof_of_work_bits.serialized_size(compress)
            + self.predicate_bytes.serialized_size(compress)
            + self.predicate_bit_order_tag.serialized_size(compress)
            + self.fold_response_attempt_bits.serialized_size(compress)
            + self.fold_response_attempts.serialized_size(compress)
            + self.query_policy_revision.serialized_size(compress)
            + self
                .fold_coordinate_oracle_revision
                .serialized_size(compress)
            + 32
    }
}

impl AkitaDeserialize for TranscriptGrindingBinding {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &Self::Context,
    ) -> Result<Self, SerializationError> {
        let mut plan_digest = [0u8; 32];
        let binding = Self {
            encoding_version: u16::deserialize_with_mode(&mut reader, compress, validate, &())?,
            target_security_bits: u16::deserialize_with_mode(&mut reader, compress, validate, &())?,
            proof_of_work_slack_bits: u8::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
            maximum_proof_of_work_bits: u8::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
            predicate_bytes: u8::deserialize_with_mode(&mut reader, compress, validate, &())?,
            predicate_bit_order_tag: u8::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
            fold_response_attempt_bits: u8::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
            fold_response_attempts: u32::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
            query_policy_revision: u16::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
            fold_coordinate_oracle_revision: u16::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
            plan_digest: {
                reader.read_exact(&mut plan_digest)?;
                plan_digest
            },
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
    use crate::{GrindingRun, GrindingSite};

    fn sample_plan() -> GrindingPlan {
        GrindingPlan::new(
            vec![
                GrindingRun::proof_of_work(GrindingSite::EvaluationBatch, 1, 128).unwrap(),
                GrindingRun::fold_response(0),
            ],
            128,
        )
        .unwrap()
    }

    #[test]
    fn binding_serializes_exact_policy_then_digest() {
        let binding = TranscriptGrindingBinding::for_plan(&sample_plan()).unwrap();
        let mut bytes = Vec::new();
        binding.serialize_uncompressed(&mut bytes).unwrap();
        assert_eq!(
            bytes,
            vec![
                1, 0, 128, 0, 7, 25, 32, 0, 12, 0, 16, 0, 0, 1, 0, 1, 0, 151, 149, 136, 70, 183,
                131, 180, 155, 68, 1, 176, 12, 94, 2, 221, 151, 63, 71, 238, 161, 158, 232, 76, 56,
                19, 17, 35, 104, 169, 100, 185, 235,
            ]
        );
        assert_eq!(binding.fold_response_attempts, 4096);
        assert_eq!(binding.fold_response_attempt_bits, 12);
    }

    #[test]
    fn binding_roundtrip_and_nonce_boundary() {
        let binding = TranscriptGrindingBinding::for_plan(&sample_plan()).unwrap();
        let mut bytes = Vec::new();
        binding.serialize_uncompressed(&mut bytes).unwrap();
        assert_eq!(
            TranscriptGrindingBinding::deserialize_uncompressed_exact(&bytes, &()).unwrap(),
            binding
        );
        TranscriptGrindingBinding::validate_fold_response_nonce(4095).unwrap();
        assert_eq!(
            TranscriptGrindingBinding::validate_fold_response_nonce(4096),
            Err(AkitaError::InvalidProof)
        );
    }
}
