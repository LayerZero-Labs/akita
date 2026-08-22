//! Protocol-wide transcript-grinding contract bound into every preamble.

use crate::{GrindingPlan, GrindingPolicy, FOLD_RESPONSE_ATTEMPTS};
use akita_error::AkitaError;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::io::{Read, Write};

/// Active transcript proof-of-work, fold-response, and plan-digest binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscriptGrindingBinding {
    pub policy: GrindingPolicy,
    pub plan_digest: [u8; 32],
}

impl TranscriptGrindingBinding {
    /// Bind the active protocol policy to one validated public plan.
    pub fn for_plan(plan: &GrindingPlan) -> Result<Self, AkitaError> {
        Ok(Self {
            policy: GrindingPolicy::ACTIVE,
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
}

impl Valid for TranscriptGrindingBinding {
    fn check(&self) -> Result<(), SerializationError> {
        if self.policy != GrindingPolicy::ACTIVE {
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
        let _ = compress;
        writer.write_all(&self.policy.canonical_bytes())?;
        writer.write_all(&self.plan_digest)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let _ = compress;
        self.policy.canonical_bytes().len() + 32
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
        let _ = compress;
        let mut policy_bytes = GrindingPolicy::ACTIVE.canonical_bytes();
        reader.read_exact(&mut policy_bytes)?;
        let mut plan_digest = [0u8; 32];
        reader.read_exact(&mut plan_digest)?;
        let binding = Self {
            policy: GrindingPolicy::from_canonical_bytes(policy_bytes),
            plan_digest,
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
        assert_eq!(binding.policy.fold_response_attempts, 4096);
        assert_eq!(binding.policy.fold_response_attempt_bits, 12);
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
        let last_valid_attempt = binding
            .policy
            .fold_response_attempts
            .checked_sub(1)
            .expect("positive fold response attempt budget");
        TranscriptGrindingBinding::validate_fold_response_nonce(last_valid_attempt).unwrap();
        assert_eq!(
            TranscriptGrindingBinding::validate_fold_response_nonce(
                binding.policy.fold_response_attempts
            ),
            Err(AkitaError::InvalidProof)
        );
    }
}
