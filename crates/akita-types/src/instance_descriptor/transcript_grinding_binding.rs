//! Protocol-wide transcript-grinding contract bound into every preamble.

use crate::GrindingPlan;
use akita_error::AkitaError;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::io::{Read, Write};

/// Active transcript proof-of-work, fold-response, and plan-digest binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscriptGrindingBinding {
    pub plan_digest: [u8; 32],
}

impl TranscriptGrindingBinding {
    /// Bind the active protocol policy to one validated public plan.
    pub fn for_plan(plan: &GrindingPlan) -> Result<Self, AkitaError> {
        Ok(Self {
            plan_digest: plan.digest()?,
        })
    }
}

impl Valid for TranscriptGrindingBinding {
    fn check(&self) -> Result<(), SerializationError> {
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
        writer.write_all(&self.plan_digest)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let _ = compress;
        32
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
        let mut plan_digest = [0u8; 32];
        reader.read_exact(&mut plan_digest)?;
        let binding = Self { plan_digest };
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
                GrindingRun::proof_of_work(GrindingSite::EvaluationBatch { level: 0 }, 1, 128)
                    .unwrap(),
                GrindingRun::fold_response(0),
            ],
            128,
        )
        .unwrap()
    }

    #[test]
    fn binding_serializes_exact_plan_digest() {
        let binding = TranscriptGrindingBinding::for_plan(&sample_plan()).unwrap();
        let mut bytes = Vec::new();
        binding.serialize_uncompressed(&mut bytes).unwrap();
        assert_eq!(bytes, sample_plan().digest().unwrap());
    }

    #[test]
    fn binding_roundtrip() {
        let binding = TranscriptGrindingBinding::for_plan(&sample_plan()).unwrap();
        let mut bytes = Vec::new();
        binding.serialize_uncompressed(&mut bytes).unwrap();
        assert_eq!(
            TranscriptGrindingBinding::deserialize_uncompressed_exact(&bytes, &()).unwrap(),
            binding
        );
    }
}
