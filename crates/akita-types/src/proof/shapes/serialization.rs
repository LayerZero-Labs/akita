use super::*;

// === Headerless shape (de)serialization ===
//
// These impls let callers bundle proof shapes alongside proofs (e.g. when
// shipping verifier inputs to a Jolt guest program), so that the proof can be
// deserialized in environments that don't reconstruct a `FoldSchedule` first.

fn deserialize_shape_vec<T, R: Read>(
    reader: &mut R,
    compress: Compress,
    validate: Validate,
) -> Result<Vec<T>, SerializationError>
where
    T: AkitaDeserialize<Context = ()>,
{
    let encoded_len = u64::deserialize_with_mode(&mut *reader, compress, validate, &())?;
    let len =
        usize::try_from(encoded_len).map_err(|_| SerializationError::LengthLimitExceeded {
            len: encoded_len,
            max: usize::MAX,
        })?;
    if matches!(validate, Validate::Yes) {
        checked_shape_sequence_len(len)?;
    }

    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| SerializationError::InvalidData("shape-backed allocation failed".into()))?;
    for _ in 0..len {
        out.push(T::deserialize_with_mode(
            &mut *reader,
            compress,
            validate,
            &(),
        )?);
    }
    Ok(out)
}

impl Valid for AkitaStage1StageShape {
    fn check(&self) -> Result<(), SerializationError> {
        checked_shape_len(self.sumcheck_proof.0)?;
        checked_shape_len(self.sumcheck_proof.1)?;
        checked_shape_len(self.child_claims)?;
        Ok(())
    }
}

impl AkitaSerialize for AkitaStage1StageShape {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let (rounds, degree) = self.sumcheck_proof;
        rounds.serialize_with_mode(&mut writer, compress)?;
        degree.serialize_with_mode(&mut writer, compress)?;
        self.child_claims
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let (rounds, degree) = self.sumcheck_proof;
        rounds.serialized_size(compress)
            + degree.serialized_size(compress)
            + self.child_claims.serialized_size(compress)
    }
}

impl AkitaDeserialize for AkitaStage1StageShape {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let rounds = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let degree = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let child_claims = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            sumcheck_proof: (rounds, degree),
            child_claims,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for PhysicalL2NormProofWireShape {
    fn check(&self) -> Result<(), SerializationError> {
        checked_shape_len(self.subclaims)?;
        checked_shape_len(self.virtual_evaluations)?;
        if self.virtual_evaluations == 0 {
            return Err(SerializationError::InvalidData(
                "L2 norm proof shape requires a virtual evaluation".into(),
            ));
        }
        checked_shape_sequence_len(self.sumcheck.len())?;
        for &degree in &self.sumcheck {
            checked_shape_len(degree)?;
        }
        Ok(())
    }
}

impl AkitaSerialize for PhysicalL2NormProofWireShape {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.subclaims.serialize_with_mode(&mut writer, compress)?;
        self.virtual_evaluations
            .serialize_with_mode(&mut writer, compress)?;
        self.sumcheck.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.subclaims.serialized_size(compress)
            + self.virtual_evaluations.serialized_size(compress)
            + self.sumcheck.serialized_size(compress)
    }
}

impl AkitaDeserialize for PhysicalL2NormProofWireShape {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let out = Self {
            subclaims: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            virtual_evaluations: usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
            sumcheck: deserialize_shape_vec(&mut reader, compress, validate)?,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for LevelProofShape {
    fn check(&self) -> Result<(), SerializationError> {
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction.check()?;
        }
        checked_shape_len(self.opening_payload_coeffs)?;
        checked_shape_sequence_len(self.stage1_stages.len())?;
        self.stage1_stages.check()?;
        if let Some(shape) = &self.stage1_norm {
            shape.check()?;
        }
        checked_shape_sequence_len(self.stage2_sumcheck_proof.len())?;
        for &degree in &self.stage2_sumcheck_proof {
            checked_shape_len(degree)?;
        }
        if let Some(shape) = &self.stage3_sumcheck {
            shape.check()?;
        }
        if let NextWitnessBindingShape::OuterPayload { coeffs } = self.next_witness_binding {
            checked_shape_len(coeffs)?;
        }
        Ok(())
    }
}

impl AkitaSerialize for LevelProofShape {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.extension_opening_reduction
            .is_some()
            .serialize_with_mode(&mut writer, compress)?;
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction
                .partials
                .serialize_with_mode(&mut writer, compress)?;
            reduction
                .final_claims
                .serialize_with_mode(&mut writer, compress)?;
            reduction
                .sumcheck
                .serialize_with_mode(&mut writer, compress)?;
        }
        self.opening_payload_coeffs
            .serialize_with_mode(&mut writer, compress)?;
        self.stage1_stages
            .serialize_with_mode(&mut writer, compress)?;
        self.stage1_norm
            .is_some()
            .serialize_with_mode(&mut writer, compress)?;
        if let Some(stage1_norm) = &self.stage1_norm {
            stage1_norm.serialize_with_mode(&mut writer, compress)?;
        }
        self.stage2_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage3_sumcheck
            .is_some()
            .serialize_with_mode(&mut writer, compress)?;
        if let Some(stage3_sumcheck) = &self.stage3_sumcheck {
            stage3_sumcheck
                .sumcheck
                .serialize_with_mode(&mut writer, compress)?;
        }
        match self.next_witness_binding {
            NextWitnessBindingShape::OuterPayload { coeffs } => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                coeffs.serialize_with_mode(&mut writer, compress)?;
            }
            NextWitnessBindingShape::TerminalInnerState => {
                1u8.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let reduction_size = true.serialized_size(compress)
            + self
                .extension_opening_reduction
                .as_ref()
                .map_or(0, |reduction| {
                    reduction.partials.serialized_size(compress)
                        + reduction.final_claims.serialized_size(compress)
                        + reduction.sumcheck.serialized_size(compress)
                });
        reduction_size
            + self.opening_payload_coeffs.serialized_size(compress)
            + self.stage1_stages.serialized_size(compress)
            + true.serialized_size(compress)
            + self
                .stage1_norm
                .as_ref()
                .map_or(0, |shape| shape.serialized_size(compress))
            + self.stage2_sumcheck_proof.serialized_size(compress)
            + true.serialized_size(compress)
            + self
                .stage3_sumcheck
                .as_ref()
                .map_or(0, |shape| shape.sumcheck.serialized_size(compress))
            + 0u8.serialized_size(compress)
            + match self.next_witness_binding {
                NextWitnessBindingShape::OuterPayload { coeffs } => {
                    coeffs.serialized_size(compress)
                }
                NextWitnessBindingShape::TerminalInnerState => 0,
            }
    }
}

impl AkitaDeserialize for LevelProofShape {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let has_extension_opening_reduction =
            bool::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let extension_opening_reduction = if has_extension_opening_reduction {
            let partials = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
            let final_claims = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
            let sumcheck = deserialize_shape_vec(&mut reader, compress, validate)?;
            Some(ExtensionOpeningReductionShape {
                partials,
                final_claims,
                sumcheck,
            })
        } else {
            None
        };
        let opening_payload_coeffs =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let stage1_stages = deserialize_shape_vec(&mut reader, compress, validate)?;
        let has_stage1_norm = bool::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let stage1_norm = if has_stage1_norm {
            Some(PhysicalL2NormProofWireShape::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?)
        } else {
            None
        };
        let stage2_sumcheck = deserialize_shape_vec(&mut reader, compress, validate)?;
        let has_stage3_sumcheck =
            bool::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let stage3_sumcheck = if has_stage3_sumcheck {
            Some(SetupProductSumcheckShape {
                sumcheck: deserialize_shape_vec(&mut reader, compress, validate)?,
            })
        } else {
            None
        };
        let next_witness_binding =
            match u8::deserialize_with_mode(&mut reader, compress, validate, &())? {
                0 => NextWitnessBindingShape::OuterPayload {
                    coeffs: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
                },
                1 => NextWitnessBindingShape::TerminalInnerState,
                tag => {
                    return Err(SerializationError::InvalidData(format!(
                        "invalid next-witness binding shape tag {tag}"
                    )))
                }
            };
        let out = Self {
            extension_opening_reduction,
            opening_payload_coeffs,
            stage1_stages,
            stage1_norm,
            stage2_sumcheck_proof: stage2_sumcheck,
            stage3_sumcheck,
            next_witness_binding,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for TerminalLevelProofShape {
    fn check(&self) -> Result<(), SerializationError> {
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction.check()?;
        }
        self.terminal_response.check()?;
        Ok(())
    }
}

impl AkitaSerialize for TerminalLevelProofShape {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.extension_opening_reduction
            .is_some()
            .serialize_with_mode(&mut writer, compress)?;
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction
                .partials
                .serialize_with_mode(&mut writer, compress)?;
            reduction
                .final_claims
                .serialize_with_mode(&mut writer, compress)?;
            reduction
                .sumcheck
                .serialize_with_mode(&mut writer, compress)?;
        }
        self.terminal_response
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let reduction_size = true.serialized_size(compress)
            + self
                .extension_opening_reduction
                .as_ref()
                .map_or(0, |reduction| {
                    reduction.partials.serialized_size(compress)
                        + reduction.final_claims.serialized_size(compress)
                        + reduction.sumcheck.serialized_size(compress)
                });
        reduction_size + self.terminal_response.serialized_size(compress)
    }
}

impl AkitaDeserialize for TerminalLevelProofShape {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let has_extension_opening_reduction =
            bool::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let extension_opening_reduction = if has_extension_opening_reduction {
            let partials = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
            let final_claims = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
            let sumcheck = deserialize_shape_vec(&mut reader, compress, validate)?;
            Some(ExtensionOpeningReductionShape {
                partials,
                final_claims,
                sumcheck,
            })
        } else {
            None
        };
        let terminal_response =
            TerminalResponseShape::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            extension_opening_reduction,
            terminal_response,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}
