use super::*;
impl<E: Field + AkitaSerialize> AkitaSerialize for PhysicalL2NormProof<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.response_l2_sq
            .serialize_with_mode(&mut writer, compress)?;
        for subclaim in &self.subclaims {
            subclaim.serialize_with_mode(&mut writer, compress)?;
        }
        for evaluation in &self.virtual_evaluations {
            evaluation.serialize_with_mode(&mut writer, compress)?;
        }
        self.sumcheck.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.response_l2_sq.serialized_size(compress)
            + self
                .subclaims
                .iter()
                .map(|claim| claim.serialized_size(compress))
                .sum::<usize>()
            + self
                .virtual_evaluations
                .iter()
                .map(|evaluation| evaluation.serialized_size(compress))
                .sum::<usize>()
            + self.sumcheck.serialized_size(compress)
    }
}

fn serialize_extension_opening_reduction<E, W>(
    reduction: Option<&ExtensionOpeningReductionProof<E>>,
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError>
where
    E: Field + AkitaSerialize,
    W: Write,
{
    if let Some(reduction) = reduction {
        for partial in &reduction.partials {
            partial.serialize_with_mode(&mut writer, compress)?;
        }
        reduction
            .sumcheck
            .serialize_with_mode(&mut writer, compress)?;
        for final_claim in &reduction.final_claims {
            final_claim.serialize_with_mode(&mut writer, compress)?;
        }
    }
    Ok(())
}

pub(super) fn extension_opening_reduction_serialized_size<E>(
    reduction: Option<&ExtensionOpeningReductionProof<E>>,
    compress: Compress,
) -> usize
where
    E: Field + AkitaSerialize,
{
    reduction.map_or(0, |reduction| {
        reduction
            .partials
            .iter()
            .map(|partial| partial.serialized_size(compress))
            .sum::<usize>()
            + { reduction.sumcheck.serialized_size(compress) }
            + reduction
                .final_claims
                .iter()
                .map(|claim| claim.serialized_size(compress))
                .sum::<usize>()
    })
}

fn serialize_next_witness_binding<F, W>(
    binding: &NextWitnessBinding<F>,
    writer: W,
    compress: Compress,
) -> Result<(), SerializationError>
where
    F: Field + AkitaSerialize,
    W: Write,
{
    match binding {
        NextWitnessBinding::OuterPayload(commitment) => {
            commitment.serialize_with_mode(writer, compress)
        }
        NextWitnessBinding::TerminalInnerState => Ok(()),
    }
}

fn next_witness_binding_serialized_size<F>(
    binding: &NextWitnessBinding<F>,
    compress: Compress,
) -> usize
where
    F: Field + AkitaSerialize,
{
    match binding {
        NextWitnessBinding::OuterPayload(commitment) => commitment.serialized_size(compress),
        NextWitnessBinding::TerminalInnerState => 0,
    }
}

fn serialize_intermediate_fold_wire_prefix<F, E, W>(
    mut writer: W,
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    opening_payload: &RingVec<F>,
    compress: Compress,
) -> Result<(), SerializationError>
where
    F: Field + AkitaSerialize,
    E: Field + AkitaSerialize,
    W: Write,
{
    serialize_extension_opening_reduction(extension_opening_reduction, &mut writer, compress)?;
    opening_payload.serialize_with_mode(writer, compress)
}

fn intermediate_fold_wire_prefix_serialized_size<F, E>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    opening_payload: &RingVec<F>,
    compress: Compress,
) -> usize
where
    F: Field + AkitaSerialize,
    E: Field + AkitaSerialize,
{
    extension_opening_reduction_serialized_size(extension_opening_reduction, compress)
        + opening_payload.serialized_size(compress)
}

fn serialize_terminal_fold_wire_prefix<E, W>(
    writer: W,
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    compress: Compress,
) -> Result<(), SerializationError>
where
    E: Field + AkitaSerialize,
    W: Write,
{
    serialize_extension_opening_reduction(extension_opening_reduction, writer, compress)
}

fn terminal_fold_wire_prefix_serialized_size<E>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    compress: Compress,
) -> usize
where
    E: Field + AkitaSerialize,
{
    extension_opening_reduction_serialized_size(extension_opening_reduction, compress)
}

fn serialize_stage3_sumcheck<E, W>(
    stage3_sumcheck: Option<&SetupSumcheckProof<E>>,
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError>
where
    E: Field + AkitaSerialize,
    W: Write,
{
    if let Some(stage3_sumcheck) = stage3_sumcheck {
        stage3_sumcheck
            .claim
            .serialize_with_mode(&mut writer, compress)?;
        stage3_sumcheck
            .setup_prefix_eval
            .serialize_with_mode(&mut writer, compress)?;
        stage3_sumcheck
            .sumcheck
            .serialize_with_mode(&mut writer, compress)?;
    }
    Ok(())
}

fn stage3_sumcheck_serialized_size<E>(
    stage3_sumcheck: Option<&SetupSumcheckProof<E>>,
    compress: Compress,
) -> usize
where
    E: Field + AkitaSerialize,
{
    stage3_sumcheck.map_or(0, |stage3_sumcheck| {
        stage3_sumcheck.claim.serialized_size(compress)
            + stage3_sumcheck.setup_prefix_eval.serialized_size(compress)
            + stage3_sumcheck.sumcheck.serialized_size(compress)
    })
}

impl<F: Field + CanonicalEncoding + AkitaSerialize, E: Field + AkitaSerialize> AkitaSerialize
    for TerminalLevelProof<F, E>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        serialize_terminal_fold_wire_prefix(
            &mut writer,
            self.extension_opening_reduction.as_ref(),
            compress,
        )?;
        self.terminal_response
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        terminal_fold_wire_prefix_serialized_size(
            self.extension_opening_reduction.as_ref(),
            compress,
        ) + self.terminal_response.serialized_size(compress)
    }
}

impl<F: Field + CanonicalEncoding + AkitaSerialize, E: Field + AkitaSerialize> AkitaSerialize
    for FoldLevelProof<F, E>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let stage2 = &self.stage2;
        serialize_intermediate_fold_wire_prefix(
            &mut writer,
            self.extension_opening_reduction.as_ref(),
            &self.opening_payload,
            compress,
        )?;
        for stage in &self.stage1.stages {
            stage
                .sumcheck_proof
                .serialize_with_mode(&mut writer, compress)?;
            for claim in &stage.child_claims {
                claim.serialize_with_mode(&mut writer, compress)?;
            }
        }
        self.stage1
            .range_image_evaluation
            .serialize_with_mode(&mut writer, compress)?;
        if let Some(norm) = &self.stage1.norm_proof {
            norm.serialize_with_mode(&mut writer, compress)?;
        }
        stage2
            .sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        serialize_stage3_sumcheck(self.stage3_sumcheck_proof.as_ref(), &mut writer, compress)?;
        serialize_next_witness_binding(&stage2.next_witness_binding, &mut writer, compress)?;
        stage2
            .next_w_eval
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let stage2 = &self.stage2;
        intermediate_fold_wire_prefix_serialized_size(
            self.extension_opening_reduction.as_ref(),
            &self.opening_payload,
            compress,
        ) + self
            .stage1
            .stages
            .iter()
            .map(|stage| {
                ({ stage.sumcheck_proof.serialized_size(compress) })
                    + stage
                        .child_claims
                        .iter()
                        .map(|claim| claim.serialized_size(compress))
                        .sum::<usize>()
            })
            .sum::<usize>()
            + self.stage1.range_image_evaluation.serialized_size(compress)
            + self
                .stage1
                .norm_proof
                .as_ref()
                .map_or(0, |norm| norm.serialized_size(compress))
            + ({ stage2.sumcheck_proof.serialized_size(compress) })
            + stage3_sumcheck_serialized_size(self.stage3_sumcheck_proof.as_ref(), compress)
            + next_witness_binding_serialized_size(&stage2.next_witness_binding, compress)
            + stage2.next_w_eval.serialized_size(compress)
    }
}
