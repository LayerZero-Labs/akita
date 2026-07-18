use super::*;

fn serialize_extension_opening_reduction<E, W>(
    reduction: Option<&ExtensionOpeningReductionProof<E>>,
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError>
where
    E: FieldCore + AkitaSerialize,
    W: Write,
{
    if let Some(reduction) = reduction {
        for partial in &reduction.partials {
            partial.serialize_with_mode(&mut writer, compress)?;
        }
        reduction
            .sumcheck
            .serialize_with_mode(&mut writer, compress)?;
    }
    Ok(())
}

pub(super) fn extension_opening_reduction_serialized_size<E>(
    reduction: Option<&ExtensionOpeningReductionProof<E>>,
    compress: Compress,
) -> usize
where
    E: FieldCore + AkitaSerialize,
{
    reduction.map_or(0, |reduction| {
        reduction
            .partials
            .iter()
            .map(|partial| partial.serialized_size(compress))
            .sum::<usize>()
            + { reduction.sumcheck.serialized_size(compress) }
    })
}

fn deserialize_extension_opening_reduction<E, R>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
    shape: Option<&ExtensionOpeningReductionShape>,
) -> Result<Option<ExtensionOpeningReductionProof<E>>, SerializationError>
where
    E: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    R: Read,
{
    let Some(shape) = shape else {
        return Ok(None);
    };
    shape.check()?;
    let mut partials = Vec::new();
    reserve_shape_len(&mut partials, shape.partials)?;
    for _ in 0..shape.partials {
        partials.push(E::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &(),
        )?);
    }
    let sumcheck =
        SumcheckProof::deserialize_with_mode(&mut reader, compress, validate, &shape.sumcheck)?;
    Ok(Some(ExtensionOpeningReductionProof { partials, sumcheck }))
}

fn deserialize_fold_grind_nonce<R: Read>(
    reader: R,
    compress: Compress,
    validate: Validate,
) -> Result<u32, SerializationError> {
    u32::deserialize_with_mode(reader, compress, validate, &())
}

fn fold_grind_nonce_serialized_size(compress: Compress) -> usize {
    0u32.serialized_size(compress)
}

fn serialize_next_witness_binding<F, W>(
    binding: &NextWitnessBinding<F>,
    writer: W,
    compress: Compress,
) -> Result<(), SerializationError>
where
    F: FieldCore + AkitaSerialize,
    W: Write,
{
    match binding {
        NextWitnessBinding::OuterCommitment(commitment) => {
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
    F: FieldCore + AkitaSerialize,
{
    match binding {
        NextWitnessBinding::OuterCommitment(commitment) => commitment.serialized_size(compress),
        NextWitnessBinding::TerminalInnerState => 0,
    }
}

fn check_next_witness_binding<F: FieldCore + Valid>(
    binding: &NextWitnessBinding<F>,
) -> Result<(), SerializationError> {
    match binding {
        NextWitnessBinding::OuterCommitment(commitment) => commitment.check(),
        NextWitnessBinding::TerminalInnerState => Ok(()),
    }
}

fn deserialize_next_witness_binding<F, R>(
    reader: R,
    compress: Compress,
    validate: Validate,
    shape: NextWitnessBindingShape,
) -> Result<NextWitnessBinding<F>, SerializationError>
where
    F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    R: Read,
{
    match shape {
        NextWitnessBindingShape::OuterCommitment { coeffs } => {
            Ok(NextWitnessBinding::OuterCommitment(
                RingVec::deserialize_with_mode(reader, compress, validate, &coeffs)?,
            ))
        }
        NextWitnessBindingShape::TerminalInnerState => Ok(NextWitnessBinding::TerminalInnerState),
    }
}

fn serialize_intermediate_fold_wire_prefix<F, E, W>(
    mut writer: W,
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    v: &RingVec<F>,
    fold_grind_nonce: u32,
    compress: Compress,
) -> Result<(), SerializationError>
where
    F: FieldCore + AkitaSerialize,
    E: FieldCore + AkitaSerialize,
    W: Write,
{
    serialize_extension_opening_reduction(extension_opening_reduction, &mut writer, compress)?;
    v.serialize_with_mode(&mut writer, compress)?;
    fold_grind_nonce.serialize_with_mode(writer, compress)
}

fn intermediate_fold_wire_prefix_serialized_size<F, E>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    v: &RingVec<F>,
    compress: Compress,
) -> usize
where
    F: FieldCore + AkitaSerialize,
    E: FieldCore + AkitaSerialize,
{
    extension_opening_reduction_serialized_size(extension_opening_reduction, compress)
        + v.serialized_size(compress)
        + fold_grind_nonce_serialized_size(compress)
}

type IntermediateFoldWirePrefix<F, E> =
    (Option<ExtensionOpeningReductionProof<E>>, RingVec<F>, u32);

fn deserialize_intermediate_fold_wire_prefix<F, E, R>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
    extension_shape: Option<&ExtensionOpeningReductionShape>,
    v_shape: &<RingVec<F> as AkitaDeserialize>::Context,
) -> Result<IntermediateFoldWirePrefix<F, E>, SerializationError>
where
    F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    E: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    R: Read,
{
    let extension_opening_reduction =
        deserialize_extension_opening_reduction(&mut reader, compress, validate, extension_shape)?;
    let v = RingVec::deserialize_with_mode(&mut reader, compress, validate, v_shape)?;
    let fold_grind_nonce = deserialize_fold_grind_nonce(&mut reader, compress, validate)?;
    Ok((extension_opening_reduction, v, fold_grind_nonce))
}

fn serialize_terminal_fold_wire_prefix<E, W>(
    mut writer: W,
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    fold_grind_nonce: u32,
    compress: Compress,
) -> Result<(), SerializationError>
where
    E: FieldCore + AkitaSerialize,
    W: Write,
{
    serialize_extension_opening_reduction(extension_opening_reduction, &mut writer, compress)?;
    fold_grind_nonce.serialize_with_mode(writer, compress)
}

fn terminal_fold_wire_prefix_serialized_size<E>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    compress: Compress,
) -> usize
where
    E: FieldCore + AkitaSerialize,
{
    extension_opening_reduction_serialized_size(extension_opening_reduction, compress)
        + fold_grind_nonce_serialized_size(compress)
}

fn deserialize_terminal_fold_wire_prefix<E, R>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
    extension_shape: Option<&ExtensionOpeningReductionShape>,
) -> Result<(Option<ExtensionOpeningReductionProof<E>>, u32), SerializationError>
where
    E: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    R: Read,
{
    let extension_opening_reduction =
        deserialize_extension_opening_reduction(&mut reader, compress, validate, extension_shape)?;
    let fold_grind_nonce = deserialize_fold_grind_nonce(&mut reader, compress, validate)?;
    Ok((extension_opening_reduction, fold_grind_nonce))
}

fn serialize_stage3_sumcheck<E, W>(
    stage3_sumcheck: Option<&SetupSumcheckProof<E>>,
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError>
where
    E: FieldCore + AkitaSerialize,
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
            .next_w_eval
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
    E: FieldCore + AkitaSerialize,
{
    stage3_sumcheck.map_or(0, |stage3_sumcheck| {
        stage3_sumcheck.claim.serialized_size(compress)
            + stage3_sumcheck.setup_prefix_eval.serialized_size(compress)
            + stage3_sumcheck.next_w_eval.serialized_size(compress)
            + stage3_sumcheck.sumcheck.serialized_size(compress)
    })
}

fn deserialize_stage3_sumcheck<E, R>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
    shape: Option<&SetupProductSumcheckShape>,
) -> Result<Option<SetupSumcheckProof<E>>, SerializationError>
where
    E: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    R: Read,
{
    let Some(shape) = shape else {
        return Ok(None);
    };
    shape.check()?;
    let claim = E::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let setup_prefix_eval = E::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let next_w_eval = E::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let sumcheck =
        SumcheckProof::deserialize_with_mode(&mut reader, compress, validate, &shape.sumcheck)?;
    Ok(Some(SetupSumcheckProof {
        claim,
        setup_prefix_eval,
        next_w_eval,
        sumcheck,
    }))
}

impl<F: FieldCore + CanonicalField + AkitaSerialize, E: FieldCore + AkitaSerialize> AkitaSerialize
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
            self.fold_grind_nonce,
            compress,
        )?;
        self.final_witness
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        terminal_fold_wire_prefix_serialized_size(
            self.extension_opening_reduction.as_ref(),
            compress,
        ) + self.final_witness.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, E: FieldCore + Valid> Valid for TerminalLevelProof<F, E> {
    fn check(&self) -> Result<(), SerializationError> {
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction.partials.check()?;
            reduction.sumcheck.check()?;
        }
        self.final_witness.check()
    }
}

impl<
        F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
        E: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    > AkitaDeserialize for TerminalLevelProof<F, E>
{
    type Context = TerminalLevelProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &TerminalLevelProofShape,
    ) -> Result<Self, SerializationError> {
        ctx.check()?;
        let (extension_opening_reduction, fold_grind_nonce) =
            deserialize_terminal_fold_wire_prefix(
                &mut reader,
                compress,
                validate,
                ctx.extension_opening_reduction.as_ref(),
            )?;
        let final_witness = CleartextWitnessProof::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.final_witness,
        )?;
        let out = Self {
            extension_opening_reduction,
            fold_grind_nonce,
            final_witness,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + CanonicalField + AkitaSerialize, E: FieldCore + AkitaSerialize> AkitaSerialize
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
            &self.v,
            self.fold_grind_nonce,
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
            .s_claim
            .serialize_with_mode(&mut writer, compress)?;
        stage2
            .sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        serialize_stage3_sumcheck(self.stage3_sumcheck_proof.as_ref(), &mut writer, compress)?;
        serialize_next_witness_binding(&stage2.next_witness_binding, &mut writer, compress)?;
        stage2
            .next_w_eval()
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let stage2 = &self.stage2;
        intermediate_fold_wire_prefix_serialized_size(
            self.extension_opening_reduction.as_ref(),
            &self.v,
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
            + self.stage1.s_claim.serialized_size(compress)
            + ({ stage2.sumcheck_proof.serialized_size(compress) })
            + stage3_sumcheck_serialized_size(self.stage3_sumcheck_proof.as_ref(), compress)
            + next_witness_binding_serialized_size(&stage2.next_witness_binding, compress)
            + stage2.next_w_eval().serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, E: FieldCore + Valid> Valid for FoldLevelProof<F, E> {
    fn check(&self) -> Result<(), SerializationError> {
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction.partials.check()?;
            reduction.sumcheck.check()?;
        }
        self.v.check()?;
        for stage in &self.stage1.stages {
            stage.sumcheck_proof.check()?;
            stage.child_claims.check()?;
        }
        self.stage1.s_claim.check()?;
        let stage2 = &self.stage2;
        stage2.sumcheck_proof.check()?;
        if let Some(stage3_sumcheck) = &self.stage3_sumcheck_proof {
            stage3_sumcheck.claim.check()?;
            stage3_sumcheck.next_w_eval.check()?;
            stage3_sumcheck.sumcheck.check()?;
        }
        check_next_witness_binding(&stage2.next_witness_binding)?;
        stage2.next_w_eval().check()
    }
}

impl<
        F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
        E: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    > AkitaDeserialize for FoldLevelProof<F, E>
{
    type Context = LevelProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &LevelProofShape,
    ) -> Result<Self, SerializationError> {
        ctx.check()?;
        let (extension_opening_reduction, v, fold_grind_nonce) =
            deserialize_intermediate_fold_wire_prefix(
                &mut reader,
                compress,
                validate,
                ctx.extension_opening_reduction.as_ref(),
                &ctx.v_coeffs,
            )?;
        let mut stage1_stages = Vec::new();
        reserve_shape_len(&mut stage1_stages, ctx.stage1_stages.len())?;
        for stage_shape in &ctx.stage1_stages {
            let sumcheck = EqFactoredSumcheckProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &stage_shape.sumcheck_proof,
            )?;
            let mut child_claims = Vec::new();
            reserve_shape_len(&mut child_claims, stage_shape.child_claims)?;
            for _ in 0..stage_shape.child_claims {
                child_claims.push(E::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?);
            }
            stage1_stages.push(AkitaStage1StageProof {
                sumcheck_proof: sumcheck,
                child_claims,
            });
        }
        let stage1 = AkitaStage1Proof {
            stages: stage1_stages,
            s_claim: E::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        let stage2_sumcheck_proof = SumcheckProof::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.stage2_sumcheck_proof,
        )?;
        let stage3_sumcheck_proof = deserialize_stage3_sumcheck(
            &mut reader,
            compress,
            validate,
            ctx.stage3_sumcheck.as_ref(),
        )?;
        let stage2 = AkitaStage2Proof {
            sumcheck_proof: stage2_sumcheck_proof,
            next_witness_binding: deserialize_next_witness_binding(
                &mut reader,
                compress,
                validate,
                ctx.next_witness_binding,
            )?,
            next_w_eval: E::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        let out = Self {
            extension_opening_reduction,
            v,
            fold_grind_nonce,
            stage1,
            stage2,
            stage3_sumcheck_proof,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + CanonicalField + AkitaSerialize, E: FieldCore + AkitaSerialize> AkitaSerialize
    for AkitaBatchedProof<F, E>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.root.serialize_with_mode(&mut writer, compress)?;
        for fold in &self.recursive_folds {
            fold.serialize_with_mode(&mut writer, compress)?;
        }
        self.terminal.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.root.serialized_size(compress)
            + self
                .recursive_folds
                .iter()
                .map(|fold| fold.serialized_size(compress))
                .sum::<usize>()
            + self.terminal.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, E: FieldCore + Valid> Valid for AkitaBatchedProof<F, E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.root.check()?;
        for fold in &self.recursive_folds {
            fold.check()?;
        }
        self.terminal.check()
    }
}

impl<
        F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
        E: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    > AkitaDeserialize for AkitaBatchedProof<F, E>
{
    type Context = AkitaBatchedProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &AkitaBatchedProofShape,
    ) -> Result<Self, SerializationError> {
        ctx.check()?;
        let root =
            FoldLevelProof::deserialize_with_mode(&mut reader, compress, validate, &ctx.root)?;
        let mut recursive_folds = Vec::new();
        reserve_shape_len(&mut recursive_folds, ctx.recursive_folds.len())?;
        for shape in &ctx.recursive_folds {
            recursive_folds.push(FoldLevelProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                shape,
            )?);
        }
        let terminal = TerminalLevelProof::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.terminal,
        )?;
        let out = Self {
            root,
            recursive_folds,
            terminal,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}
