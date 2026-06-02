use super::*;

fn serialize_extension_opening_reduction<L, W>(
    reduction: Option<&ExtensionOpeningReductionProof<L>>,
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError>
where
    L: FieldCore + AkitaSerialize,
    W: Write,
{
    if let Some(reduction) = reduction {
        for partial in &reduction.partials {
            partial.serialize_with_mode(&mut writer, compress)?;
        }
        #[cfg(not(feature = "zk"))]
        reduction
            .sumcheck
            .serialize_with_mode(&mut writer, compress)?;
        #[cfg(feature = "zk")]
        reduction
            .sumcheck_proof_masked
            .serialize_with_mode(&mut writer, compress)?;
    }
    Ok(())
}

pub(super) fn extension_opening_reduction_serialized_size<L>(
    reduction: Option<&ExtensionOpeningReductionProof<L>>,
    compress: Compress,
) -> usize
where
    L: FieldCore + AkitaSerialize,
{
    reduction.map_or(0, |reduction| {
        reduction
            .partials
            .iter()
            .map(|partial| partial.serialized_size(compress))
            .sum::<usize>()
            + {
                #[cfg(not(feature = "zk"))]
                {
                    reduction.sumcheck.serialized_size(compress)
                }
                #[cfg(feature = "zk")]
                {
                    reduction.sumcheck_proof_masked.serialized_size(compress)
                }
            }
    })
}

fn deserialize_extension_opening_reduction<L, R>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
    shape: Option<&ExtensionOpeningReductionShape>,
) -> Result<Option<ExtensionOpeningReductionProof<L>>, SerializationError>
where
    L: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    R: Read,
{
    let Some(shape) = shape else {
        return Ok(None);
    };
    shape.check()?;
    let mut partials = Vec::new();
    reserve_shape_len(&mut partials, shape.partials)?;
    for _ in 0..shape.partials {
        partials.push(L::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &(),
        )?);
    }
    #[cfg(not(feature = "zk"))]
    let sumcheck =
        SumcheckProof::deserialize_with_mode(&mut reader, compress, validate, &shape.sumcheck)?;
    #[cfg(feature = "zk")]
    let sumcheck_proof_masked = SumcheckProofMasked::deserialize_with_mode(
        &mut reader,
        compress,
        validate,
        &shape.sumcheck,
    )?;
    Ok(Some(ExtensionOpeningReductionProof {
        partials,
        #[cfg(not(feature = "zk"))]
        sumcheck,
        #[cfg(feature = "zk")]
        sumcheck_proof_masked,
    }))
}

fn serialize_stage3_sumcheck<L, W>(
    stage3_sumcheck: Option<&SetupSumcheckProof<L>>,
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError>
where
    L: FieldCore + AkitaSerialize,
    W: Write,
{
    if let Some(stage3_sumcheck) = stage3_sumcheck {
        stage3_sumcheck
            .claim
            .serialize_with_mode(&mut writer, compress)?;
        stage3_sumcheck
            .sumcheck
            .serialize_with_mode(&mut writer, compress)?;
    }
    Ok(())
}

fn stage3_sumcheck_serialized_size<L>(
    stage3_sumcheck: Option<&SetupSumcheckProof<L>>,
    compress: Compress,
) -> usize
where
    L: FieldCore + AkitaSerialize,
{
    stage3_sumcheck.map_or(0, |stage3_sumcheck| {
        stage3_sumcheck.claim.serialized_size(compress)
            + stage3_sumcheck.sumcheck.serialized_size(compress)
    })
}

fn deserialize_stage3_sumcheck<L, R>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
    shape: Option<&SetupProductSumcheckShape>,
) -> Result<Option<SetupSumcheckProof<L>>, SerializationError>
where
    L: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    R: Read,
{
    let Some(shape) = shape else {
        return Ok(None);
    };
    shape.check()?;
    let claim = L::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let sumcheck =
        SumcheckProof::deserialize_with_mode(&mut reader, compress, validate, &shape.sumcheck)?;
    Ok(Some(SetupSumcheckProof { claim, sumcheck }))
}

impl<F: FieldCore + AkitaSerialize, L: FieldCore + AkitaSerialize> AkitaSerialize
    for AkitaLevelProof<F, L>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.y_ring.serialize_with_mode(&mut writer, compress)?;
        serialize_extension_opening_reduction(
            self.extension_opening_reduction.as_ref(),
            &mut writer,
            compress,
        )?;
        self.v.serialize_with_mode(&mut writer, compress)?;
        for stage in &self.stage1.stages {
            #[cfg(not(feature = "zk"))]
            stage
                .sumcheck_proof
                .serialize_with_mode(&mut writer, compress)?;
            #[cfg(feature = "zk")]
            stage
                .sumcheck_proof_masked
                .serialize_with_mode(&mut writer, compress)?;
            for claim in &stage.child_claims {
                claim.serialize_with_mode(&mut writer, compress)?;
            }
        }
        self.stage1
            .s_claim
            .serialize_with_mode(&mut writer, compress)?;
        #[cfg(not(feature = "zk"))]
        self.stage2
            .sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        #[cfg(feature = "zk")]
        self.stage2
            .sumcheck_proof_masked
            .serialize_with_mode(&mut writer, compress)?;
        serialize_stage3_sumcheck(self.stage3_sumcheck_proof.as_ref(), &mut writer, compress)?;
        self.stage2
            .next_w_commitment
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2
            .next_w_eval()
            .serialize_with_mode(&mut writer, compress)
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        let base = self.y_ring.serialized_size(compress)
            + extension_opening_reduction_serialized_size(
                self.extension_opening_reduction.as_ref(),
                compress,
            )
            + self.v.serialized_size(compress);
        base + self
            .stage1
            .stages
            .iter()
            .map(|stage| {
                ({
                    #[cfg(not(feature = "zk"))]
                    {
                        stage.sumcheck_proof.serialized_size(compress)
                    }
                    #[cfg(feature = "zk")]
                    {
                        stage.sumcheck_proof_masked.serialized_size(compress)
                    }
                }) + stage
                    .child_claims
                    .iter()
                    .map(|claim| claim.serialized_size(compress))
                    .sum::<usize>()
            })
            .sum::<usize>()
            + self.stage1.s_claim.serialized_size(compress)
            + ({
                #[cfg(not(feature = "zk"))]
                {
                    self.stage2.sumcheck_proof.serialized_size(compress)
                }
                #[cfg(feature = "zk")]
                {
                    self.stage2.sumcheck_proof_masked.serialized_size(compress)
                }
            })
            + stage3_sumcheck_serialized_size(self.stage3_sumcheck_proof.as_ref(), compress)
            + self.stage2.next_w_commitment.serialized_size(compress)
            + self.stage2.next_w_eval().serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, L: FieldCore + Valid> Valid for AkitaLevelProof<F, L> {
    fn check(&self) -> Result<(), SerializationError> {
        self.y_ring.check()?;
        if self.y_ring.coeff_len() == 0 {
            return Err(SerializationError::InvalidData(
                "Akita level y_ring must contain exactly one ring element".to_string(),
            ));
        }
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction.partials.check()?;
            #[cfg(not(feature = "zk"))]
            reduction.sumcheck.check()?;
            #[cfg(feature = "zk")]
            reduction.sumcheck_proof_masked.check()?;
        }
        self.v.check()?;
        for stage in &self.stage1.stages {
            #[cfg(not(feature = "zk"))]
            stage.sumcheck_proof.check()?;
            #[cfg(feature = "zk")]
            stage.sumcheck_proof_masked.check()?;
            stage.child_claims.check()?;
        }
        self.stage1.s_claim.check()?;
        #[cfg(not(feature = "zk"))]
        self.stage2.sumcheck_proof.check()?;
        #[cfg(feature = "zk")]
        self.stage2.sumcheck_proof_masked.check()?;
        if let Some(stage3_sumcheck) = &self.stage3_sumcheck_proof {
            stage3_sumcheck.claim.check()?;
            stage3_sumcheck.sumcheck.check()?;
        }
        self.stage2.next_w_commitment.check()?;
        self.stage2.next_w_eval().check()
    }
}

impl<
        F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
        L: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    > AkitaDeserialize for AkitaLevelProof<F, L>
{
    type Context = LevelProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &LevelProofShape,
    ) -> Result<Self, SerializationError> {
        ctx.check()?;
        let y_ring = FlatRingVec::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.y_ring_coeffs,
        )?;
        let extension_opening_reduction = deserialize_extension_opening_reduction(
            &mut reader,
            compress,
            validate,
            ctx.extension_opening_reduction.as_ref(),
        )?;
        let v = FlatRingVec::deserialize_with_mode(&mut reader, compress, validate, &ctx.v_coeffs)?;
        let mut stage1_stages = Vec::new();
        reserve_shape_len(&mut stage1_stages, ctx.stage1_stages.len())?;
        for stage_shape in &ctx.stage1_stages {
            #[cfg(not(feature = "zk"))]
            let sumcheck = EqFactoredSumcheckProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &stage_shape.sumcheck_proof,
            )?;
            #[cfg(feature = "zk")]
            let sumcheck_proof_masked = EqFactoredSumcheckProofMasked::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &stage_shape.sumcheck_proof,
            )?;
            let mut child_claims = Vec::new();
            reserve_shape_len(&mut child_claims, stage_shape.child_claims)?;
            for _ in 0..stage_shape.child_claims {
                child_claims.push(L::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?);
            }
            stage1_stages.push(AkitaStage1StageProof {
                #[cfg(not(feature = "zk"))]
                sumcheck_proof: sumcheck,
                #[cfg(feature = "zk")]
                sumcheck_proof_masked,
                child_claims,
            });
        }
        let stage1 = AkitaStage1Proof {
            stages: stage1_stages,
            s_claim: L::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        #[cfg(not(feature = "zk"))]
        let stage2_sumcheck_proof = SumcheckProof::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.stage2_sumcheck_proof,
        )?;
        #[cfg(feature = "zk")]
        let stage2_sumcheck_proof_masked = SumcheckProofMasked::deserialize_with_mode(
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
            #[cfg(not(feature = "zk"))]
            sumcheck_proof: stage2_sumcheck_proof,
            #[cfg(feature = "zk")]
            sumcheck_proof_masked: stage2_sumcheck_proof_masked,
            next_w_commitment: FlatRingVec::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.next_commit_coeffs,
            )?,
            #[cfg(not(feature = "zk"))]
            next_w_eval: L::deserialize_with_mode(&mut reader, compress, validate, &())?,
            #[cfg(feature = "zk")]
            next_w_eval_masked: L::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        let out = Self {
            y_ring,
            extension_opening_reduction,
            v,
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

impl<F: FieldCore + AkitaSerialize, L: FieldCore + AkitaSerialize> AkitaSerialize
    for TerminalLevelProof<F, L>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.y_rings.serialize_with_mode(&mut writer, compress)?;
        serialize_extension_opening_reduction(
            self.extension_opening_reduction.as_ref(),
            &mut writer,
            compress,
        )?;
        #[cfg(not(feature = "zk"))]
        self.stage2_sumcheck
            .serialize_with_mode(&mut writer, compress)?;
        #[cfg(feature = "zk")]
        self.stage2_sumcheck_proof_masked
            .serialize_with_mode(&mut writer, compress)?;
        serialize_stage3_sumcheck(self.stage3_sumcheck_proof.as_ref(), &mut writer, compress)?;
        self.final_witness
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.y_rings.serialized_size(compress)
            + extension_opening_reduction_serialized_size(
                self.extension_opening_reduction.as_ref(),
                compress,
            )
            + {
                #[cfg(not(feature = "zk"))]
                {
                    self.stage2_sumcheck.serialized_size(compress)
                }
                #[cfg(feature = "zk")]
                {
                    self.stage2_sumcheck_proof_masked.serialized_size(compress)
                }
            }
            + stage3_sumcheck_serialized_size(self.stage3_sumcheck_proof.as_ref(), compress)
            + self.final_witness.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, L: FieldCore + Valid> Valid for TerminalLevelProof<F, L> {
    fn check(&self) -> Result<(), SerializationError> {
        self.y_rings.check()?;
        if self.y_rings.coeff_len() == 0 {
            return Err(SerializationError::InvalidData(
                "terminal level y_rings must contain at least one ring element".to_string(),
            ));
        }
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction.partials.check()?;
            #[cfg(not(feature = "zk"))]
            reduction.sumcheck.check()?;
            #[cfg(feature = "zk")]
            reduction.sumcheck_proof_masked.check()?;
        }
        #[cfg(not(feature = "zk"))]
        self.stage2_sumcheck.check()?;
        #[cfg(feature = "zk")]
        self.stage2_sumcheck_proof_masked.check()?;
        if let Some(stage3_sumcheck) = &self.stage3_sumcheck_proof {
            stage3_sumcheck.claim.check()?;
            stage3_sumcheck.sumcheck.check()?;
        }
        self.final_witness.check()
    }
}

impl<
        F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
        L: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    > AkitaDeserialize for TerminalLevelProof<F, L>
{
    type Context = TerminalLevelProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &TerminalLevelProofShape,
    ) -> Result<Self, SerializationError> {
        let y_rings = FlatRingVec::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.y_rings_coeffs,
        )?;
        let extension_opening_reduction = deserialize_extension_opening_reduction(
            &mut reader,
            compress,
            validate,
            ctx.extension_opening_reduction.as_ref(),
        )?;
        #[cfg(not(feature = "zk"))]
        let stage2_sumcheck = SumcheckProof::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.stage2_sumcheck,
        )?;
        #[cfg(feature = "zk")]
        let stage2_sumcheck_proof_masked = SumcheckProofMasked::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.stage2_sumcheck,
        )?;
        let stage3_sumcheck_proof = deserialize_stage3_sumcheck(
            &mut reader,
            compress,
            validate,
            ctx.stage3_sumcheck.as_ref(),
        )?;
        let final_witness = DirectWitnessProof::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.final_witness,
        )?;
        let out = Self {
            y_rings,
            extension_opening_reduction,
            #[cfg(not(feature = "zk"))]
            stage2_sumcheck,
            #[cfg(feature = "zk")]
            stage2_sumcheck_proof_masked,
            stage3_sumcheck_proof,
            final_witness,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + AkitaSerialize, L: FieldCore + AkitaSerialize> AkitaSerialize
    for AkitaProofStep<F, L>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::Intermediate(level) => level.serialize_with_mode(&mut writer, compress),
            Self::Terminal(terminal) => terminal.serialize_with_mode(&mut writer, compress),
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            Self::Intermediate(level) => level.serialized_size(compress),
            Self::Terminal(terminal) => terminal.serialized_size(compress),
        }
    }
}

impl<F: FieldCore + Valid, L: FieldCore + Valid> Valid for AkitaProofStep<F, L> {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Self::Intermediate(level) => level.check(),
            Self::Terminal(terminal) => terminal.check(),
        }
    }
}

impl<
        F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
        L: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    > AkitaDeserialize for AkitaProofStep<F, L>
{
    type Context = AkitaProofStepShape;

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &AkitaProofStepShape,
    ) -> Result<Self, SerializationError> {
        let out = match ctx {
            AkitaProofStepShape::Intermediate(shape) => Self::Intermediate(
                AkitaLevelProof::deserialize_with_mode(&mut reader, compress, validate, shape)?,
            ),
            AkitaProofStepShape::Terminal(shape) => Self::Terminal(
                TerminalLevelProof::deserialize_with_mode(&mut reader, compress, validate, shape)?,
            ),
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + AkitaSerialize, L: FieldCore + AkitaSerialize> AkitaSerialize
    for AkitaBatchedFoldRoot<F, L>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.y_rings.serialize_with_mode(&mut writer, compress)?;
        serialize_extension_opening_reduction(
            self.extension_opening_reduction.as_ref(),
            &mut writer,
            compress,
        )?;
        self.v.serialize_with_mode(&mut writer, compress)?;
        for stage in &self.stage1.stages {
            #[cfg(not(feature = "zk"))]
            stage
                .sumcheck_proof
                .serialize_with_mode(&mut writer, compress)?;
            #[cfg(feature = "zk")]
            stage
                .sumcheck_proof_masked
                .serialize_with_mode(&mut writer, compress)?;
            for claim in &stage.child_claims {
                claim.serialize_with_mode(&mut writer, compress)?;
            }
        }
        self.stage1
            .s_claim
            .serialize_with_mode(&mut writer, compress)?;
        #[cfg(not(feature = "zk"))]
        self.stage2
            .sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        #[cfg(feature = "zk")]
        self.stage2
            .sumcheck_proof_masked
            .serialize_with_mode(&mut writer, compress)?;
        serialize_stage3_sumcheck(self.stage3_sumcheck_proof.as_ref(), &mut writer, compress)?;
        self.stage2
            .next_w_commitment
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2
            .next_w_eval()
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.y_rings.serialized_size(compress)
            + extension_opening_reduction_serialized_size(
                self.extension_opening_reduction.as_ref(),
                compress,
            )
            + self.v.serialized_size(compress)
            + self
                .stage1
                .stages
                .iter()
                .map(|stage| {
                    ({
                        #[cfg(not(feature = "zk"))]
                        {
                            stage.sumcheck_proof.serialized_size(compress)
                        }
                        #[cfg(feature = "zk")]
                        {
                            stage.sumcheck_proof_masked.serialized_size(compress)
                        }
                    }) + stage
                        .child_claims
                        .iter()
                        .map(|claim| claim.serialized_size(compress))
                        .sum::<usize>()
                })
                .sum::<usize>()
            + self.stage1.s_claim.serialized_size(compress)
            + ({
                #[cfg(not(feature = "zk"))]
                {
                    self.stage2.sumcheck_proof.serialized_size(compress)
                }
                #[cfg(feature = "zk")]
                {
                    self.stage2.sumcheck_proof_masked.serialized_size(compress)
                }
            })
            + stage3_sumcheck_serialized_size(self.stage3_sumcheck_proof.as_ref(), compress)
            + self.stage2.next_w_commitment.serialized_size(compress)
            + self.stage2.next_w_eval().serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, L: FieldCore + Valid> Valid for AkitaBatchedFoldRoot<F, L> {
    fn check(&self) -> Result<(), SerializationError> {
        self.y_rings.check()?;
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction.partials.check()?;
            #[cfg(not(feature = "zk"))]
            reduction.sumcheck.check()?;
            #[cfg(feature = "zk")]
            reduction.sumcheck_proof_masked.check()?;
        }
        self.v.check()?;
        for stage in &self.stage1.stages {
            #[cfg(not(feature = "zk"))]
            stage.sumcheck_proof.check()?;
            #[cfg(feature = "zk")]
            stage.sumcheck_proof_masked.check()?;
            stage.child_claims.check()?;
        }
        self.stage1.s_claim.check()?;
        #[cfg(not(feature = "zk"))]
        self.stage2.sumcheck_proof.check()?;
        #[cfg(feature = "zk")]
        self.stage2.sumcheck_proof_masked.check()?;
        if let Some(stage3_sumcheck) = &self.stage3_sumcheck_proof {
            stage3_sumcheck.claim.check()?;
            stage3_sumcheck.sumcheck.check()?;
        }
        self.stage2.next_w_commitment.check()?;
        self.stage2.next_w_eval().check()
    }
}

impl<
        F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
        L: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    > AkitaDeserialize for AkitaBatchedFoldRoot<F, L>
{
    type Context = LevelProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &LevelProofShape,
    ) -> Result<Self, SerializationError> {
        ctx.check()?;
        let y_rings = FlatRingVec::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.y_ring_coeffs,
        )?;
        let extension_opening_reduction = deserialize_extension_opening_reduction(
            &mut reader,
            compress,
            validate,
            ctx.extension_opening_reduction.as_ref(),
        )?;
        let v = FlatRingVec::deserialize_with_mode(&mut reader, compress, validate, &ctx.v_coeffs)?;
        let mut stage1_stages = Vec::new();
        reserve_shape_len(&mut stage1_stages, ctx.stage1_stages.len())?;
        for stage_shape in &ctx.stage1_stages {
            #[cfg(not(feature = "zk"))]
            let sumcheck = EqFactoredSumcheckProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &stage_shape.sumcheck_proof,
            )?;
            #[cfg(feature = "zk")]
            let sumcheck_proof_masked = EqFactoredSumcheckProofMasked::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &stage_shape.sumcheck_proof,
            )?;
            let mut child_claims = Vec::new();
            reserve_shape_len(&mut child_claims, stage_shape.child_claims)?;
            for _ in 0..stage_shape.child_claims {
                child_claims.push(L::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?);
            }
            stage1_stages.push(AkitaStage1StageProof {
                #[cfg(not(feature = "zk"))]
                sumcheck_proof: sumcheck,
                #[cfg(feature = "zk")]
                sumcheck_proof_masked,
                child_claims,
            });
        }
        let stage1 = AkitaStage1Proof {
            stages: stage1_stages,
            s_claim: L::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        #[cfg(not(feature = "zk"))]
        let stage2_sumcheck_proof = SumcheckProof::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.stage2_sumcheck_proof,
        )?;
        #[cfg(feature = "zk")]
        let stage2_sumcheck_proof_masked = SumcheckProofMasked::deserialize_with_mode(
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
            #[cfg(not(feature = "zk"))]
            sumcheck_proof: stage2_sumcheck_proof,
            #[cfg(feature = "zk")]
            sumcheck_proof_masked: stage2_sumcheck_proof_masked,
            next_w_commitment: FlatRingVec::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.next_commit_coeffs,
            )?,
            #[cfg(not(feature = "zk"))]
            next_w_eval: L::deserialize_with_mode(&mut reader, compress, validate, &())?,
            #[cfg(feature = "zk")]
            next_w_eval_masked: L::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        let out = Self {
            y_rings,
            extension_opening_reduction,
            v,
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

impl<F: FieldCore + AkitaSerialize, L: FieldCore + AkitaSerialize> AkitaSerialize
    for AkitaBatchedRootProof<F, L>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::Fold(fold) => fold.serialize_with_mode(&mut writer, compress),
            Self::Terminal(terminal) => terminal.serialize_with_mode(&mut writer, compress),
            Self::Direct {
                witnesses,
                #[cfg(feature = "zk")]
                b_blinding_digits,
            } => {
                for witness in witnesses {
                    witness.serialize_with_mode(&mut writer, compress)?;
                }
                #[cfg(feature = "zk")]
                b_blinding_digits.serialize_with_mode(&mut writer, compress)?;
                Ok(())
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            Self::Fold(fold) => fold.serialized_size(compress),
            Self::Terminal(terminal) => terminal.serialized_size(compress),
            Self::Direct {
                witnesses,
                #[cfg(feature = "zk")]
                b_blinding_digits,
            } => {
                let witness_size = witnesses
                    .iter()
                    .map(|witness| witness.serialized_size(compress))
                    .sum::<usize>();
                #[cfg(feature = "zk")]
                {
                    witness_size + b_blinding_digits.serialized_size(compress)
                }
                #[cfg(not(feature = "zk"))]
                {
                    witness_size
                }
            }
        }
    }
}

impl<F: FieldCore + Valid, L: FieldCore + Valid> Valid for AkitaBatchedRootProof<F, L> {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Self::Fold(fold) => fold.check(),
            Self::Terminal(terminal) => terminal.check(),
            Self::Direct {
                witnesses,
                #[cfg(feature = "zk")]
                b_blinding_digits,
            } => {
                for witness in witnesses {
                    witness.check()?;
                }
                #[cfg(feature = "zk")]
                b_blinding_digits.check()?;
                Ok(())
            }
        }
    }
}

impl<F: FieldCore + AkitaSerialize, L: FieldCore + AkitaSerialize> AkitaSerialize
    for AkitaBatchedProof<F, L>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        #[cfg(feature = "zk")]
        self.zk_hiding.serialize_with_mode(&mut writer, compress)?;
        self.root.serialize_with_mode(&mut writer, compress)?;
        for step in &self.steps {
            step.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        #[cfg(feature = "zk")]
        let zk_size = self.zk_hiding.serialized_size(compress);
        #[cfg(not(feature = "zk"))]
        let zk_size = 0;
        zk_size
            + self.root.serialized_size(compress)
            + self
                .steps
                .iter()
                .map(|step| step.serialized_size(compress))
                .sum::<usize>()
    }
}

impl<F: FieldCore + Valid, L: FieldCore + Valid> Valid for AkitaBatchedProof<F, L> {
    fn check(&self) -> Result<(), SerializationError> {
        #[cfg(feature = "zk")]
        self.zk_hiding.check()?;
        self.root.check()?;
        for step in &self.steps {
            step.check()?;
        }
        match &self.root {
            AkitaBatchedRootProof::Fold(_) => {
                let Some(AkitaProofStep::Terminal(_)) = self.steps.last() else {
                    return Err(SerializationError::InvalidData(
                        "fold-rooted batched Akita proof must terminate with a terminal step"
                            .to_string(),
                    ));
                };
                if self.steps[..self.steps.len().saturating_sub(1)]
                    .iter()
                    .any(|step| !matches!(step, AkitaProofStep::Intermediate(_)))
                {
                    return Err(SerializationError::InvalidData(
                        "fold-rooted batched Akita proof may only contain intermediate steps before the terminal step"
                            .to_string(),
                    ));
                }
                // Headerless validity cannot infer the ring dimension from
                // `y_ring`: multipoint levels store one D-sized ring per
                // public row. Schedule-shaped deserialization and verifier
                // replay own the cross-level dimension checks.
            }
            AkitaBatchedRootProof::Terminal(_) => {
                if !self.steps.is_empty() {
                    return Err(SerializationError::InvalidData(
                        "terminal-rooted batched proof must not carry recursive-suffix steps"
                            .to_string(),
                    ));
                }
            }
            AkitaBatchedRootProof::Direct { .. } => {
                #[cfg(feature = "zk")]
                if !self.zk_hiding.is_empty() {
                    return Err(SerializationError::InvalidData(
                        "root-direct ZK hiding payload must be empty".to_string(),
                    ));
                }
                if !self.steps.is_empty() {
                    return Err(SerializationError::InvalidData(
                        "root-direct batched proof must not carry recursive-suffix steps"
                            .to_string(),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl<
        F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
        L: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    > AkitaDeserialize for AkitaBatchedProof<F, L>
{
    type Context = AkitaBatchedProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &AkitaBatchedProofShape,
    ) -> Result<Self, SerializationError> {
        ctx.check()?;
        #[cfg(feature = "zk")]
        let zk_hiding =
            ZkHidingProof::<F>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = match ctx {
            AkitaBatchedProofShape::Fold {
                root_shape,
                step_shapes,
            } => {
                let fold = AkitaBatchedFoldRoot::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    root_shape,
                )?;
                let mut steps = Vec::new();
                reserve_shape_len(&mut steps, step_shapes.len())?;
                for shape in step_shapes {
                    steps.push(AkitaProofStep::deserialize_with_mode(
                        &mut reader,
                        compress,
                        validate,
                        shape,
                    )?);
                }
                Self {
                    #[cfg(feature = "zk")]
                    zk_hiding,
                    root: AkitaBatchedRootProof::Fold(fold),
                    steps,
                }
            }
            AkitaBatchedProofShape::Terminal(terminal_shape) => {
                let terminal = TerminalLevelProof::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    terminal_shape,
                )?;
                Self {
                    #[cfg(feature = "zk")]
                    zk_hiding,
                    root: AkitaBatchedRootProof::Terminal(terminal),
                    steps: Vec::new(),
                }
            }
            AkitaBatchedProofShape::Direct { witness_shapes } => {
                let mut witnesses = Vec::new();
                reserve_shape_len(&mut witnesses, witness_shapes.len())?;
                for shape in witness_shapes {
                    witnesses.push(DirectWitnessProof::deserialize_with_mode(
                        &mut reader,
                        compress,
                        validate,
                        shape,
                    )?);
                }
                #[cfg(feature = "zk")]
                let b_blinding_digits =
                    Vec::<Vec<i8>>::deserialize_with_mode(&mut reader, compress, validate, &())?;
                Self {
                    #[cfg(feature = "zk")]
                    zk_hiding,
                    root: AkitaBatchedRootProof::Direct {
                        witnesses,
                        #[cfg(feature = "zk")]
                        b_blinding_digits,
                    },
                    steps: Vec::new(),
                }
            }
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}
