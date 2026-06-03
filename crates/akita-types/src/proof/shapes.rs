use super::*;

/// Headerless shape context for one stage in the stage-1 range-check tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AkitaStage1StageShape {
    /// Eq-factored sumcheck shape `(num_rounds, q_degree)`.
    pub sumcheck_proof: EqFactoredSumcheckProofShape,
    /// Number of child claims serialized after the stage proof.
    pub child_claims: usize,
}

/// Headerless shape for [`ExtensionOpeningReductionProof`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionOpeningReductionShape {
    /// Number of partial evaluations serialized before the sumcheck.
    pub partials: usize,
    /// Reduction sumcheck shape: one compact coefficient count per round.
    pub sumcheck: SumcheckProofShape,
}

impl ExtensionOpeningReductionShape {
    /// Construct the standard degree-two reduction shape.
    pub fn standard(partials: usize, num_rounds: usize) -> Self {
        Self {
            partials,
            sumcheck: uniform_sumcheck_shape(num_rounds, EXTENSION_OPENING_REDUCTION_DEGREE),
        }
    }
}

impl Valid for ExtensionOpeningReductionShape {
    fn check(&self) -> Result<(), SerializationError> {
        checked_shape_len(self.partials)?;
        checked_shape_len(self.sumcheck.len())?;
        for &degree in &self.sumcheck {
            checked_shape_len(degree)?;
            if degree != EXTENSION_OPENING_REDUCTION_DEGREE {
                return Err(SerializationError::InvalidData(format!(
                    "extension opening reduction degree {} does not match expected degree {}",
                    degree, EXTENSION_OPENING_REDUCTION_DEGREE
                )));
            }
        }
        Ok(())
    }
}

/// Shape descriptor for deserializing a [`TerminalLevelProof`] without
/// headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalLevelProofShape {
    /// Number of field coefficients in `y_rings`.
    pub y_rings_coeffs: usize,
    /// Shape of the optional extension-opening reduction payload.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionShape>,
    /// Stage-2 sumcheck shape: one compact coefficient count per round.
    pub stage2_sumcheck: SumcheckProofShape,
    /// Shape of the terminal cleartext witness.
    pub final_witness: CleartextWitnessShape,
}

/// Shape descriptor for deserializing a [`AkitaLevelProof`] without headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelProofShape {
    /// Number of field coefficients in `y_ring`.
    pub y_ring_coeffs: usize,
    /// Shape of the optional extension-opening reduction payload.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionShape>,
    /// Number of field coefficients in `v`.
    pub v_coeffs: usize,
    /// Stage-1 tree stage shapes in root-to-leaf order.
    pub stage1_stages: Vec<AkitaStage1StageShape>,
    /// Stage-2 sumcheck shape: `(num_rounds, degree)`.
    pub stage2_sumcheck_proof: SumcheckProofShape,
    /// Number of field coefficients in `next_w_commitment`.
    pub next_commit_coeffs: usize,
}

/// Shape descriptor for deserializing an [`AkitaBatchedProof`] without
/// headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkitaBatchedProofShape {
    /// Standard fold-rooted batched proof with a recursive suffix. The
    /// recursive suffix is a (possibly empty) sequence of
    /// [`AkitaProofStepShape::Intermediate`] step shapes followed by exactly
    /// one [`AkitaProofStepShape::Terminal`].
    Fold {
        /// Root-level shape (same field layout as a regular level).
        root_shape: LevelProofShape,
        /// Recursive proof step shapes following the batched root level.
        step_shapes: Vec<AkitaProofStepShape>,
    },
    /// Terminal-rooted batched proof (1-fold case): the root is itself the
    /// terminal fold level and no steps follow.
    Terminal(TerminalLevelProofShape),
    /// Zero-fold batched proof: one cleartext witness per claim.
    ZeroFold {
        /// Per-claim cleartext witness shapes.
        witness_shapes: Vec<CleartextWitnessShape>,
    },
}

/// Shape descriptor for deserializing a proof step without headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkitaProofStepShape {
    /// Shape of an intermediate fold level.
    Intermediate(LevelProofShape),
    /// Shape of the terminal fold level.
    Terminal(TerminalLevelProofShape),
}

#[cfg(not(feature = "zk"))]
pub(super) fn sumcheck_shape<F: FieldCore>(sc: &SumcheckProof<F>) -> SumcheckProofShape {
    sc.round_polys
        .iter()
        .map(|p| p.coeffs_except_linear_term.len())
        .collect()
}

#[cfg(feature = "zk")]
pub(super) fn sumcheck_proof_masked_shape<F: FieldCore>(
    masks: &SumcheckProofMasked<F>,
) -> SumcheckProofShape {
    masks
        .masked_round_polys
        .iter()
        .map(|p| p.coeffs_except_linear_term.len())
        .collect()
}

#[cfg(not(feature = "zk"))]
fn eq_factored_sumcheck_shape<F: FieldCore>(
    sc: &EqFactoredSumcheckProof<F>,
) -> EqFactoredSumcheckProofShape {
    let degree = sc
        .round_polys
        .first()
        .map_or(0, |p| p.coeffs_except_linear_term.len());
    (sc.round_polys.len(), degree)
}

#[cfg(feature = "zk")]
fn eq_factored_sumcheck_proof_masked_shape<F: FieldCore>(
    masks: &EqFactoredSumcheckProofMasked<F>,
) -> EqFactoredSumcheckProofShape {
    let degree = masks
        .masked_round_polys
        .first()
        .map_or(0, |p| p.coeffs_except_linear_term.len());
    (masks.masked_round_polys.len(), degree)
}

pub(super) fn level_proof_shape<F: FieldCore, L: FieldCore>(
    y_coeffs: usize,
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<L>>,
    v: &FlatRingVec<F>,
    stage1: &AkitaStage1Proof<L>,
    stage2: &AkitaStage2Proof<F, L>,
) -> LevelProofShape {
    LevelProofShape {
        y_ring_coeffs: y_coeffs,
        extension_opening_reduction: extension_opening_reduction
            .map(ExtensionOpeningReductionProof::shape),
        v_coeffs: v.coeff_len(),
        stage1_stages: stage1
            .stages
            .iter()
            .map(|stage| AkitaStage1StageShape {
                #[cfg(not(feature = "zk"))]
                sumcheck_proof: eq_factored_sumcheck_shape(&stage.sumcheck_proof),
                #[cfg(feature = "zk")]
                sumcheck_proof: eq_factored_sumcheck_proof_masked_shape(
                    &stage.sumcheck_proof_masked,
                ),
                child_claims: stage.child_claims.len(),
            })
            .collect(),
        #[cfg(not(feature = "zk"))]
        stage2_sumcheck_proof: sumcheck_shape(&stage2.sumcheck_proof),
        #[cfg(feature = "zk")]
        stage2_sumcheck_proof: sumcheck_proof_masked_shape(&stage2.sumcheck_proof_masked),
        next_commit_coeffs: stage2.next_w_commitment.coeff_len(),
    }
}

// === Headerless shape (de)serialization ===
//
// These impls let callers bundle proof shapes alongside proofs (e.g. when
// shipping verifier inputs to a Jolt guest program), so that the proof can be
// deserialized in environments that don't reconstruct a `Schedule` first.

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

impl Valid for LevelProofShape {
    fn check(&self) -> Result<(), SerializationError> {
        checked_shape_len(self.y_ring_coeffs)?;
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction.check()?;
        }
        checked_shape_len(self.v_coeffs)?;
        checked_shape_len(self.stage1_stages.len())?;
        self.stage1_stages.check()?;
        checked_shape_len(self.stage2_sumcheck_proof.len())?;
        for &degree in &self.stage2_sumcheck_proof {
            checked_shape_len(degree)?;
        }
        checked_shape_len(self.next_commit_coeffs)?;
        Ok(())
    }
}

impl AkitaSerialize for LevelProofShape {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.y_ring_coeffs
            .serialize_with_mode(&mut writer, compress)?;
        self.extension_opening_reduction
            .is_some()
            .serialize_with_mode(&mut writer, compress)?;
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction
                .partials
                .serialize_with_mode(&mut writer, compress)?;
            reduction
                .sumcheck
                .serialize_with_mode(&mut writer, compress)?;
        }
        self.v_coeffs.serialize_with_mode(&mut writer, compress)?;
        self.stage1_stages
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.next_commit_coeffs
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
                        + reduction.sumcheck.serialized_size(compress)
                });
        self.y_ring_coeffs.serialized_size(compress)
            + reduction_size
            + self.v_coeffs.serialized_size(compress)
            + self.stage1_stages.serialized_size(compress)
            + self.stage2_sumcheck_proof.serialized_size(compress)
            + self.next_commit_coeffs.serialized_size(compress)
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
        let y_ring_coeffs = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let has_extension_opening_reduction =
            bool::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let extension_opening_reduction = if has_extension_opening_reduction {
            let partials = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
            let sumcheck =
                SumcheckProofShape::deserialize_with_mode(&mut reader, compress, validate, &())?;
            Some(ExtensionOpeningReductionShape { partials, sumcheck })
        } else {
            None
        };
        let v_coeffs = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let stage1_stages = Vec::<AkitaStage1StageShape>::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &(),
        )?;
        let stage2_sumcheck =
            SumcheckProofShape::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let next_commit_coeffs =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            y_ring_coeffs,
            extension_opening_reduction,
            v_coeffs,
            stage1_stages,
            stage2_sumcheck_proof: stage2_sumcheck,
            next_commit_coeffs,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for CleartextWitnessShape {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Self::PackedDigits((num_elems, bits_per_elem)) => {
                if *bits_per_elem == 0 || *bits_per_elem > 6 {
                    return Err(SerializationError::InvalidData(
                        "bits_per_elem out of range".to_string(),
                    ));
                }
                checked_shape_len(*num_elems)?;
            }
            Self::FieldElements(coeff_len) => checked_shape_len(*coeff_len)?,
        }
        Ok(())
    }
}

impl AkitaSerialize for CleartextWitnessShape {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::PackedDigits((num_elems, bits_per_elem)) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                num_elems.serialize_with_mode(&mut writer, compress)?;
                bits_per_elem.serialize_with_mode(&mut writer, compress)?;
            }
            Self::FieldElements(coeff_len) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                coeff_len.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let tag = 1usize;
        match self {
            Self::PackedDigits((num_elems, bits_per_elem)) => {
                tag + num_elems.serialized_size(compress) + bits_per_elem.serialized_size(compress)
            }
            Self::FieldElements(coeff_len) => tag + coeff_len.serialized_size(compress),
        }
    }
}

impl AkitaDeserialize for CleartextWitnessShape {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = match tag {
            0 => {
                let num_elems = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
                let bits_per_elem =
                    u32::deserialize_with_mode(&mut reader, compress, validate, &())?;
                Self::PackedDigits((num_elems, bits_per_elem))
            }
            1 => {
                let coeff_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
                Self::FieldElements(coeff_len)
            }
            other => {
                return Err(SerializationError::InvalidData(format!(
                    "unknown CleartextWitnessShape tag {other}"
                )))
            }
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for TerminalLevelProofShape {
    fn check(&self) -> Result<(), SerializationError> {
        checked_shape_len(self.y_rings_coeffs)?;
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction.check()?;
        }
        checked_shape_len(self.stage2_sumcheck.len())?;
        for &degree in &self.stage2_sumcheck {
            checked_shape_len(degree)?;
        }
        self.final_witness.check()?;
        Ok(())
    }
}

impl AkitaSerialize for TerminalLevelProofShape {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.y_rings_coeffs
            .serialize_with_mode(&mut writer, compress)?;
        self.extension_opening_reduction
            .is_some()
            .serialize_with_mode(&mut writer, compress)?;
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction
                .partials
                .serialize_with_mode(&mut writer, compress)?;
            reduction
                .sumcheck
                .serialize_with_mode(&mut writer, compress)?;
        }
        self.stage2_sumcheck
            .serialize_with_mode(&mut writer, compress)?;
        self.final_witness
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
                        + reduction.sumcheck.serialized_size(compress)
                });
        self.y_rings_coeffs.serialized_size(compress)
            + reduction_size
            + self.stage2_sumcheck.serialized_size(compress)
            + self.final_witness.serialized_size(compress)
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
        let y_rings_coeffs = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let has_extension_opening_reduction =
            bool::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let extension_opening_reduction = if has_extension_opening_reduction {
            let partials = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
            let sumcheck =
                SumcheckProofShape::deserialize_with_mode(&mut reader, compress, validate, &())?;
            Some(ExtensionOpeningReductionShape { partials, sumcheck })
        } else {
            None
        };
        let stage2_sumcheck =
            SumcheckProofShape::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let final_witness =
            CleartextWitnessShape::deserialize_with_mode(&mut reader, compress, validate, &())?;
        Ok(Self {
            y_rings_coeffs,
            extension_opening_reduction,
            stage2_sumcheck,
            final_witness,
        })
    }
}

impl Valid for AkitaProofStepShape {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Self::Intermediate(level) => level.check()?,
            Self::Terminal(terminal) => terminal.check()?,
        }
        Ok(())
    }
}

impl AkitaSerialize for AkitaProofStepShape {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::Intermediate(level) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                level.serialize_with_mode(&mut writer, compress)?;
            }
            Self::Terminal(terminal) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                terminal.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + match self {
            Self::Intermediate(level) => level.serialized_size(compress),
            Self::Terminal(terminal) => terminal.serialized_size(compress),
        }
    }
}

impl AkitaDeserialize for AkitaProofStepShape {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = match tag {
            0 => Self::Intermediate(LevelProofShape::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?),
            1 => Self::Terminal(TerminalLevelProofShape::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?),
            other => {
                return Err(SerializationError::InvalidData(format!(
                    "unknown AkitaProofStepShape tag {other}"
                )))
            }
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for AkitaBatchedProofShape {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Self::Fold {
                root_shape,
                step_shapes,
            } => {
                root_shape.check()?;
                checked_shape_len(step_shapes.len())?;
                step_shapes.check()?;
            }
            Self::Terminal(terminal) => {
                terminal.check()?;
            }
            Self::ZeroFold { witness_shapes } => {
                checked_shape_len(witness_shapes.len())?;
                witness_shapes.check()?;
            }
        }
        Ok(())
    }
}

impl AkitaSerialize for AkitaBatchedProofShape {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::Fold {
                root_shape,
                step_shapes,
            } => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                root_shape.serialize_with_mode(&mut writer, compress)?;
                step_shapes.serialize_with_mode(&mut writer, compress)?;
            }
            Self::Terminal(terminal_shape) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                terminal_shape.serialize_with_mode(&mut writer, compress)?;
            }
            Self::ZeroFold { witness_shapes } => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                witness_shapes.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + match self {
            Self::Fold {
                root_shape,
                step_shapes,
            } => root_shape.serialized_size(compress) + step_shapes.serialized_size(compress),
            Self::Terminal(terminal_shape) => terminal_shape.serialized_size(compress),
            Self::ZeroFold { witness_shapes } => witness_shapes.serialized_size(compress),
        }
    }
}

impl AkitaDeserialize for AkitaBatchedProofShape {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate, &())?;
        match tag {
            0 => {
                let root_shape =
                    LevelProofShape::deserialize_with_mode(&mut reader, compress, validate, &())?;
                let step_shapes = Vec::<AkitaProofStepShape>::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?;
                let out = Self::Fold {
                    root_shape,
                    step_shapes,
                };
                if matches!(validate, Validate::Yes) {
                    out.check()?;
                }
                Ok(out)
            }
            1 => {
                let terminal_shape = TerminalLevelProofShape::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?;
                let out = Self::Terminal(terminal_shape);
                if matches!(validate, Validate::Yes) {
                    out.check()?;
                }
                Ok(out)
            }
            2 => {
                let witness_shapes = Vec::<CleartextWitnessShape>::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?;
                let out = Self::ZeroFold { witness_shapes };
                if matches!(validate, Validate::Yes) {
                    out.check()?;
                }
                Ok(out)
            }
            other => Err(SerializationError::InvalidData(format!(
                "unknown AkitaBatchedProofShape tag {other}"
            ))),
        }
    }
}
