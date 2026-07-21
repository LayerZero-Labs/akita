use super::*;

/// Degree bound for the setup-product sumcheck (`S(lambda, y) * omega(lambda) * alpha(y)`).
pub const SETUP_SUMCHECK_DEGREE: usize = 2;

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

/// Headerless shape for [`SetupSumcheckProof`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupProductSumcheckShape {
    /// Product-sumcheck shape: one compact coefficient count per round.
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

impl Valid for SetupProductSumcheckShape {
    fn check(&self) -> Result<(), SerializationError> {
        checked_shape_sequence_len(self.sumcheck.len())?;
        for &degree in &self.sumcheck {
            checked_shape_len(degree)?;
            if degree != SETUP_SUMCHECK_DEGREE {
                return Err(SerializationError::InvalidData(format!(
                    "setup product sumcheck degree {} does not match expected degree {}",
                    degree, SETUP_SUMCHECK_DEGREE
                )));
            }
        }
        Ok(())
    }
}

impl Valid for ExtensionOpeningReductionShape {
    fn check(&self) -> Result<(), SerializationError> {
        checked_shape_len(self.partials)?;
        checked_shape_sequence_len(self.sumcheck.len())?;
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
    /// Shape of the optional extension-opening reduction payload.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionShape>,
    /// Shape of the terminal cleartext witness.
    pub terminal_response: TerminalResponseShape,
}

/// Shape-selected outgoing witness binding for an intermediate fold.
///
/// This tag is serialized only in the proof-shape descriptor. The proof body
/// itself remains tag-free.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextWitnessBindingShape {
    /// Number of base-field coefficients in the outer commitment `u`.
    OuterCommitment { coeffs: usize },
    /// The following terminal proof owns the canonical `t` state bytes.
    TerminalInnerState,
}

/// Shape descriptor for deserializing a [`FoldLevelProof`] without headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelProofShape {
    /// Shape of the optional extension-opening reduction payload.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionShape>,
    /// Number of field coefficients in `v`.
    pub v_coeffs: usize,
    /// Stage-1 tree stage shapes in root-to-leaf order.
    pub stage1_stages: Vec<AkitaStage1StageShape>,
    /// Stage-2 sumcheck shape: `(num_rounds, degree)`.
    pub stage2_sumcheck_proof: SumcheckProofShape,
    /// Shape of the optional stage-3 setup product-sumcheck payload.
    pub stage3_sumcheck: Option<SetupProductSumcheckShape>,
    /// Shape-selected outgoing witness binding.
    pub next_witness_binding: NextWitnessBindingShape,
}

/// Shape descriptor for deserializing an [`AkitaBatchedProof`] without
/// headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaBatchedProofShape {
    /// Root fold shape.
    pub root: LevelProofShape,
    /// Non-terminal recursive fold shapes in execution order.
    pub recursive_folds: Vec<LevelProofShape>,
    /// Required terminal fold shape.
    pub terminal: TerminalLevelProofShape,
}

pub(super) fn sumcheck_shape<F: FieldCore>(sc: &SumcheckProof<F>) -> SumcheckProofShape {
    sc.round_polys
        .iter()
        .map(|p| p.coeffs_except_linear_term.len())
        .collect()
}

fn eq_factored_sumcheck_shape<F: FieldCore>(
    sc: &EqFactoredSumcheckProof<F>,
) -> EqFactoredSumcheckProofShape {
    let degree = sc
        .round_polys
        .first()
        .map_or(0, |p| p.coeffs_except_linear_term.len());
    (sc.round_polys.len(), degree)
}

pub(super) fn level_proof_shape<F: FieldCore, E: FieldCore>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    v: &RingVec<F>,
    stage1: &AkitaStage1Proof<E>,
    stage2: &AkitaStage2Proof<F, E>,
    stage3_sumcheck_proof: Option<&SetupSumcheckProof<E>>,
) -> LevelProofShape {
    LevelProofShape {
        extension_opening_reduction: extension_opening_reduction
            .map(ExtensionOpeningReductionProof::shape),
        v_coeffs: v.coeff_len(),
        stage1_stages: stage1
            .stages
            .iter()
            .map(|stage| AkitaStage1StageShape {
                sumcheck_proof: eq_factored_sumcheck_shape(&stage.sumcheck_proof),
                child_claims: stage.child_claims.len(),
            })
            .collect(),
        stage2_sumcheck_proof: sumcheck_shape(&stage2.sumcheck_proof),
        stage3_sumcheck: stage3_sumcheck_proof.map(SetupSumcheckProof::shape),
        next_witness_binding: match &stage2.next_witness_binding {
            NextWitnessBinding::OuterCommitment(commitment) => {
                NextWitnessBindingShape::OuterCommitment {
                    coeffs: commitment.coeff_len(),
                }
            }
            NextWitnessBinding::TerminalInnerState => NextWitnessBindingShape::TerminalInnerState,
        },
    }
}

// === Headerless shape (de)serialization ===
//
// These impls let callers bundle proof shapes alongside proofs (e.g. when
// shipping verifier inputs to a Jolt guest program), so that the proof can be
// deserialized in environments that don't reconstruct a `Schedule` first.

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

impl Valid for LevelProofShape {
    fn check(&self) -> Result<(), SerializationError> {
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction.check()?;
        }
        checked_shape_len(self.v_coeffs)?;
        checked_shape_sequence_len(self.stage1_stages.len())?;
        self.stage1_stages.check()?;
        checked_shape_sequence_len(self.stage2_sumcheck_proof.len())?;
        for &degree in &self.stage2_sumcheck_proof {
            checked_shape_len(degree)?;
        }
        if let Some(shape) = &self.stage3_sumcheck {
            shape.check()?;
        }
        if let NextWitnessBindingShape::OuterCommitment { coeffs } = self.next_witness_binding {
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
                .sumcheck
                .serialize_with_mode(&mut writer, compress)?;
        }
        self.v_coeffs.serialize_with_mode(&mut writer, compress)?;
        self.stage1_stages
            .serialize_with_mode(&mut writer, compress)?;
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
            NextWitnessBindingShape::OuterCommitment { coeffs } => {
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
                        + reduction.sumcheck.serialized_size(compress)
                });
        reduction_size
            + self.v_coeffs.serialized_size(compress)
            + self.stage1_stages.serialized_size(compress)
            + self.stage2_sumcheck_proof.serialized_size(compress)
            + true.serialized_size(compress)
            + self
                .stage3_sumcheck
                .as_ref()
                .map_or(0, |shape| shape.sumcheck.serialized_size(compress))
            + 0u8.serialized_size(compress)
            + match self.next_witness_binding {
                NextWitnessBindingShape::OuterCommitment { coeffs } => {
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
            let sumcheck = deserialize_shape_vec(&mut reader, compress, validate)?;
            Some(ExtensionOpeningReductionShape { partials, sumcheck })
        } else {
            None
        };
        let v_coeffs = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let stage1_stages = deserialize_shape_vec(&mut reader, compress, validate)?;
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
                0 => NextWitnessBindingShape::OuterCommitment {
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
            v_coeffs,
            stage1_stages,
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
            let sumcheck = deserialize_shape_vec(&mut reader, compress, validate)?;
            Some(ExtensionOpeningReductionShape { partials, sumcheck })
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

impl Valid for AkitaBatchedProofShape {
    fn check(&self) -> Result<(), SerializationError> {
        self.root.check()?;
        checked_shape_sequence_len(self.recursive_folds.len())?;
        self.recursive_folds.check()?;
        self.terminal.check()?;
        Ok(())
    }
}

impl AkitaSerialize for AkitaBatchedProofShape {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.root.serialize_with_mode(&mut writer, compress)?;
        self.recursive_folds
            .serialize_with_mode(&mut writer, compress)?;
        self.terminal.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.root.serialized_size(compress)
            + self.recursive_folds.serialized_size(compress)
            + self.terminal.serialized_size(compress)
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
        let out = Self {
            root: LevelProofShape::deserialize_with_mode(&mut reader, compress, validate, &())?,
            recursive_folds: deserialize_shape_vec(&mut reader, compress, validate)?,
            terminal: TerminalLevelProofShape::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}
