use super::*;
use crate::{CommittedGroupParams, FoldSchedule};

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
    /// Number of individual terminal claims serialized after the sumcheck.
    pub final_claims: usize,
    /// One compact coefficient count per round of the batched reduction.
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
    pub fn standard(partials: usize, num_rounds: usize, num_claims: usize) -> Self {
        Self {
            partials,
            final_claims: num_claims,
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
        checked_shape_len(self.final_claims)?;
        checked_shape_sequence_len(self.sumcheck.len())?;
        if self.final_claims == 0 {
            return Err(SerializationError::InvalidData(
                "extension opening reduction shape must contain terminal claims".to_string(),
            ));
        }
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
    /// Number of base-field coefficients in the compressed outer payload.
    OuterPayload { coeffs: usize },
    /// The following terminal proof owns the canonical `t` state bytes.
    TerminalInnerState,
}

/// Shape descriptor for deserializing a [`FoldLevelProof`] without headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelProofShape {
    /// Shape of the optional extension-opening reduction payload.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionShape>,
    /// Number of field coefficients in the compressed opening payload.
    pub opening_payload_coeffs: usize,
    /// Stage-1 tree stage shapes in root-to-leaf order.
    pub stage1_stages: Vec<AkitaStage1StageShape>,
    /// Shape of the optional schedule-selected physical norm payload.
    pub stage1_norm: Option<PhysicalL2NormProofWireShape>,
    /// Stage-2 sumcheck shape: `(num_rounds, degree)`.
    pub stage2_sumcheck_proof: SumcheckProofShape,
    /// Shape of the optional stage-3 setup product-sumcheck payload.
    pub stage3_sumcheck: Option<SetupProductSumcheckShape>,
    /// Shape-selected outgoing witness binding.
    pub next_witness_binding: NextWitnessBindingShape,
}

/// Headerless wire shape of [`PhysicalL2NormProof`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalL2NormProofWireShape {
    /// Number of blockwise limb claims; zero for direct mode.
    pub subclaims: usize,
    /// Number of final response/limb virtual evaluations.
    pub virtual_evaluations: usize,
    /// General final-leaf sumcheck shape.
    pub sumcheck: SumcheckProofShape,
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

/// Derive the only accepted headerless proof shape for a base-field schedule.
///
/// This is the verifier-side owner for decoding proofs whose claim field is the
/// base field. Proper extension fields add an extension-opening reduction shape
/// and must use their configuration-specific derivation.
pub fn canonical_base_field_proof_shape(
    schedule: &FoldSchedule,
) -> Result<AkitaBatchedProofShape, AkitaError> {
    fn stage3_shape(
        successor: Option<&CommittedGroupParams>,
    ) -> Result<Option<SetupProductSumcheckShape>, AkitaError> {
        let Some(prefix) = successor.and_then(|params| params.setup_prefix.as_ref()) else {
            return Ok(None);
        };
        let n_prefix = prefix.n_prefix()?;
        let setup_ring_len = n_prefix.checked_div(prefix.d_setup()).ok_or_else(|| {
            AkitaError::InvalidSetup("setup-prefix ring dimension is zero".to_string())
        })?;
        if setup_ring_len == 0 || !n_prefix.is_multiple_of(prefix.d_setup()) {
            return Err(AkitaError::InvalidSetup(
                "setup-prefix field length does not align with its ring dimension".to_string(),
            ));
        }
        let rounds = (prefix.d_setup().trailing_zeros() as usize)
            .checked_add(setup_ring_len.next_power_of_two().trailing_zeros() as usize)
            .ok_or_else(|| AkitaError::InvalidSetup("stage-3 round count overflow".to_string()))?;
        Ok(Some(SetupProductSumcheckShape {
            sumcheck: vec![SETUP_SUMCHECK_DEGREE; rounds],
        }))
    }

    fn level_shape(
        params: &CommittedGroupParams,
        output_witness_len: usize,
        successor: Option<&CommittedGroupParams>,
    ) -> Result<LevelProofShape, AkitaError> {
        let rounds = crate::sumcheck_rounds(params.d_a(), output_witness_len);
        let basis = 1usize.checked_shl(params.log_basis_open).ok_or_else(|| {
            AkitaError::InvalidSetup("digit-range basis does not fit usize".to_string())
        })?;
        let (stage1_stages, stage1_norm) = DigitRangePlan::new(basis)?
            .proof_shapes_for_route(rounds, params.inner_commit_matrix.security_route())?;
        let next_witness_binding = match successor {
            Some(next) => NextWitnessBindingShape::OuterPayload {
                coeffs: next.outer_payload_geometry()?.transmitted_coefficients(),
            },
            None => NextWitnessBindingShape::TerminalInnerState,
        };
        Ok(LevelProofShape {
            extension_opening_reduction: None,
            opening_payload_coeffs: params
                .opening_payload_geometry()?
                .transmitted_coefficients(),
            stage1_stages,
            stage1_norm,
            stage2_sumcheck_proof: vec![3; rounds],
            stage3_sumcheck: stage3_shape(successor)?,
            next_witness_binding,
        })
    }

    let root_successor = schedule
        .recursive_folds
        .first()
        .map(|step| &step.params.witness);
    let root = level_shape(
        &schedule.root.params.final_group.commitment,
        schedule.root.output_witness_len,
        root_successor,
    )?;
    let mut recursive_folds = Vec::with_capacity(schedule.recursive_folds.len());
    for (index, step) in schedule.recursive_folds.iter().enumerate() {
        let successor = schedule
            .recursive_folds
            .get(index + 1)
            .map(|next| &next.params.witness);
        recursive_folds.push(level_shape(
            &step.params.witness,
            step.output_witness_len,
            successor,
        )?);
    }
    Ok(AkitaBatchedProofShape {
        root,
        recursive_folds,
        terminal: TerminalLevelProofShape {
            extension_opening_reduction: None,
            terminal_response: schedule.terminal.params.response_shape.clone(),
        },
    })
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
    opening_payload: &RingVec<F>,
    stage1: &AkitaStage1Proof<E>,
    stage2: &AkitaStage2Proof<F, E>,
    stage3_sumcheck_proof: Option<&SetupSumcheckProof<E>>,
) -> LevelProofShape {
    LevelProofShape {
        extension_opening_reduction: extension_opening_reduction
            .map(ExtensionOpeningReductionProof::shape),
        opening_payload_coeffs: opening_payload.coeff_len(),
        stage1_stages: stage1
            .stages
            .iter()
            .map(|stage| AkitaStage1StageShape {
                sumcheck_proof: eq_factored_sumcheck_shape(&stage.sumcheck_proof),
                child_claims: stage.child_claims.len(),
            })
            .collect(),
        stage1_norm: stage1
            .norm_proof
            .as_ref()
            .map(|proof| PhysicalL2NormProofWireShape {
                subclaims: proof.subclaims.len(),
                virtual_evaluations: proof.virtual_evaluations.len(),
                sumcheck: sumcheck_shape(&proof.sumcheck),
            }),
        stage2_sumcheck_proof: sumcheck_shape(&stage2.sumcheck_proof),
        stage3_sumcheck: stage3_sumcheck_proof.map(SetupSumcheckProof::shape),
        next_witness_binding: match &stage2.next_witness_binding {
            NextWitnessBinding::OuterPayload(commitment) => NextWitnessBindingShape::OuterPayload {
                coeffs: commitment.coeff_len(),
            },
            NextWitnessBinding::TerminalInnerState => NextWitnessBindingShape::TerminalInnerState,
        },
    }
}

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

impl Valid for AkitaBatchedProofShape {
    fn check(&self) -> Result<(), SerializationError> {
        self.root.check()?;
        checked_shape_sequence_len(self.recursive_folds.len())?;
        self.recursive_folds.check()?;
        self.terminal.check()?;
        Ok(())
    }
}

impl AkitaBatchedProofShape {
    /// Reject a proof shape whose aggregate declared field payload cannot fit
    /// in the bytes available to the proof decoder.
    pub fn validate_decode_budget(
        &self,
        available_bytes: usize,
        base_field_bytes: usize,
        extension_field_bytes: usize,
    ) -> Result<(), SerializationError> {
        if base_field_bytes == 0 || extension_field_bytes == 0 {
            return Err(SerializationError::InvalidData(
                "base and extension field wire sizes must be nonzero".to_string(),
            ));
        }
        fn add(total: &mut usize, value: usize) -> Result<(), SerializationError> {
            *total = total.checked_add(value).ok_or_else(|| {
                SerializationError::InvalidData(
                    "aggregate proof shape field count overflow".to_string(),
                )
            })?;
            Ok(())
        }
        fn add_sumcheck(total: &mut usize, shape: &[usize]) -> Result<(), SerializationError> {
            for &stored_coefficients in shape {
                add(total, stored_coefficients)?;
            }
            Ok(())
        }
        fn add_extension_opening(
            total: &mut usize,
            shape: Option<&ExtensionOpeningReductionShape>,
        ) -> Result<(), SerializationError> {
            if let Some(shape) = shape {
                add(total, shape.partials)?;
                add_sumcheck(total, &shape.sumcheck)?;
                add(total, shape.final_claims)?;
            }
            Ok(())
        }
        fn add_level(
            base: &mut usize,
            extension: &mut usize,
            shape: &LevelProofShape,
        ) -> Result<(), SerializationError> {
            add_extension_opening(extension, shape.extension_opening_reduction.as_ref())?;
            add(base, shape.opening_payload_coeffs)?;
            for stage in &shape.stage1_stages {
                let (rounds, stored_coefficients) = stage.sumcheck_proof;
                add(
                    extension,
                    rounds.checked_mul(stored_coefficients).ok_or_else(|| {
                        SerializationError::InvalidData(
                            "stage-1 proof shape field count overflow".to_string(),
                        )
                    })?,
                )?;
                add(extension, stage.child_claims)?;
            }
            add(extension, 1)?;
            if let Some(norm) = &shape.stage1_norm {
                add(extension, norm.subclaims)?;
                add(extension, norm.virtual_evaluations)?;
                add_sumcheck(extension, &norm.sumcheck)?;
            }
            add_sumcheck(extension, &shape.stage2_sumcheck_proof)?;
            match shape.next_witness_binding {
                NextWitnessBindingShape::OuterPayload { coeffs } => add(base, coeffs)?,
                NextWitnessBindingShape::TerminalInnerState => {}
            }
            add(extension, 1)?;
            if let Some(stage3) = &shape.stage3_sumcheck {
                add(extension, 2)?;
                add_sumcheck(extension, &stage3.sumcheck)?;
            }
            Ok(())
        }

        let mut base_field_elements = 0usize;
        let mut extension_field_elements = 0usize;
        add_level(
            &mut base_field_elements,
            &mut extension_field_elements,
            &self.root,
        )?;
        for shape in &self.recursive_folds {
            add_level(
                &mut base_field_elements,
                &mut extension_field_elements,
                shape,
            )?;
        }
        add_extension_opening(
            &mut extension_field_elements,
            self.terminal.extension_opening_reduction.as_ref(),
        )?;
        add(
            &mut base_field_elements,
            self.terminal.terminal_response.layout.e_field_elems(),
        )?;
        add(
            &mut base_field_elements,
            self.terminal.terminal_response.layout.t_field_elems(),
        )?;
        let required_bytes = base_field_elements
            .checked_mul(base_field_bytes)
            .and_then(|base_bytes| {
                extension_field_elements
                    .checked_mul(extension_field_bytes)
                    .and_then(|extension_bytes| base_bytes.checked_add(extension_bytes))
            })
            .ok_or_else(|| {
                SerializationError::InvalidData("aggregate proof byte budget overflow".to_string())
            })?;
        if required_bytes > available_bytes {
            return Err(SerializationError::LengthLimitExceeded {
                len: u64::try_from(required_bytes).unwrap_or(u64::MAX),
                max: available_bytes,
            });
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
