use super::*;
use crate::{
    CommittedGroupParams, FoldSchedule, FoldSuccessor, OpeningClaimsLayout, OpeningMethod,
    PolynomialGroupLayout,
};

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

/// Derive the only accepted extension-opening reduction shape for one opening batch.
pub fn canonical_extension_opening_reduction_shape(
    opening_layout: &OpeningClaimsLayout,
    extension_degree: usize,
) -> Result<ExtensionOpeningReductionShape, AkitaError> {
    if extension_degree <= 1 || !extension_degree.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "extension opening degree must be a power of two greater than one".to_string(),
        ));
    }
    opening_layout.check()?;
    let split_bits = extension_degree.trailing_zeros() as usize;
    let num_rounds = opening_layout
        .max_num_vars()
        .checked_sub(split_bits)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "extension opening split exceeds the opening arity".to_string(),
            )
        })?;
    let num_claims = opening_layout.num_total_polynomials();
    let partials = extension_degree.checked_mul(num_claims).ok_or_else(|| {
        AkitaError::InvalidSetup("extension opening partial count overflow".to_string())
    })?;
    Ok(ExtensionOpeningReductionShape::standard(
        partials, num_rounds, num_claims,
    ))
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
    /// Exact packed nonce-stream width derived from the public grinding plan.
    pub nonce_stream_bits: usize,
    /// Root fold shape.
    pub root: LevelProofShape,
    /// Non-terminal recursive fold shapes in execution order.
    pub recursive_folds: Vec<LevelProofShape>,
    /// Required terminal fold shape.
    pub terminal: TerminalLevelProofShape,
}

/// Derive the only accepted headerless proof shape for a schedule and opening layout.
pub fn canonical_proof_shape(
    schedule: &FoldSchedule,
    root_opening_layout: &OpeningClaimsLayout,
    extension_degree: usize,
    grinding_plan: &crate::GrindingPlan,
) -> Result<AkitaBatchedProofShape, AkitaError> {
    if !extension_degree.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "proof-shape extension degree must be a nonzero power of two".to_string(),
        ));
    }

    fn level_extension_shape(
        params: &CommittedGroupParams,
        opening_layout: &OpeningClaimsLayout,
        extension_degree: usize,
    ) -> Result<Option<ExtensionOpeningReductionShape>, AkitaError> {
        let first_method = params.group_params(opening_layout, 0)?.opening_method();
        for group_index in 1..opening_layout.num_groups() {
            if params
                .group_params(opening_layout, group_index)?
                .opening_method()
                != first_method
            {
                return Err(AkitaError::InvalidSetup(
                    "one fold cannot mix opening-method families".to_string(),
                ));
            }
        }
        if extension_degree == 1 || !matches!(first_method, OpeningMethod::EvaluationTrace) {
            return Ok(None);
        }
        canonical_extension_opening_reduction_shape(opening_layout, extension_degree).map(Some)
    }

    fn stage3_shape(
        successor: FoldSuccessor<'_>,
    ) -> Result<Option<SetupProductSumcheckShape>, AkitaError> {
        let Some(prefix) = (match successor {
            FoldSuccessor::Recursive(params) => params.setup_prefix(),
            FoldSuccessor::Terminal(_) => None,
        }) else {
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
        opening_layout: &OpeningClaimsLayout,
        extension_degree: usize,
        output_witness_len: usize,
        successor: FoldSuccessor<'_>,
    ) -> Result<(LevelProofShape, usize), AkitaError> {
        let rounds = params
            .relation_address_geometry(
                opening_layout,
                extension_degree,
                successor.ring_dimension(),
                output_witness_len,
            )?
            .relation_point_variable_count();
        let basis = 1usize
            .checked_shl(params.open().digits.log_basis)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("digit-range basis does not fit usize".to_string())
            })?;
        let (stage1_stages, stage1_norm) = DigitRangePlan::new(basis)?
            .proof_shapes_for_route(rounds, params.inner().matrix.security_route())?;
        let next_witness_binding = match successor {
            FoldSuccessor::Recursive(next) => NextWitnessBindingShape::OuterPayload {
                coeffs: next.outer_payload_geometry()?.transmitted_coefficients(),
            },
            FoldSuccessor::Terminal(_) => NextWitnessBindingShape::TerminalInnerState,
        };
        Ok((
            LevelProofShape {
                extension_opening_reduction: level_extension_shape(
                    params,
                    opening_layout,
                    extension_degree,
                )?,
                opening_payload_coeffs: params
                    .opening_payload_geometry()?
                    .transmitted_coefficients(),
                stage1_stages,
                stage1_norm,
                stage2_sumcheck_proof: vec![3; rounds],
                stage3_sumcheck: stage3_shape(successor)?,
                next_witness_binding,
            },
            rounds,
        ))
    }

    let root_successor = schedule
        .recursive_folds
        .first()
        .map_or(FoldSuccessor::Terminal(&schedule.terminal), |step| {
            FoldSuccessor::Recursive(&step.params)
        });
    let (root, mut predecessor_rounds) = level_shape(
        &schedule.root.params,
        root_opening_layout,
        extension_degree,
        schedule.root.output_witness_len,
        root_successor,
    )?;
    let mut recursive_folds = Vec::with_capacity(schedule.recursive_folds.len());
    for (index, step) in schedule.recursive_folds.iter().enumerate() {
        let successor = schedule
            .recursive_folds
            .get(index + 1)
            .map_or(FoldSuccessor::Terminal(&schedule.terminal), |next| {
                FoldSuccessor::Recursive(&next.params)
            });
        let opening_layout = step
            .params
            .opening_layout_for_final_group(PolynomialGroupLayout::singleton(predecessor_rounds))?;
        let (shape, rounds) = level_shape(
            &step.params,
            &opening_layout,
            extension_degree,
            step.output_witness_len,
            successor,
        )?;
        recursive_folds.push(shape);
        predecessor_rounds = rounds;
    }
    Ok(AkitaBatchedProofShape {
        nonce_stream_bits: grinding_plan.total_nonce_bits(),
        root,
        recursive_folds,
        terminal: TerminalLevelProofShape {
            extension_opening_reduction: if extension_degree == 1 {
                None
            } else {
                Some(canonical_extension_opening_reduction_shape(
                    &OpeningClaimsLayout::new(predecessor_rounds, 1)?,
                    extension_degree,
                )?)
            },
            terminal_response: schedule.terminal.response_shape.clone(),
        },
    })
}

pub(super) fn sumcheck_shape<F: Field>(sc: &SumcheckProof<F>) -> SumcheckProofShape {
    sc.round_polys
        .iter()
        .map(|p| p.coeffs_except_linear_term.len())
        .collect()
}

fn eq_factored_sumcheck_shape<F: Field>(
    sc: &EqFactoredSumcheckProof<F>,
) -> EqFactoredSumcheckProofShape {
    let degree = sc
        .round_polys
        .first()
        .map_or(0, |p| p.coeffs_except_linear_term.len());
    (sc.round_polys.len(), degree)
}

pub(super) fn level_proof_shape<F: Field, E: Field>(
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

mod serialization;
