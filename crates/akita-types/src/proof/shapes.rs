use super::*;
use crate::OpeningClaimsLayout;
use akita_sumcheck::{
    EqFactoredSumcheckProof, EqFactoredSumcheckProofShape, SumcheckProof, SumcheckProofShape,
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

/// Public shape of the native extension-opening-reduction messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionOpeningReductionShape {
    /// Number of partial evaluations serialized before the sumcheck.
    pub partials: usize,
    /// Number of individual terminal claims serialized after the sumcheck.
    pub final_claims: usize,
    /// One compact coefficient count per round of the batched reduction.
    pub sumcheck: SumcheckProofShape,
}

/// Public shape of the native setup-product sumcheck messages.
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

/// Public layout of a terminal level's native messages.
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

/// Public layout of one fold level's native messages.
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

/// Public layout of the native physical-L2 messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalL2NormProofWireShape {
    /// Number of blockwise limb claims; zero for direct mode.
    pub subclaims: usize,
    /// Number of final response/limb virtual evaluations.
    pub virtual_evaluations: usize,
    /// General final-leaf sumcheck shape.
    pub sumcheck: SumcheckProofShape,
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
        .map_or(0, |p| p.coeffs_except_constant_term.len());
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
