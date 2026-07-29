use super::shapes::level_proof_shape;
use super::shapes::sumcheck_shape;
use super::*;
use crate::{CommittedGroupParams, SetupContributionMode};

/// One stage in the stage-1 range-check tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaStage1StageProof<F: FieldCore> {
    /// Eq-factored sumcheck proof for this stage.
    pub sumcheck_proof: EqFactoredSumcheckProof<F>,
    /// Claimed child-node evaluations at this stage's output point.
    ///
    /// Non-leaf stages populate these so the verifier can seed the next stage;
    /// the leaf stage leaves this empty and instead carries `range_image_evaluation` below.
    pub child_claims: Vec<F>,
}

/// Proof payload for stage 1 of a single Akita level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaStage1Proof<F: FieldCore> {
    /// Root-to-leaf range-check stages.
    pub stages: Vec<AkitaStage1StageProof<F>>,
    /// Claimed evaluation of `S` at the final stage-1 output point.
    pub range_image_evaluation: F,
}

/// FoldSchedule-shaped outgoing witness binding for an intermediate fold.
///
/// The proof stream carries no variant tag. Headerless decoding obtains the
/// variant from [`NextWitnessBindingShape`]: ordinary recursive edges carry an
/// outer `u`, while an edge into the suffix terminal binds the `t` segment
/// owned by the following [`TerminalLevelProof`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NextWitnessBinding<F: FieldCore> {
    /// Outer commitment `u = B * decompose(t)` for an ordinary recursive edge.
    OuterCommitment(RingVec<F>),
    /// The following terminal proof's canonical `t` segment is the state.
    TerminalInnerState,
}

impl<F: FieldCore> NextWitnessBinding<F> {
    /// Borrow the outer commitment when this is an ordinary recursive edge.
    #[must_use]
    pub fn outer_commitment(&self) -> Option<&RingVec<F>> {
        match self {
            Self::OuterCommitment(commitment) => Some(commitment),
            Self::TerminalInnerState => None,
        }
    }
}

/// Intermediate-stage payload for stage 2 of a fold level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaStage2Proof<F: FieldCore, E: FieldCore> {
    /// Stage-2 fused sumcheck proof.
    pub sumcheck_proof: SumcheckProof<E>,
    /// FoldSchedule-shaped binding for the next witness.
    pub next_witness_binding: NextWitnessBinding<F>,
    /// Claimed evaluation of the next witness `w` at the stage-2 challenge point.
    pub next_w_eval: E,
}

impl<F: FieldCore, E: FieldCore> AkitaStage2Proof<F, E> {
    /// Wire value for the next-witness evaluation claim at stage 2.
    pub fn next_w_eval(&self) -> E {
        self.next_w_eval
    }
}

/// Optional proof that reduces a logical extension-field opening into one
/// ordinary opening of the transformed committed witness.
///
/// This object is not serialized with a tag or length. Its presence and shape
/// are determined by the verifier's expected proof shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionOpeningReductionProof<E: FieldCore> {
    /// Transcript-bound partial evaluations used by the basis-conversion
    /// check.
    pub partials: Vec<E>,
    /// Degree-two reduction sumcheck.
    pub sumcheck: SumcheckProof<E>,
}

/// Fused stage-3 proof for the public setup contribution and carried witness opening.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupSumcheckProof<E: FieldCore> {
    /// Claimed setup contribution fed into the stage-2 final row evaluation.
    pub claim: E,
    /// Claimed setup-prefix opening carried into the next fold as a precommitted group.
    pub setup_prefix_eval: E,
    /// Claimed next-witness opening after the batched stage-3 point projection.
    pub next_w_eval: E,
    /// Degree-two batched product sumcheck carrying setup and next-witness terms.
    pub sumcheck: SumcheckProof<E>,
}

impl<E: FieldCore> SetupSumcheckProof<E> {
    /// Shape descriptor required for headerless deserialization.
    pub fn shape(&self) -> SetupProductSumcheckShape {
        SetupProductSumcheckShape {
            sumcheck: sumcheck_shape(&self.sumcheck),
        }
    }
}

impl<E: FieldCore> ExtensionOpeningReductionProof<E> {
    /// Shape descriptor required for headerless deserialization.
    pub fn shape(&self) -> ExtensionOpeningReductionShape {
        ExtensionOpeningReductionShape {
            partials: self.partials.len(),
            sumcheck: sumcheck_shape(&self.sumcheck),
        }
    }

    /// Number of sumcheck rounds in the reduction proof.
    pub fn num_rounds(&self) -> usize {
        self.sumcheck.round_polys.len()
    }
}

/// Proof for one non-terminal fold level, including the root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoldLevelProof<F: FieldCore, E: FieldCore> {
    /// Optional extension-opening reduction payload.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionProof<E>>,
    /// `v = D · ŵ` in the current level's ring dimension.
    pub v: RingVec<F>,
    /// Accepted fold-l∞ grind nonce (`0` under deterministic policy).
    pub fold_grind_nonce: u32,
    /// Stage-1 norm-check payload.
    pub stage1: AkitaStage1Proof<E>,
    /// Stage-2 fused payload.
    pub stage2: AkitaStage2Proof<F, E>,
    /// Optional stage-3 setup product-sumcheck proof.
    pub stage3_sumcheck_proof: Option<SetupSumcheckProof<E>>,
}

impl<F: FieldCore, E: FieldCore> FoldLevelProof<F, E> {
    /// Construct from typed ring elements for the current level and its
    /// inline norm-check payloads.
    pub fn new<const D: usize>(
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<E>,
        stage2: AkitaStage2Proof<F, E>,
    ) -> Self {
        Self {
            extension_opening_reduction: None,
            v: RingVec::from_ring_elems(&v).into_compact(),
            fold_grind_nonce: 0,
            stage1,
            stage2,
            stage3_sumcheck_proof: None,
        }
    }

    /// Accepted fold grind nonce (`0` under deterministic policy).
    pub fn fold_grind_nonce(&self) -> u32 {
        self.fold_grind_nonce
    }

    /// Borrow the optional extension-opening reduction payload.
    pub fn extension_opening_reduction(&self) -> Option<&ExtensionOpeningReductionProof<E>> {
        self.extension_opening_reduction.as_ref()
    }

    /// Borrow the `v` payload.
    pub fn v(&self) -> &RingVec<F> {
        &self.v
    }

    /// Mutably borrow the `v` payload.
    pub fn v_mut(&mut self) -> &mut RingVec<F> {
        &mut self.v
    }

    /// Borrow the stage-1 payload.
    pub fn stage1(&self) -> &AkitaStage1Proof<E> {
        &self.stage1
    }

    /// Mutably borrow the stage-1 payload.
    pub fn stage1_mut(&mut self) -> &mut AkitaStage1Proof<E> {
        &mut self.stage1
    }

    /// Borrow the stage-2 payload.
    pub fn stage2(&self) -> &AkitaStage2Proof<F, E> {
        &self.stage2
    }

    /// Mutably borrow the stage-2 payload.
    pub fn stage2_mut(&mut self) -> &mut AkitaStage2Proof<F, E> {
        &mut self.stage2
    }

    /// Borrow the optional stage-3 setup sumcheck proof.
    pub fn stage3_sumcheck_proof(&self) -> Option<&SetupSumcheckProof<E>> {
        self.stage3_sumcheck_proof.as_ref()
    }

    /// Borrow and validate the optional stage-3 setup sumcheck proof.
    pub fn stage3_for_mode<'a>(
        &'a self,
        mode: SetupContributionMode,
        next_fold_level_params: Option<&'a CommittedGroupParams>,
    ) -> Result<Option<(&'a SetupSumcheckProof<E>, &'a CommittedGroupParams)>, AkitaError> {
        match (mode, self.stage3_sumcheck_proof.as_ref()) {
            (SetupContributionMode::Direct, None) => Ok(None),
            (SetupContributionMode::Direct, Some(_)) => Err(AkitaError::InvalidSetup(
                "direct setup-contribution mode received stage3_sumcheck_proof".to_string(),
            )),
            (SetupContributionMode::Recursive, Some(proof)) => {
                let next_fold_level_params = next_fold_level_params.ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "recursive setup-contribution mode is missing next-level params"
                            .to_string(),
                    )
                })?;
                Ok(Some((proof, next_fold_level_params)))
            }
            (SetupContributionMode::Recursive, None) => Err(AkitaError::InvalidSetup(
                "recursive setup-contribution mode is missing stage3_sumcheck_proof".to_string(),
            )),
        }
    }

    /// Reconstruct typed `v`, returning `InvalidProof` on shape mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored `v` payload is not
    /// well-formed for ring dimension `D`.
    pub fn try_v_typed<const D: usize>(&self) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        self.v.try_to_vec()
    }

    /// Borrow the next witness's outer commitment when this level has one.
    pub fn next_w_commitment(&self) -> Option<&RingVec<F>> {
        self.stage2.next_witness_binding.outer_commitment()
    }

    /// Claimed evaluation of the next witness `w` at the norm-check output point.
    pub fn next_w_eval(&self) -> E {
        self.stage2.next_w_eval()
    }

    /// Derive the [`LevelProofShape`] for this level proof.
    pub fn shape(&self) -> LevelProofShape {
        level_proof_shape(
            self.extension_opening_reduction.as_ref(),
            &self.v,
            &self.stage1,
            &self.stage2,
            self.stage3_sumcheck_proof.as_ref(),
        )
    }
}

/// Terminal fold-level proof.
///
/// Ships the terminal response in cleartext. Its raw `e` segment is bound before the
/// terminal sparse challenge. The predecessor first binds canonical `t` as its
/// outgoing state; terminal replay rebinds the same `t` as current state before
/// absorbing `e`, sampling challenges, and absorbing the `z` response.
///
/// Drops the redundant proof components at the terminal: `stage1`
/// (the terminal response codec enforces its range), the stage-2 outgoing binding
/// (replaced by the terminal response), and `next_w_eval` (verifier computes
/// directly from the response). All terminal schedules drop commitment and
/// D-row blocks, so neither an outer `u` nor `v` is serialized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalLevelProof<F: FieldCore, E: FieldCore> {
    /// Optional extension-opening reduction payload.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionProof<E>>,
    /// Accepted Fiat-Shamir grind nonce for fold-l∞ rejection (0 under deterministic policy).
    pub fold_grind_nonce: u32,
    /// Quotient-free terminal response checked directly by the verifier.
    pub terminal_response: TerminalResponse<F>,
}

impl<F: FieldCore, E: FieldCore> TerminalLevelProof<F, E> {
    /// Construct from typed ring elements and a clear terminal response.
    ///
    /// Pass `extension_opening_reduction = None` for opening shapes that do
    /// not use extension-opening reduction.
    pub fn new_with_extension_opening_reduction(
        extension_opening_reduction: Option<ExtensionOpeningReductionProof<E>>,
        terminal_response: TerminalResponse<F>,
        fold_grind_nonce: u32,
    ) -> Self {
        Self {
            extension_opening_reduction,
            fold_grind_nonce,
            terminal_response,
        }
    }

    /// Borrow the clear terminal response.
    pub fn terminal_response(&self) -> &TerminalResponse<F> {
        &self.terminal_response
    }

    /// Mutably borrow the clear terminal response.
    pub fn terminal_response_mut(&mut self) -> &mut TerminalResponse<F> {
        &mut self.terminal_response
    }

    /// Derive the [`TerminalLevelProofShape`] for this terminal-level proof.
    pub fn shape(&self) -> TerminalLevelProofShape {
        TerminalLevelProofShape {
            extension_opening_reduction: self
                .extension_opening_reduction
                .as_ref()
                .map(ExtensionOpeningReductionProof::shape),
            terminal_response: self.terminal_response().shape(),
        }
    }
}

/// Akita PCS proof for fused batched openings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaBatchedProof<F: FieldCore, E: FieldCore> {
    /// Root fold over all original-polynomial claims.
    pub root: FoldLevelProof<F, E>,
    /// Non-terminal recursive folds between the root and terminal fold.
    pub recursive_folds: Vec<FoldLevelProof<F, E>>,
    /// Required terminal fold carrying the clear terminal response.
    pub terminal: TerminalLevelProof<F, E>,
}

impl<F: FieldCore, E: FieldCore> AkitaBatchedProof<F, E> {
    /// Access the clear terminal response.
    pub fn terminal_response(&self) -> &TerminalResponse<F> {
        self.terminal.terminal_response()
    }

    /// Iterate over every non-terminal fold in execution order.
    pub fn nonterminal_folds(&self) -> impl Iterator<Item = &FoldLevelProof<F, E>> {
        std::iter::once(&self.root).chain(self.recursive_folds.iter())
    }

    /// Total number of fold levels, including root and terminal.
    pub fn num_fold_levels(&self) -> usize {
        2 + self.recursive_folds.len()
    }

    /// Derive the [`AkitaBatchedProofShape`] for this proof.
    pub fn shape(&self) -> AkitaBatchedProofShape {
        AkitaBatchedProofShape {
            root: self.root.shape(),
            recursive_folds: self
                .recursive_folds
                .iter()
                .map(FoldLevelProof::shape)
                .collect(),
            terminal: self.terminal.shape(),
        }
    }
}

impl<F: FieldCore + CanonicalField + AkitaSerialize, E: FieldCore + AkitaSerialize>
    AkitaBatchedProof<F, E>
{
    /// Returns the proof size in bytes (uncompressed).
    pub fn size(&self) -> usize {
        self.serialized_size(Compress::No)
    }
}
