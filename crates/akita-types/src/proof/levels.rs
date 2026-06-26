use super::shapes::level_proof_shape;
use super::shapes::sumcheck_shape;
use super::*;
use crate::{LevelParams, MRowLayout, SetupContributionMode};

/// One stage in the stage-1 range-check tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaStage1StageProof<F: FieldCore> {
    /// Eq-factored sumcheck proof for this stage.
    pub sumcheck_proof: EqFactoredSumcheckProof<F>,
    /// Claimed child-node evaluations at this stage's output point.
    ///
    /// Non-leaf stages populate these so the verifier can seed the next stage;
    /// the leaf stage leaves this empty and instead carries `s_claim` below.
    pub child_claims: Vec<F>,
}

/// Proof payload for stage 1 of a single Akita level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaStage1Proof<F: FieldCore> {
    /// Root-to-leaf range-check stages.
    pub stages: Vec<AkitaStage1StageProof<F>>,
    /// Claimed evaluation of `S` at the final stage-1 output point.
    pub s_claim: F,
}

/// Intermediate-stage payload for stage 2 of a fold level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaIntermediateStage2Proof<F: FieldCore, L: FieldCore> {
    /// Stage-2 fused sumcheck proof.
    pub sumcheck_proof: SumcheckProof<L>,
    /// Commitment to the next witness `w`
    /// (ring dim = next level's D, may differ from `v`).
    pub next_w_commitment: FlatRingVec<F>,
    /// Claimed evaluation of the next witness `w` at the stage-2 challenge point.
    pub next_w_eval: L,
}

impl<F: FieldCore, L: FieldCore> AkitaIntermediateStage2Proof<F, L> {
    /// Wire value for the next-witness evaluation claim at stage 2.
    pub fn next_w_eval(&self) -> L {
        self.next_w_eval
    }
}

/// Terminal-stage payload for stage 2 of a fold level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaTerminalStage2Proof<F: FieldCore, L: FieldCore> {
    /// Stage-2 fused sumcheck proof.
    pub sumcheck_proof: SumcheckProof<L>,
    /// Terminal witness, absorbed via `ABSORB_NEXT_LEVEL_WITNESS_BINDING` in place of
    /// `next_w_commitment`.
    pub final_witness: CleartextWitnessProof<F>,
}

impl<F: FieldCore, L: FieldCore> AkitaTerminalStage2Proof<F, L> {
    /// Borrow the terminal cleartext witness.
    pub fn final_witness(&self) -> &CleartextWitnessProof<F> {
        &self.final_witness
    }
}

/// Proof payload for stage 2 of a fold level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkitaStage2Proof<F: FieldCore, L: FieldCore> {
    /// Intermediate stage-2 payload with a recursive next-witness claim.
    Intermediate(AkitaIntermediateStage2Proof<F, L>),
    /// Terminal stage-2 payload with a cleartext final witness.
    Terminal(AkitaTerminalStage2Proof<F, L>),
}

impl<F: FieldCore, L: FieldCore> AkitaStage2Proof<F, L> {
    /// Borrow the intermediate stage-2 payload.
    pub fn as_intermediate(&self) -> Option<&AkitaIntermediateStage2Proof<F, L>> {
        match self {
            Self::Intermediate(proof) => Some(proof),
            Self::Terminal(_) => None,
        }
    }

    /// Mutably borrow the intermediate stage-2 payload.
    pub fn as_intermediate_mut(&mut self) -> Option<&mut AkitaIntermediateStage2Proof<F, L>> {
        match self {
            Self::Intermediate(proof) => Some(proof),
            Self::Terminal(_) => None,
        }
    }

    /// Borrow the terminal stage-2 payload.
    pub fn as_terminal(&self) -> Option<&AkitaTerminalStage2Proof<F, L>> {
        match self {
            Self::Intermediate(_) => None,
            Self::Terminal(proof) => Some(proof),
        }
    }

    /// Mutably borrow the terminal stage-2 payload.
    pub fn as_terminal_mut(&mut self) -> Option<&mut AkitaTerminalStage2Proof<F, L>> {
        match self {
            Self::Intermediate(_) => None,
            Self::Terminal(proof) => Some(proof),
        }
    }

    /// Borrow the transparent stage-2 sumcheck proof.
    pub fn sumcheck(&self) -> &SumcheckProof<L> {
        match self {
            Self::Intermediate(proof) => &proof.sumcheck_proof,
            Self::Terminal(proof) => &proof.sumcheck_proof,
        }
    }

    /// Wire value for the next-witness evaluation claim on intermediate levels.
    ///
    /// # Panics
    ///
    /// Panics if called on a terminal stage-2 proof.
    pub fn next_w_eval(&self) -> L {
        self.as_intermediate()
            .expect("next_w_eval() called on terminal stage-2 proof")
            .next_w_eval()
    }

    /// Borrow the terminal cleartext witness.
    pub fn final_witness(&self) -> Option<&CleartextWitnessProof<F>> {
        self.as_terminal()
            .map(AkitaTerminalStage2Proof::final_witness)
    }

    /// Mutably borrow the terminal cleartext witness.
    pub fn final_witness_mut(&mut self) -> Option<&mut CleartextWitnessProof<F>> {
        self.as_terminal_mut().map(|proof| &mut proof.final_witness)
    }
}

/// Optional proof that reduces a logical extension-field opening into one
/// ordinary opening of the transformed committed witness.
///
/// This object is not serialized with a tag or length. Its presence and shape
/// are determined by the verifier's expected proof shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionOpeningReductionProof<L: FieldCore> {
    /// Transcript-bound partial evaluations used by the basis-conversion
    /// check.
    pub partials: Vec<L>,
    /// Degree-two reduction sumcheck.
    pub sumcheck: SumcheckProof<L>,
}

/// Fused stage-3 proof for the public setup contribution and carried witness opening.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupSumcheckProof<L: FieldCore> {
    /// Claimed setup contribution fed into the stage-2 final row evaluation.
    pub claim: L,
    /// Claimed next-witness opening after the batched stage-3 point projection.
    pub next_w_eval: L,
    /// Degree-two batched product sumcheck carrying setup and next-witness terms.
    pub sumcheck: SumcheckProof<L>,
}

impl<L: FieldCore> SetupSumcheckProof<L> {
    /// Shape descriptor required for headerless deserialization.
    pub fn shape(&self) -> SetupProductSumcheckShape {
        SetupProductSumcheckShape {
            sumcheck: sumcheck_shape(&self.sumcheck),
        }
    }
}

impl<L: FieldCore> ExtensionOpeningReductionProof<L> {
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

/// Proof for one recursive fold level.
///
/// Intermediate levels carry the next-witness commitment and stage-1 range
/// proof. Terminal levels carry the final witness inside their terminal stage-2
/// payload and remember the scheduled final witness length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkitaLevelProof<F: FieldCore, L: FieldCore> {
    /// Intermediate recursive fold level.
    Intermediate {
        /// Optional extension-opening reduction payload. `None` for degree-one
        /// openings and proof paths that do not use extension-opening reduction.
        extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
        /// `v = D · ŵ` (ring dim = current level's D).
        v: FlatRingVec<F>,
        /// Accepted Fiat-Shamir grind nonce for fold-l∞ rejection (0 under deterministic policy).
        fold_grind_nonce: u32,
        /// Stage-1 norm-check payload.
        stage1: AkitaStage1Proof<L>,
        /// Stage-2 fused payload.
        stage2: AkitaStage2Proof<F, L>,
        /// Optional stage-3 setup product-sumcheck proof.
        stage3_sumcheck_proof: Option<SetupSumcheckProof<L>>,
    },
    /// Terminal recursive fold level.
    Terminal {
        /// Optional extension-opening reduction payload.
        extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
        /// Accepted Fiat-Shamir grind nonce for fold-l∞ rejection (0 under deterministic policy).
        fold_grind_nonce: u32,
        /// Terminal stage-2 payload.
        stage2: AkitaStage2Proof<F, L>,
        /// Scheduled final witness length in field elements.
        final_w_len: usize,
    },
}

impl<F: FieldCore, L: FieldCore> AkitaLevelProof<F, L> {
    /// Construct from typed ring elements for the current level and its
    /// inline two-stage norm-check payloads.
    pub fn new<const D: usize>(
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<L>,
        stage2: AkitaStage2Proof<F, L>,
    ) -> Self {
        Self::Intermediate {
            extension_opening_reduction: None,
            v: FlatRingVec::from_ring_elems(&v).into_compact(),
            fold_grind_nonce: 0,
            stage1,
            stage2,
            stage3_sumcheck_proof: None,
        }
    }

    /// Construct a level proof for the two-stage norm-check.
    #[allow(clippy::too_many_arguments)]
    pub fn new_two_stage<const D: usize>(
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<L>,
        stage2_sumcheck_proof: SumcheckProof<L>,
        next_w_commitment: FlatRingVec<F>,
        next_w_eval: L,
    ) -> Self {
        Self::new::<D>(
            v,
            stage1,
            AkitaStage2Proof::Intermediate(AkitaIntermediateStage2Proof {
                sumcheck_proof: stage2_sumcheck_proof,
                next_w_commitment: next_w_commitment.into_compact(),
                next_w_eval,
            }),
        )
    }

    /// Construct a level proof for a multi-row public opening relation.
    #[allow(clippy::too_many_arguments)]
    pub fn new_two_stage_many<const D: usize>(
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<L>,
        stage2_sumcheck_proof: SumcheckProof<L>,
        next_w_commitment: FlatRingVec<F>,
        next_w_eval: L,
    ) -> Self {
        Self::new_two_stage_many_with_extension_opening_reduction::<D>(
            None,
            v,
            stage1,
            stage2_sumcheck_proof,
            next_w_commitment,
            next_w_eval,
        )
    }

    /// Construct a level proof for a multi-row public opening relation with
    /// extension-opening reduction payloads already produced.
    #[allow(clippy::too_many_arguments)]
    pub fn new_two_stage_many_with_extension_opening_reduction<const D: usize>(
        extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<L>,
        stage2_sumcheck_proof: SumcheckProof<L>,
        next_w_commitment: FlatRingVec<F>,
        next_w_eval: L,
    ) -> Self {
        Self::Intermediate {
            extension_opening_reduction,
            v: FlatRingVec::from_ring_elems(&v).into_compact(),
            fold_grind_nonce: 0,
            stage1,
            stage2: AkitaStage2Proof::Intermediate(AkitaIntermediateStage2Proof {
                sumcheck_proof: stage2_sumcheck_proof,
                next_w_commitment: next_w_commitment.into_compact(),
                next_w_eval,
            }),
            stage3_sumcheck_proof: None,
        }
    }

    /// Construct a terminal level proof.
    #[allow(clippy::too_many_arguments)]
    pub fn new_terminal_with_extension_opening_reduction(
        extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
        stage2_sumcheck: SumcheckProof<L>,
        final_witness: CleartextWitnessProof<F>,
        final_w_len: usize,
        fold_grind_nonce: u32,
    ) -> Self {
        Self::Terminal {
            extension_opening_reduction,
            fold_grind_nonce,
            stage2: AkitaStage2Proof::Terminal(AkitaTerminalStage2Proof {
                sumcheck_proof: stage2_sumcheck,
                final_witness,
            }),
            final_w_len,
        }
    }

    /// Accepted fold grind nonce (`0` under deterministic policy).
    pub fn fold_grind_nonce(&self) -> u32 {
        match self {
            Self::Intermediate {
                fold_grind_nonce, ..
            } => *fold_grind_nonce,
            Self::Terminal {
                fold_grind_nonce, ..
            } => *fold_grind_nonce,
        }
    }

    /// Borrow the optional extension-opening reduction payload.
    pub fn extension_opening_reduction(&self) -> Option<&ExtensionOpeningReductionProof<L>> {
        match self {
            Self::Intermediate {
                extension_opening_reduction,
                ..
            }
            | Self::Terminal {
                extension_opening_reduction,
                ..
            } => extension_opening_reduction.as_ref(),
        }
    }

    /// M-row layout implied by this recursive proof level.
    pub fn m_row_layout(&self) -> MRowLayout {
        match self {
            Self::Intermediate { .. } => MRowLayout::WithDBlock,
            Self::Terminal { .. } => MRowLayout::WithoutDBlock,
        }
    }

    /// Borrow the intermediate `v` payload.
    ///
    /// # Panics
    ///
    /// Panics if called on a terminal proof.
    pub fn v(&self) -> &FlatRingVec<F> {
        match self {
            Self::Intermediate { v, .. } => v,
            Self::Terminal { .. } => panic!("v() called on terminal level proof"),
        }
    }

    /// Reconstruct typed `v` as a borrowed slice, returning an empty slice for
    /// terminal levels.
    pub fn v_as_ring_slice<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        match self {
            Self::Intermediate { v, .. } => v.as_ring_slice::<D>(),
            Self::Terminal { .. } => Ok(&[]),
        }
    }

    /// Mutably borrow the intermediate `v` payload.
    ///
    /// # Panics
    ///
    /// Panics if called on a terminal proof.
    pub fn v_mut(&mut self) -> &mut FlatRingVec<F> {
        match self {
            Self::Intermediate { v, .. } => v,
            Self::Terminal { .. } => panic!("v_mut() called on terminal level proof"),
        }
    }

    /// Borrow the intermediate stage-1 payload.
    ///
    /// # Panics
    ///
    /// Panics if called on a terminal proof.
    pub fn stage1(&self) -> &AkitaStage1Proof<L> {
        match self {
            Self::Intermediate { stage1, .. } => stage1,
            Self::Terminal { .. } => panic!("stage1() called on terminal level proof"),
        }
    }

    /// Borrow the intermediate stage-1 payload, if present.
    pub fn stage1_proof(&self) -> Option<&AkitaStage1Proof<L>> {
        match self {
            Self::Intermediate { stage1, .. } => Some(stage1),
            Self::Terminal { .. } => None,
        }
    }

    /// Mutably borrow the intermediate stage-1 payload.
    ///
    /// # Panics
    ///
    /// Panics if called on a terminal proof.
    pub fn stage1_mut(&mut self) -> &mut AkitaStage1Proof<L> {
        match self {
            Self::Intermediate { stage1, .. } => stage1,
            Self::Terminal { .. } => panic!("stage1_mut() called on terminal level proof"),
        }
    }

    /// Borrow the stage-2 payload.
    pub fn stage2(&self) -> &AkitaStage2Proof<F, L> {
        match self {
            Self::Intermediate { stage2, .. } | Self::Terminal { stage2, .. } => stage2,
        }
    }

    /// Mutably borrow the stage-2 payload.
    pub fn stage2_mut(&mut self) -> &mut AkitaStage2Proof<F, L> {
        match self {
            Self::Intermediate { stage2, .. } | Self::Terminal { stage2, .. } => stage2,
        }
    }

    /// Borrow the optional intermediate stage-3 setup sumcheck proof.
    pub fn stage3_sumcheck_proof(&self) -> Option<&SetupSumcheckProof<L>> {
        match self {
            Self::Intermediate {
                stage3_sumcheck_proof,
                ..
            } => stage3_sumcheck_proof.as_ref(),
            Self::Terminal { .. } => None,
        }
    }

    /// Borrow and validate the optional stage-3 setup sumcheck proof.
    pub fn stage3_for_mode<'a>(
        &'a self,
        mode: SetupContributionMode,
        next_fold_level_params: Option<&'a LevelParams>,
    ) -> Result<Option<(&'a SetupSumcheckProof<L>, &'a LevelParams)>, AkitaError> {
        match self {
            Self::Terminal { .. } => Ok(None),
            Self::Intermediate {
                stage3_sumcheck_proof,
                ..
            } => match (mode, stage3_sumcheck_proof.as_ref()) {
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
                    "recursive setup-contribution mode is missing stage3_sumcheck_proof"
                        .to_string(),
                )),
            },
        }
    }

    /// Reconstruct typed `v`, returning `InvalidProof` on shape mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored `v` payload is not
    /// well-formed for ring dimension `D`.
    pub fn try_v_typed<const D: usize>(&self) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        match self {
            Self::Intermediate { v, .. } => v.try_to_vec(),
            Self::Terminal { .. } => Err(AkitaError::InvalidProof),
        }
    }

    /// Commitment to the next witness `w`.
    pub fn next_w_commitment(&self) -> &FlatRingVec<F> {
        &self
            .stage2()
            .as_intermediate()
            .expect("next_w_commitment() called on terminal stage-2 proof")
            .next_w_commitment
    }

    /// Borrow the next witness commitment if this is an intermediate level.
    pub fn next_w_commitment_opt(&self) -> Option<&FlatRingVec<F>> {
        match self {
            Self::Intermediate { .. } => Some(self.next_w_commitment()),
            Self::Terminal { .. } => None,
        }
    }

    /// Reconstruct typed `w_commitment`.
    ///
    /// # Panics
    ///
    /// Panics if `D` does not match the stored ring dimension.
    pub fn w_commitment_typed<const D: usize>(&self) -> RingCommitment<F, D> {
        RingCommitment {
            u: self.next_w_commitment().to_vec(),
        }
    }

    /// Reconstruct typed `w_commitment`, returning `InvalidProof` on shape mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored next-level commitment
    /// is not well-formed for ring dimension `D`.
    pub fn try_w_commitment_typed<const D: usize>(
        &self,
    ) -> Result<RingCommitment<F, D>, AkitaError> {
        Ok(RingCommitment {
            u: self.next_w_commitment().try_to_vec()?,
        })
    }

    /// Claimed evaluation of the next witness `w` at the norm-check output point.
    pub fn next_w_eval(&self) -> L {
        self.stage2().next_w_eval()
    }

    /// Scheduled terminal final witness length.
    pub fn final_w_len(&self) -> Option<usize> {
        match self {
            Self::Terminal { final_w_len, .. } => Some(*final_w_len),
            Self::Intermediate { .. } => None,
        }
    }

    /// Borrow this proof if it is an intermediate recursive level.
    pub fn as_intermediate(&self) -> Option<&Self> {
        match self {
            Self::Intermediate { .. } => Some(self),
            Self::Terminal { .. } => None,
        }
    }

    /// Mutably borrow this proof if it is an intermediate recursive level.
    pub fn as_intermediate_mut(&mut self) -> Option<&mut Self> {
        match self {
            Self::Intermediate { .. } => Some(self),
            Self::Terminal { .. } => None,
        }
    }

    /// Borrow this proof if it is a terminal recursive level.
    pub fn as_terminal(&self) -> Option<&Self> {
        match self {
            Self::Intermediate { .. } => None,
            Self::Terminal { .. } => Some(self),
        }
    }

    /// Mutably borrow this proof if it is a terminal recursive level.
    pub fn as_terminal_mut(&mut self) -> Option<&mut Self> {
        match self {
            Self::Intermediate { .. } => None,
            Self::Terminal { .. } => Some(self),
        }
    }

    /// Derive the [`LevelProofShape`] for this level proof.
    pub fn shape(&self) -> LevelProofShape {
        let Self::Intermediate {
            extension_opening_reduction,
            v,
            stage1,
            stage2,
            stage3_sumcheck_proof,
            ..
        } = self
        else {
            panic!("shape() called on terminal level proof");
        };
        level_proof_shape(
            extension_opening_reduction.as_ref(),
            v,
            stage1,
            stage2,
            stage3_sumcheck_proof.as_ref(),
        )
    }

    /// Derive the [`TerminalLevelProofShape`] for a terminal level proof.
    pub fn terminal_shape(&self) -> TerminalLevelProofShape {
        let Self::Terminal {
            extension_opening_reduction,
            stage2,
            ..
        } = self
        else {
            panic!("terminal_shape() called on intermediate level proof");
        };
        TerminalLevelProofShape {
            extension_opening_reduction: extension_opening_reduction
                .as_ref()
                .map(ExtensionOpeningReductionProof::shape),
            stage2_sumcheck: { sumcheck_shape(stage2.sumcheck()) },
            final_witness: self
                .stage2()
                .final_witness()
                .expect("terminal level proof must carry final witness")
                .shape(),
        }
    }

    /// Derive the shape for this recursive level proof.
    pub fn step_shape(&self) -> AkitaProofStepShape {
        match self {
            Self::Intermediate { .. } => AkitaProofStepShape::Intermediate(self.shape()),
            Self::Terminal { .. } => AkitaProofStepShape::Terminal(self.terminal_shape()),
        }
    }
}

/// Terminal fold-level proof.
///
/// Ships `final_witness` in cleartext, absorbed into the transcript at the
/// `ABSORB_NEXT_LEVEL_WITNESS_BINDING` position in place of the prior `next_w_commitment`.
/// Drops the redundant proof components at the terminal: `stage1`
/// (segment-typed tail encodes digit range), `next_w_commitment`
/// (replaced by `final_witness`), and `next_w_eval` (verifier computes
/// directly from `final_witness`). The terminal M-row layout also drops the
/// D-row block, so `v` is not serialized at the terminal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalLevelProof<F: FieldCore, L: FieldCore> {
    /// Optional extension-opening reduction payload.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
    /// Accepted Fiat-Shamir grind nonce for fold-l∞ rejection (0 under deterministic policy).
    pub fold_grind_nonce: u32,
    /// Terminal stage-2 payload.
    pub stage2: AkitaStage2Proof<F, L>,
}

impl<F: FieldCore, L: FieldCore> TerminalLevelProof<F, L> {
    /// Construct from typed ring elements and a terminal cleartext witness.
    ///
    /// Pass `extension_opening_reduction = None` for opening shapes that do
    /// not use extension-opening reduction.
    pub fn new_with_extension_opening_reduction(
        extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
        stage2_sumcheck: SumcheckProof<L>,
        final_witness: CleartextWitnessProof<F>,
        fold_grind_nonce: u32,
    ) -> Self {
        Self {
            extension_opening_reduction,
            fold_grind_nonce,
            stage2: AkitaStage2Proof::Terminal(AkitaTerminalStage2Proof {
                sumcheck_proof: stage2_sumcheck,
                final_witness,
            }),
        }
    }

    /// Borrow the terminal cleartext witness.
    pub fn final_witness(&self) -> &CleartextWitnessProof<F> {
        self.stage2
            .final_witness()
            .expect("final_witness() called on intermediate stage-2 proof")
    }

    /// Derive the [`TerminalLevelProofShape`] for this terminal-level proof.
    pub fn shape(&self) -> TerminalLevelProofShape {
        TerminalLevelProofShape {
            extension_opening_reduction: self
                .extension_opening_reduction
                .as_ref()
                .map(ExtensionOpeningReductionProof::shape),
            stage2_sumcheck: { sumcheck_shape(self.stage2.sumcheck()) },
            final_witness: self.final_witness().shape(),
        }
    }
}

/// Fused batched-root payload for the two-stage folding protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaBatchedFoldRoot<F: FieldCore, L: FieldCore> {
    /// Optional extension-opening reduction payload. `None` until the
    /// extension-opening reduction cutover is wired into the root path.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
    /// Aggregated `v = D · ŵ`.
    pub v: FlatRingVec<F>,
    /// Accepted Fiat-Shamir grind nonce for fold-l∞ rejection (0 under deterministic policy).
    pub fold_grind_nonce: u32,
    /// Stage-1 norm-check payload.
    pub stage1: AkitaStage1Proof<L>,
    /// Stage-2 fused payload.
    pub stage2: AkitaStage2Proof<F, L>,
    /// Optional stage-3 setup product-sumcheck proof.
    pub stage3_sumcheck_proof: Option<SetupSumcheckProof<L>>,
}

/// Root proof payload for fused batched openings.
///
/// Three-way split:
///
/// * `Fold` — standard two-stage folded root proof followed by intermediate
///   steps and a terminal step.
/// * `Terminal` — 1-fold case where the root itself is the terminal level.
///   No recursive-suffix steps follow.
/// * `ZeroFold` — 0-fold batched fast path: one cleartext field-element
///   witness per claim, in the normalized opening batch claim order
///   used by the prover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkitaBatchedRootProof<F: FieldCore, L: FieldCore> {
    /// Standard two-stage folded root proof.
    Fold(AkitaBatchedFoldRoot<F, L>),
    /// 1-fold root: the root level is itself the terminal fold level.
    Terminal(TerminalLevelProof<F, L>),
    /// Zero-fold batched fast path: one cleartext field-element witness per
    /// claim, in the normalized opening batch claim order used by the prover.
    ZeroFold {
        /// Per-claim cleartext witnesses.
        witnesses: Vec<CleartextWitnessProof<F>>,
    },
}

impl<F: FieldCore, L: FieldCore> AkitaBatchedRootProof<F, L> {
    /// Construct a batched root proof from the root fold-level payload.
    pub fn new(root: AkitaLevelProof<F, L>) -> Self {
        let AkitaLevelProof::Intermediate {
            extension_opening_reduction,
            v,
            fold_grind_nonce,
            stage1,
            stage2,
            stage3_sumcheck_proof,
        } = root
        else {
            panic!("AkitaBatchedRootProof::new called with terminal level proof");
        };
        Self::Fold(AkitaBatchedFoldRoot {
            extension_opening_reduction,
            v,
            fold_grind_nonce,
            stage1,
            stage2,
            stage3_sumcheck_proof,
        })
    }

    /// Attach a stage-3 setup sumcheck proof to a folded root proof.
    pub fn with_stage3_sumcheck_proof(
        mut self,
        stage3_sumcheck_proof: Option<SetupSumcheckProof<L>>,
    ) -> Self {
        if let Self::Fold(fold) = &mut self {
            fold.stage3_sumcheck_proof = stage3_sumcheck_proof;
        }
        self
    }

    /// Attach extension-opening reduction payloads to a folded root proof.
    pub fn with_extension_opening_reduction(
        mut self,
        extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
    ) -> Self {
        if let Self::Fold(fold) = &mut self {
            fold.extension_opening_reduction = extension_opening_reduction;
        }
        self
    }

    /// Construct the terminal-root variant (1-fold case): the root itself is
    /// the terminal fold level.
    pub fn new_terminal(terminal: TerminalLevelProof<F, L>) -> Self {
        Self::Terminal(terminal)
    }

    /// Construct the zero-fold batched variant with one witness per claim.
    pub fn new_zero_fold(witnesses: Vec<CleartextWitnessProof<F>>) -> Self {
        Self::ZeroFold { witnesses }
    }

    /// Borrow the fold payload when this is a fold root.
    pub fn as_fold(&self) -> Option<&AkitaBatchedFoldRoot<F, L>> {
        match self {
            Self::Fold(fold) => Some(fold),
            Self::Terminal(_) | Self::ZeroFold { .. } => None,
        }
    }

    /// Mutably borrow the fold payload when this is a fold root.
    pub fn as_fold_mut(&mut self) -> Option<&mut AkitaBatchedFoldRoot<F, L>> {
        match self {
            Self::Fold(fold) => Some(fold),
            Self::Terminal(_) | Self::ZeroFold { .. } => None,
        }
    }

    /// Borrow the terminal-root payload when this is a terminal root.
    pub fn as_terminal_root(&self) -> Option<&TerminalLevelProof<F, L>> {
        match self {
            Self::Terminal(terminal) => Some(terminal),
            Self::Fold(_) | Self::ZeroFold { .. } => None,
        }
    }

    /// Mutably borrow the terminal-root payload when this is a terminal root.
    pub fn as_terminal_root_mut(&mut self) -> Option<&mut TerminalLevelProof<F, L>> {
        match self {
            Self::Terminal(terminal) => Some(terminal),
            Self::Fold(_) | Self::ZeroFold { .. } => None,
        }
    }

    /// Accepted fold grind nonce for root proofs that run fold challenge sampling.
    pub fn fold_grind_nonce(&self) -> Result<u32, AkitaError> {
        match self {
            Self::Fold(fold) => Ok(fold.fold_grind_nonce),
            Self::Terminal(terminal) => Ok(terminal.fold_grind_nonce),
            Self::ZeroFold { .. } => Err(AkitaError::InvalidProof),
        }
    }

    /// Borrow the per-claim cleartext witnesses when this is a zero-fold
    /// batched proof.
    pub fn as_zero_fold(&self) -> Option<&[CleartextWitnessProof<F>]> {
        match self {
            Self::ZeroFold { witnesses, .. } => Some(witnesses.as_slice()),
            Self::Fold(_) | Self::Terminal(_) => None,
        }
    }

    /// Row layout used by the root fold verifier for fold and terminal-root proofs.
    pub fn fold_m_row_layout(&self) -> Option<MRowLayout> {
        match self {
            Self::Fold(_) => Some(MRowLayout::WithDBlock),
            Self::Terminal(_) => Some(MRowLayout::WithoutDBlock),
            Self::ZeroFold { .. } => None,
        }
    }

    /// Borrow the optional root extension-opening reduction payload.
    pub fn fold_extension_opening_reduction(&self) -> Option<&ExtensionOpeningReductionProof<L>> {
        match self {
            Self::Fold(fold) => fold.extension_opening_reduction.as_ref(),
            Self::Terminal(terminal) => terminal.extension_opening_reduction.as_ref(),
            Self::ZeroFold { .. } => None,
        }
    }

    /// Borrow typed root `v` for fold proofs; terminal roots have no D-block rows.
    pub fn fold_v<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        match self {
            Self::Fold(fold) => fold.v.as_ring_slice::<D>(),
            Self::Terminal(_) => Ok(&[]),
            Self::ZeroFold { .. } => Err(AkitaError::InvalidProof),
        }
    }

    /// Borrow root stage-1 for fold proofs; terminal roots run relation-only stage 2.
    pub fn fold_stage1(&self) -> Result<Option<&AkitaStage1Proof<L>>, AkitaError> {
        match self {
            Self::Fold(fold) => Ok(Some(&fold.stage1)),
            Self::Terminal(_) => Ok(None),
            Self::ZeroFold { .. } => Err(AkitaError::InvalidProof),
        }
    }

    /// Borrow the root next-witness commitment for fold proofs.
    pub fn fold_next_w_commitment(&self) -> Result<Option<&FlatRingVec<F>>, AkitaError> {
        match self {
            Self::Fold(fold) => Ok(Some(
                &fold
                    .stage2
                    .as_intermediate()
                    .ok_or(AkitaError::InvalidProof)?
                    .next_w_commitment,
            )),
            Self::Terminal(_) => Ok(None),
            Self::ZeroFold { .. } => Err(AkitaError::InvalidProof),
        }
    }

    /// Borrow root stage-2 for fold and terminal-root proofs.
    pub fn fold_stage2(&self) -> Result<&AkitaStage2Proof<F, L>, AkitaError> {
        match self {
            Self::Fold(fold) => Ok(&fold.stage2),
            Self::Terminal(terminal) => Ok(&terminal.stage2),
            Self::ZeroFold { .. } => Err(AkitaError::InvalidProof),
        }
    }

    /// Borrow and validate the optional root stage-3 setup sumcheck proof.
    pub fn fold_stage3_sumcheck_proof(
        &self,
        mode: SetupContributionMode,
    ) -> Result<Option<&SetupSumcheckProof<L>>, AkitaError> {
        match self {
            Self::Fold(fold) => match (mode, fold.stage3_sumcheck_proof.as_ref()) {
                (SetupContributionMode::Direct, None) => Ok(None),
                (SetupContributionMode::Direct, Some(_)) => Err(AkitaError::InvalidSetup(
                    "direct setup-contribution mode received stage3_sumcheck_proof".to_string(),
                )),
                (SetupContributionMode::Recursive, Some(proof)) => Ok(Some(proof)),
                (SetupContributionMode::Recursive, None) => Err(AkitaError::InvalidSetup(
                    "recursive setup-contribution mode is missing stage3_sumcheck_proof"
                        .to_string(),
                )),
            },
            Self::Terminal(_) => Ok(None),
            Self::ZeroFold { .. } => Err(AkitaError::InvalidProof),
        }
    }

    /// True when this root proof is a zero-fold batched fast path.
    pub fn is_zero_fold(&self) -> bool {
        matches!(self, Self::ZeroFold { .. })
    }

    /// True when this root proof is itself the terminal fold level.
    pub fn is_terminal_root(&self) -> bool {
        matches!(self, Self::Terminal(_))
    }

    /// Borrow the stored root `v` ring vector (Fold only).
    ///
    /// # Panics
    ///
    /// Panics on terminal-root and zero-fold batched proofs.
    pub fn v(&self) -> &FlatRingVec<F> {
        &self
            .as_fold()
            .expect("v() called on a non-fold root proof")
            .v
    }

    /// Commitment to the next witness `w` (Fold only).
    ///
    /// # Panics
    ///
    /// Panics on terminal-root and zero-fold batched proofs.
    pub fn next_w_commitment(&self) -> &FlatRingVec<F> {
        &self
            .as_fold()
            .expect("next_w_commitment() called on a non-fold root proof")
            .stage2
            .as_intermediate()
            .expect("next_w_commitment() called on terminal stage-2 proof")
            .next_w_commitment
    }

    /// Claimed evaluation of the next witness `w` (Fold only).
    ///
    /// # Panics
    ///
    /// Panics on terminal-root and zero-fold batched proofs.
    pub fn next_w_eval(&self) -> L {
        self.as_fold()
            .expect("next_w_eval() called on a non-fold root proof")
            .stage2
            .next_w_eval()
    }
}

impl<F: FieldCore, L: FieldCore> AkitaBatchedFoldRoot<F, L> {
    /// Derive the [`LevelProofShape`] for this fold root.
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

/// Akita PCS proof for fused batched openings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaBatchedProof<F: FieldCore, L: FieldCore> {
    /// Batched root proof over all original-polynomial claims.
    pub root: AkitaBatchedRootProof<F, L>,
    /// Recursive proof steps following the batched root proof.
    pub steps: Vec<AkitaLevelProof<F, L>>,
}

impl<F: FieldCore, L: FieldCore> AkitaBatchedProof<F, L> {
    /// Access the terminal cleartext witness of the recursive-suffix path.
    ///
    /// Returns the `final_witness` from the terminal level: either the
    /// terminal step at the tail of a fold-rooted suffix, or directly from
    /// the [`AkitaBatchedRootProof::Terminal`] root (1-fold case).
    ///
    /// # Panics
    ///
    /// Panics on a zero-fold batched proof (use
    /// [`AkitaBatchedRootProof::as_zero_fold`] to access the per-claim witnesses
    /// in that case), and panics if a fold-rooted proof does not terminate
    /// with a terminal step.
    pub fn final_witness(&self) -> &CleartextWitnessProof<F> {
        match &self.root {
            AkitaBatchedRootProof::Terminal(terminal) => terminal.final_witness(),
            AkitaBatchedRootProof::Fold(_) => self
                .steps
                .last()
                .and_then(AkitaLevelProof::as_terminal)
                .expect("fold-rooted Akita proof must terminate with a terminal step")
                .stage2()
                .final_witness()
                .expect("terminal Akita level proof must carry final witness"),
            AkitaBatchedRootProof::ZeroFold { .. } => {
                panic!("final_witness() called on a zero-fold batched proof")
            }
        }
    }

    /// Iterate over the intermediate (non-terminal) fold levels of the
    /// recursive suffix.
    pub fn fold_levels(&self) -> impl Iterator<Item = &AkitaLevelProof<F, L>> {
        self.steps
            .iter()
            .filter_map(AkitaLevelProof::as_intermediate)
    }

    /// Number of intermediate recursive fold levels.
    pub fn num_fold_levels(&self) -> usize {
        self.fold_levels().count()
    }

    /// True when this proof uses the zero-fold batched fast path (no
    /// two-stage root proof and no recursive suffix).
    pub fn is_root_direct(&self) -> bool {
        self.root.is_zero_fold()
    }

    /// True when the batched root is itself the terminal fold level (1-fold
    /// case).
    pub fn is_root_terminal(&self) -> bool {
        self.root.is_terminal_root()
    }

    /// Derive the [`AkitaBatchedProofShape`] for this proof.
    pub fn shape(&self) -> AkitaBatchedProofShape {
        match &self.root {
            AkitaBatchedRootProof::Fold(fold) => AkitaBatchedProofShape::Fold {
                root_shape: fold.shape(),
                step_shapes: self.steps.iter().map(AkitaLevelProof::step_shape).collect(),
            },
            AkitaBatchedRootProof::Terminal(terminal) => {
                AkitaBatchedProofShape::Terminal(terminal.shape())
            }
            AkitaBatchedRootProof::ZeroFold { witnesses, .. } => AkitaBatchedProofShape::ZeroFold {
                witness_shapes: witnesses.iter().map(CleartextWitnessProof::shape).collect(),
            },
        }
    }
}

impl<F: FieldCore + CanonicalField + AkitaSerialize, L: FieldCore + AkitaSerialize>
    AkitaBatchedProof<F, L>
{
    /// Returns the proof size in bytes (uncompressed).
    pub fn size(&self) -> usize {
        self.serialized_size(Compress::No)
    }
}
