use super::shapes::level_proof_shape;
#[cfg(feature = "zk")]
use super::shapes::sumcheck_proof_masked_shape;
#[cfg(not(feature = "zk"))]
use super::shapes::sumcheck_shape;
use super::*;
use crate::BasisMode;

/// One stage in the stage-1 range-check tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaStage1StageProof<F: FieldCore> {
    /// Eq-factored sumcheck proof for this stage.
    #[cfg(not(feature = "zk"))]
    pub sumcheck_proof: EqFactoredSumcheckProof<F>,
    /// ZK plain-opening masked round payload.
    #[cfg(feature = "zk")]
    pub sumcheck_proof_masked: EqFactoredSumcheckProofMasked<F>,
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

/// Proof payload for stage 2 of a single Akita level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaStage2Proof<F: FieldCore, L: FieldCore> {
    /// Stage-2 fused sumcheck proof.
    #[cfg(not(feature = "zk"))]
    pub sumcheck_proof: SumcheckProof<L>,
    /// ZK plain-opening masked compressed round payload.
    #[cfg(feature = "zk")]
    pub sumcheck_proof_masked: SumcheckProofMasked<L>,
    /// Commitment to the next witness `w`
    /// (ring dim = next level's D, may differ from y_ring/v).
    pub next_w_commitment: FlatRingVec<F>,
    /// Claimed evaluation of the next witness `w` at the stage-2 challenge point.
    #[cfg(not(feature = "zk"))]
    pub next_w_eval: L,
    /// Masked claimed evaluation of the next witness `w` at the stage-2 challenge point.
    #[cfg(feature = "zk")]
    pub next_w_eval_masked: L,
    /// Additional non-witness opening claims carried into the next recursive level.
    pub extra_carried_openings: Vec<CarriedOpeningProof<F, L>>,
}

/// Proof-visible opening claim carried into the next recursive level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CarriedOpeningProof<F: FieldCore, L: FieldCore> {
    /// Commitment opened by this claim.
    pub commitment: FlatRingVec<F>,
    /// Evaluation point for the carried claim.
    pub point: Vec<L>,
    /// Claimed opening value at `point`.
    pub value: L,
    /// Basis used to interpret the point.
    pub basis: BasisMode,
    /// Unpadded logical field length of the opened object.
    pub natural_len: usize,
    /// Common padded field-domain length used by the carried batch.
    pub padded_len: usize,
    /// Logical source of this carried opening.
    pub kind: CarriedOpeningKind,
}

impl<F: FieldCore, L: FieldCore> AkitaStage2Proof<F, L> {
    /// Wire value for the next-witness evaluation claim.
    ///
    /// In transparent builds this is the true evaluation; in ZK builds this is
    /// the masked evaluation carried on the proof transcript.
    pub fn next_w_eval(&self) -> L {
        #[cfg(not(feature = "zk"))]
        {
            self.next_w_eval
        }
        #[cfg(feature = "zk")]
        {
            self.next_w_eval_masked
        }
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
    #[cfg(not(feature = "zk"))]
    pub sumcheck: SumcheckProof<L>,
    /// ZK plain-opening masked compressed degree-two reduction sumcheck.
    #[cfg(feature = "zk")]
    pub sumcheck_proof_masked: SumcheckProofMasked<L>,
}

impl<L: FieldCore> ExtensionOpeningReductionProof<L> {
    /// Shape descriptor required for headerless deserialization.
    pub fn shape(&self) -> ExtensionOpeningReductionShape {
        ExtensionOpeningReductionShape {
            partials: self.partials.len(),
            #[cfg(not(feature = "zk"))]
            sumcheck: sumcheck_shape(&self.sumcheck),
            #[cfg(feature = "zk")]
            sumcheck: sumcheck_proof_masked_shape(&self.sumcheck_proof_masked),
        }
    }

    /// Number of sumcheck rounds in the reduction proof.
    pub fn num_rounds(&self) -> usize {
        #[cfg(not(feature = "zk"))]
        {
            self.sumcheck.round_polys.len()
        }
        #[cfg(feature = "zk")]
        {
            self.sumcheck_proof_masked.masked_round_polys.len()
        }
    }
}

/// Proof for a single fold level (quad_eq + ring_switch + sumcheck).
///
/// D-agnostic: proof-owned ring vectors are stored in compact mode
/// (`ring_dim = 0`), and callers recover the typed ring dimension from the
/// surrounding proof shape or runtime context.
///
/// One recursive Akita level proof with inline stage-1 and stage-2 sumchecks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaLevelProof<F: FieldCore, L: FieldCore> {
    /// `y_ring` from the §3.1 reduction (ring dim = current level's D).
    pub y_ring: FlatRingVec<F>,
    /// Optional extension-opening reduction payload. `None` for degree-one
    /// openings and proof paths that do not use extension-opening reduction.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
    /// `v = D · ŵ` (ring dim = current level's D).
    pub v: FlatRingVec<F>,
    /// Stage-1 norm-check payload.
    pub stage1: AkitaStage1Proof<L>,
    /// Stage-2 fused payload.
    pub stage2: AkitaStage2Proof<F, L>,
}

impl<F: FieldCore, L: FieldCore> AkitaLevelProof<F, L> {
    /// Construct from typed ring elements for the current level and its
    /// inline two-stage norm-check payloads.
    pub fn new<const D: usize>(
        y_ring: CyclotomicRing<F, D>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<L>,
        stage2: AkitaStage2Proof<F, L>,
    ) -> Self {
        Self {
            y_ring: FlatRingVec::from_single(&y_ring).into_compact(),
            extension_opening_reduction: None,
            v: FlatRingVec::from_ring_elems(&v).into_compact(),
            stage1,
            stage2,
        }
    }

    /// Construct a level proof for the two-stage norm-check.
    #[allow(clippy::too_many_arguments)]
    pub fn new_two_stage<const D: usize>(
        y_ring: CyclotomicRing<F, D>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<L>,
        #[cfg(not(feature = "zk"))] stage2_sumcheck_proof: SumcheckProof<L>,
        #[cfg(feature = "zk")] stage2_sumcheck_proof_masked: SumcheckProofMasked<L>,
        next_w_commitment: FlatRingVec<F>,
        next_w_eval: L,
    ) -> Self {
        Self::new::<D>(
            y_ring,
            v,
            stage1,
            AkitaStage2Proof {
                #[cfg(not(feature = "zk"))]
                sumcheck_proof: stage2_sumcheck_proof,
                #[cfg(feature = "zk")]
                sumcheck_proof_masked: stage2_sumcheck_proof_masked,
                next_w_commitment: next_w_commitment.into_compact(),
                #[cfg(not(feature = "zk"))]
                next_w_eval,
                #[cfg(feature = "zk")]
                next_w_eval_masked: next_w_eval,
                extra_carried_openings: Vec::new(),
            },
        )
    }

    /// Construct a level proof for a multi-row public opening relation.
    ///
    /// The singleton recursive path is the `y_rings.len() == 1`
    /// specialization.
    #[allow(clippy::too_many_arguments)]
    pub fn new_two_stage_many<const D: usize>(
        y_rings: Vec<CyclotomicRing<F, D>>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<L>,
        #[cfg(not(feature = "zk"))] stage2_sumcheck_proof: SumcheckProof<L>,
        #[cfg(feature = "zk")] stage2_sumcheck_proof_masked: SumcheckProofMasked<L>,
        next_w_commitment: FlatRingVec<F>,
        next_w_eval: L,
    ) -> Self {
        Self::new_two_stage_many_with_extension_opening_reduction::<D>(
            y_rings,
            None,
            v,
            stage1,
            #[cfg(not(feature = "zk"))]
            stage2_sumcheck_proof,
            #[cfg(feature = "zk")]
            stage2_sumcheck_proof_masked,
            next_w_commitment,
            next_w_eval,
        )
    }

    /// Construct a level proof for a multi-row public opening relation with
    /// extension-opening reduction payloads already produced.
    #[allow(clippy::too_many_arguments)]
    pub fn new_two_stage_many_with_extension_opening_reduction<const D: usize>(
        y_rings: Vec<CyclotomicRing<F, D>>,
        extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<L>,
        #[cfg(not(feature = "zk"))] stage2_sumcheck_proof: SumcheckProof<L>,
        #[cfg(feature = "zk")] stage2_sumcheck_proof_masked: SumcheckProofMasked<L>,
        next_w_commitment: FlatRingVec<F>,
        next_w_eval: L,
    ) -> Self {
        Self {
            y_ring: FlatRingVec::from_ring_elems(&y_rings).into_compact(),
            extension_opening_reduction,
            v: FlatRingVec::from_ring_elems(&v).into_compact(),
            stage1,
            stage2: AkitaStage2Proof {
                #[cfg(not(feature = "zk"))]
                sumcheck_proof: stage2_sumcheck_proof,
                #[cfg(feature = "zk")]
                sumcheck_proof_masked: stage2_sumcheck_proof_masked,
                next_w_commitment: next_w_commitment.into_compact(),
                #[cfg(not(feature = "zk"))]
                next_w_eval,
                #[cfg(feature = "zk")]
                next_w_eval_masked: next_w_eval,
                extra_carried_openings: Vec::new(),
            },
        }
    }

    /// Reconstruct typed `y_ring`.
    ///
    /// # Panics
    ///
    /// Panics if `D` does not match the stored ring dimension.
    pub fn y_ring_typed<const D: usize>(&self) -> CyclotomicRing<F, D> {
        self.y_ring.to_single()
    }

    /// Reconstruct typed `y_ring`, returning `InvalidProof` on shape mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored `y_ring` does not
    /// encode exactly one ring element at dimension `D`.
    pub fn try_y_ring_typed<const D: usize>(&self) -> Result<CyclotomicRing<F, D>, AkitaError> {
        self.y_ring.try_to_single()
    }

    /// Reconstruct typed public opening rings.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored payload is not
    /// well-formed for ring dimension `D`.
    pub fn try_y_rings_typed<const D: usize>(
        &self,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        self.y_ring.try_to_vec()
    }

    /// Reconstruct typed `v`.
    ///
    /// # Panics
    ///
    /// Panics if `D` does not match the stored ring dimension.
    pub fn v_typed<const D: usize>(&self) -> Vec<CyclotomicRing<F, D>> {
        self.v.to_vec()
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

    /// Commitment to the next witness `w`.
    pub fn next_w_commitment(&self) -> &FlatRingVec<F> {
        &self.stage2.next_w_commitment
    }

    /// Number of stored field coefficients for the next witness commitment.
    pub fn next_w_commitment_coeff_len(&self) -> usize {
        self.stage2.next_w_commitment.coeff_len()
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
        self.stage2.next_w_eval()
    }

    /// Derive the [`LevelProofShape`] for this level proof.
    pub fn shape(&self) -> LevelProofShape {
        level_proof_shape(
            self.y_ring.coeff_len(),
            self.extension_opening_reduction.as_ref(),
            &self.v,
            &self.stage1,
            &self.stage2,
        )
    }
}

/// Terminal fold-level proof.
///
/// Ships `final_witness` in cleartext, absorbed into the transcript at the
/// `ABSORB_SUMCHECK_W` position in place of the prior `next_w_commitment`.
/// Drops the redundant proof components at the terminal: `stage1`
/// (`PackedDigits` structurally enforces digit range), `next_w_commitment`
/// (replaced by `final_witness`), and `next_w_eval` (verifier computes
/// directly from `final_witness`). The terminal M-row layout also drops the
/// D-row block, so `v` is not serialized at the terminal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalLevelProof<F: FieldCore, L: FieldCore> {
    /// Public output ring(s). At a non-root terminal step this carries
    /// exactly one ring; at the root terminal (1-fold case) it carries one
    /// ring per opening point.
    pub y_rings: FlatRingVec<F>,
    /// Optional extension-opening reduction payload.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
    /// Stage-2 fused sumcheck proof.
    #[cfg(not(feature = "zk"))]
    pub stage2_sumcheck: SumcheckProof<L>,
    #[cfg(feature = "zk")]
    pub stage2_sumcheck_proof_masked: SumcheckProofMasked<L>,
    /// Terminal witness, absorbed via `ABSORB_SUMCHECK_W` in place of
    /// `next_w_commitment`.
    pub final_witness: DirectWitnessProof<F>,
}

impl<F: FieldCore, L: FieldCore> TerminalLevelProof<F, L> {
    /// Construct from typed ring elements and a terminal direct witness.
    ///
    /// Pass `extension_opening_reduction = None` for opening shapes that do
    /// not use extension-opening reduction.
    pub fn new_with_extension_opening_reduction<const D: usize>(
        y_rings: Vec<CyclotomicRing<F, D>>,
        extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
        #[cfg(not(feature = "zk"))] stage2_sumcheck: SumcheckProof<L>,
        #[cfg(feature = "zk")] stage2_sumcheck_proof_masked: SumcheckProofMasked<L>,
        final_witness: DirectWitnessProof<F>,
    ) -> Self {
        Self {
            y_rings: FlatRingVec::from_ring_elems(&y_rings).into_compact(),
            extension_opening_reduction,
            #[cfg(not(feature = "zk"))]
            stage2_sumcheck,
            #[cfg(feature = "zk")]
            stage2_sumcheck_proof_masked,
            final_witness,
        }
    }

    /// Reconstruct typed public opening rings.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored payload is not
    /// well-formed for ring dimension `D`.
    pub fn try_y_rings_typed<const D: usize>(
        &self,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        self.y_rings.try_to_vec()
    }

    /// Derive the [`TerminalLevelProofShape`] for this terminal-level proof.
    pub fn shape(&self) -> TerminalLevelProofShape {
        TerminalLevelProofShape {
            y_rings_coeffs: self.y_rings.coeff_len(),
            extension_opening_reduction: self
                .extension_opening_reduction
                .as_ref()
                .map(ExtensionOpeningReductionProof::shape),
            stage2_sumcheck: {
                #[cfg(not(feature = "zk"))]
                {
                    sumcheck_shape(&self.stage2_sumcheck)
                }
                #[cfg(feature = "zk")]
                {
                    sumcheck_proof_masked_shape(&self.stage2_sumcheck_proof_masked)
                }
            },
            final_witness: self.final_witness.shape(),
        }
    }
}

/// Fused batched-root payload for the two-stage folding protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaBatchedFoldRoot<F: FieldCore, L: FieldCore> {
    /// Per-point batched public outputs `(y_j)_j`, stored as a flat ring vector.
    pub y_rings: FlatRingVec<F>,
    /// Optional extension-opening reduction payload. `None` until the
    /// extension-opening reduction cutover is wired into the root path.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
    /// Aggregated `v = Σ_ell D_ell · w_hat_ell`.
    pub v: FlatRingVec<F>,
    /// Stage-1 norm-check payload.
    pub stage1: AkitaStage1Proof<L>,
    /// Stage-2 fused payload.
    pub stage2: AkitaStage2Proof<F, L>,
}

/// Root proof payload for fused batched openings.
///
/// Three-way split:
///
/// * `Fold` — standard two-stage folded root proof followed by intermediate
///   steps and a terminal step.
/// * `Terminal` — 1-fold case where the root itself is the terminal level.
///   No recursive-suffix steps follow.
/// * `Direct` — 0-fold (root-direct) batched fast path: one direct
///   field-element witness per claim, in the normalized incidence claim order
///   used by the prover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkitaBatchedRootProof<F: FieldCore, L: FieldCore> {
    /// Standard two-stage folded root proof.
    Fold(AkitaBatchedFoldRoot<F, L>),
    /// 1-fold root: the root level is itself the terminal fold level.
    Terminal(TerminalLevelProof<F, L>),
    /// Root-direct batched fast path: one direct field-element witness per
    /// claim, in the normalized incidence claim order used by the prover.
    Direct {
        /// Per-claim direct witnesses.
        witnesses: Vec<DirectWitnessProof<F>>,
        /// Per-commitment B-blinding digit streams revealed for verifier
        /// recommitment in the root-direct zk fast path.
        #[cfg(feature = "zk")]
        b_blinding_digits: Vec<Vec<i8>>,
    },
}

impl<F: FieldCore, L: FieldCore> AkitaBatchedRootProof<F, L> {
    /// Construct from typed ring elements for the batched root level.
    pub fn new<const D: usize>(
        y_rings: Vec<CyclotomicRing<F, D>>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<L>,
        stage2: AkitaStage2Proof<F, L>,
    ) -> Self {
        Self::Fold(AkitaBatchedFoldRoot {
            y_rings: FlatRingVec::from_ring_elems(&y_rings).into_compact(),
            extension_opening_reduction: None,
            v: FlatRingVec::from_ring_elems(&v).into_compact(),
            stage1,
            stage2,
        })
    }

    /// Construct a batched root proof for the two-stage norm-check.
    #[allow(clippy::too_many_arguments)]
    pub fn new_two_stage<const D: usize>(
        y_rings: Vec<CyclotomicRing<F, D>>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<L>,
        #[cfg(not(feature = "zk"))] stage2_sumcheck_proof: SumcheckProof<L>,
        #[cfg(feature = "zk")] stage2_sumcheck_proof_masked: SumcheckProofMasked<L>,
        next_w_commitment: FlatRingVec<F>,
        next_w_eval: L,
    ) -> Self {
        Self::new_two_stage_with_extension_opening_reduction::<D>(
            y_rings,
            None,
            v,
            stage1,
            #[cfg(not(feature = "zk"))]
            stage2_sumcheck_proof,
            #[cfg(feature = "zk")]
            stage2_sumcheck_proof_masked,
            next_w_commitment,
            next_w_eval,
        )
    }

    /// Construct a batched root proof for the two-stage norm-check with
    /// extension-opening reduction payloads already produced.
    #[allow(clippy::too_many_arguments)]
    pub fn new_two_stage_with_extension_opening_reduction<const D: usize>(
        y_rings: Vec<CyclotomicRing<F, D>>,
        extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<L>,
        #[cfg(not(feature = "zk"))] stage2_sumcheck_proof: SumcheckProof<L>,
        #[cfg(feature = "zk")] stage2_sumcheck_proof_masked: SumcheckProofMasked<L>,
        next_w_commitment: FlatRingVec<F>,
        next_w_eval: L,
    ) -> Self {
        Self::new::<D>(
            y_rings,
            v,
            stage1,
            AkitaStage2Proof {
                #[cfg(not(feature = "zk"))]
                sumcheck_proof: stage2_sumcheck_proof,
                #[cfg(feature = "zk")]
                sumcheck_proof_masked: stage2_sumcheck_proof_masked,
                next_w_commitment: next_w_commitment.into_compact(),
                #[cfg(not(feature = "zk"))]
                next_w_eval,
                #[cfg(feature = "zk")]
                next_w_eval_masked: next_w_eval,
                extra_carried_openings: Vec::new(),
            },
        )
        .with_extension_opening_reduction(extension_opening_reduction)
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

    /// Construct the root-direct batched variant with one witness per claim.
    #[cfg(not(feature = "zk"))]
    pub fn new_direct(witnesses: Vec<DirectWitnessProof<F>>) -> Self {
        Self::Direct { witnesses }
    }

    /// Construct the root-direct batched variant with one witness per claim and
    /// one revealed B-blinding payload per opening-point commitment.
    #[cfg(feature = "zk")]
    pub fn new_direct(
        witnesses: Vec<DirectWitnessProof<F>>,
        b_blinding_digits: Vec<Vec<i8>>,
    ) -> Self {
        Self::Direct {
            witnesses,
            b_blinding_digits,
        }
    }

    /// Borrow the fold payload when this is a fold root.
    pub fn as_fold(&self) -> Option<&AkitaBatchedFoldRoot<F, L>> {
        match self {
            Self::Fold(fold) => Some(fold),
            Self::Terminal(_) | Self::Direct { .. } => None,
        }
    }

    /// Mutably borrow the fold payload when this is a fold root.
    pub fn as_fold_mut(&mut self) -> Option<&mut AkitaBatchedFoldRoot<F, L>> {
        match self {
            Self::Fold(fold) => Some(fold),
            Self::Terminal(_) | Self::Direct { .. } => None,
        }
    }

    /// Borrow the terminal-root payload when this is a terminal root.
    pub fn as_terminal_root(&self) -> Option<&TerminalLevelProof<F, L>> {
        match self {
            Self::Terminal(terminal) => Some(terminal),
            Self::Fold(_) | Self::Direct { .. } => None,
        }
    }

    /// Mutably borrow the terminal-root payload when this is a terminal root.
    pub fn as_terminal_root_mut(&mut self) -> Option<&mut TerminalLevelProof<F, L>> {
        match self {
            Self::Terminal(terminal) => Some(terminal),
            Self::Fold(_) | Self::Direct { .. } => None,
        }
    }

    /// Borrow the per-claim direct witnesses when this is a root-direct
    /// batched proof.
    pub fn as_direct(&self) -> Option<&[DirectWitnessProof<F>]> {
        match self {
            Self::Direct { witnesses, .. } => Some(witnesses.as_slice()),
            Self::Fold(_) | Self::Terminal(_) => None,
        }
    }

    /// Borrow the revealed root-direct B-blinding payloads.
    #[cfg(feature = "zk")]
    pub fn direct_b_blinding_digits(&self) -> Option<&[Vec<i8>]> {
        match self {
            Self::Direct {
                b_blinding_digits, ..
            } => Some(b_blinding_digits.as_slice()),
            Self::Fold(_) | Self::Terminal(_) => None,
        }
    }

    /// True when this root proof is a root-direct batched fast path.
    pub fn is_direct(&self) -> bool {
        matches!(self, Self::Direct { .. })
    }

    /// True when this root proof is itself the terminal fold level.
    pub fn is_terminal_root(&self) -> bool {
        matches!(self, Self::Terminal(_))
    }

    /// Borrow the stored root per-point `y_rings` payload (Fold only).
    ///
    /// # Panics
    ///
    /// Panics on terminal-root and root-direct batched proofs.
    pub fn y_rings(&self) -> &FlatRingVec<F> {
        &self
            .as_fold()
            .expect("y_rings() called on a non-fold root proof")
            .y_rings
    }

    /// Borrow the stored root `v` ring vector (Fold only).
    ///
    /// # Panics
    ///
    /// Panics on terminal-root and root-direct batched proofs.
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
    /// Panics on terminal-root and root-direct batched proofs.
    pub fn next_w_commitment(&self) -> &FlatRingVec<F> {
        &self
            .as_fold()
            .expect("next_w_commitment() called on a non-fold root proof")
            .stage2
            .next_w_commitment
    }

    /// Claimed evaluation of the next witness `w` (Fold only).
    ///
    /// # Panics
    ///
    /// Panics on terminal-root and root-direct batched proofs.
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
            self.y_rings.coeff_len(),
            self.extension_opening_reduction.as_ref(),
            &self.v,
            &self.stage1,
            &self.stage2,
        )
    }
}

/// Akita PCS proof for fused batched openings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaBatchedProof<F: FieldCore, L: FieldCore> {
    /// Plain-opening ZK hiding-factor commitment and opening payload.
    #[cfg(feature = "zk")]
    pub zk_hiding: ZkHidingProof<F>,
    /// Batched root proof over all original-polynomial claims.
    pub root: AkitaBatchedRootProof<F, L>,
    /// Recursive proof steps following the batched root proof.
    pub steps: Vec<AkitaProofStep<F, L>>,
}

impl<F: FieldCore, L: FieldCore> AkitaBatchedProof<F, L> {
    /// Access the terminal direct witness of the recursive-suffix path.
    ///
    /// Returns the `final_witness` from the terminal level: either the
    /// terminal step at the tail of a fold-rooted suffix, or directly from
    /// the [`AkitaBatchedRootProof::Terminal`] root (1-fold case).
    ///
    /// # Panics
    ///
    /// Panics on a root-direct batched proof (use
    /// [`AkitaBatchedRootProof::as_direct`] to access the per-claim witnesses
    /// in that case), and panics if a fold-rooted proof does not terminate
    /// with a terminal step.
    pub fn final_witness(&self) -> &DirectWitnessProof<F> {
        match &self.root {
            AkitaBatchedRootProof::Terminal(terminal) => &terminal.final_witness,
            AkitaBatchedRootProof::Fold(_) => {
                &self
                    .steps
                    .last()
                    .and_then(AkitaProofStep::as_terminal)
                    .expect("fold-rooted Akita proof must terminate with a terminal step")
                    .final_witness
            }
            AkitaBatchedRootProof::Direct { .. } => {
                panic!("final_witness() called on a root-direct batched proof")
            }
        }
    }

    /// Iterate over the intermediate (non-terminal) fold levels of the
    /// recursive suffix.
    pub fn fold_levels(&self) -> impl Iterator<Item = &AkitaLevelProof<F, L>> {
        self.steps
            .iter()
            .filter_map(AkitaProofStep::as_intermediate)
    }

    /// Number of intermediate recursive fold levels.
    pub fn num_fold_levels(&self) -> usize {
        self.fold_levels().count()
    }

    /// True when this proof uses the root-direct batched fast path (no
    /// two-stage root proof and no recursive suffix).
    pub fn is_root_direct(&self) -> bool {
        self.root.is_direct()
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
                step_shapes: self.steps.iter().map(AkitaProofStep::shape).collect(),
            },
            AkitaBatchedRootProof::Terminal(terminal) => {
                AkitaBatchedProofShape::Terminal(terminal.shape())
            }
            AkitaBatchedRootProof::Direct { witnesses, .. } => AkitaBatchedProofShape::Direct {
                witness_shapes: witnesses.iter().map(DirectWitnessProof::shape).collect(),
            },
        }
    }
}

impl<F: FieldCore + AkitaSerialize, L: FieldCore + AkitaSerialize> AkitaBatchedProof<F, L> {
    /// Returns the proof size in bytes (uncompressed).
    pub fn size(&self) -> usize {
        self.serialized_size(Compress::No)
    }
}

/// A recursive proof step.
///
/// Hard-split between intermediate fold levels (which still ship a recursive
/// `next_w_commitment`) and the terminal fold level (which ships the witness
/// in cleartext via `TerminalLevelProof::final_witness`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkitaProofStep<F: FieldCore, L: FieldCore> {
    /// Intermediate (non-terminal) fold level. Ships `next_w_commitment` and
    /// the stage-1 range-check tree.
    Intermediate(AkitaLevelProof<F, L>),
    /// Terminal fold level. Ships `final_witness` in cleartext (absorbed via
    /// `ABSORB_SUMCHECK_W`) and drops `stage1`, `next_w_commitment`,
    /// `next_w_eval`.
    Terminal(TerminalLevelProof<F, L>),
}

impl<F: FieldCore, L: FieldCore> AkitaProofStep<F, L> {
    /// Borrow the intermediate fold proof when this is an intermediate step.
    pub fn as_intermediate(&self) -> Option<&AkitaLevelProof<F, L>> {
        match self {
            Self::Intermediate(level) => Some(level),
            Self::Terminal(_) => None,
        }
    }

    /// Mutably borrow the intermediate fold proof when this is an
    /// intermediate step.
    pub fn as_intermediate_mut(&mut self) -> Option<&mut AkitaLevelProof<F, L>> {
        match self {
            Self::Intermediate(level) => Some(level),
            Self::Terminal(_) => None,
        }
    }

    /// Borrow the terminal level proof when this is a terminal step.
    pub fn as_terminal(&self) -> Option<&TerminalLevelProof<F, L>> {
        match self {
            Self::Intermediate(_) => None,
            Self::Terminal(terminal) => Some(terminal),
        }
    }

    /// Mutably borrow the terminal level proof when this is a terminal step.
    pub fn as_terminal_mut(&mut self) -> Option<&mut TerminalLevelProof<F, L>> {
        match self {
            Self::Intermediate(_) => None,
            Self::Terminal(terminal) => Some(terminal),
        }
    }

    /// Derive the shape for this proof step.
    pub fn shape(&self) -> AkitaProofStepShape {
        match self {
            Self::Intermediate(level) => AkitaProofStepShape::Intermediate(level.shape()),
            Self::Terminal(terminal) => AkitaProofStepShape::Terminal(terminal.shape()),
        }
    }
}
