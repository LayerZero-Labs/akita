//! Typed per-level proof values.
//!
//! The protocol wire is the spongefish byte stream produced by the prover and
//! replayed by the verifier; nothing in production builds these values. They
//! remain as a test oracle: [`super::wire`] serializes them so the planner byte
//! formulas in `proof_size` and `schedule_tests` can be checked against an
//! independent encoding. Only `AkitaSerialize` is implemented; nothing decodes
//! them.

use super::*;
use akita_sumcheck::{EqFactoredSumcheckProof, SumcheckProof};

/// One stage in the stage-1 range-check tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AkitaStage1StageProof<F: Field> {
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
pub(crate) struct AkitaStage1Proof<F: Field> {
    /// Root-to-leaf range-check stages.
    pub stages: Vec<AkitaStage1StageProof<F>>,
    /// Claimed evaluation of `S` at the final stage-1 output point.
    pub range_image_evaluation: F,
    /// Optional schedule-selected proof of the complete physical response
    /// square sum. Its presence and exact vector lengths are derived from the
    /// A matrix security route.
    pub norm_proof: Option<PhysicalL2NormProof<F>>,
}

/// Stage-1 payload for one schedule-selected physical L2 norm proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PhysicalL2NormProof<F: Field> {
    /// Exact nonnegative integer square sum reconstructed by the verifier.
    pub response_l2_sq: u128,
    /// Direct mode leaves this empty. Limb-Gram mode carries the canonical
    /// block-major, upper-triangular inner-product claims.
    pub subclaims: Vec<F>,
    /// Final virtual response or limb evaluations consumed by Stage 2.
    pub virtual_evaluations: Vec<F>,
    /// General final-leaf sumcheck batching range and norm terms.
    pub sumcheck: SumcheckProof<F>,
}

/// FoldSchedule-shaped outgoing witness binding for an intermediate fold.
///
/// The encoding carries no variant tag. Ordinary recursive edges carry a
/// compressed outer payload, while an edge into the suffix terminal binds the
/// `t` segment owned by the following [`TerminalLevelProof`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NextWitnessBinding<F: Field> {
    /// Terminal compressed commitment payload for an ordinary recursive edge.
    OuterPayload(RingVec<F>),
    /// The following terminal proof's canonical `t` segment is the state.
    TerminalInnerState,
}

/// Intermediate-stage payload for stage 2 of a fold level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AkitaStage2Proof<F: Field, E: Field> {
    /// Stage-2 fused sumcheck proof.
    pub sumcheck_proof: SumcheckProof<E>,
    /// FoldSchedule-shaped binding for the next witness.
    pub next_witness_binding: NextWitnessBinding<F>,
    /// Claimed evaluation of the next witness `w` at the stage-2 challenge point.
    pub next_w_eval: E,
}

/// Optional proof that reduces a logical extension-field opening into one
/// ordinary opening of the transformed committed witness.
///
/// This object is serialized without a tag or length; the schedule determines
/// its presence and lengths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExtensionOpeningReductionProof<E: Field> {
    /// Transcript-bound partial evaluations used by the basis-conversion
    /// check.
    pub partials: Vec<E>,
    /// One degree-two sumcheck for an early random linear combination of all
    /// extension-opening claims.
    pub sumcheck: SumcheckProof<E>,
    /// Individual terminal claims at the common sumcheck point. These are
    /// absorbed before the later application batching challenge.
    pub final_claims: Vec<E>,
}

/// Stage-3 proof for the public setup contribution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SetupSumcheckProof<E: Field> {
    /// Claimed setup contribution fed into the stage-2 final row evaluation.
    pub claim: E,
    /// Claimed setup-prefix opening carried into the next fold as a precommitted group.
    pub setup_prefix_eval: E,
    /// Degree-two setup-product sumcheck.
    pub sumcheck: SumcheckProof<E>,
}

/// Proof for one non-terminal fold level, including the root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FoldLevelProof<F: Field, E: Field> {
    /// Optional extension-opening reduction payload.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionProof<E>>,
    /// Terminal compressed opening payload `p_H`.
    pub opening_payload: RingVec<F>,
    /// Stage-1 norm-check payload.
    pub stage1: AkitaStage1Proof<E>,
    /// Stage-2 fused payload.
    pub stage2: AkitaStage2Proof<F, E>,
    /// Optional stage-3 setup product-sumcheck proof.
    pub stage3_sumcheck_proof: Option<SetupSumcheckProof<E>>,
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
pub(crate) struct TerminalLevelProof<F: Field, E: Field> {
    /// Optional extension-opening reduction payload.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionProof<E>>,
    /// Quotient-free terminal response checked directly by the verifier.
    pub terminal_response: TerminalResponse<F>,
}

impl<F: Field, E: Field> TerminalLevelProof<F, E> {
    /// Construct from typed ring elements and a clear terminal response.
    ///
    /// Pass `extension_opening_reduction = None` for opening shapes that do
    /// not use extension-opening reduction.
    pub(crate) fn new_with_extension_opening_reduction(
        extension_opening_reduction: Option<ExtensionOpeningReductionProof<E>>,
        terminal_response: TerminalResponse<F>,
    ) -> Self {
        Self {
            extension_opening_reduction,
            terminal_response,
        }
    }
}
