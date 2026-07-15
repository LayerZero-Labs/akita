//! Helpers for transcript-binding terminal cleartext witnesses.

use akita_field::FieldCore;

/// Transcript byte slices for terminal direct-witness replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalWitnessTranscriptParts {
    /// Logical terminal `e_hat` bytes, bound before sparse challenge sampling.
    pub e_hat: Vec<u8>,
    /// Remaining final-witness bytes, bound before ring-switch challenges.
    pub remainder: Vec<u8>,
}

/// Stage-2 inputs for terminal relation-only replay.
///
/// Terminal folds have no stage-1 norm-check claim. Setting
/// `batching_coeff = 0` removes the virtual norm contribution from every
/// stage-2 round, so `s_claim` and `stage1_point` are structural zeros.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationOnlyStage2Inputs<E: FieldCore> {
    /// Zero coefficient for the omitted virtual norm-check oracle.
    pub batching_coeff: E,
    /// Zero claim for the omitted stage-1 sumcheck.
    pub s_claim: E,
    /// Zero challenge vector with length `col_bits + ring_bits`.
    pub stage1_point: Vec<E>,
}

impl<E: FieldCore> RelationOnlyStage2Inputs<E> {
    /// Build the terminal relation-only stage-2 input bundle.
    #[must_use]
    pub fn new(num_stage1_vars: usize) -> Self {
        Self {
            batching_coeff: E::zero(),
            s_claim: E::zero(),
            stage1_point: vec![E::zero(); num_stage1_vars],
        }
    }
}
