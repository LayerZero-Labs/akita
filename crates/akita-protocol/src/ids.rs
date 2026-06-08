//! Concrete identifier types for Akita's sumcheck descriptors.
//!
//! These are the `O`/`P`/`C` instantiations of the generic
//! `akita_sumcheck::descriptor` algebra. They name the specific MLE openings,
//! public weights, and Fiat-Shamir scalars that Akita's stage formulas refer
//! to. The generic algebra in `akita-sumcheck` names none of these.

/// MLE opening identifiers.
///
/// The prover resolves an opening to a witness view; the verifier resolves it
/// to a claimed evaluation at the round point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AkitaOpeningId {
    /// The committed next-witness MLE `W`, opened at the round challenges.
    Witness,
}

/// Public, verifier-evaluable weight identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AkitaPublicId {
    /// `eq(stage1_point, ·)` evaluated at the full round point.
    EqStage1Point,
    /// The stage-1 sparse-challenge weight MLE `alpha(·)` over the ring (y)
    /// variables.
    Alpha,
    /// The ring-switch relation-row evaluation over the column (x) variables.
    RelationRow,
}

/// Fiat-Shamir scalar identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AkitaChallengeId {
    /// The gamma batching coefficient fusing the virtual-claim sumcheck into
    /// the relation sumcheck.
    BatchingCoeff,
}
