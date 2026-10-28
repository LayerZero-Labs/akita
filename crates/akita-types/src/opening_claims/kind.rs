//! Relation kinds carried by canonical grouped opening claims.

/// Extension seam: identify a group's relation independently of its commitment payload.
///
/// Tracking issue: <https://github.com/LayerZero-Labs/akita/issues/285>.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupClaimKind {
    /// The ordinary negacyclic-ring opening relation.
    Ordinary,
}
