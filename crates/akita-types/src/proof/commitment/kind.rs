//! Source relation kinds for self-describing group commitments.

/// Extension seam: freeze a source relation kind alongside an ordinary payload.
///
/// Tracking issue: <https://github.com/LayerZero-Labs/akita/issues/285>.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommittedGroupKind {
    /// A commitment to an ordinary negacyclic-ring source.
    Ordinary,
}
