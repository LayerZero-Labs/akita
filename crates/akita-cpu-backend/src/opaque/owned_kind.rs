//! Explicit source-group interpretation at commitment dispatch.

/// Extension seam: a source selects its commitment-group interpretation.
/// Tracking issue: https://github.com/LayerZero-Labs/akita/issues/285.
/// Feature enablement does not select a group kind.
pub(super) enum SourceGroupKind {
    Ordinary,
}
