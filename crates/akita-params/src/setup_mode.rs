//! Direct setup verification policy, shared by provisioning and replay.

/// Direct setup verification policy. Stage 3 reductions are checked in both modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetupMode {
    /// Export direct evaluations as obligations, using authenticated terminal transforms.
    Defer,
    /// Evaluate direct setup queries against materialized public coefficients.
    Check,
}
