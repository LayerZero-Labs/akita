//! Fold-l∞ Fiat–Shamir grind contract shared by prover reroll and verifier replay.

use akita_field::AkitaError;

use super::FoldLinfThresholdPolicy;

/// Per-fold-level grind policy: acceptance threshold and nonce cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldLinfGrindContract {
    /// Whether this level rerolls under a proved sub-Gaussian tail bound.
    pub policy: FoldLinfThresholdPolicy,
    /// Prover accepts when `centered_inf_norm <= inf_threshold`.
    pub inf_threshold: u128,
    /// Exclusive upper bound on the wire `fold_grind_nonce`.
    pub max_nonce_exclusive: u32,
}

impl FoldLinfGrindContract {
    /// Reject malformed fold grind nonces before challenge replay.
    ///
    /// Deterministic `β_inf` policies forbid reroll (`nonce = 0` only).
    /// Tail-bound-with-grind policies accept `nonce < max_nonce_exclusive`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] when the nonce is out of policy range.
    pub fn validate_nonce(&self, fold_grind_nonce: u32) -> Result<(), AkitaError> {
        match self.policy {
            FoldLinfThresholdPolicy::WorstCaseBetaOnly if fold_grind_nonce != 0 => {
                Err(AkitaError::InvalidProof)
            }
            FoldLinfThresholdPolicy::TailBoundWithGrind
                if fold_grind_nonce >= self.max_nonce_exclusive =>
            {
                Err(AkitaError::InvalidProof)
            }
            _ => Ok(()),
        }
    }
}
