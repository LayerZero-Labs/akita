//! Fold-l∞ Fiat–Shamir grind contract shared by prover reroll and verifier replay.

use akita_field::AkitaError;

use super::FoldWitnessLinfCapPolicy;

/// Preview absorb label for ZK grind probe permutations (prover-only).
pub const FOLD_GRIND_PROBE_ORDER_ABSORB: &[u8] = b"ak/a/fgpo";

/// Per-fold-level grind policy: acceptance threshold and nonce cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldWitnessGrindContract {
    /// Whether this level rerolls under a proved sub-Gaussian tail bound.
    pub policy: FoldWitnessLinfCapPolicy,
    /// Prover accepts when `centered_inf_norm <= witness_linf_cap`.
    pub witness_linf_cap: u128,
    /// Exclusive upper bound on the wire `fold_grind_nonce`.
    pub max_nonce_exclusive: u32,
}

impl FoldWitnessGrindContract {
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
            FoldWitnessLinfCapPolicy::WorstCaseBetaOnly if fold_grind_nonce != 0 => {
                Err(AkitaError::InvalidProof)
            }
            FoldWitnessLinfCapPolicy::TailBoundWithGrind
            | FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind
                if fold_grind_nonce >= self.max_nonce_exclusive =>
            {
                Err(AkitaError::InvalidProof)
            }
            _ => Ok(()),
        }
    }
}

/// One shared grind transaction over every fold group in transcript order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldWitnessGrindBatchContract {
    group_contracts: Vec<FoldWitnessGrindContract>,
}

impl FoldWitnessGrindBatchContract {
    /// Build a nonempty batch contract.
    pub fn new(group_contracts: Vec<FoldWitnessGrindContract>) -> Result<Self, AkitaError> {
        if group_contracts.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "fold grind batch requires at least one group".to_string(),
            ));
        }
        Ok(Self { group_contracts })
    }

    /// Group-local contracts in transcript order.
    #[inline]
    pub fn group_contracts(&self) -> &[FoldWitnessGrindContract] {
        &self.group_contracts
    }

    /// Exclusive shared nonce bound accepted by every group.
    pub fn max_nonce_exclusive(&self) -> u32 {
        self.group_contracts
            .iter()
            .map(|contract| contract.max_nonce_exclusive)
            .min()
            .unwrap_or(0)
    }

    /// Whether every group permits Fiat-Shamir rerolls.
    pub fn allows_grind(&self) -> bool {
        self.group_contracts
            .iter()
            .all(|contract| contract.policy.allows_grind())
    }

    /// Reject a nonce unless every group-local contract accepts it.
    pub fn validate_nonce(&self, fold_grind_nonce: u32) -> Result<(), AkitaError> {
        self.group_contracts
            .iter()
            .try_for_each(|contract| contract.validate_nonce(fold_grind_nonce))
    }
}
