//! Checked batch geometry and transcript-owned sumcheck executor shell.

mod eq_factored;
mod eq_factored_verifier;
mod plan;
mod shared;
mod standard;

pub use eq_factored::{prove_eq_factored_executor_batch, EqFactoredBatchExecution};
pub use eq_factored_verifier::{
    verify_eq_factored_executor_batch_rounds, EqFactoredBatchRoundResult,
    EqFactoredTerminalObligation,
};
pub use plan::{
    CheckedEqFactoredBatch, CheckedStandardBatch, CheckedSumcheckGroup, SumcheckGroupSpec,
    SumcheckMemberShape,
};
pub use standard::{prove_standard_executor_batch, StandardBatchExecution};

use crate::{EqFactoredUniPoly, UniPoly};
use akita_error::AkitaError;
use akita_field::FieldCore;

/// Checked local-round coordinates supplied by the protocol engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckedLocalRound {
    /// Index in the shared master challenge point.
    pub master_round: usize,
    /// Index in this group's local challenge point.
    pub local_round: usize,
    /// Total local rounds for this group.
    pub local_rounds: usize,
}

/// Protocol-owned values needed by one executor round.
pub enum CheckedRoundContext<'a, E: FieldCore> {
    /// Standard sumcheck recurrence state.
    Standard {
        /// Group claim before this round.
        previous_claim: E,
        /// Batching coefficient for each member in group order.
        batching_coefficients: &'a [E],
    },
    /// Eq-factored recurrence state.
    EqFactored {
        /// Scaled group claim before this round.
        scaled_claim: E,
        /// Protocol-owned accumulated claim scale.
        claim_scale: E,
        /// Current equality factor evaluated at zero.
        factor_at_zero: E,
        /// Current equality factor evaluated at one.
        factor_at_one: E,
        /// Equality scalar contributed by virtual master-prefix rounds.
        ///
        /// The lifted relation and the master factor both contain this scalar,
        /// so the executor returns its ordinary suffix-local `q` without
        /// multiplying by or dividing through this value.
        master_lift_prefix: E,
        /// Batching coefficient for each member in group order.
        batching_coefficients: &'a [E],
    },
}

/// One fused round request. The previous local challenge is delivered with the next request.
pub struct CheckedRoundRequest<'a, E: FieldCore> {
    /// Checked master and local round coordinates.
    pub round: CheckedLocalRound,
    /// Previous local challenge, or `None` for the first local round.
    pub previous_challenge: Option<E>,
    /// Protocol-owned recurrence context.
    pub context: CheckedRoundContext<'a, E>,
}

/// Group-level round output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GroupRoundMessage<E: FieldCore> {
    /// One batching-weighted standard polynomial for the group.
    Standard(UniPoly<E>),
    /// One batching-weighted compact inner polynomial for the eq-factored group.
    EqFactored(EqFactoredUniPoly<E>),
}

/// Raw terminal claim for each logical member in one group.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GroupTerminalClaims<E: FieldCore> {
    /// Claims in the checked group's member order.
    pub claims: Vec<E>,
}

/// Transcript-free, object-safe group executor.
pub trait SumcheckRoundExecutor<E: FieldCore>: Send {
    /// Submit or compute one local round without waiting for its result.
    fn start_round(&mut self, request: CheckedRoundRequest<'_, E>) -> Result<(), AkitaError>;

    /// Finish the submitted round and return its group-level message.
    fn finish_round(&mut self) -> Result<GroupRoundMessage<E>, AkitaError>;

    /// Apply the final active challenge and return raw member terminal claims.
    fn finish_binding(
        &mut self,
        final_challenge: Option<E>,
    ) -> Result<GroupTerminalClaims<E>, AkitaError>;
}

#[cfg(test)]
mod tests;
