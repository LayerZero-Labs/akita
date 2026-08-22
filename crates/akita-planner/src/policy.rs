//! Offline interpretation of catalog-bound planner policy data.

use akita_error::AkitaError;
use akita_schedules::{ChunkedWitnessCfg, DecompositionParams, PlannerPolicy, SelectionPolicyId};
use akita_types::MAX_I16_LOG_BASIS;

/// Coefficient source whose A-matrix decomposition basis is being selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InnerBasisSource {
    /// Raw coefficients that may benefit from a full basis sweep.
    RawCoefficients { log_bound: u32 },
    /// Unit one-hot coefficients already represented by one exact digit.
    UnitOneHot,
    /// A recursive witness already represented as balanced digits.
    BalancedDigits { log_basis: u32 },
}

impl InnerBasisSource {
    /// Inclusive A/source basis domain for this coefficient shape.
    pub fn search_range(self, policy: &PlannerPolicy) -> Result<(u32, u32), AkitaError> {
        let (min, max) = policy.inner_basis_range;
        match self {
            Self::RawCoefficients { log_bound } => Ok((min, max.min(log_bound.max(min)))),
            Self::UnitOneHot => Ok((min, min)),
            Self::BalancedDigits { log_basis }
                if (1..=MAX_I16_LOG_BASIS).contains(&log_basis) =>
            {
                Ok((log_basis, log_basis))
            }
            Self::BalancedDigits { log_basis } => Err(AkitaError::InvalidSetup(format!(
                "recursive digit basis {log_basis} is outside the supported range [1, {MAX_I16_LOG_BASIS}]"
            ))),
        }
    }

    /// Exact source digit depth for a selected A basis.
    pub fn num_digits_inner(
        self,
        decomposition: DecompositionParams,
        selected_log_basis: u32,
    ) -> Result<usize, AkitaError> {
        match self {
            Self::RawCoefficients { log_bound } => Ok(akita_types::sis::num_digits_inner_for_bound(
                DecompositionParams {
                    log_basis: selected_log_basis,
                    ..decomposition
                },
                log_bound,
            )),
            Self::UnitOneHot => Ok(1),
            Self::BalancedDigits { log_basis } if log_basis == selected_log_basis => Ok(1),
            Self::BalancedDigits { log_basis } => Err(AkitaError::InvalidSetup(format!(
                "balanced source basis {log_basis} cannot be re-decomposed at basis {selected_log_basis}"
            ))),
        }
    }
}

pub(crate) fn direct_only_policy(mut policy: PlannerPolicy) -> PlannerPolicy {
    policy.recursive_setup_planning = false;
    policy.selection_policy =
        SelectionPolicyId::for_policy(false, policy.ring_dimension_schedule_mode);
    policy
}

pub(crate) fn witness_chunk_at_level(
    policy: &PlannerPolicy,
    fold_level: usize,
) -> ChunkedWitnessCfg {
    let num_chunks = policy.chunks_at_level(fold_level);
    if num_chunks > 1 {
        ChunkedWitnessCfg {
            num_chunks,
            num_activated_levels: policy.witness_chunk.num_activated_levels,
        }
    } else {
        ChunkedWitnessCfg::default()
    }
}

pub(crate) fn log_basis_search_range_at_level(policy: &PlannerPolicy, level: usize) -> (u32, u32) {
    let (configured_min, max) = policy.opening_basis_range;
    if level == 0 {
        (configured_min, configured_min)
    } else {
        (configured_min, max)
    }
}
