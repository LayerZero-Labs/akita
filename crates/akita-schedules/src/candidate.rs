//! Shared candidate-construction helpers.

use crate::runtime::PlannerPolicy;
use akita_field::AkitaError;
use akita_types::sis::{
    min_secure_rank, projected_role_ring_count, rounded_up_collision_inf_norm, SisTableKey,
};
use akita_types::{CommitmentRingDims, SisMatrixRole};

/// Exact or adaptively derived dimensions for one planner candidate.
#[derive(Clone, Copy, Debug)]
pub enum RingDimensionCandidate<'a> {
    Fixed(CommitmentRingDims),
    Adaptive {
        inner: usize,
        outer_dimensions: &'a [usize],
        opening_dimensions: &'a [usize],
        ceiling: CommitmentRingDims,
    },
}

impl RingDimensionCandidate<'_> {
    pub fn inner(self) -> usize {
        match self {
            Self::Fixed(dimensions) => dimensions.d_a(),
            Self::Adaptive { inner, .. } => inner,
        }
    }

    pub fn validate(self) -> Result<(), AkitaError> {
        match self {
            Self::Fixed(dimensions) => dimensions.validate_role_projection(),
            Self::Adaptive { inner, ceiling, .. } => {
                ceiling.validate_role_projection()?;
                if inner == 0 || !inner.is_power_of_two() || inner > ceiling.d_a() {
                    return Err(AkitaError::InvalidSetup(format!(
                        "adaptive A dimension D{inner} is invalid under D{} ceiling",
                        ceiling.d_a()
                    )));
                }
                Ok(())
            }
        }
    }

    pub fn collision_role_price(
        self,
        policy: &PlannerPolicy,
        role: SisMatrixRole,
        native_width: usize,
        log_basis: u32,
    ) -> Option<(SisTableKey, usize)> {
        let carrier_dimension = self.inner();
        match self {
            Self::Fixed(dimensions) => {
                let role_dimension = match role {
                    SisMatrixRole::Outer => dimensions.d_b(),
                    SisMatrixRole::Open => dimensions.d_d(),
                    SisMatrixRole::Inner => return None,
                };
                projected_collision_role_price(
                    policy,
                    role,
                    carrier_dimension,
                    role_dimension,
                    native_width,
                    log_basis,
                )
            }
            Self::Adaptive {
                outer_dimensions,
                opening_dimensions,
                ceiling,
                ..
            } => {
                let (dimensions, role_ceiling) = match role {
                    SisMatrixRole::Outer => (outer_dimensions, ceiling.d_b()),
                    SisMatrixRole::Open => (opening_dimensions, ceiling.d_d()),
                    SisMatrixRole::Inner => return None,
                };
                let mut best = None;
                for &role_dimension in dimensions {
                    if role_dimension > carrier_dimension || role_dimension > role_ceiling {
                        continue;
                    }
                    let Some((key, width)) = projected_collision_role_price(
                        policy,
                        role,
                        carrier_dimension,
                        role_dimension,
                        native_width,
                        log_basis,
                    ) else {
                        continue;
                    };
                    let rank = min_secure_rank(key, u64::try_from(width).ok()?)?;
                    if best.as_ref().is_none_or(|(best_rank, best_d, _, _)| {
                        (rank, role_dimension) < (*best_rank, *best_d)
                    }) {
                        best = Some((rank, role_dimension, key, width));
                    }
                    if rank == 1 {
                        break;
                    }
                }
                best.map(|(_, _, key, width)| (key, width))
            }
        }
    }
}

/// Construct the canonical SIS-table key for one role and ring dimension.
pub fn sis_key_at_dimension(
    policy: &PlannerPolicy,
    role: SisMatrixRole,
    ring_dimension: usize,
    coeff_linf_bound: u128,
) -> SisTableKey {
    SisTableKey {
        policy: policy.sis_security_policy,
        table_digest: policy.sis_table_digest,
        modulus_profile: policy.sis_modulus_profile,
        role,
        ring_dimension: ring_dimension as u32,
        coeff_linf_bound,
    }
}

/// Price one projected B/D collision role using canonical physical width and
/// coefficient bounds.
pub fn projected_collision_role_price(
    policy: &PlannerPolicy,
    role: SisMatrixRole,
    carrier_dimension: usize,
    role_dimension: usize,
    native_width: usize,
    log_basis: u32,
) -> Option<(SisTableKey, usize)> {
    if role == SisMatrixRole::Inner
        || role_dimension == 0
        || !carrier_dimension.is_multiple_of(role_dimension)
    {
        return None;
    }
    let coeff_linf_bound = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        role,
        role_dimension,
        log_basis,
    )?;
    let physical_width =
        projected_role_ring_count(carrier_dimension, role_dimension, native_width)?;
    Some((
        sis_key_at_dimension(policy, role, role_dimension, coeff_linf_bound),
        physical_width,
    ))
}
