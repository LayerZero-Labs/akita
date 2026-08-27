//! Direct scalarized SIS estimates for every canonical occurrence in a schedule.

use akita_types::{FoldSchedule, ScheduleSisOccurrence};
pub use akita_types::{ScheduleSisBound, ScheduleSisRole};

use crate::{
    estimate, scalar_sis_from_ring_euclidean, scalar_sis_from_ring_wide, CostValue, EstimateConfig,
    EstimatorError, LatticeCost, Result, SisNorm,
};

/// Direct estimator result for one schedule-derived SIS instance.
#[derive(Clone, Debug, PartialEq)]
pub struct ScheduleSisInstanceEstimate {
    /// Stable human-readable location within the schedule.
    pub location: String,
    /// Protocol matrix role.
    pub role: ScheduleSisRole,
    /// Module output rank.
    pub output_rank: usize,
    /// Module input width.
    pub input_width: usize,
    /// Ring dimension.
    pub ring_dimension: usize,
    /// Exact modulus profile supplied to the scalar estimator.
    pub modulus_profile: akita_types::SisModulusProfileId,
    /// Collision bound passed to the estimator.
    pub bound: ScheduleSisBound,
    /// Complete estimator output under the table's ADPS16 quantum model.
    pub cost: LatticeCost,
}

impl ScheduleSisInstanceEstimate {
    /// Return the estimated `log2(rop)` security value.
    ///
    /// A finite estimate is exact under the configured model. A
    /// [`CostValue::ProvenAboveTarget`] value returns its certified lower bound,
    /// and [`CostValue::Infinity`] returns positive infinity.
    #[must_use]
    pub fn security_bits(&self) -> f64 {
        cost_security_bits(self.cost.rop)
    }

    /// Return the norm family used by this instance.
    #[must_use]
    pub const fn norm(&self) -> SisNorm {
        match self.bound {
            ScheduleSisBound::Linf(_) => SisNorm::Infinity,
            ScheduleSisBound::L2Squared(_) => SisNorm::Euclidean,
        }
    }
}

/// Direct modeled attack-cost estimates for all scalarized SIS instances in one schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct ScheduleSecurityEstimate {
    instances: Vec<ScheduleSisInstanceEstimate>,
    minimum_index: usize,
}

impl ScheduleSecurityEstimate {
    /// Return every matrix occurrence, including compression maps, frozen
    /// groups, and setup prefixes.
    #[must_use]
    pub fn instances(&self) -> &[ScheduleSisInstanceEstimate] {
        &self.instances
    }

    /// Return the instance attaining the minimum estimated security.
    #[must_use]
    pub fn minimum(&self) -> &ScheduleSisInstanceEstimate {
        &self.instances[self.minimum_index]
    }

    /// Return the minimum modeled SIS attack cost over all schedule instances.
    #[must_use]
    pub fn minimum_security_bits(&self) -> f64 {
        self.minimum().security_bits()
    }
}

/// Estimate every canonical commitment and compression SIS occurrence in
/// `schedule` and retain the minimum `log2(rop)` attack cost.
///
/// This is an offline diagnostic path. It calls the estimator directly with
/// each matrix's scheduled rank, width, ring dimension, modulus, and collision
/// bound; it does not consult the generated sufficient-security tables.
/// [`FoldSchedule::sis_occurrences`] validates the complete schedule and owns
/// the protocol occurrence topology.
///
/// # Errors
///
/// Returns an estimator input error when the schedule is structurally invalid,
/// a coordinate cannot be represented by the scalar estimator, or an
/// individual estimate fails.
pub fn estimate_schedule_security(schedule: &FoldSchedule) -> Result<ScheduleSecurityEstimate> {
    let instances = schedule
        .sis_occurrences()
        .map_err(|error| invalid_schedule_coordinate(&error.to_string()))?
        .into_iter()
        .map(estimate_occurrence)
        .collect::<Result<Vec<_>>>()?;
    let minimum_index = instances
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| left.security_bits().total_cmp(&right.security_bits()))
        .map(|(index, _)| index)
        .ok_or_else(|| invalid_schedule_coordinate("schedule has no SIS instances"))?;
    Ok(ScheduleSecurityEstimate {
        instances,
        minimum_index,
    })
}

fn estimate_occurrence(occurrence: ScheduleSisOccurrence) -> Result<ScheduleSisInstanceEstimate> {
    let ring_dimension = u32::try_from(occurrence.ring_dimension)
        .map_err(|_| invalid_schedule_coordinate("ring dimension exceeds u32"))?;
    let output_rank = u32::try_from(occurrence.output_rank)
        .map_err(|_| invalid_schedule_coordinate("output rank exceeds u32"))?;
    let input_width = u64::try_from(occurrence.input_width)
        .map_err(|_| invalid_schedule_coordinate("input width exceeds u64"))?;
    let (params, config) = match occurrence.bound {
        ScheduleSisBound::Linf(bound) => {
            let bound = u64::try_from(bound)
                .map_err(|_| invalid_schedule_coordinate("L-infinity bound exceeds u64"))?;
            (
                scalar_sis_from_ring_wide(
                    occurrence.modulus_profile.into(),
                    ring_dimension,
                    output_rank,
                    input_width,
                    bound,
                )?,
                EstimateConfig::akita_infinity_table(),
            )
        }
        ScheduleSisBound::L2Squared(bound) => (
            scalar_sis_from_ring_euclidean(
                occurrence.modulus_profile.into(),
                ring_dimension,
                output_rank,
                input_width,
                bound,
            )?,
            EstimateConfig::akita_euclidean_table(),
        ),
    };
    let cost = estimate(&params, &config)?;
    Ok(ScheduleSisInstanceEstimate {
        location: occurrence.location,
        role: occurrence.role,
        output_rank: occurrence.output_rank,
        input_width: occurrence.input_width,
        ring_dimension: occurrence.ring_dimension,
        modulus_profile: occurrence.modulus_profile,
        bound: occurrence.bound,
        cost,
    })
}

fn cost_security_bits(cost: CostValue) -> f64 {
    match cost {
        CostValue::Finite(value) | CostValue::ProvenAboveTarget(value) => value.log2,
        CostValue::Infinity => f64::INFINITY,
    }
}

fn invalid_schedule_coordinate(reason: &str) -> EstimatorError {
    EstimatorError::InvalidParameter {
        field: "schedule",
        reason: reason.to_string(),
    }
}
