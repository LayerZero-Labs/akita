//! Direct scalarized SIS estimates for every commitment and compression instance in a schedule.

use akita_types::sis::{
    InnerCommitMatrixParams, InnerCommitSecurityRoute, SisMatrixRole, SisModulusProfileId,
};
use akita_types::{CompressionChainPlan, CompressionMapPlan, FoldSchedule, GroupOpenPhaseParams};

use crate::{
    estimate, scalar_sis_from_ring_euclidean, scalar_sis_from_ring_wide, CostValue, EstimateConfig,
    EstimatorError, LatticeCost, Result, SisNorm,
};

/// The norm bound used for one concrete schedule SIS estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleSisBound {
    /// Coefficient-L-infinity collision bound.
    Linf(u128),
    /// Squared L2 bound on the complete scalar collision vector.
    L2Squared(u128),
}

/// Protocol role of one SIS instance selected or derived by a schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleSisRole {
    /// Inner commitment matrix A.
    Inner,
    /// Outer commitment matrix B.
    Outer,
    /// Shared opening matrix D.
    Open,
    /// Rank-one compressed-commitment map.
    Compression,
}

impl From<SisMatrixRole> for ScheduleSisRole {
    fn from(role: SisMatrixRole) -> Self {
        match role {
            SisMatrixRole::Inner => Self::Inner,
            SisMatrixRole::Outer => Self::Outer,
            SisMatrixRole::Open => Self::Open,
        }
    }
}

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
    pub modulus_profile: SisModulusProfileId,
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

/// Estimate every scalarized commitment and compression SIS instance in
/// `schedule` and retain the minimum `log2(rop)` attack cost.
///
/// This is an offline diagnostic path. It calls the estimator directly with
/// each matrix's scheduled rank, width, ring dimension, modulus, and collision
/// bound; it does not consult the generated sufficient-security tables.
///
/// # Errors
///
/// Returns an estimator input error when a schedule coordinate cannot be
/// represented by the scalar estimator or an individual estimate fails.
pub fn estimate_schedule_security(schedule: &FoldSchedule) -> Result<ScheduleSecurityEstimate> {
    let mut instances = Vec::new();

    estimate_nonterminal("root fold", &schedule.root.params, &mut instances)?;
    for (index, fold) in schedule.recursive_folds.iter().enumerate() {
        estimate_nonterminal(
            &format!("recursive fold {index}"),
            &fold.params,
            &mut instances,
        )?;
    }
    estimate_inner(
        "terminal fold A".to_string(),
        &schedule.terminal.inner.matrix,
        &mut instances,
    )?;

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

fn estimate_nonterminal(
    location: &str,
    params: &akita_types::CommittedGroupParams,
    instances: &mut Vec<ScheduleSisInstanceEstimate>,
) -> Result<()> {
    for (index, group) in params.preceding_group_iter().enumerate() {
        let group_location = if group.setup_natural_len.is_some() {
            format!("{location} setup prefix")
        } else {
            format!("{location} precommitted group {index}")
        };
        estimate_group(&group_location, group, instances)?;
    }
    estimate_group(
        &format!("{location} final group"),
        params.own_group(),
        instances,
    )?;
    let matrix = &params.open_matrix;
    estimate_linf(
        format!("{location} shared D"),
        SisMatrixRole::Open.into(),
        matrix.sis_modulus_profile(),
        matrix.ring_dimension(),
        matrix.output_rank(),
        matrix.input_width(),
        matrix.coeff_linf_bound(),
        instances,
    )?;
    estimate_compression_chains(location, params, instances)
}

fn estimate_group(
    location: &str,
    group: &GroupOpenPhaseParams,
    instances: &mut Vec<ScheduleSisInstanceEstimate>,
) -> Result<()> {
    estimate_inner(
        format!("{location} A"),
        &group.profile.inner.matrix,
        instances,
    )?;
    let matrix = &group.profile.outer.matrix;
    estimate_linf(
        format!("{location} B"),
        SisMatrixRole::Outer.into(),
        matrix.sis_modulus_profile(),
        matrix.ring_dimension(),
        matrix.output_rank(),
        matrix.input_width(),
        matrix.coeff_linf_bound(),
        instances,
    )
}

fn estimate_inner(
    location: String,
    matrix: &InnerCommitMatrixParams,
    instances: &mut Vec<ScheduleSisInstanceEstimate>,
) -> Result<()> {
    match matrix.security_route() {
        InnerCommitSecurityRoute::Linf(key) => estimate_linf(
            location,
            SisMatrixRole::Inner.into(),
            key.modulus_profile,
            matrix.ring_dimension(),
            matrix.output_rank(),
            matrix.input_width(),
            key.coeff_linf_bound,
            instances,
        ),
        InnerCommitSecurityRoute::L2 { table_key, .. } => {
            let ring_dimension = u32::try_from(matrix.ring_dimension())
                .map_err(|_| invalid_schedule_coordinate("ring dimension exceeds u32"))?;
            let output_rank = u32::try_from(matrix.output_rank())
                .map_err(|_| invalid_schedule_coordinate("output rank exceeds u32"))?;
            let input_width = u64::try_from(matrix.input_width())
                .map_err(|_| invalid_schedule_coordinate("input width exceeds u64"))?;
            let params = scalar_sis_from_ring_euclidean(
                table_key.modulus_profile.into(),
                ring_dimension,
                output_rank,
                input_width,
                table_key.collision_l2_sq,
            )?;
            let cost = estimate(&params, &EstimateConfig::akita_euclidean_table())?;
            instances.push(ScheduleSisInstanceEstimate {
                location,
                role: ScheduleSisRole::Inner,
                output_rank: matrix.output_rank(),
                input_width: matrix.input_width(),
                ring_dimension: matrix.ring_dimension(),
                modulus_profile: table_key.modulus_profile,
                bound: ScheduleSisBound::L2Squared(table_key.collision_l2_sq),
                cost,
            });
            Ok(())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn estimate_linf(
    location: String,
    role: ScheduleSisRole,
    modulus_profile: SisModulusProfileId,
    ring_dimension: usize,
    output_rank: usize,
    input_width: usize,
    coeff_linf_bound: u128,
    instances: &mut Vec<ScheduleSisInstanceEstimate>,
) -> Result<()> {
    let ring_dimension_u32 = u32::try_from(ring_dimension)
        .map_err(|_| invalid_schedule_coordinate("ring dimension exceeds u32"))?;
    let output_rank_u32 = u32::try_from(output_rank)
        .map_err(|_| invalid_schedule_coordinate("output rank exceeds u32"))?;
    let input_width_u64 = u64::try_from(input_width)
        .map_err(|_| invalid_schedule_coordinate("input width exceeds u64"))?;
    let coeff_linf_bound_u64 = u64::try_from(coeff_linf_bound)
        .map_err(|_| invalid_schedule_coordinate("L-infinity bound exceeds u64"))?;
    let params = scalar_sis_from_ring_wide(
        modulus_profile.into(),
        ring_dimension_u32,
        output_rank_u32,
        input_width_u64,
        coeff_linf_bound_u64,
    )?;
    let cost = estimate(&params, &EstimateConfig::akita_infinity_table())?;
    instances.push(ScheduleSisInstanceEstimate {
        location,
        role,
        output_rank,
        input_width,
        ring_dimension,
        modulus_profile,
        bound: ScheduleSisBound::Linf(coeff_linf_bound),
        cost,
    });
    Ok(())
}

fn estimate_compression_chains(
    location: &str,
    params: &akita_types::CommittedGroupParams,
    instances: &mut Vec<ScheduleSisInstanceEstimate>,
) -> Result<()> {
    if !params.payload_mode.is_compressed() {
        return Ok(());
    }

    let own = params.own_group();
    estimate_group_compression(&format!("{location} final group B"), own, instances)?;
    for (index, group) in params.preceding_group_iter().enumerate() {
        let group_location = if group.setup_natural_len.is_some() {
            format!("{location} setup prefix B")
        } else {
            format!("{location} precommitted group {index} B")
        };
        estimate_group_compression(&group_location, group, instances)?;
    }
    let matrix = &params.open_matrix;
    let source_coefficients = matrix
        .output_rank()
        .checked_mul(matrix.ring_dimension())
        .ok_or_else(|| invalid_schedule_coordinate("D compression source length overflow"))?;
    estimate_compression_chain(
        &format!("{location} shared D"),
        matrix.sis_modulus_profile(),
        source_coefficients,
        instances,
    )
}

fn estimate_group_compression(
    location: &str,
    group: &GroupOpenPhaseParams,
    instances: &mut Vec<ScheduleSisInstanceEstimate>,
) -> Result<()> {
    let matrix = &group.profile.outer.matrix;
    let source_coefficients = group
        .profile
        .outer_slice_count
        .complete_source_coefficients(matrix.output_rank(), matrix.ring_dimension())
        .map_err(|error| invalid_schedule_coordinate(&error.to_string()))?;
    estimate_compression_chain(
        location,
        matrix.sis_modulus_profile(),
        source_coefficients,
        instances,
    )
}

fn estimate_compression_chain(
    location: &str,
    modulus_profile: SisModulusProfileId,
    source_coefficients: usize,
    instances: &mut Vec<ScheduleSisInstanceEstimate>,
) -> Result<()> {
    let plan = CompressionChainPlan::for_complete_source(modulus_profile, source_coefficients)
        .map_err(|error| invalid_schedule_coordinate(&error.to_string()))?;
    for (index, map) in plan.maps().iter().enumerate() {
        estimate_compression_map(
            format!("{location} compression map {index}"),
            *map,
            instances,
        )?;
    }
    Ok(())
}

fn estimate_compression_map(
    location: String,
    map: CompressionMapPlan,
    instances: &mut Vec<ScheduleSisInstanceEstimate>,
) -> Result<()> {
    estimate_linf(
        location,
        ScheduleSisRole::Compression,
        map.modulus_profile(),
        map.ring_dimension(),
        map.output_rank(),
        map.input_width(),
        akita_types::sis::compression::COMPRESSION_SIS_COEFF_LINF_BOUND,
        instances,
    )
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
