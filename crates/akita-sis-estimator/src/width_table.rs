//! ADPS16 quantum infinity norm scalar table generation.
//!
//! The generator is offline only. The selected profile controls both boundary
//! discovery and certification without dimension-specific search behavior.

use crate::{
    akita::{scalar_sis_from_ring_wide, AkitaModulusProfileId},
    config::{EstimateConfig, OptimizerConfig, ReductionCostModel, SearchMode, SisSecurityPolicy},
    cost::{CostValue, LatticeCost},
    error::{EstimatorError, Result},
    estimate,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;

mod boundary;

use boundary::{certified_boundary_from_hint, max_true_in_prefix};

/// Estimator projection of the runtime's canonical coefficient buckets.
pub static COEFF_LINF_BUCKETS: LazyLock<Vec<u64>> = LazyLock::new(|| {
    akita_types::sis::COEFF_LINF_BUCKETS
        .iter()
        .map(|&bound| u64::try_from(bound).expect("runtime SIS bucket exceeds u64"))
        .collect()
});

fn estimator_profile(profile: akita_types::sis::SisModulusProfileId) -> AkitaModulusProfileId {
    match profile {
        akita_types::sis::SisModulusProfileId::Q32Offset99 => AkitaModulusProfileId::Q32Offset99,
        akita_types::sis::SisModulusProfileId::Q64Offset59 => AkitaModulusProfileId::Q64Offset59,
        akita_types::sis::SisModulusProfileId::Q128OffsetA7F7 => {
            AkitaModulusProfileId::Q128OffsetA7F7
        }
    }
}

fn canonical_scalar_origins() -> Vec<(AkitaModulusProfileId, u32, u64)> {
    let mut origins = BTreeSet::new();
    origins.extend(akita_types::sis::sis_role_cells().into_iter().map(|cell| {
        (
            estimator_profile(cell.modulus_profile),
            cell.ring_dimension,
            u64::try_from(cell.coeff_linf_bound).expect("canonical SIS bound exceeds u64"),
        )
    }));
    origins.extend(
        akita_types::sis::compression::compression_sis_cells().map(|cell| {
            (
                estimator_profile(cell.modulus_profile),
                cell.ring_dimension,
                u64::try_from(akita_types::sis::compression::COMPRESSION_SIS_COEFF_LINF_BOUND)
                    .expect("compression SIS bound exceeds u64"),
            )
        }),
    );
    origins.into_iter().collect()
}

#[cfg(test)]
fn scalar_origin_is_canonical(
    profile: AkitaModulusProfileId,
    ring_dimension: u32,
    coeff_linf_bound: u64,
) -> bool {
    canonical_scalar_origins().contains(&(profile, ring_dimension, coeff_linf_bound))
}

/// Ring dimensions derived from the exact canonical role and compression cells.
pub static RING_DIMS: LazyLock<Vec<u32>> = LazyLock::new(|| {
    canonical_scalar_origins()
        .into_iter()
        .map(|(_, dimension, _)| dimension)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
});

/// Modulus profiles derived from the exact canonical generation cells.
pub static FAMILIES: LazyLock<Vec<AkitaModulusProfileId>> = LazyLock::new(|| {
    canonical_scalar_origins()
        .into_iter()
        .map(|(profile, _, _)| profile)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
});

/// Maximum module rank emitted for each scalar row.
pub const DEFAULT_MAX_RANK: u32 = 20;

/// Policy table search cap.
pub const DEFAULT_SEARCH_CAP: u64 = 6_400_000_000_000;

/// Legacy L2 generator cap retained for the independent Euclidean table.
/// The quantum infinity table itself uses [`DEFAULT_SEARCH_CAP`] uniformly.
pub const D128_SEARCH_CAP: u64 = DEFAULT_SEARCH_CAP;

/// Search domain recorded for production boundary certificates.
pub const PRODUCTION_CERTIFICATE_DOMAIN: &str = concat!(
    "proven-pruned beta from 40 to the capped Euclidean baseline, ",
    "with ADPS16 lower-bound early stop; for each visited beta, ",
    "LGSA complete-profile transition and predecessor plus zeta 0 and 1"
);

/// Optimizer profile used to discover and certify scalar boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InfinityWidthProfile {
    /// Local minimum discovery followed by proven-pruned boundary certification.
    LocalMinimum,
    /// Pinned lattice-estimator-compatible local-minimum beta and zeta search.
    LatticeEstimatorParity,
    /// Serial exhaustive beta and zeta search.
    ExhaustiveSerial,
    /// Parallel exhaustive beta and zeta search.
    ExhaustiveParallel,
}

impl InfinityWidthProfile {
    /// Stable provenance label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::LocalMinimum => "local-minimum+proven-pruned-certification",
            Self::LatticeEstimatorParity => "lattice-estimator-local-minimum",
            Self::ExhaustiveSerial => "exhaustive-serial",
            Self::ExhaustiveParallel => "exhaustive-parallel",
        }
    }

    /// Estimator configuration for the selected profile.
    pub fn config(self) -> EstimateConfig {
        match self {
            Self::LocalMinimum | Self::LatticeEstimatorParity => EstimateConfig {
                red_cost_model: ReductionCostModel::Adps16 {
                    mode: crate::config::Adps16Mode::Quantum,
                },
                ..EstimateConfig::lattice_estimator_parity()
            },
            Self::ExhaustiveSerial => EstimateConfig {
                red_cost_model: ReductionCostModel::Adps16 {
                    mode: crate::config::Adps16Mode::Quantum,
                },
                optimizer: OptimizerConfig::OptimizeZeta {
                    beta: SearchMode::Exhaustive,
                    zeta: SearchMode::Exhaustive,
                },
                ..EstimateConfig::default()
            },
            Self::ExhaustiveParallel => EstimateConfig {
                red_cost_model: ReductionCostModel::Adps16 {
                    mode: crate::config::Adps16Mode::Quantum,
                },
                optimizer: OptimizerConfig::OptimizeZeta {
                    beta: SearchMode::ExhaustiveParallel,
                    zeta: SearchMode::ExhaustiveParallel,
                },
                ..EstimateConfig::default()
            },
        }
    }
}

/// One scalar table generation request domain.
#[derive(Clone, Debug, PartialEq)]
pub struct InfinityWidthTableConfig {
    /// Exact modulus profiles.
    pub profiles: Vec<AkitaModulusProfileId>,
    /// Ring dimensions used to expand role origins.
    pub ring_dims: Vec<u32>,
    /// Role coefficient cells.
    pub coeff_linf_bounds: Vec<u64>,
    /// Maximum module rank.
    pub max_rank: u32,
    /// ADPS16 quantum policy.
    pub policy: SisSecurityPolicy,
    /// Optional generation cap.
    pub search_cap: Option<u64>,
    /// Search profile.
    pub profile: InfinityWidthProfile,
    /// Progress report interval.
    pub progress_every: Option<usize>,
}

impl Default for InfinityWidthTableConfig {
    fn default() -> Self {
        Self {
            profiles: FAMILIES.to_vec(),
            ring_dims: RING_DIMS.to_vec(),
            coeff_linf_bounds: COEFF_LINF_BUCKETS.clone(),
            max_rank: DEFAULT_MAX_RANK,
            policy: SisSecurityPolicy::Quantum128BitADPS16,
            search_cap: None,
            profile: InfinityWidthProfile::LocalMinimum,
            progress_every: None,
        }
    }
}

/// Whether a config may publish the canonical production artifact.
///
/// Comparison profiles may generate CSV output, but production Rust output
/// must use the profile that certifies every discovered boundary.
pub fn is_production_infinity_width_table_config(config: &InfinityWidthTableConfig) -> bool {
    same_set(&config.profiles, FAMILIES.as_slice())
        && same_set(&config.ring_dims, RING_DIMS.as_slice())
        && same_set(&config.coeff_linf_bounds, &COEFF_LINF_BUCKETS)
        && config.max_rank == DEFAULT_MAX_RANK
        && config.policy == SisSecurityPolicy::Quantum128BitADPS16
        && config.search_cap.is_none()
        && config.profile == InfinityWidthProfile::LocalMinimum
}

/// ADPS16 quantum certificate costs for one accepted or rejected boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct InfinityWidthPolicyCosts {
    /// The only hard model.
    pub adps16_quantum: LatticeCost,
}

/// One generated ring-origin row. The emitted artifact deduplicates these rows
/// by `(profile, B, n = rank * d)`.
#[derive(Clone, Debug, PartialEq)]
pub struct InfinityWidthRow {
    /// Exact modulus profile.
    pub modulus_profile: AkitaModulusProfileId,
    /// Ring dimension of this role origin.
    pub d: u32,
    /// Module rank of this role origin.
    pub rank: u32,
    /// Coefficient infinity bound.
    pub coeff_linf_bound: u64,
    /// Largest accepted ring width.
    pub max_width: u64,
    /// Policy identity.
    pub policy: SisSecurityPolicy,
    /// Search cap.
    pub search_cap: u64,
    /// Whether the cap was reached.
    pub hit_cap: bool,
    /// Discovery and certification profile.
    pub profile: InfinityWidthProfile,
    /// Accepted boundary certificate.
    pub max_costs: Option<InfinityWidthPolicyCosts>,
    /// Immediate rejected successor certificate.
    pub next_costs: Option<InfinityWidthPolicyCosts>,
}

impl InfinityWidthRow {
    /// CSV header for the single hard model and both certificates.
    pub const fn csv_header() -> &'static str {
        "policy,modulus_profile,d,rank,coeff_linf_bound,max_width,scalar_n,search_cap,hit_cap,profile,target_bits,max_adps16_quantum_rop_log2,next_adps16_quantum_rop_log2,max_beta,max_zeta,next_beta,next_zeta,cutoff_kind"
    }

    /// Format a deterministic audit row.
    pub fn to_csv_record(&self) -> String {
        let n = u64::from(self.d) * u64::from(self.rank);
        let kind = if self.hit_cap { "AtLeast" } else { "Exact" };
        format!(
            "{},{},{},{},{},{},{},{},{},{},{:.1},{},{},{},{},{},{},{}",
            self.policy.label(),
            self.modulus_profile.label(),
            self.d,
            self.rank,
            self.coeff_linf_bound,
            self.max_width,
            n,
            self.search_cap,
            self.hit_cap,
            self.profile.label(),
            self.policy.adps16_quantum_constraint().minimum_log2_rop,
            cost_log2_text(
                self.max_costs
                    .as_ref()
                    .map(|costs| costs.adps16_quantum.rop)
            ),
            cost_log2_text(
                self.next_costs
                    .as_ref()
                    .map(|costs| costs.adps16_quantum.rop)
            ),
            self.max_costs
                .as_ref()
                .and_then(|costs| costs.adps16_quantum.beta)
                .map_or_else(String::new, |value| value.to_string()),
            self.max_costs
                .as_ref()
                .and_then(|costs| costs.adps16_quantum.zeta)
                .map_or_else(String::new, |value| value.to_string()),
            self.next_costs
                .as_ref()
                .and_then(|costs| costs.adps16_quantum.beta)
                .map_or_else(String::new, |value| value.to_string()),
            self.next_costs
                .as_ref()
                .and_then(|costs| costs.adps16_quantum.zeta)
                .map_or_else(String::new, |value| value.to_string()),
            kind,
        )
    }
}

/// Generate ring-origin rows under the ADPS16 quantum policy.
pub fn generate_infinity_width_rows(
    config: &InfinityWidthTableConfig,
) -> Result<Vec<InfinityWidthRow>> {
    validate_table_config(config)?;
    let estimator_config = config.profile.config();
    let mut work = Vec::new();
    for (modulus_profile, d, bound) in canonical_scalar_origins() {
        if !config.profiles.contains(&modulus_profile)
            || !config.ring_dims.contains(&d)
            || !config.coeff_linf_bounds.contains(&bound)
        {
            continue;
        }
        for rank in 1..=config.max_rank {
            work.push((modulus_profile, d, rank, bound));
        }
    }
    if work.is_empty() {
        return invalid_config(
            "coverage",
            "the requested dimensions and coefficient bounds contain no canonical SIS role cells",
        );
    }
    generate_rows_from_work(work, config, &estimator_config)
}

#[cfg(feature = "parallel")]
fn generate_rows_from_work(
    work: Vec<(AkitaModulusProfileId, u32, u32, u64)>,
    config: &InfinityWidthTableConfig,
    estimator_config: &EstimateConfig,
) -> Result<Vec<InfinityWidthRow>> {
    let total = work.len();
    let completed = AtomicUsize::new(0);
    let rows: Result<Vec<_>> = work
        .into_par_iter()
        .map(|request| {
            let row = max_secure_width_row(
                request.0,
                request.1,
                request.2,
                request.3,
                config,
                estimator_config,
            )
            .map_err(|error| EstimatorError::InvalidConfig {
                field: "width_table_row",
                reason: format!(
                    "profile={} d={} rank={} bound={}: {error}",
                    request.0.label(),
                    request.1,
                    request.2,
                    request.3
                ),
            });
            report_progress(config.progress_every, &completed, total);
            row
        })
        .collect();
    let mut rows = rows?;
    rows.sort_by_key(|row| (row.modulus_profile, row.coeff_linf_bound, row.d, row.rank));
    Ok(rows)
}

#[cfg(not(feature = "parallel"))]
fn generate_rows_from_work(
    work: Vec<(AkitaModulusProfileId, u32, u32, u64)>,
    config: &InfinityWidthTableConfig,
    estimator_config: &EstimateConfig,
) -> Result<Vec<InfinityWidthRow>> {
    let total = work.len();
    let mut rows = Vec::with_capacity(work.len());
    for (completed, (modulus_profile, d, rank, bound)) in work.into_iter().enumerate() {
        rows.push(
            max_secure_width_row(modulus_profile, d, rank, bound, config, estimator_config)
                .map_err(|error| EstimatorError::InvalidConfig {
                    field: "width_table_row",
                    reason: format!(
                        "profile={} d={} rank={} bound={}: {error}",
                        modulus_profile.label(),
                        d,
                        rank,
                        bound
                    ),
                })?,
        );
        report_progress(config.progress_every, completed + 1, total);
    }
    rows.sort_by_key(|row| (row.modulus_profile, row.coeff_linf_bound, row.d, row.rank));
    Ok(rows)
}

#[cfg(feature = "parallel")]
fn report_progress(progress_every: Option<usize>, completed: &AtomicUsize, total: usize) {
    let Some(every) = progress_every.filter(|value| *value > 0) else {
        return;
    };
    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
    if done == total || done.is_multiple_of(every) {
        eprintln!("infinity width table progress: {done}/{total} rows");
    }
}

#[cfg(not(feature = "parallel"))]
fn report_progress(progress_every: Option<usize>, completed: usize, total: usize) {
    let Some(every) = progress_every.filter(|value| *value > 0) else {
        return;
    };
    if completed == total || completed.is_multiple_of(every) {
        eprintln!("infinity width table progress: {completed}/{total} rows");
    }
}

/// Validate ADPS16 quantum certificates and monotonicity.
pub fn validate_infinity_width_rows(rows: &[InfinityWidthRow]) -> Result<()> {
    for row in rows {
        let target = row.policy.adps16_quantum_constraint().minimum_log2_rop;
        if row.max_width > 0 {
            let costs = row
                .max_costs
                .as_ref()
                .ok_or_else(|| EstimatorError::InvalidConfig {
                    field: "rows",
                    reason: "accepted width is missing its ADPS16 quantum certificate".to_string(),
                })?;
            if !security_met(costs.adps16_quantum.rop, target) {
                return invalid_config(
                    "rows",
                    "accepted ADPS16 quantum certificate is below target",
                );
            }
        }
        if let Some(costs) = row.next_costs.as_ref() {
            if security_met(costs.adps16_quantum.rop, target) {
                return invalid_config(
                    "rows",
                    "rejected successor still meets ADPS16 quantum target",
                );
            }
        }
    }
    validate_rank_monotonicity(rows)?;
    validate_bound_monotonicity(rows)
}

/// One runtime `(profile, d, B) -> widths[rank]` table row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeWidthRow {
    /// Exact modulus profile.
    pub modulus_profile: AkitaModulusProfileId,
    /// Ring dimension.
    pub d: u32,
    /// Coefficient infinity bound.
    pub coeff_linf_bound: u64,
    /// Maximum secure input width at each one-based module rank.
    pub widths: Vec<u64>,
}

/// Project certified ring-origin rows to runtime `(d, B) -> widths[rank]` rows.
///
/// Scalar certification still groups by `(B, n)` and takes the min `m`. The
/// emitted runtime table projects those cutoffs onto each reachable ring
/// dimension as `width[r - 1] = cutoff_m(B, n = r * d) / d`.
pub fn runtime_width_rows(
    rows: &[InfinityWidthRow],
    max_rank: u32,
) -> Result<Vec<RuntimeWidthRow>> {
    if max_rank == 0 {
        return invalid_config("max_rank", "max_rank must be positive");
    }
    let mut scalar = BTreeMap::<(AkitaModulusProfileId, u64, u64), u64>::new();
    let mut pairs = BTreeSet::<(AkitaModulusProfileId, u32, u64)>::new();
    for row in rows {
        let n = u64::from(row.d) * u64::from(row.rank);
        let scalar_m = row.max_width.checked_mul(u64::from(row.d)).ok_or_else(|| {
            EstimatorError::InvalidConfig {
                field: "rows",
                reason: format!(
                    "max_width * d overflowed for profile={} d={} rank={} bound={}",
                    row.modulus_profile.label(),
                    row.d,
                    row.rank,
                    row.coeff_linf_bound
                ),
            }
        })?;
        scalar
            .entry((row.modulus_profile, row.coeff_linf_bound, n))
            .and_modify(|current| {
                if scalar_m < *current {
                    *current = scalar_m;
                }
            })
            .or_insert(scalar_m);
        pairs.insert((row.modulus_profile, row.d, row.coeff_linf_bound));
    }
    let mut runtime_rows = Vec::new();
    for (profile, d, bound) in pairs {
        let mut widths = Vec::with_capacity(max_rank as usize);
        for rank in 1..=max_rank {
            let n = u64::from(d) * u64::from(rank);
            match scalar.get(&(profile, bound, n)) {
                Some(&scalar_m) => widths.push(scalar_m / u64::from(d)),
                None => {
                    return invalid_config(
                        "rows",
                        &format!(
                            "missing scalar row for profile={} d={d} rank={rank} bound={bound}",
                            profile.label()
                        ),
                    );
                }
            }
        }
        runtime_rows.push(RuntimeWidthRow {
            modulus_profile: profile,
            d,
            coeff_linf_bound: bound,
            widths,
        });
    }
    Ok(runtime_rows)
}

fn max_secure_width_row(
    modulus_profile: AkitaModulusProfileId,
    d: u32,
    rank: u32,
    coeff_linf_bound: u64,
    table_config: &InfinityWidthTableConfig,
    estimator_config: &EstimateConfig,
) -> Result<InfinityWidthRow> {
    let search_cap = row_search_cap(d, table_config.search_cap)?;
    let policy = table_config.policy;
    let target = policy.adps16_quantum_constraint().minimum_log2_rop;
    let discovery = |width| {
        if width < u64::from(rank) {
            return Ok(true);
        }
        let cost = estimate_width(
            modulus_profile,
            d,
            rank,
            width,
            coeff_linf_bound,
            estimator_config,
        )?;
        secure_or_error(cost.rop, target)
    };
    let discovered = max_true_in_prefix(1, search_cap, discovery)?;
    let (max_width, next_width, hit_cap) =
        if table_config.profile == InfinityWidthProfile::LocalMinimum {
            certify_boundary(
                modulus_profile,
                d,
                rank,
                coeff_linf_bound,
                search_cap,
                discovered.max_value,
                &EstimateConfig::akita_infinity_table(),
                target,
            )?
        } else {
            (
                discovered.max_value,
                discovered.next_value,
                discovered.hit_cap,
            )
        };
    let boundary_certificate_config = EstimateConfig::akita_infinity_table();
    let boundary_config = if table_config.profile == InfinityWidthProfile::LocalMinimum {
        &boundary_certificate_config
    } else {
        estimator_config
    };
    let max_costs = (max_width > 0)
        .then(|| {
            estimate_width(
                modulus_profile,
                d,
                rank,
                max_width,
                coeff_linf_bound,
                boundary_config,
            )
        })
        .transpose()?
        .map(|adps16_quantum| InfinityWidthPolicyCosts { adps16_quantum });
    let next_costs = next_width
        .map(|width| {
            estimate_width(
                modulus_profile,
                d,
                rank,
                width,
                coeff_linf_bound,
                boundary_config,
            )
        })
        .transpose()?
        .map(|adps16_quantum| InfinityWidthPolicyCosts { adps16_quantum });
    Ok(InfinityWidthRow {
        modulus_profile,
        d,
        rank,
        coeff_linf_bound,
        max_width,
        policy,
        search_cap,
        hit_cap,
        profile: table_config.profile,
        max_costs,
        next_costs,
    })
}

#[allow(clippy::too_many_arguments)]
fn certify_boundary(
    modulus_profile: AkitaModulusProfileId,
    d: u32,
    rank: u32,
    bound: u64,
    cap: u64,
    discovered: u64,
    config: &EstimateConfig,
    target: f64,
) -> Result<(u64, Option<u64>, bool)> {
    let result = certified_boundary_from_hint(cap, discovered, |width| {
        let cost = estimate_width(modulus_profile, d, rank, width, bound, config)?;
        secure_or_error(cost.rop, target)
    })?;
    Ok((result.max_value, result.next_value, result.hit_cap))
}

fn row_search_cap(d: u32, requested: Option<u64>) -> Result<u64> {
    if d == 0 {
        return Err(EstimatorError::InvalidParameter {
            field: "d",
            reason: "ring dimension must be positive".to_string(),
        });
    }
    let cap = requested.unwrap_or(DEFAULT_SEARCH_CAP);
    if cap == 0 {
        return Err(EstimatorError::InvalidParameter {
            field: "search_cap",
            reason: "search cap must be positive".to_string(),
        });
    }
    Ok(cap)
}

#[allow(clippy::too_many_arguments)]
fn estimate_width(
    modulus_profile: AkitaModulusProfileId,
    d: u32,
    rank: u32,
    width: u64,
    bound: u64,
    config: &EstimateConfig,
) -> Result<LatticeCost> {
    estimate(
        &scalar_sis_from_ring_wide(modulus_profile, d, rank, width, bound)?,
        config,
    )
}

fn secure_or_error(rop: CostValue, target: f64) -> Result<bool> {
    match rop {
        CostValue::Finite(cost) if cost.log2.is_finite() => Ok(cost.log2 >= target),
        CostValue::ProvenAboveTarget(lower_bound)
            if lower_bound.log2.is_finite() && lower_bound.log2 >= target
                || lower_bound.log2.is_infinite() && lower_bound.log2.is_sign_positive() =>
        {
            Ok(true)
        }
        // An unclassified infinite result is never evidence that a point
        // passes. Stop generation rather than guessing whether it is a
        // numerical underflow, unsupported input, or a genuinely large cost.
        CostValue::Infinity => Err(EstimatorError::Unsupported {
            feature: "unclassified infinite ADPS16 quantum estimate",
        }),
        CostValue::Finite(_) | CostValue::ProvenAboveTarget(_) => {
            Err(EstimatorError::Unsupported {
                feature: "non-finite ADPS16 quantum estimate",
            })
        }
    }
}

fn security_met(rop: CostValue, target: f64) -> bool {
    matches!(rop, CostValue::Finite(cost) if cost.log2.is_finite() && cost.log2 >= target)
        || matches!(rop, CostValue::ProvenAboveTarget(lower_bound)
            if (lower_bound.log2.is_finite() && lower_bound.log2 >= target)
                || (lower_bound.log2.is_infinite() && lower_bound.log2.is_sign_positive()))
}

fn validate_rank_monotonicity(rows: &[InfinityWidthRow]) -> Result<()> {
    let mut groups = BTreeMap::<(AkitaModulusProfileId, u32, u64), Vec<&InfinityWidthRow>>::new();
    for row in rows {
        groups
            .entry((row.modulus_profile, row.d, row.coeff_linf_bound))
            .or_default()
            .push(row);
    }
    for group in groups.values_mut() {
        group.sort_by_key(|row| row.rank);
        for pair in group.windows(2) {
            if pair[1].max_width < pair[0].max_width {
                return invalid_config("rows", "width decreases with rank");
            }
        }
    }
    Ok(())
}

fn validate_bound_monotonicity(rows: &[InfinityWidthRow]) -> Result<()> {
    let mut groups = BTreeMap::<(AkitaModulusProfileId, u32, u32), Vec<&InfinityWidthRow>>::new();
    for row in rows {
        groups
            .entry((row.modulus_profile, row.d, row.rank))
            .or_default()
            .push(row);
    }
    for group in groups.values_mut() {
        group.sort_by_key(|row| row.coeff_linf_bound);
        for pair in group.windows(2) {
            if pair[1].max_width > pair[0].max_width {
                return invalid_config("rows", "width increases with coefficient bound");
            }
        }
    }
    Ok(())
}

fn validate_table_config(config: &InfinityWidthTableConfig) -> Result<()> {
    if config.profiles.is_empty() {
        return invalid_config("profiles", "at least one profile is required");
    }
    if config.ring_dims.is_empty() {
        return invalid_config("ring_dims", "at least one ring dimension is required");
    }
    if config.coeff_linf_bounds.is_empty() {
        return invalid_config("coeff_linf_bounds", "at least one bound is required");
    }
    if config.max_rank == 0 {
        return invalid_config("max_rank", "max_rank must be positive");
    }
    Ok(())
}

fn invalid_config<T>(field: &'static str, reason: &str) -> Result<T> {
    Err(EstimatorError::InvalidConfig {
        field,
        reason: reason.to_string(),
    })
}

fn cost_log2_text(value: Option<CostValue>) -> String {
    match value {
        Some(CostValue::Finite(cost)) if cost.log2.is_finite() => format!("{:.12}", cost.log2),
        Some(CostValue::ProvenAboveTarget(lower_bound)) => {
            format!("above-target:{:.12}", lower_bound.log2)
        }
        Some(CostValue::Infinity) => "unclassified-infinity".to_string(),
        Some(CostValue::Finite(_)) => "non-finite".to_string(),
        None => String::new(),
    }
}

fn same_set<T: Copy + Ord>(left: &[T], right: &[T]) -> bool {
    left.len() == right.len()
        && left.iter().copied().collect::<BTreeSet<_>>()
            == right.iter().copied().collect::<BTreeSet<_>>()
}

#[cfg(test)]
mod tests;
