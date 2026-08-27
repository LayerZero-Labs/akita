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
    work_cache::WorkId,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;

mod boundary;

use boundary::{certified_boundary_from_hint, max_true_in_prefix};

/// Estimator projection of every canonical role coefficient bound.
pub static COEFF_LINF_BOUNDS: LazyLock<Vec<u64>> = LazyLock::new(|| {
    canonical_scalar_origins()
        .into_iter()
        .map(|(_, _, bound)| bound)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
});

fn canonical_scalar_origins() -> Vec<(AkitaModulusProfileId, u32, u64)> {
    let mut origins = BTreeSet::new();
    origins.extend(akita_types::sis::sis_role_cells().into_iter().map(|cell| {
        (
            cell.modulus_profile.into(),
            cell.ring_dimension,
            u64::try_from(cell.coeff_linf_bound).expect("canonical SIS bound exceeds u64"),
        )
    }));
    origins.extend(
        akita_types::sis::compression::compression_sis_cells().map(|cell| {
            (
                cell.modulus_profile.into(),
                cell.ring_dimension,
                u64::try_from(akita_types::sis::compression::COMPRESSION_SIS_COEFF_LINF_BOUND)
                    .expect("compression SIS bound exceeds u64"),
            )
        }),
    );
    for profile in [
        akita_types::sis::SisModulusProfileId::Q32Offset99,
        akita_types::sis::SisModulusProfileId::Q64Offset59,
        akita_types::sis::SisModulusProfileId::Q128OffsetA7F7,
    ] {
        let dimensions = akita_types::compression_ring_dimensions(profile);
        let doubled = dimensions
            .into_iter()
            .max()
            .and_then(|dimension| dimension.checked_mul(2))
            .expect("compression diagnostic dimension fits usize");
        origins.insert((
            profile.into(),
            u32::try_from(doubled).expect("compression diagnostic dimension fits u32"),
            u64::try_from(akita_types::sis::compression::COMPRESSION_SIS_COEFF_LINF_BOUND)
                .expect("compression SIS bound exceeds u64"),
        ));
    }
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
pub const DEFAULT_MAX_RANK: u32 = akita_types::sis::SIS_MAX_MODULE_RANK;

/// Policy table search cap.
pub const DEFAULT_SEARCH_CAP: u64 = akita_types::sis::SIS_REQUIRED_MAX_WIDTH;

/// Legacy L2 generator cap retained for the independent Euclidean table.
/// The quantum infinity table itself uses [`DEFAULT_SEARCH_CAP`] uniformly.
pub const D128_SEARCH_CAP: u64 = DEFAULT_SEARCH_CAP;

/// Search domain recorded for production boundary certificates.
pub const PRODUCTION_CERTIFICATE_DOMAIN: &str = concat!(
    "proven-pruned beta from 40 to the capped Euclidean baseline, ",
    "with ADPS16 best-cost and 128-bit decision lower-bound early stops; ",
    "for each visited beta, ",
    "every pre-stable LGSA dimension plus both stable-tail endpoints, ",
    "plus both sides of any active-dimension probability transition, ",
    "restricted to the tall q-ary domain 0 <= zeta < d - n"
);

/// Semantic identity of the current infinity-width evaluator.
///
/// Change this value whenever an estimator or certification change can alter
/// a work result. Operational changes such as parallelism and progress output
/// do not change it.
pub const INFINITY_WIDTH_EVALUATOR_ID: &str = "akita-infinity-width-v3";

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

    fn parse_label(label: &str) -> Result<Self> {
        match label {
            "local-minimum+proven-pruned-certification" => Ok(Self::LocalMinimum),
            "lattice-estimator-local-minimum" => Ok(Self::LatticeEstimatorParity),
            "exhaustive-serial" => Ok(Self::ExhaustiveSerial),
            "exhaustive-parallel" => Ok(Self::ExhaustiveParallel),
            _ => invalid_config("work_result", "unknown infinity-width profile label"),
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

/// One independently evaluable infinity-width table row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InfinityWidthWorkItem {
    /// Exact modulus profile.
    pub modulus_profile: AkitaModulusProfileId,
    /// Ring dimension of the semantic origin.
    pub d: u32,
    /// Module rank of the semantic origin.
    pub rank: u32,
    /// Coefficient infinity bound.
    pub coeff_linf_bound: u64,
}

impl InfinityWidthWorkItem {
    /// Content address under one complete table-generation configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the table search cap is malformed.
    pub fn work_id(self, config: &InfinityWidthTableConfig) -> Result<WorkId> {
        let search_cap = row_search_cap(self.d, config.search_cap)?;
        let target_bits = config
            .policy
            .adps16_quantum_constraint()
            .minimum_log2_rop
            .to_bits();
        let canonical = format!(
            "evaluator={INFINITY_WIDTH_EVALUATOR_ID}\npolicy={}\ntarget_bits={target_bits:016x}\nprofile={}\ncertificate_domain={PRODUCTION_CERTIFICATE_DOMAIN}\nmodulus_profile={}\nmodulus={}\nd={}\nrank={}\ncoeff_linf_bound={}\nsearch_cap={}\n",
            config.policy.label(),
            config.profile.label(),
            self.modulus_profile.label(),
            self.modulus_profile.modulus(),
            self.d,
            self.rank,
            self.coeff_linf_bound,
            search_cap,
        );
        Ok(WorkId::new(
            b"akita-sis-estimator/infinity-width-row",
            canonical.as_bytes(),
        ))
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
            coeff_linf_bounds: COEFF_LINF_BOUNDS.clone(),
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
        && same_set(&config.coeff_linf_bounds, &COEFF_LINF_BOUNDS)
        && config.max_rank == DEFAULT_MAX_RANK
        && config.policy == SisSecurityPolicy::Quantum128BitADPS16
        && config.search_cap.is_none()
        && config.profile == InfinityWidthProfile::LocalMinimum
}

/// Compact attack certificate for one accepted or rejected boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InfinityWidthCertificate {
    /// Total attack cost or certified lower bound.
    pub rop: CostValue,
    /// BKZ block size selected by the attack.
    pub beta: Option<u32>,
    /// Number of projected coordinates selected by the attack.
    pub zeta: Option<u64>,
}

impl From<LatticeCost> for InfinityWidthCertificate {
    fn from(cost: LatticeCost) -> Self {
        Self {
            rop: cost.rop,
            beta: cost.beta,
            zeta: cost.zeta,
        }
    }
}

/// ADPS16 quantum certificate for one accepted or rejected boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct InfinityWidthPolicyCosts {
    /// The only hard model.
    pub adps16_quantum: InfinityWidthCertificate,
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
    const WORK_RESULT_SCHEMA: &'static str = "akita-infinity-width-row-result-v1";

    /// CSV header for the single hard model and both certificates.
    pub const fn csv_header() -> &'static str {
        "policy,modulus_profile,d,rank,coeff_linf_bound,max_width,scalar_n,search_cap,hit_cap,profile,target_bits,max_adps16_quantum_rop_log2,next_adps16_quantum_rop_log2,max_beta,max_zeta,next_beta,next_zeta,cutoff_kind"
    }

    /// Format a deterministic audit row.
    pub fn to_csv_record(&self) -> String {
        self.to_record(cost_log2_text)
    }

    fn to_record(&self, format_cost: fn(Option<CostValue>) -> String) -> String {
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
            format_cost(
                self.max_costs
                    .as_ref()
                    .map(|costs| costs.adps16_quantum.rop)
            ),
            format_cost(
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

    /// Encode one self-describing immutable work-result payload.
    #[must_use]
    pub fn to_work_result(&self) -> String {
        format!(
            "{}\n{}\n",
            Self::WORK_RESULT_SCHEMA,
            self.to_record(cost_log2_work_text)
        )
    }

    /// Decode one immutable work-result payload.
    ///
    /// # Errors
    ///
    /// Returns an error when the schema or row is malformed.
    pub fn from_work_result(payload: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(payload).map_err(|error| {
            invalid_config_value("work_result", format!("result is not UTF-8: {error}"))
        })?;
        let mut lines = text.lines();
        if lines.next() != Some(Self::WORK_RESULT_SCHEMA) {
            return invalid_config("work_result", "unknown infinity-width result schema");
        }
        let record = lines
            .next()
            .ok_or_else(|| invalid_config_value("work_result", "result row is missing"))?;
        if lines.next().is_some() {
            return invalid_config("work_result", "result contains more than one row");
        }
        parse_csv_record(record)
    }

    /// Validate that a cached row satisfies exactly one planned work item.
    ///
    /// # Errors
    ///
    /// Returns an error when the row belongs to another item or configuration,
    /// or when its certificates fail normal table validation.
    pub fn validate_for_work_item(
        &self,
        item: InfinityWidthWorkItem,
        config: &InfinityWidthTableConfig,
    ) -> Result<()> {
        if self.modulus_profile != item.modulus_profile
            || self.d != item.d
            || self.rank != item.rank
            || self.coeff_linf_bound != item.coeff_linf_bound
            || self.policy != config.policy
            || self.profile != config.profile
            || self.search_cap != row_search_cap(item.d, config.search_cap)?
        {
            return invalid_config(
                "work_result",
                "cached infinity-width row does not match its planned work item",
            );
        }
        validate_infinity_width_rows(std::slice::from_ref(self))
    }
}

/// Generate ring-origin rows under the ADPS16 quantum policy.
pub fn generate_infinity_width_rows(
    config: &InfinityWidthTableConfig,
) -> Result<Vec<InfinityWidthRow>> {
    let work = infinity_width_work_items(config)?;
    generate_rows_from_work(work, config, &config.profile.config())
}

/// Resolve the deterministic work set for one table-generation request.
///
/// # Errors
///
/// Returns an error when the request is malformed or selects no canonical
/// coverage cells.
pub fn infinity_width_work_items(
    config: &InfinityWidthTableConfig,
) -> Result<Vec<InfinityWidthWorkItem>> {
    validate_table_config(config)?;
    let mut work = Vec::new();
    for (modulus_profile, d, bound) in canonical_scalar_origins() {
        if !config.profiles.contains(&modulus_profile)
            || !config.ring_dims.contains(&d)
            || !config.coeff_linf_bounds.contains(&bound)
        {
            continue;
        }
        for rank in 1..=config.max_rank {
            work.push(InfinityWidthWorkItem {
                modulus_profile,
                d,
                rank,
                coeff_linf_bound: bound,
            });
        }
    }
    if work.is_empty() {
        return invalid_config(
            "coverage",
            "the requested dimensions and coefficient bounds contain no canonical SIS role cells",
        );
    }
    work.sort_unstable();
    Ok(work)
}

/// Evaluate one planned infinity-width work item.
///
/// # Errors
///
/// Returns an error when the item is outside the requested canonical coverage
/// or when estimation or boundary certification fails.
pub fn generate_infinity_width_row(
    item: InfinityWidthWorkItem,
    config: &InfinityWidthTableConfig,
) -> Result<InfinityWidthRow> {
    validate_table_config(config)?;
    if !config.profiles.contains(&item.modulus_profile)
        || !config.ring_dims.contains(&item.d)
        || !config.coeff_linf_bounds.contains(&item.coeff_linf_bound)
        || item.rank == 0
        || item.rank > config.max_rank
        || !canonical_scalar_origins().contains(&(
            item.modulus_profile,
            item.d,
            item.coeff_linf_bound,
        ))
    {
        return invalid_config(
            "work_item",
            "infinity-width work item is outside the requested canonical coverage",
        );
    }
    max_secure_width_row(
        item.modulus_profile,
        item.d,
        item.rank,
        item.coeff_linf_bound,
        config,
        &config.profile.config(),
    )
}

#[cfg(feature = "parallel")]
fn generate_rows_from_work(
    work: Vec<InfinityWidthWorkItem>,
    config: &InfinityWidthTableConfig,
    estimator_config: &EstimateConfig,
) -> Result<Vec<InfinityWidthRow>> {
    let total = work.len();
    let completed = AtomicUsize::new(0);
    let rows: Result<Vec<_>> = work
        .into_par_iter()
        .map(|request| {
            let row = max_secure_width_row(
                request.modulus_profile,
                request.d,
                request.rank,
                request.coeff_linf_bound,
                config,
                estimator_config,
            )
            .map_err(|error| EstimatorError::InvalidConfig {
                field: "width_table_row",
                reason: format!(
                    "profile={} d={} rank={} bound={}: {error}",
                    request.modulus_profile.label(),
                    request.d,
                    request.rank,
                    request.coeff_linf_bound
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
    work: Vec<InfinityWidthWorkItem>,
    config: &InfinityWidthTableConfig,
    estimator_config: &EstimateConfig,
) -> Result<Vec<InfinityWidthRow>> {
    let total = work.len();
    let mut rows = Vec::with_capacity(work.len());
    for (completed, request) in work.into_iter().enumerate() {
        rows.push(
            max_secure_width_row(
                request.modulus_profile,
                request.d,
                request.rank,
                request.coeff_linf_bound,
                config,
                estimator_config,
            )
            .map_err(|error| EstimatorError::InvalidConfig {
                field: "width_table_row",
                reason: format!(
                    "profile={} d={} rank={} bound={}: {error}",
                    request.modulus_profile.label(),
                    request.d,
                    request.rank,
                    request.coeff_linf_bound
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
    let first_tall_width = u64::from(rank).saturating_add(1);
    let policy = table_config.policy;
    let target = policy.adps16_quantum_constraint().minimum_log2_rop;
    let discovery = |width| {
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
    let discovered = max_true_in_prefix(first_tall_width, search_cap, discovery)?;
    let (max_width, next_width, hit_cap) =
        if table_config.profile == InfinityWidthProfile::LocalMinimum {
            certify_boundary(
                modulus_profile,
                d,
                rank,
                coeff_linf_bound,
                first_tall_width,
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
        .map(|cost| InfinityWidthPolicyCosts {
            adps16_quantum: cost.into(),
        });
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
        .map(|cost| InfinityWidthPolicyCosts {
            adps16_quantum: cost.into(),
        });
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
    start: u64,
    cap: u64,
    discovered: u64,
    config: &EstimateConfig,
    target: f64,
) -> Result<(u64, Option<u64>, bool)> {
    let result = certified_boundary_from_hint(start, cap, discovered, |width| {
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

fn cost_log2_work_text(value: Option<CostValue>) -> String {
    match value {
        Some(CostValue::Finite(cost)) if cost.log2.is_finite() => cost.log2.to_string(),
        Some(CostValue::ProvenAboveTarget(lower_bound)) => {
            format!("above-target:{}", lower_bound.log2)
        }
        Some(CostValue::Infinity) => "unclassified-infinity".to_string(),
        Some(CostValue::Finite(_)) => "non-finite".to_string(),
        None => String::new(),
    }
}

fn parse_csv_record(record: &str) -> Result<InfinityWidthRow> {
    let fields = record.split(',').collect::<Vec<_>>();
    if fields.len() != 18 {
        return invalid_config(
            "work_result",
            "infinity-width result row must contain exactly 18 fields",
        );
    }
    let policy = match fields[0] {
        "Quantum128BitADPS16" => SisSecurityPolicy::Quantum128BitADPS16,
        _ => return invalid_config("work_result", "unknown SIS policy label"),
    };
    let modulus_profile = AkitaModulusProfileId::parse(fields[1])?;
    let d = parse_field(fields[2], "d")?;
    let rank = parse_field(fields[3], "rank")?;
    let coeff_linf_bound = parse_field(fields[4], "coeff_linf_bound")?;
    let max_width = parse_field(fields[5], "max_width")?;
    let scalar_n: u64 = parse_field(fields[6], "scalar_n")?;
    let expected_n = u64::from(d)
        .checked_mul(u64::from(rank))
        .ok_or_else(|| invalid_config_value("work_result", "d * rank overflowed"))?;
    if scalar_n != expected_n {
        return invalid_config("work_result", "scalar_n does not equal d * rank");
    }
    let search_cap = parse_field(fields[7], "search_cap")?;
    let hit_cap = parse_field(fields[8], "hit_cap")?;
    let profile = InfinityWidthProfile::parse_label(fields[9])?;
    let target: f64 = parse_field(fields[10], "target_bits")?;
    if target.to_bits()
        != policy
            .adps16_quantum_constraint()
            .minimum_log2_rop
            .to_bits()
    {
        return invalid_config("work_result", "target does not match the SIS policy");
    }
    let max_costs = parse_certificate(fields[11], fields[13], fields[14])?
        .map(|adps16_quantum| InfinityWidthPolicyCosts { adps16_quantum });
    let next_costs = parse_certificate(fields[12], fields[15], fields[16])?
        .map(|adps16_quantum| InfinityWidthPolicyCosts { adps16_quantum });
    let cutoff_hit_cap = match fields[17] {
        "Exact" => false,
        "AtLeast" => true,
        _ => return invalid_config("work_result", "unknown scalar cutoff kind"),
    };
    if cutoff_hit_cap != hit_cap {
        return invalid_config("work_result", "cutoff kind disagrees with hit_cap");
    }
    if max_width > search_cap || hit_cap != (max_width == search_cap) {
        return invalid_config("work_result", "cutoff is inconsistent with search_cap");
    }
    if (max_width > 0) != max_costs.is_some() {
        return invalid_config(
            "work_result",
            "accepted cutoff and accepted certificate disagree",
        );
    }
    if hit_cap == next_costs.is_some() {
        return invalid_config(
            "work_result",
            "rejected-successor certificate disagrees with cutoff kind",
        );
    }
    Ok(InfinityWidthRow {
        modulus_profile,
        d,
        rank,
        coeff_linf_bound,
        max_width,
        policy,
        search_cap,
        hit_cap,
        profile,
        max_costs,
        next_costs,
    })
}

fn parse_certificate(
    cost: &str,
    beta: &str,
    zeta: &str,
) -> Result<Option<InfinityWidthCertificate>> {
    if cost.is_empty() {
        if !beta.is_empty() || !zeta.is_empty() {
            return invalid_config(
                "work_result",
                "empty cost has non-empty optimizer coordinates",
            );
        }
        return Ok(None);
    }
    let rop = if let Some(lower_bound) = cost.strip_prefix("above-target:") {
        CostValue::ProvenAboveTarget(crate::cost::LogCost::new(parse_field(
            lower_bound,
            "above_target_cost",
        )?))
    } else if cost == "unclassified-infinity" {
        CostValue::Infinity
    } else if cost == "non-finite" {
        return invalid_config("work_result", "non-finite cost is not cacheable");
    } else {
        CostValue::finite_log2(parse_field(cost, "cost")?)
    };
    let beta = parse_optional_field(beta, "beta")?;
    let zeta = parse_optional_field(zeta, "zeta")?;
    Ok(Some(InfinityWidthCertificate { rop, beta, zeta }))
}

fn parse_field<T>(value: &str, field: &'static str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|error| EstimatorError::InvalidConfig {
            field: "work_result",
            reason: format!("invalid {field}: {error}"),
        })
}

fn parse_optional_field<T>(value: &str, field: &'static str) -> Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    if value.is_empty() {
        Ok(None)
    } else {
        parse_field(value, field).map(Some)
    }
}

fn invalid_config_value(field: &'static str, reason: impl Into<String>) -> EstimatorError {
    EstimatorError::InvalidConfig {
        field,
        reason: reason.into(),
    }
}

fn same_set<T: Copy + Ord>(left: &[T], right: &[T]) -> bool {
    left.len() == right.len()
        && left.iter().copied().collect::<BTreeSet<_>>()
            == right.iter().copied().collect::<BTreeSet<_>>()
}

#[cfg(test)]
mod tests;
