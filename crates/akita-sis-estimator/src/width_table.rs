//! ADPS16 quantum infinity norm scalar table generation.
//!
//! The generator is offline only. It discovers a boundary with the local
//! optimizer, then certifies the accepted point and its rejected successor
//! with the exhaustive beta and zeta search.

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

/// Role-independent A coefficient cells used by the current planner domain.
pub const COEFF_LINF_BUCKETS: &[u64] = &[
    2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131_071,
    262_143, 524_287, 1_048_575, 2_097_151, 4_194_303, 8_388_607, 16_777_215, 33_554_431,
    67_108_863,
];

/// Ring dimensions included in the current reachable generation domain.
pub const RING_DIMS: &[u32] = &[32, 64, 128, 256];

/// Exact modulus profiles included in the generated artifact.
pub const FAMILIES: &[AkitaModulusProfileId] = &[
    AkitaModulusProfileId::Q32Offset99,
    AkitaModulusProfileId::Q64Offset59,
    AkitaModulusProfileId::Q128OffsetA7F7,
];

/// Maximum module rank emitted for each scalar row.
pub const DEFAULT_MAX_RANK: u32 = 20;

/// Policy table search cap.
pub const DEFAULT_SEARCH_CAP: u64 = 6_400_000_000_000;

/// Legacy L2 generator cap retained for the independent Euclidean table.
/// The quantum infinity table itself uses [`DEFAULT_SEARCH_CAP`] uniformly.
pub const D128_SEARCH_CAP: u64 = DEFAULT_SEARCH_CAP;

/// Optimizer profile used to discover and certify scalar boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InfinityWidthProfile {
    /// Local minimum discovery followed by exhaustive boundary certification.
    LocalMinimum,
    /// Serial exhaustive beta and zeta search.
    ExhaustiveSerial,
    /// Parallel exhaustive beta and zeta search.
    ExhaustiveParallel,
}

impl InfinityWidthProfile {
    /// Stable provenance label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::LocalMinimum => "local-minimum+exhaustive-certification",
            Self::ExhaustiveSerial => "exhaustive-serial",
            Self::ExhaustiveParallel => "exhaustive-parallel",
        }
    }

    /// Estimator configuration for the selected profile.
    pub fn config(self) -> EstimateConfig {
        match self {
            Self::LocalMinimum => EstimateConfig {
                red_cost_model: ReductionCostModel::Adps16 {
                    mode: crate::config::Adps16Mode::Quantum,
                },
                ..EstimateConfig::lattice_estimator_parity()
            },
            Self::ExhaustiveSerial => EstimateConfig::akita_infinity_table(),
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
            coeff_linf_bounds: COEFF_LINF_BUCKETS.to_vec(),
            max_rank: DEFAULT_MAX_RANK,
            policy: SisSecurityPolicy::Quantum128BitADPS16,
            search_cap: None,
            profile: InfinityWidthProfile::LocalMinimum,
            progress_every: None,
        }
    }
}

/// Whether a config is the complete current generation domain.
pub fn is_full_infinity_width_table_config(config: &InfinityWidthTableConfig) -> bool {
    same_set(&config.profiles, FAMILIES)
        && same_set(&config.ring_dims, RING_DIMS)
        && same_set(&config.coeff_linf_bounds, COEFF_LINF_BUCKETS)
        && config.max_rank == DEFAULT_MAX_RANK
        && config.policy == SisSecurityPolicy::Quantum128BitADPS16
        && config.search_cap.is_none()
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
    for &modulus_profile in &config.profiles {
        for &d in &config.ring_dims {
            for &bound in &config.coeff_linf_bounds {
                if !reachable_role_cell(modulus_profile, d, bound) {
                    continue;
                }
                for rank in 1..=config.max_rank {
                    work.push((modulus_profile, d, rank, bound));
                }
            }
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

/// Return whether a scalar origin is reachable from at least one canonical
/// matrix-role cell. The shared coverage declaration lives in `akita-types`;
/// this adapter only maps the estimator's modulus enum to that declaration.
fn reachable_role_cell(
    modulus_profile: AkitaModulusProfileId,
    ring_dimension: u32,
    coeff_linf_bound: u64,
) -> bool {
    let profile = match modulus_profile {
        AkitaModulusProfileId::Q32Offset99 => akita_types::sis::SisModulusProfileId::Q32Offset99,
        AkitaModulusProfileId::Q64Offset59 => akita_types::sis::SisModulusProfileId::Q64Offset59,
        AkitaModulusProfileId::Q128OffsetA7F7 => {
            akita_types::sis::SisModulusProfileId::Q128OffsetA7F7
        }
    };
    akita_types::sis::SIS_MATRIX_ROLES
        .iter()
        .copied()
        .any(|role| {
            akita_types::sis::ajtai_key::sis_role_cell(
                role,
                profile,
                ring_dimension,
                u128::from(coeff_linf_bound),
            )
            .is_some()
        })
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
            );
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
        rows.push(max_secure_width_row(
            modulus_profile,
            d,
            rank,
            bound,
            config,
            estimator_config,
        )?);
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

/// Convert ring-origin rows to runtime `(d, B) -> widths[rank]` match arms.
///
/// Scalar certification still groups by `(B, n)` and takes the min `m`. The
/// emitted runtime table projects those cutoffs onto each reachable ring
/// dimension as `width[r - 1] = cutoff_m(B, n = r * d) / d`.
pub fn rust_table_arms(
    rows: &[InfinityWidthRow],
    max_rank: u32,
) -> BTreeMap<AkitaModulusProfileId, Vec<String>> {
    let mut scalar = BTreeMap::<(AkitaModulusProfileId, u64, u64), u64>::new();
    let mut pairs = BTreeSet::<(AkitaModulusProfileId, u32, u64)>::new();
    for row in rows {
        let n = u64::from(row.d) * u64::from(row.rank);
        let scalar_m = row.max_width.checked_mul(u64::from(row.d)).unwrap_or(0);
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
    let mut arms = BTreeMap::<AkitaModulusProfileId, Vec<String>>::new();
    for (profile, d, bound) in pairs {
        let mut widths = Vec::with_capacity(max_rank as usize);
        let mut complete = true;
        for rank in 1..=max_rank {
            let n = u64::from(d) * u64::from(rank);
            match scalar.get(&(profile, bound, n)) {
                Some(&scalar_m) => widths.push(scalar_m / u64::from(d)),
                None => {
                    complete = false;
                    break;
                }
            }
        }
        if !complete {
            continue;
        }
        let body = widths
            .iter()
            .map(|width| width.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        arms.entry(profile)
            .or_default()
            .push(format!("({d}, {bound}) => Some(&[{body}]),"));
    }
    arms
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
    let discovered = max_true_in_prefix(search_cap, discovery)?;
    let (max_width, next_width, hit_cap) =
        if table_config.profile == InfinityWidthProfile::LocalMinimum {
            let cert_config = InfinityWidthProfile::ExhaustiveSerial.config();
            certify_boundary(
                modulus_profile,
                d,
                rank,
                coeff_linf_bound,
                search_cap,
                discovered.max_value,
                &cert_config,
                target,
            )?
        } else {
            (
                discovered.max_value,
                discovered.next_value,
                discovered.hit_cap,
            )
        };
    let max_costs = (max_width > 0)
        .then(|| {
            estimate_width(
                modulus_profile,
                d,
                rank,
                max_width,
                coeff_linf_bound,
                &InfinityWidthProfile::ExhaustiveSerial.config(),
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
                &InfinityWidthProfile::ExhaustiveSerial.config(),
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
    let mut accepted = discovered.min(cap);
    while accepted > 0 {
        let cost = estimate_width(modulus_profile, d, rank, accepted, bound, config)?;
        if secure_or_error(cost.rop, target)? {
            break;
        }
        accepted -= 1;
    }
    let Some(mut successor) = accepted.checked_add(1) else {
        return Ok((accepted, None, false));
    };
    while successor <= cap {
        let cost = estimate_width(modulus_profile, d, rank, successor, bound, config)?;
        if !secure_or_error(cost.rop, target)? {
            return Ok((accepted, Some(successor), false));
        }
        accepted = successor;
        successor = successor
            .checked_add(1)
            .ok_or_else(|| EstimatorError::InvalidConfig {
                field: "search_cap",
                reason: "width successor overflow".to_string(),
            })?;
    }
    Ok((cap, None, true))
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

fn max_true_in_prefix<F>(cap: u64, mut predicate: F) -> Result<PrefixSearchResult>
where
    F: FnMut(u64) -> Result<bool>,
{
    if cap == 0 {
        return invalid_config("cap", "cap must be positive");
    }
    let mut first = 1;
    while first <= cap && !predicate(first)? {
        first = first
            .checked_add(1)
            .ok_or_else(|| EstimatorError::InvalidConfig {
                field: "search_cap",
                reason: "search probe overflow".to_string(),
            })?;
    }
    if first > cap {
        return Ok(PrefixSearchResult {
            max_value: 0,
            next_value: Some(1),
            hit_cap: false,
        });
    }
    let mut low = first;
    let mut high = first.checked_mul(2).unwrap_or(cap).min(cap);
    if high == low && high < cap {
        high = high
            .checked_add(1)
            .ok_or_else(|| EstimatorError::InvalidConfig {
                field: "search_cap",
                reason: "search probe overflow".to_string(),
            })?;
    }
    while high < cap && predicate(high)? {
        low = high;
        high = high.checked_mul(2).unwrap_or(cap).min(cap);
    }
    if high == cap && predicate(cap)? {
        return Ok(PrefixSearchResult {
            max_value: cap,
            next_value: None,
            hit_cap: true,
        });
    }
    while low + 1 < high {
        let mid = low + (high - low) / 2;
        if predicate(mid)? {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(PrefixSearchResult {
        max_value: low,
        next_value: Some(high),
        hit_cap: false,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PrefixSearchResult {
    max_value: u64,
    next_value: Option<u64>,
    hit_cap: bool,
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
mod tests {
    use super::*;

    #[test]
    fn prefix_search_finds_last_true_value() {
        assert_eq!(
            max_true_in_prefix(16, |value| Ok(value <= 9))
                .unwrap()
                .max_value,
            9
        );
    }

    #[test]
    fn infinity_never_counts_as_secure() {
        assert!(!security_met(CostValue::Infinity, 128.0));
        assert!(secure_or_error(CostValue::Infinity, 128.0).is_err());
    }

    #[test]
    fn csv_has_no_classical_columns() {
        assert!(!InfinityWidthRow::csv_header().contains("classical"));
    }

    #[test]
    fn generation_filters_to_canonical_role_cells() {
        assert!(reachable_role_cell(
            AkitaModulusProfileId::Q128OffsetA7F7,
            32,
            15
        ));
        assert!(!reachable_role_cell(
            AkitaModulusProfileId::Q128OffsetA7F7,
            32,
            2
        ));
        assert!(reachable_role_cell(
            AkitaModulusProfileId::Q128OffsetA7F7,
            64,
            2
        ));
        assert!(!reachable_role_cell(
            AkitaModulusProfileId::Q128OffsetA7F7,
            16,
            15
        ));
    }
}
