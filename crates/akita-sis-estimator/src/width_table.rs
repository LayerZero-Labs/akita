//! Infinity-norm max-width table generation helpers.
//!
//! The planner-facing infinity table shape is:
//!
//! ```text
//! (family, ring_dimension, coeff_linf_bound) -> max widths by rank
//! ```
//!
//! This module is offline-only. It generates production Rust tables and CSV
//! audit artifacts for the infinity estimator.

use crate::{
    akita::{scalar_sis_from_ring_wide, AkitaModulusFamily},
    config::{EstimateConfig, OptimizerConfig, SearchMode},
    cost::{CostValue, LatticeCost},
    error::{EstimatorError, Result},
    estimate,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicUsize, Ordering};

/// Coefficient-L∞ buckets needed by Akita planner envelopes.
///
/// Keep in lockstep with `crates/akita-types/src/sis/ajtai_key.rs`.
pub const COEFF_LINF_BUCKETS: &[u64] = &[
    2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131_071,
    262_143, 524_287, 1_048_575, 2_097_151, 4_194_303, 8_388_607, 16_777_215, 33_554_431,
    67_108_863,
];

/// Ring dimensions covered by Akita SIS table generation.
pub const RING_DIMS: &[u32] = &[32, 64, 128, 256];

/// Modulus families covered by Akita SIS table generation.
pub const FAMILIES: &[AkitaModulusFamily] = &[
    AkitaModulusFamily::Q32,
    AkitaModulusFamily::Q64,
    AkitaModulusFamily::Q128,
];

/// Default target security bits for the infinity-norm comparison table.
pub const DEFAULT_INFINITY_TARGET_BITS: f64 = 138.0;

/// Default maximum rank emitted by the current Euclidean SIS table.
pub const DEFAULT_MAX_RANK: u32 = 20;

/// Default ring-width search cap used by the legacy Euclidean generator.
pub const DEFAULT_SEARCH_CAP: u64 = 10_000_000_000;

/// Wider legacy cap for `d=128`, where shipped schedules have needed it.
pub const D128_SEARCH_CAP: u64 = 50_000_000_000;

/// Optimizer profile used while generating the infinity width table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InfinityWidthProfile {
    /// Lattice-estimator parity search. Rows may be generated in parallel, but
    /// each row uses Python-compatible local-minimum beta and zeta search.
    LocalMinimum,
    /// Serial exhaustive beta/zeta search.
    ExhaustiveSerial,
    /// Parallel exhaustive beta/zeta search.
    ExhaustiveParallel,
}

impl InfinityWidthProfile {
    /// Stable profile label for CSV metadata and CLI flags.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::LocalMinimum => "local-minimum",
            Self::ExhaustiveSerial => "exhaustive-serial",
            Self::ExhaustiveParallel => "exhaustive-parallel",
        }
    }

    /// Estimator configuration for this profile.
    #[must_use]
    pub fn config(self) -> EstimateConfig {
        match self {
            Self::LocalMinimum => EstimateConfig::lattice_estimator_parity(),
            Self::ExhaustiveSerial => EstimateConfig::akita_infinity_table(),
            Self::ExhaustiveParallel => EstimateConfig {
                optimizer: OptimizerConfig::OptimizeZeta {
                    beta: SearchMode::ExhaustiveParallel,
                    zeta: SearchMode::ExhaustiveParallel,
                },
                ..EstimateConfig::default()
            },
        }
    }
}

/// One max-width table generation request.
#[derive(Clone, Debug, PartialEq)]
pub struct InfinityWidthTableConfig {
    /// Modulus families to generate.
    pub families: Vec<AkitaModulusFamily>,
    /// Ring dimensions to generate.
    pub ring_dims: Vec<u32>,
    /// Coefficient-L∞ buckets to generate.
    pub coeff_linf_bounds: Vec<u64>,
    /// Maximum module rank.
    pub max_rank: u32,
    /// Security threshold for `rop_log2`.
    pub target_bits: f64,
    /// Optional caller cap on ring-element width.
    pub search_cap: Option<u64>,
    /// Optimizer profile.
    pub profile: InfinityWidthProfile,
    /// Optional progress report interval, in completed rows.
    pub progress_every: Option<usize>,
}

impl Default for InfinityWidthTableConfig {
    fn default() -> Self {
        Self {
            families: FAMILIES.to_vec(),
            ring_dims: RING_DIMS.to_vec(),
            coeff_linf_bounds: COEFF_LINF_BUCKETS.to_vec(),
            max_rank: DEFAULT_MAX_RANK,
            target_bits: DEFAULT_INFINITY_TARGET_BITS,
            search_cap: None,
            profile: InfinityWidthProfile::LocalMinimum,
            progress_every: None,
        }
    }
}

/// Return whether `config` covers the complete production infinity SIS width-table keyspace.
#[must_use]
pub fn is_full_infinity_width_table_config(config: &InfinityWidthTableConfig) -> bool {
    same_set(&config.families, FAMILIES)
        && same_set(&config.ring_dims, RING_DIMS)
        && same_set(&config.coeff_linf_bounds, COEFF_LINF_BUCKETS)
        && config.max_rank == DEFAULT_MAX_RANK
        && config.target_bits == DEFAULT_INFINITY_TARGET_BITS
        && config.search_cap.is_none()
}

/// One generated comparison row.
#[derive(Clone, Debug, PartialEq)]
pub struct InfinityWidthRow {
    /// Modulus family.
    pub family: AkitaModulusFamily,
    /// Ring dimension.
    pub d: u32,
    /// Module rank.
    pub rank: u32,
    /// Coefficient-L∞ bound.
    pub coeff_linf_bound: u64,
    /// Largest secure ring-element width found within the search cap.
    pub max_width: u64,
    /// Security threshold.
    pub target_bits: f64,
    /// Actual cap used for this row.
    pub search_cap: u64,
    /// Whether `max_width == search_cap`, so the row is a lower bound.
    pub hit_cap: bool,
    /// Optimizer profile label.
    pub profile: InfinityWidthProfile,
    /// Cost at `max_width`, if `max_width > 0`.
    pub max_cost: Option<LatticeCost>,
    /// Cost at `max_width + 1`, when that probe is representable.
    pub next_cost: Option<LatticeCost>,
}

impl InfinityWidthRow {
    /// Whether the row is a strict bracket rather than a lower bound.
    #[must_use]
    pub const fn is_tight(&self) -> bool {
        !self.hit_cap
    }

    /// CSV header for row-oriented comparison artifacts.
    #[must_use]
    pub const fn csv_header() -> &'static str {
        "family,d,rank,coeff_linf_bound,max_width,target_bits,search_cap,hit_cap,profile,max_rop_log2,next_rop_log2,max_security_margin_bits,next_failure_margin_bits,max_beta,max_zeta,next_beta,next_zeta"
    }

    /// Format one CSV row.
    #[must_use]
    pub fn to_csv_record(&self) -> String {
        format!(
            "{},{},{},{},{},{:.17},{},{},{},{},{},{},{},{},{},{},{}",
            self.family.label(),
            self.d,
            self.rank,
            self.coeff_linf_bound,
            self.max_width,
            self.target_bits,
            self.search_cap,
            self.hit_cap,
            self.profile.label(),
            cost_log2_text(self.max_cost.as_ref().map(|cost| cost.rop)),
            cost_log2_text(self.next_cost.as_ref().map(|cost| cost.rop)),
            signed_margin_text(
                self.max_cost
                    .as_ref()
                    .and_then(|cost| security_margin_bits(cost.rop, self.target_bits)),
            ),
            signed_margin_text(
                self.next_cost
                    .as_ref()
                    .and_then(|cost| security_failure_margin_bits(cost.rop, self.target_bits)),
            ),
            optional_u32_text(self.max_cost.as_ref().and_then(|cost| cost.beta)),
            optional_u64_text(self.max_cost.as_ref().and_then(|cost| cost.zeta)),
            optional_u32_text(self.next_cost.as_ref().and_then(|cost| cost.beta)),
            optional_u64_text(self.next_cost.as_ref().and_then(|cost| cost.zeta)),
        )
    }
}

/// Generate row-oriented infinity max-width comparison data.
///
/// # Errors
///
/// Returns estimator errors for malformed rows or unsupported profile choices.
pub fn generate_infinity_width_rows(
    config: &InfinityWidthTableConfig,
) -> Result<Vec<InfinityWidthRow>> {
    validate_table_config(config)?;
    let estimator_config = config.profile.config();
    let mut work = Vec::new();
    for &family in &config.families {
        for &d in &config.ring_dims {
            for &coeff_linf_bound in &config.coeff_linf_bounds {
                for rank in 1..=config.max_rank {
                    work.push((family, d, rank, coeff_linf_bound));
                }
            }
        }
    }
    generate_rows_from_work(work, config, &estimator_config)
}

#[cfg(feature = "parallel")]
fn generate_rows_from_work(
    work: Vec<(AkitaModulusFamily, u32, u32, u64)>,
    config: &InfinityWidthTableConfig,
    estimator_config: &EstimateConfig,
) -> Result<Vec<InfinityWidthRow>> {
    let total = work.len();
    let completed = AtomicUsize::new(0);
    work.into_par_iter()
        .map(|(family, d, rank, coeff_linf_bound)| {
            let row =
                max_secure_width_row(family, d, rank, coeff_linf_bound, config, estimator_config);
            report_progress(config.progress_every, &completed, total);
            row
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn generate_rows_from_work(
    work: Vec<(AkitaModulusFamily, u32, u32, u64)>,
    config: &InfinityWidthTableConfig,
    estimator_config: &EstimateConfig,
) -> Result<Vec<InfinityWidthRow>> {
    let total = work.len();
    let mut completed = 0usize;
    work.into_iter()
        .map(|(family, d, rank, coeff_linf_bound)| {
            let row =
                max_secure_width_row(family, d, rank, coeff_linf_bound, config, estimator_config);
            completed += 1;
            report_progress(config.progress_every, completed, total);
            row
        })
        .collect()
}

#[cfg(feature = "parallel")]
fn report_progress(progress_every: Option<usize>, completed: &AtomicUsize, total: usize) {
    let Some(progress_every) = progress_every else {
        return;
    };
    if progress_every == 0 {
        return;
    }
    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
    if done == total || done.is_multiple_of(progress_every) {
        eprintln!("infinity width table progress: {done}/{total} rows");
    }
}

#[cfg(not(feature = "parallel"))]
fn report_progress(progress_every: Option<usize>, completed: usize, total: usize) {
    let Some(progress_every) = progress_every else {
        return;
    };
    if progress_every == 0 {
        return;
    }
    if completed == total || completed.is_multiple_of(progress_every) {
        eprintln!("infinity width table progress: {completed}/{total} rows");
    }
}

/// Validate monotonicity and security brackets for generated rows.
///
/// # Errors
///
/// Returns an error when a row's stored costs do not bracket the target, widths
/// decrease as rank increases, or widths increase as the coefficient bound gets
/// looser.
pub fn validate_infinity_width_rows(rows: &[InfinityWidthRow]) -> Result<()> {
    for row in rows {
        if let Some(cost) = &row.max_cost {
            if !security_met(cost.rop, row.target_bits) {
                return invalid_config("rows", "max_width row does not meet target_bits");
            }
        } else if row.max_width != 0 {
            return invalid_config("rows", "positive max_width is missing max_cost");
        }
        if let Some(cost) = &row.next_cost {
            if security_met(cost.rop, row.target_bits) {
                return invalid_config("rows", "next_width row still meets target_bits");
            }
        }
    }
    validate_rank_monotonicity(rows)?;
    validate_bound_monotonicity(rows)
}

/// Convert rows into generated table match arms grouped by family.
#[must_use]
pub fn rust_table_arms(
    rows: &[InfinityWidthRow],
    max_rank: u32,
) -> BTreeMap<AkitaModulusFamily, Vec<String>> {
    let mut grouped = BTreeMap::<(AkitaModulusFamily, u32, u64), Vec<&InfinityWidthRow>>::new();
    for row in rows {
        grouped
            .entry((row.family, row.d, row.coeff_linf_bound))
            .or_default()
            .push(row);
    }

    let mut arms = BTreeMap::<AkitaModulusFamily, Vec<String>>::new();
    for ((family, d, coeff_linf_bound), mut group) in grouped {
        group.sort_by_key(|row| row.rank);
        if group.len() != max_rank as usize {
            continue;
        }
        if group.iter().all(|row| row.max_width == 0) {
            continue;
        }
        let widths = group
            .into_iter()
            .map(|row| format_u64_underscored(row.max_width))
            .collect::<Vec<_>>()
            .join(", ");
        arms.entry(family).or_default().push(format!(
            "({}, {}) => Some(&[{}]),",
            d, coeff_linf_bound, widths
        ));
    }
    arms
}

fn validate_rank_monotonicity(rows: &[InfinityWidthRow]) -> Result<()> {
    let mut groups = BTreeMap::<(AkitaModulusFamily, u32, u64), Vec<&InfinityWidthRow>>::new();
    for row in rows {
        groups
            .entry((row.family, row.d, row.coeff_linf_bound))
            .or_default()
            .push(row);
    }
    for group in groups.values_mut() {
        group.sort_by_key(|row| row.rank);
        let mut prior = None::<&InfinityWidthRow>;
        for row in group {
            if let Some(prior_row) = prior {
                if row.max_width < prior_row.max_width {
                    return invalid_config(
                        "rows",
                        &format!(
                            "max_width decreased as rank increased for family={} d={} coeff_linf_bound={}: rank {} width {} -> rank {} width {}",
                            row.family.label(),
                            row.d,
                            row.coeff_linf_bound,
                            prior_row.rank,
                            prior_row.max_width,
                            row.rank,
                            row.max_width,
                        ),
                    );
                }
            }
            prior = Some(row);
        }
    }
    Ok(())
}

fn validate_bound_monotonicity(rows: &[InfinityWidthRow]) -> Result<()> {
    let mut groups = BTreeMap::<(AkitaModulusFamily, u32, u32), Vec<&InfinityWidthRow>>::new();
    for row in rows {
        groups
            .entry((row.family, row.d, row.rank))
            .or_default()
            .push(row);
    }
    for group in groups.values_mut() {
        group.sort_by_key(|row| row.coeff_linf_bound);
        let mut prior = None::<&InfinityWidthRow>;
        for row in group {
            if let Some(prior_row) = prior {
                if row.max_width > prior_row.max_width {
                    return invalid_config(
                        "rows",
                        &format!(
                            "max_width increased as coeff_linf_bound increased for family={} d={} rank={}: bound {} width {} -> bound {} width {}",
                            row.family.label(),
                            row.d,
                            row.rank,
                            prior_row.coeff_linf_bound,
                            prior_row.max_width,
                            row.coeff_linf_bound,
                            row.max_width,
                        ),
                    );
                }
            }
            prior = Some(row);
        }
    }
    Ok(())
}

fn validate_table_config(config: &InfinityWidthTableConfig) -> Result<()> {
    if config.families.is_empty() {
        return invalid_config("families", "at least one family is required");
    }
    if config.ring_dims.is_empty() {
        return invalid_config("ring_dims", "at least one ring dimension is required");
    }
    if config.coeff_linf_bounds.is_empty() {
        return invalid_config(
            "coeff_linf_bounds",
            "at least one coefficient bound is required",
        );
    }
    if config.max_rank == 0 {
        return invalid_config("max_rank", "max_rank must be positive");
    }
    if !config.target_bits.is_finite() || config.target_bits <= 0.0 {
        return invalid_config("target_bits", "target_bits must be finite and positive");
    }
    if config.search_cap == Some(0) {
        return invalid_config("search_cap", "search_cap must be positive when present");
    }
    Ok(())
}

fn invalid_config<T>(field: &'static str, reason: &str) -> Result<T> {
    Err(EstimatorError::InvalidConfig {
        field,
        reason: reason.to_string(),
    })
}

fn max_secure_width_row(
    family: AkitaModulusFamily,
    d: u32,
    rank: u32,
    coeff_linf_bound: u64,
    table_config: &InfinityWidthTableConfig,
    estimator_config: &EstimateConfig,
) -> Result<InfinityWidthRow> {
    let search_cap = row_search_cap(d, table_config.search_cap)?;
    let mut probe =
        |width| estimate_width(family, d, rank, width, coeff_linf_bound, estimator_config);
    let result = max_true_in_prefix(search_cap, |width| {
        probe(width).map(|cost| security_met(cost.rop, table_config.target_bits))
    })?;
    let max_cost = if result.max_value == 0 {
        None
    } else {
        Some(probe(result.max_value)?)
    };
    let next_cost = result.next_value.map(&mut probe).transpose()?;
    Ok(InfinityWidthRow {
        family,
        d,
        rank,
        coeff_linf_bound,
        max_width: result.max_value,
        target_bits: table_config.target_bits,
        search_cap,
        hit_cap: result.hit_cap,
        profile: table_config.profile,
        max_cost,
        next_cost,
    })
}

fn row_search_cap(d: u32, requested_cap: Option<u64>) -> Result<u64> {
    if d == 0 {
        return Err(EstimatorError::InvalidParameter {
            field: "d",
            reason: "ring dimension must be positive".to_string(),
        });
    }
    let default_cap = if d == 128 {
        D128_SEARCH_CAP
    } else {
        DEFAULT_SEARCH_CAP
    };
    let cap = requested_cap.unwrap_or(default_cap);
    if cap == 0 {
        return Err(EstimatorError::InvalidParameter {
            field: "search_cap",
            reason: "search cap must be positive".to_string(),
        });
    }
    Ok(cap)
}

fn estimate_width(
    family: AkitaModulusFamily,
    d: u32,
    rank: u32,
    width: u64,
    coeff_linf_bound: u64,
    config: &EstimateConfig,
) -> Result<LatticeCost> {
    let params = scalar_sis_from_ring_wide(family, d, rank, width, coeff_linf_bound)?;
    estimate(&params, config)
}

fn security_met(rop: CostValue, target_bits: f64) -> bool {
    match rop {
        CostValue::Infinity => true,
        CostValue::Finite(cost) => cost.log2 >= target_bits,
    }
}

fn security_margin_bits(rop: CostValue, target_bits: f64) -> Option<f64> {
    match rop {
        CostValue::Infinity => None,
        CostValue::Finite(cost) => Some(cost.log2 - target_bits),
    }
}

fn security_failure_margin_bits(rop: CostValue, target_bits: f64) -> Option<f64> {
    match rop {
        CostValue::Infinity => None,
        CostValue::Finite(cost) => Some(target_bits - cost.log2),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PrefixSearchResult {
    max_value: u64,
    next_value: Option<u64>,
    hit_cap: bool,
}

fn max_true_in_prefix<F>(cap: u64, mut predicate: F) -> Result<PrefixSearchResult>
where
    F: FnMut(u64) -> Result<bool>,
{
    if cap == 0 {
        return invalid_config("cap", "cap must be positive");
    }
    if !predicate(1)? {
        return Ok(PrefixSearchResult {
            max_value: 0,
            next_value: Some(1),
            hit_cap: false,
        });
    }

    let mut low = 1;
    let mut high = 2.min(cap);
    while high < cap && predicate(high)? {
        low = high;
        high = high.saturating_mul(2).min(cap);
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

fn cost_log2_text(value: Option<CostValue>) -> String {
    match value {
        Some(CostValue::Finite(cost)) => format!("{:.12}", cost.log2),
        Some(CostValue::Infinity) => "inf".to_string(),
        None => String::new(),
    }
}

fn signed_margin_text(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.12}"))
}

fn optional_u32_text(value: Option<u32>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn optional_u64_text(value: Option<u64>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn format_u64_underscored(value: u64) -> String {
    let raw = value.to_string();
    let mut out = String::new();
    for (index, ch) in raw.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            out.push('_');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

fn same_set<T>(left: &[T], right: &[T]) -> bool
where
    T: Copy + Ord,
{
    left.len() == right.len()
        && left.iter().copied().collect::<BTreeSet<_>>()
            == right.iter().copied().collect::<BTreeSet<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_search_finds_last_true_value() {
        let result = max_true_in_prefix(16, |value| Ok(value <= 9)).unwrap();
        assert_eq!(
            result,
            PrefixSearchResult {
                max_value: 9,
                next_value: Some(10),
                hit_cap: false,
            }
        );
    }

    #[test]
    fn prefix_search_marks_cap_hit_as_lower_bound() {
        let result = max_true_in_prefix(16, |_| Ok(true)).unwrap();
        assert_eq!(
            result,
            PrefixSearchResult {
                max_value: 16,
                next_value: None,
                hit_cap: true,
            }
        );
    }

    #[test]
    fn prefix_search_ignores_late_secure_islands() {
        let result = max_true_in_prefix(32, |value| Ok(value <= 9 || value >= 17)).unwrap();
        assert_eq!(
            result,
            PrefixSearchResult {
                max_value: 9,
                next_value: Some(10),
                hit_cap: false,
            }
        );
    }

    #[test]
    fn infinity_csv_header_includes_boundary_margins() {
        let header = InfinityWidthRow::csv_header();
        assert!(header.contains("max_security_margin_bits"));
        assert!(header.contains("next_failure_margin_bits"));
    }

    #[test]
    fn default_buckets_cover_planner_digit_bounds() {
        for bound in [3, 7, 15, 31, 63] {
            assert!(COEFF_LINF_BUCKETS.contains(&bound));
        }
    }

    #[test]
    fn row_search_cap_uses_durable_generation_defaults() {
        assert_eq!(row_search_cap(32, None).unwrap(), DEFAULT_SEARCH_CAP);
        assert_eq!(row_search_cap(128, None).unwrap(), D128_SEARCH_CAP);
        assert_eq!(row_search_cap(32, Some(100)).unwrap(), 100);
        assert_eq!(
            row_search_cap(32, Some(u64::from(u32::MAX))).unwrap(),
            u64::from(u32::MAX)
        );
    }

    #[test]
    fn smoke_generates_small_secure_row() {
        let config = InfinityWidthTableConfig {
            families: vec![AkitaModulusFamily::Q32],
            ring_dims: vec![32],
            coeff_linf_bounds: vec![15],
            max_rank: 1,
            search_cap: Some(2),
            profile: InfinityWidthProfile::LocalMinimum,
            ..InfinityWidthTableConfig::default()
        };
        let rows = generate_infinity_width_rows(&config).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].max_width, 2);
        assert!(rows[0].hit_cap);
        assert!(rows[0].max_cost.is_some());
    }

    #[test]
    fn generated_rows_pass_monotonicity_and_bracket_checks() {
        let config = InfinityWidthTableConfig {
            families: vec![AkitaModulusFamily::Q32],
            ring_dims: vec![32],
            coeff_linf_bounds: vec![15, 255],
            max_rank: 3,
            search_cap: Some(8),
            profile: InfinityWidthProfile::LocalMinimum,
            ..InfinityWidthTableConfig::default()
        };
        let rows = generate_infinity_width_rows(&config).unwrap();
        validate_infinity_width_rows(&rows).unwrap();
    }
}
