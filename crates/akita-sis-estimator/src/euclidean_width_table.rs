//! Euclidean L2 max-width table generation helpers.

use crate::{
    akita::{scalar_sis_from_ring_euclidean, AkitaModulusFamily},
    config::EstimateConfig,
    cost::{CostValue, EstimateTag, LatticeCost, LogCost},
    error::{EstimatorError, Result},
    estimate,
    width_table::{D128_SEARCH_CAP, DEFAULT_MAX_RANK, DEFAULT_SEARCH_CAP, RING_DIMS},
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

/// Current shipped L2 table target security level.
pub const DEFAULT_EUCLIDEAN_TARGET_BITS: f64 = 128.0;
/// Smallest squared-collision power-of-two bucket in the shipped L2 table.
pub const MIN_LOG_BUCKET: u32 = 1;
/// Largest squared-collision power-of-two bucket in the shipped L2 table.
pub const MAX_LOG_BUCKET: u32 = 84;
/// Coefficient-L∞ buckets used to derive extra L2 collision keys.
pub const COEFF_LINF_BUCKETS: &[u64] = &[
    2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131_071,
    262_143, 524_287, 1_048_575, 2_097_151, 4_194_303, 8_388_607, 16_777_215, 33_554_431,
    67_108_863,
];
/// Modulus families covered by Akita SIS table generation.
pub const FAMILIES: &[AkitaModulusFamily] = &[
    AkitaModulusFamily::Q32,
    AkitaModulusFamily::Q64,
    AkitaModulusFamily::Q128,
];

/// One Euclidean max-width generation request.
#[derive(Clone, Debug, PartialEq)]
pub struct EuclideanWidthTableConfig {
    /// Modulus families to generate.
    pub families: Vec<AkitaModulusFamily>,
    /// Ring dimensions to generate.
    pub ring_dims: Vec<u32>,
    /// Squared L2 collision keys to generate.
    pub collision_l2_sq: Vec<u128>,
    /// Maximum module rank.
    pub max_rank: u32,
    /// Security threshold for `rop_log2`.
    pub target_bits: f64,
    /// Optional caller cap on ring-element width.
    pub search_cap: Option<u64>,
}

impl Default for EuclideanWidthTableConfig {
    fn default() -> Self {
        Self {
            families: FAMILIES.to_vec(),
            ring_dims: RING_DIMS.to_vec(),
            collision_l2_sq: l2_table_collision_keys(),
            max_rank: DEFAULT_MAX_RANK,
            target_bits: DEFAULT_EUCLIDEAN_TARGET_BITS,
            search_cap: None,
        }
    }
}

/// Return whether `config` is the complete production L2 SIS width-table job.
#[must_use]
pub fn is_full_euclidean_width_table_config(config: &EuclideanWidthTableConfig) -> bool {
    same_set(&config.families, FAMILIES)
        && same_set(&config.ring_dims, RING_DIMS)
        && same_set(&config.collision_l2_sq, &l2_table_collision_keys())
        && config.max_rank == DEFAULT_MAX_RANK
        && config.target_bits == DEFAULT_EUCLIDEAN_TARGET_BITS
        && config.search_cap.is_none()
}

/// One generated Euclidean comparison row.
#[derive(Clone, Debug, PartialEq)]
pub struct EuclideanWidthRow {
    /// Modulus family.
    pub family: AkitaModulusFamily,
    /// Ring dimension.
    pub d: u32,
    /// Module rank.
    pub rank: u32,
    /// Squared L2 collision key.
    pub collision_l2_sq: u128,
    /// Largest secure ring-element width found within the search cap.
    pub max_width: u64,
    /// Security threshold.
    pub target_bits: f64,
    /// Actual cap used for this row.
    pub search_cap: u64,
    /// Whether `max_width == search_cap`, so the row is a lower bound.
    pub hit_cap: bool,
    /// Cost at `max_width`, if `max_width > 0`.
    pub max_cost: Option<LatticeCost>,
    /// Cost at `max_width + 1`, when that probe is representable.
    pub next_cost: Option<LatticeCost>,
}

impl EuclideanWidthRow {
    /// CSV header for row-oriented comparison artifacts.
    #[must_use]
    pub const fn csv_header() -> &'static str {
        "family,d,rank,collision_l2_sq,max_width,target_bits,search_cap,hit_cap,max_rop_log2,next_rop_log2,max_beta,next_beta"
    }

    /// Format one CSV row.
    #[must_use]
    pub fn to_csv_record(&self) -> String {
        format!(
            "{},{},{},{},{},{:.17},{},{},{},{},{},{}",
            self.family.label(),
            self.d,
            self.rank,
            self.collision_l2_sq,
            self.max_width,
            self.target_bits,
            self.search_cap,
            self.hit_cap,
            cost_log2_text(self.max_cost.as_ref().map(|cost| cost.rop)),
            cost_log2_text(self.next_cost.as_ref().map(|cost| cost.rop)),
            optional_u32_text(self.max_cost.as_ref().and_then(|cost| cost.beta)),
            optional_u32_text(self.next_cost.as_ref().and_then(|cost| cost.beta)),
        )
    }
}

/// Return the full collision-key set used by the shipped L2 table.
#[must_use]
pub fn l2_table_collision_keys() -> Vec<u128> {
    let mut keys = BTreeSet::new();
    keys.extend(power_of_two_collision_keys());
    keys.extend(derived_l2_collision_keys());
    keys.into_iter().collect()
}

/// Return `2^MIN_LOG_BUCKET ..= 2^MAX_LOG_BUCKET`.
#[must_use]
pub fn power_of_two_collision_keys() -> Vec<u128> {
    (MIN_LOG_BUCKET..=MAX_LOG_BUCKET)
        .map(|power| 1u128 << power)
        .collect()
}

/// Return the derived keys `d * B^2` used by the production table.
#[must_use]
pub fn derived_l2_collision_keys() -> Vec<u128> {
    let mut keys = BTreeSet::new();
    for &d in RING_DIMS {
        for &bound in COEFF_LINF_BUCKETS {
            keys.insert(u128::from(d) * coeff_linf_bucket_sq(bound));
        }
    }
    keys.into_iter().collect()
}

/// Return `B²` for a coefficient-L∞ bucket without float rounding.
#[must_use]
pub fn coeff_linf_bucket_sq(bound: u64) -> u128 {
    if bound <= 3 {
        return u128::from(bound) * u128::from(bound);
    }
    let k = (bound + 1).ilog2();
    (1u128 << (2 * k)) - (1u128 << (k + 1)) + 1
}

/// Generate row-oriented Euclidean max-width comparison data.
pub fn generate_euclidean_width_rows(
    config: &EuclideanWidthTableConfig,
) -> Result<Vec<EuclideanWidthRow>> {
    validate_table_config(config)?;
    let estimator_config = EstimateConfig::akita_euclidean_table();
    let mut work = Vec::new();
    for &family in &config.families {
        for &d in &config.ring_dims {
            for &collision_l2_sq in &config.collision_l2_sq {
                for rank in 1..=config.max_rank {
                    work.push((family, d, rank, collision_l2_sq));
                }
            }
        }
    }
    generate_rows_from_work(work, config, &estimator_config)
}

#[cfg(feature = "parallel")]
fn generate_rows_from_work(
    work: Vec<(AkitaModulusFamily, u32, u32, u128)>,
    config: &EuclideanWidthTableConfig,
    estimator_config: &EstimateConfig,
) -> Result<Vec<EuclideanWidthRow>> {
    work.into_par_iter()
        .map(|(family, d, rank, collision_l2_sq)| {
            max_secure_width_row(family, d, rank, collision_l2_sq, config, estimator_config)
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn generate_rows_from_work(
    work: Vec<(AkitaModulusFamily, u32, u32, u128)>,
    config: &EuclideanWidthTableConfig,
    estimator_config: &EstimateConfig,
) -> Result<Vec<EuclideanWidthRow>> {
    work.into_iter()
        .map(|(family, d, rank, collision_l2_sq)| {
            max_secure_width_row(family, d, rank, collision_l2_sq, config, estimator_config)
        })
        .collect()
}

/// Validate monotonicity and security brackets for generated rows.
pub fn validate_euclidean_width_rows(rows: &[EuclideanWidthRow]) -> Result<()> {
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
    validate_rank_monotonicity(rows)
}

/// Convert rows into generated table match arms grouped by family.
#[must_use]
pub fn rust_table_arms(
    rows: &[EuclideanWidthRow],
    max_rank: u32,
) -> BTreeMap<AkitaModulusFamily, Vec<String>> {
    let mut grouped = BTreeMap::<(AkitaModulusFamily, u32, u128), Vec<&EuclideanWidthRow>>::new();
    for row in rows {
        grouped
            .entry((row.family, row.d, row.collision_l2_sq))
            .or_default()
            .push(row);
    }

    let mut arms = BTreeMap::<AkitaModulusFamily, Vec<String>>::new();
    for ((family, d, collision), mut group) in grouped {
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
        arms.entry(family)
            .or_default()
            .push(format!("({}, {}) => Some(&[{}]),", d, collision, widths));
    }
    arms
}

fn validate_rank_monotonicity(rows: &[EuclideanWidthRow]) -> Result<()> {
    let mut groups = BTreeMap::<(AkitaModulusFamily, u32, u128), Vec<&EuclideanWidthRow>>::new();
    for row in rows {
        groups
            .entry((row.family, row.d, row.collision_l2_sq))
            .or_default()
            .push(row);
    }
    for group in groups.values_mut() {
        group.sort_by_key(|row| row.rank);
        let mut prior = None::<&EuclideanWidthRow>;
        for row in group {
            if let Some(prior_row) = prior {
                if row.max_width < prior_row.max_width {
                    return invalid_config("rows", "max_width decreased as rank increased");
                }
            }
            prior = Some(row);
        }
    }
    Ok(())
}

fn validate_table_config(config: &EuclideanWidthTableConfig) -> Result<()> {
    if config.families.is_empty() {
        return invalid_config("families", "at least one family is required");
    }
    if config.ring_dims.is_empty() {
        return invalid_config("ring_dims", "at least one ring dimension is required");
    }
    if config.collision_l2_sq.is_empty() {
        return invalid_config("collision_l2_sq", "at least one collision key is required");
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
    collision_l2_sq: u128,
    table_config: &EuclideanWidthTableConfig,
    estimator_config: &EstimateConfig,
) -> Result<EuclideanWidthRow> {
    let search_cap = row_search_cap(d, table_config.search_cap)?;
    let mut probe =
        |width| estimate_width(family, d, rank, width, collision_l2_sq, estimator_config);
    let result = max_true_in_prefix(search_cap, |width| {
        probe(width).map(|cost| security_met(cost.rop, table_config.target_bits))
    })?;
    let max_cost = if result.max_value == 0 {
        None
    } else {
        Some(probe(result.max_value)?)
    };
    let next_cost = result.next_value.map(&mut probe).transpose()?;
    Ok(EuclideanWidthRow {
        family,
        d,
        rank,
        collision_l2_sq,
        max_width: result.max_value,
        target_bits: table_config.target_bits,
        search_cap,
        hit_cap: result.hit_cap,
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
    collision_l2_sq: u128,
    config: &EstimateConfig,
) -> Result<LatticeCost> {
    let params = scalar_sis_from_ring_euclidean(family, d, rank, width, collision_l2_sq)?;
    match estimate(&params, config) {
        Ok(cost) => Ok(cost),
        Err(EstimatorError::InvalidParameter {
            field: "length_bound",
            reason,
        }) if reason.contains("SIS trivially easy") => Ok(trivially_easy_cost()),
        Err(error) => Err(error),
    }
}

fn trivially_easy_cost() -> LatticeCost {
    LatticeCost {
        rop: CostValue::Finite(LogCost::new(f64::NEG_INFINITY)),
        red: None,
        sieve: None,
        delta: None,
        beta: None,
        eta: None,
        zeta: None,
        d: 0,
        prob: None,
        repetitions: None,
        tag: EstimateTag::empty(),
    }
}

fn security_met(rop: CostValue, target_bits: f64) -> bool {
    match rop {
        CostValue::Infinity => true,
        CostValue::Finite(cost) => cost.log2 >= target_bits,
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

fn optional_u32_text(value: Option<u32>) -> String {
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
    fn derived_collision_keys_match_python_contract() {
        assert_eq!(coeff_linf_bucket_sq(15), 225);
        assert_eq!(coeff_linf_bucket_sq(31), 961);
        assert!(derived_l2_collision_keys().contains(&(32 * 15 * 15)));
        assert!(l2_table_collision_keys().contains(&(1u128 << 84)));
    }

    #[test]
    fn smoke_generates_euclidean_row() {
        let config = EuclideanWidthTableConfig {
            families: vec![AkitaModulusFamily::Q32],
            ring_dims: vec![32],
            collision_l2_sq: vec![128],
            max_rank: 1,
            search_cap: Some(8),
            ..EuclideanWidthTableConfig::default()
        };
        let rows = generate_euclidean_width_rows(&config).unwrap();
        assert_eq!(rows.len(), 1);
        validate_euclidean_width_rows(&rows).unwrap();
    }

    #[test]
    fn production_config_detection_rejects_partial_tables() {
        assert!(is_full_euclidean_width_table_config(
            &EuclideanWidthTableConfig::default()
        ));
        assert!(!is_full_euclidean_width_table_config(
            &EuclideanWidthTableConfig {
                families: vec![AkitaModulusFamily::Q32],
                ..EuclideanWidthTableConfig::default()
            }
        ));
        assert!(!is_full_euclidean_width_table_config(
            &EuclideanWidthTableConfig {
                target_bits: 138.0,
                ..EuclideanWidthTableConfig::default()
            }
        ));
        assert!(!is_full_euclidean_width_table_config(
            &EuclideanWidthTableConfig {
                search_cap: Some(DEFAULT_SEARCH_CAP),
                ..EuclideanWidthTableConfig::default()
            }
        ));
    }
}
