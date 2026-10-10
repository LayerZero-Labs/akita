//! Tie the checked-in production SIS tables to their audit certificates.
//!
//! Verification is table-only: `akita-params` reads `q*.rs` and never runs the
//! estimator. The generator writes those tables, `policy_audit.csv`, and the
//! digest over both in one pass, so the digest test accepts any consistent
//! re-paste. These tests check that the audit covers exactly the canonical
//! origins, that every certificate sits on the correct side of the policy
//! target, that the runtime widths are the projection of the audit rows, and
//! that the current estimator still reproduces the boundary certificates at
//! production geometry.

use akita_sis_estimator::width_table::{
    infinity_width_work_items, runtime_width_rows, validate_infinity_width_rows,
    InfinityWidthCertificate, InfinityWidthProfile, InfinityWidthRow, InfinityWidthTableConfig,
    InfinityWidthWorkItem, DEFAULT_MAX_RANK, DEFAULT_SEARCH_CAP,
};
use akita_sis_estimator::{
    estimate, scalar_sis_from_ring_wide, AkitaModulusProfileId, CostValue, EstimateConfig,
};
use num_bigint::BigUint;
use std::collections::{BTreeMap, BTreeSet};

const POLICY_AUDIT_CSV: &str =
    include_str!("../../akita-params/src/sis/generated_sis_table/policy_audit.csv");

const RUNTIME_TABLES: [(AkitaModulusProfileId, &str); 3] = [
    (
        AkitaModulusProfileId::Q32Offset99,
        include_str!("../../akita-params/src/sis/generated_sis_table/q32.rs"),
    ),
    (
        AkitaModulusProfileId::Q64Offset59,
        include_str!("../../akita-params/src/sis/generated_sis_table/q64.rs"),
    ),
    (
        AkitaModulusProfileId::Q128OffsetA7F7,
        include_str!("../../akita-params/src/sis/generated_sis_table/q128.rs"),
    ),
];

/// Recorded costs are printed with twelve decimals.
const COST_LOG2_TOLERANCE: f64 = 1e-9;

type RuntimeKey = (AkitaModulusProfileId, u32, u64);

fn policy_audit_rows() -> Vec<InfinityWidthRow> {
    let mut lines = POLICY_AUDIT_CSV.lines();
    assert_eq!(lines.next(), Some(InfinityWidthRow::csv_header()));
    lines
        .map(|line| {
            InfinityWidthRow::from_csv_record(line)
                .unwrap_or_else(|error| panic!("malformed audit row {line}: {error}"))
        })
        .collect()
}

fn checked_in_runtime_widths() -> BTreeMap<RuntimeKey, Vec<u64>> {
    let mut widths = BTreeMap::new();
    for (profile, source) in RUNTIME_TABLES {
        for line in source.lines() {
            let Some(arm) = line.trim().strip_prefix('(') else {
                continue;
            };
            let (key, values) = arm
                .split_once(") => Some(&[")
                .unwrap_or_else(|| panic!("unexpected generated table line: {line}"));
            let values = values
                .strip_suffix("]),")
                .unwrap_or_else(|| panic!("unexpected generated table line: {line}"));
            let (d, bound) = key
                .split_once(", ")
                .unwrap_or_else(|| panic!("unexpected generated table key: {key}"));
            let previous = widths.insert(
                (
                    profile,
                    d.parse().expect("generated d"),
                    bound.parse().expect("generated bound"),
                ),
                values
                    .split(", ")
                    .map(|width| width.parse().expect("generated width"))
                    .collect(),
            );
            assert!(
                previous.is_none(),
                "duplicate generated key ({key}) in {}",
                profile.label()
            );
        }
    }
    widths
}

fn estimate_boundary(
    row: &InfinityWidthRow,
    width: u64,
    config: &EstimateConfig,
) -> InfinityWidthCertificate {
    let params = scalar_sis_from_ring_wide(
        row.modulus_profile,
        row.d,
        row.rank,
        width,
        row.coeff_linf_bound,
    )
    .expect("audit geometry is a valid SIS instance");
    estimate(&params, config)
        .expect("audit geometry is estimable")
        .into()
}

fn assert_same_boundary(
    row: &InfinityWidthRow,
    width: u64,
    recorded: &InfinityWidthCertificate,
    recomputed: &InfinityWidthCertificate,
) {
    let same_cost = match (recorded.rop, recomputed.rop) {
        (CostValue::Finite(left), CostValue::Finite(right))
        | (CostValue::ProvenAboveTarget(left), CostValue::ProvenAboveTarget(right)) => {
            (left.log2 - right.log2).abs() <= COST_LOG2_TOLERANCE
        }
        _ => false,
    };
    assert!(
        same_cost && recorded.beta == recomputed.beta && recorded.zeta == recomputed.zeta,
        "{} d={} rank={} bound={} width={width}: recorded (rop {:?}, beta {:?}, zeta {:?}), \
         estimator now gives (rop {:?}, beta {:?}, zeta {:?})",
        row.modulus_profile.label(),
        row.d,
        row.rank,
        row.coeff_linf_bound,
        recorded.rop,
        recorded.beta,
        recorded.zeta,
        recomputed.rop,
        recomputed.beta,
        recomputed.zeta,
    );
}

#[test]
fn policy_audit_covers_exactly_the_canonical_origins() {
    let config = InfinityWidthTableConfig::default();
    let expected = infinity_width_work_items(&config).expect("canonical work items");
    let mut recorded = policy_audit_rows()
        .iter()
        .map(|row| {
            assert_eq!(row.policy, config.policy);
            assert_eq!(row.profile, InfinityWidthProfile::LocalMinimum);
            assert_eq!(row.search_cap, DEFAULT_SEARCH_CAP);
            InfinityWidthWorkItem {
                modulus_profile: row.modulus_profile,
                d: row.d,
                rank: row.rank,
                coeff_linf_bound: row.coeff_linf_bound,
            }
        })
        .collect::<Vec<_>>();
    recorded.sort_unstable();
    assert_eq!(recorded, expected);
}

#[test]
fn policy_audit_certificates_meet_the_policy_target() {
    validate_infinity_width_rows(&policy_audit_rows()).expect("checked-in audit certificates");
}

#[test]
fn runtime_tables_are_the_projection_of_the_audit_rows() {
    let projected = runtime_width_rows(&policy_audit_rows(), DEFAULT_MAX_RANK)
        .expect("audit rows project to runtime rows")
        .into_iter()
        .map(|row| {
            (
                (row.modulus_profile, row.d, row.coeff_linf_bound),
                row.widths,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let checked_in = checked_in_runtime_widths();
    for (key, widths) in &projected {
        assert_eq!(
            checked_in.get(key),
            Some(widths),
            "runtime widths for {} d={} bound={} are not the audit projection",
            key.0.label(),
            key.1,
            key.2,
        );
    }
    for key in checked_in.keys() {
        assert!(
            projected.contains_key(key),
            "runtime table has {} d={} bound={} with no audit rows",
            key.0.label(),
            key.1,
            key.2,
        );
    }
}

/// Width of the rejected successor the generator certified for `row`.
fn rejected_successor_width(row: &InfinityWidthRow) -> u64 {
    if row.max_width == 0 {
        u64::from(row.rank) + 1
    } else {
        row.max_width + 1
    }
}

/// Whether the successor sits in the large-box probability regime
/// `sqrt(m) * B > q`, which the small-geometry goldens never reach.
fn successor_in_large_box_regime(row: &InfinityWidthRow) -> bool {
    let scalar_m = BigUint::from(rejected_successor_width(row)) * row.d;
    let bound = BigUint::from(row.coeff_linf_bound);
    let modulus = row.modulus_profile.modulus();
    scalar_m * &bound * &bound > &modulus * &modulus
}

/// Deterministic production-geometry sample of exact boundaries.
///
/// Per `(profile, d)`: the largest scalar `n`, the largest coefficient bound,
/// and the widest accepted cutoff. Per profile: the smallest and largest
/// large-box successors and the largest rank with no secure width.
fn boundary_samples(rows: &[InfinityWidthRow]) -> Vec<&InfinityWidthRow> {
    let exact = rows.iter().filter(|row| !row.hit_cap).collect::<Vec<_>>();
    let mut groups = BTreeMap::<(AkitaModulusProfileId, u32), Vec<&InfinityWidthRow>>::new();
    for &row in &exact {
        groups
            .entry((row.modulus_profile, row.d))
            .or_default()
            .push(row);
    }
    let mut selected = BTreeSet::new();
    let mut select = |row: Option<&&InfinityWidthRow>| {
        if let Some(row) = row {
            selected.insert((row.modulus_profile, row.d, row.rank, row.coeff_linf_bound));
        }
    };
    for group in groups.values() {
        select(
            group
                .iter()
                .max_by_key(|row| (row.rank, row.coeff_linf_bound)),
        );
        select(
            group
                .iter()
                .max_by_key(|row| (row.coeff_linf_bound, row.rank)),
        );
        select(group.iter().max_by_key(|row| row.max_width));
    }
    for profile in [
        AkitaModulusProfileId::Q32Offset99,
        AkitaModulusProfileId::Q64Offset59,
        AkitaModulusProfileId::Q128OffsetA7F7,
    ] {
        let scalar_n = |row: &&&InfinityWidthRow| u64::from(row.d) * u64::from(row.rank);
        let large_box = exact
            .iter()
            .filter(|row| row.modulus_profile == profile && successor_in_large_box_regime(row))
            .collect::<Vec<_>>();
        select(large_box.iter().copied().min_by_key(scalar_n));
        select(large_box.iter().copied().max_by_key(scalar_n));
        select(
            exact
                .iter()
                .filter(|row| row.modulus_profile == profile && row.max_width == 0)
                .max_by_key(scalar_n),
        );
    }
    exact
        .into_iter()
        .filter(|row| {
            selected.contains(&(row.modulus_profile, row.d, row.rank, row.coeff_linf_bound))
        })
        .collect()
}

#[test]
fn production_boundaries_reproduce_their_certificates() {
    let config = EstimateConfig::akita_infinity_table();
    let rows = policy_audit_rows();
    let samples = boundary_samples(&rows);
    assert!(samples.iter().any(|row| successor_in_large_box_regime(row)));
    assert!(samples.iter().any(|row| row.max_width == 0));
    for row in samples {
        if let Some(accepted) = &row.max_costs {
            assert_same_boundary(
                row,
                row.max_width,
                &accepted.adps16_quantum,
                &estimate_boundary(row, row.max_width, &config),
            );
        }
        let width = rejected_successor_width(row);
        let rejected = row
            .next_costs
            .as_ref()
            .expect("exact cutoff records its rejected successor");
        assert_same_boundary(
            row,
            width,
            &rejected.adps16_quantum,
            &estimate_boundary(row, width, &config),
        );
    }
}
