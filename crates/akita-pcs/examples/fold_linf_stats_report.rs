//! Aggregate fold-linf observations across repeated prove runs.

use akita_prover::FoldGrindObservation;
use akita_types::sis::{
    fold_witness_verifier_linf_bound, snap_min_tstar_retain_floor, FoldWitnessLinfCapPolicy,
};

#[derive(Clone, Debug)]
pub struct FoldLinfLevelSample {
    pub iteration: u32,
    pub observation: FoldGrindObservation,
}

#[derive(Clone, Debug)]
pub struct SnapCandidate {
    pub delta_fold: usize,
    pub verifier_cap: u128,
    pub tstar_reduction_fraction: f64,
    pub fits_observed_max: bool,
    pub fits_p90: bool,
}

#[derive(Clone, Debug)]
pub struct FoldLinfLevelAggregate {
    pub level_index: u32,
    pub r_vars: u32,
    pub num_claims: u32,
    pub policy: FoldWitnessLinfCapPolicy,
    pub log_basis: u32,
    pub samples: usize,
    pub beta_inf: u128,
    pub t_star: Option<u128>,
    pub honest_cap: u128,
    pub delta_fold: usize,
    pub verifier_linf_bound: u128,
    pub observed_min: u32,
    pub observed_max: u32,
    pub observed_p90: u32,
    pub observed_sum: u128,
    pub grind_probe_min: u32,
    pub grind_probe_max: u32,
    pub grind_probe_sum: u64,
}

fn percentile(sorted: &[u32], p: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn tstar_reduction_fraction(t_star: u128, cap: u128) -> f64 {
    if t_star == 0 {
        0.0
    } else {
        (t_star.saturating_sub(cap)) as f64 / t_star as f64
    }
}

fn snap_retain_rational(max_reduction_fraction: f64) -> (u128, u128) {
    let retain = (1.0 - max_reduction_fraction).clamp(0.0, 1.0);
    const DEN: u128 = 10_000;
    let num = (retain * DEN as f64).round() as u128;
    (num.max(1), DEN)
}

/// Tightest downward digit snap with `z_ver(δ') >= floor` and observed quantiles inside cap.
pub fn best_snap_below_tstar(
    log_basis: u32,
    current_delta: usize,
    t_star: u128,
    observed_max: u32,
    observed_p90: u32,
    max_tstar_reduction_fraction: f64,
) -> Option<SnapCandidate> {
    if current_delta <= 1 || t_star == 0 {
        return None;
    }
    let (retain_num, retain_den) = snap_retain_rational(max_tstar_reduction_fraction);
    let floor = snap_min_tstar_retain_floor(t_star, retain_num, retain_den);
    for delta in (1..current_delta).rev() {
        let cap = fold_witness_verifier_linf_bound(log_basis, delta);
        if cap < floor {
            continue;
        }
        let fits_max = u128::from(observed_max) <= cap;
        let fits_p90 = u128::from(observed_p90) <= cap;
        if fits_p90 {
            return Some(SnapCandidate {
                delta_fold: delta,
                verifier_cap: cap,
                tstar_reduction_fraction: tstar_reduction_fraction(t_star, cap),
                fits_observed_max: fits_max,
                fits_p90,
            });
        }
    }
    None
}

pub fn aggregate_fold_linf_samples(samples: &[FoldLinfLevelSample]) -> Vec<FoldLinfLevelAggregate> {
    use std::collections::BTreeMap;

    let mut by_level: BTreeMap<u32, Vec<&FoldGrindObservation>> = BTreeMap::new();
    for sample in samples {
        by_level
            .entry(sample.observation.level_index)
            .or_default()
            .push(&sample.observation);
    }

    by_level
        .into_iter()
        .map(|(level_index, obs)| {
            let first = obs[0];
            let mut observed_values: Vec<u32> = obs.iter().map(|o| o.observed_linf).collect();
            observed_values.sort_unstable();
            let observed_p90 = percentile(&observed_values, 0.9);

            FoldLinfLevelAggregate {
                level_index,
                r_vars: first.r_vars,
                num_claims: first.num_claims,
                policy: first.policy,
                log_basis: first.log_basis,
                samples: obs.len(),
                beta_inf: first.beta_inf,
                t_star: first.t_star,
                honest_cap: first.honest_cap,
                delta_fold: first.delta_fold,
                verifier_linf_bound: first.verifier_linf_bound,
                observed_min: *observed_values.first().unwrap_or(&0),
                observed_max: *observed_values.last().unwrap_or(&0),
                observed_p90,
                observed_sum: obs.iter().map(|o| u128::from(o.observed_linf)).sum(),
                grind_probe_min: obs.iter().map(|o| o.grind_probe_count).min().unwrap_or(0),
                grind_probe_max: obs.iter().map(|o| o.grind_probe_count).max().unwrap_or(0),
                grind_probe_sum: obs.iter().map(|o| u64::from(o.grind_probe_count)).sum(),
            }
        })
        .collect()
}

pub fn print_fold_linf_aggregate_report(
    label: &str,
    iterations: u32,
    samples: &[FoldLinfLevelSample],
    max_tstar_reduction_fraction: f64,
) {
    let aggregates = aggregate_fold_linf_samples(samples);
    let retain_pct = (1.0 - max_tstar_reduction_fraction) * 100.0;
    let drop_pct = max_tstar_reduction_fraction * 100.0;

    eprintln!("=== fold-linf stats: {label} ({iterations} prove iterations) ===");
    eprintln!(
        "snap policy: choose tightest δ' < δ with p90(obs) ≤ z_ver(δ') and z_ver(δ') ≥ {retain_pct:.0}%·t* (≤{drop_pct:.0}% reduction vs t*)"
    );
    eprintln!(
        "summary: level | r | β | t* | cap | δ | z_ver | obs[max p90 mean] | obs/β obs/t* | grind mean | snap δ' cap | Δvs t* | fits max?"
    );

    for agg in &aggregates {
        let snap = match (agg.policy, agg.t_star) {
            (FoldWitnessLinfCapPolicy::TailBoundWithGrind, Some(t_star)) => best_snap_below_tstar(
                agg.log_basis,
                agg.delta_fold,
                t_star,
                agg.observed_max,
                agg.observed_p90,
                max_tstar_reduction_fraction,
            ),
            _ => None,
        };

        let snap_line = snap.as_ref().map_or_else(
            || {
                if agg.t_star.is_some() {
                    "none".to_string()
                } else {
                    "n/a (no tail t*)".to_string()
                }
            },
            |s| {
                format!(
                    "δ'={} cap={} Δt*={:.1}% max_ok={}",
                    s.delta_fold,
                    s.verifier_cap,
                    s.tstar_reduction_fraction * 100.0,
                    s.fits_observed_max
                )
            },
        );

        let t_star_display = agg
            .t_star
            .map(|t| t.to_string())
            .unwrap_or_else(|| "n/a".to_string());
        let obs_over_tstar = agg.t_star.map_or(0.0, |t_star| {
            if t_star > 0 {
                agg.observed_max as f64 / t_star as f64
            } else {
                0.0
            }
        });

        eprintln!(
            "L{} | r={} {:?} | β={} t*={} cap={} δ={} z_ver={} | obs max={} p90={} mean={:.1} | obs/β={:.4} obs/t*={:.4} | grind mean={:.2} | {snap_line}",
            agg.level_index,
            agg.r_vars,
            agg.policy,
            agg.beta_inf,
            t_star_display,
            agg.honest_cap,
            agg.delta_fold,
            agg.verifier_linf_bound,
            agg.observed_max,
            agg.observed_p90,
            agg.observed_sum as f64 / agg.samples as f64,
            agg.observed_max as f64 / agg.beta_inf as f64,
            obs_over_tstar,
            agg.grind_probe_sum as f64 / agg.samples as f64,
        );
    }

    eprintln!("--- per-iteration raw rows ---");
    for sample in samples {
        let o = &sample.observation;
        let t_star_display = o
            .t_star
            .map(|t| t.to_string())
            .unwrap_or_else(|| "n/a".to_string());
        let obs_over_tstar = o.t_star.map_or(0.0, |t_star| {
            if t_star > 0 {
                o.observed_linf as f64 / t_star as f64
            } else {
                0.0
            }
        });
        eprintln!(
            "iter={} L{} r={} obs={} β={} t*={} cap={} δ={} obs/β={:.4} obs/t*={:.4} grind={} nonce={}",
            sample.iteration,
            o.level_index,
            o.r_vars,
            o.observed_linf,
            o.beta_inf,
            t_star_display,
            o.honest_cap,
            o.delta_fold,
            o.observed_linf as f64 / o.beta_inf as f64,
            obs_over_tstar,
            o.grind_probe_count,
            o.grind_nonce,
        );
    }
}
