//! Beta and zeta search for infinity-norm SIS estimates.

use std::collections::HashSet;

use crate::{
    config::{EstimateConfig, OptimizerConfig, ReductionCostModel, SearchMode, ShapeModel},
    cost::{CostValue, LatticeCost},
    error::{EstimatorError, Result},
    lattice::cost_infinity_fixed,
    math::{log2_biguint, log2_positive},
    params::{Bound, SisParameters},
    reduction::{
        adps16_log2_cost,
        delta::{beta as beta_from_delta, BETA_SEARCH_MAX},
    },
    simulator::lgsa_stable_dimension,
};
use num_traits::{One, ToPrimitive};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const MIN_BETA: u32 = 40;
const SAGE_SANITY_MAX_LOG2: f64 = 10_000.0;

/// Estimate the best infinity-norm attack under the configured optimizer.
pub fn estimate_infinity(params: &SisParameters, config: &EstimateConfig) -> Result<LatticeCost> {
    let cost = match config.optimizer {
        OptimizerConfig::Fixed { beta, zeta } => cost_infinity_fixed(beta, params, zeta, config),
        OptimizerConfig::OptimizeBeta { zeta, beta } => {
            cost_zeta_with_mode(zeta, beta, params, config)
        }
        OptimizerConfig::OptimizeZeta { beta, zeta } => {
            cost_zeta_search(beta, zeta, params, config)
        }
    }?;
    Ok(sage_sanity_check(cost))
}

/// Estimate the best beta for one fixed zeta.
pub fn cost_zeta_infinity(
    zeta: u64,
    params: &SisParameters,
    config: &EstimateConfig,
) -> Result<LatticeCost> {
    let beta_mode = match config.optimizer {
        OptimizerConfig::Fixed {
            beta,
            zeta: fixed_zeta,
        } => {
            return if fixed_zeta == zeta {
                cost_infinity_fixed(beta, params, zeta, config)
            } else {
                Err(EstimatorError::InvalidConfig {
                    field: "optimizer.zeta",
                    reason: "fixed optimizer zeta does not match cost_zeta argument".to_string(),
                })
            };
        }
        OptimizerConfig::OptimizeBeta { beta, .. } | OptimizerConfig::OptimizeZeta { beta, .. } => {
            beta
        }
    };
    cost_zeta_with_mode(zeta, beta_mode, params, config)
}

fn cost_zeta_search(
    beta_mode: SearchMode,
    zeta_mode: SearchMode,
    params: &SisParameters,
    config: &EstimateConfig,
) -> Result<LatticeCost> {
    let zeta_stop = zeta_search_stop(params, config)?;
    match zeta_mode {
        SearchMode::PythonLocalMinimum => {
            let best = local_minimum(0, zeta_stop, 1, |zeta| {
                cost_zeta_with_mode(zeta, beta_mode, params, config)
            })?;
            let zero = cost_zeta_with_mode(0, beta_mode, params, config)?;
            Ok(match best {
                Some(best) if cost_lt(&zero, &best) => zero,
                Some(best) => best,
                None => zero,
            })
        }
        SearchMode::Exhaustive => best_in_range(0, exhaustive_zeta_stop(zeta_stop)?, |zeta| {
            cost_zeta_with_mode(u64::from(zeta), beta_mode, params, config)
        })?
        .ok_or_else(empty_range_error("zeta")),
        SearchMode::ExhaustiveParallel => {
            let inner_beta_mode = match beta_mode {
                SearchMode::ExhaustiveParallel => SearchMode::Exhaustive,
                mode => mode,
            };
            best_in_range_parallel(0, exhaustive_zeta_stop(zeta_stop)?, |zeta| {
                cost_zeta_with_mode(u64::from(zeta), inner_beta_mode, params, config)
            })?
            .ok_or_else(empty_range_error("zeta"))
        }
        SearchMode::ProvenPruned => proven_pruned_zeta_search(beta_mode, params, config),
    }
}

fn proven_pruned_zeta_search(
    beta_mode: SearchMode,
    params: &SisParameters,
    config: &EstimateConfig,
) -> Result<LatticeCost> {
    let adps16_mode = match config.red_cost_model {
        ReductionCostModel::Adps16 { mode } => mode,
        _ => {
            return Err(EstimatorError::Unsupported {
                feature: "proven-pruned zeta search outside exhaustive ADPS16/LGSA",
            })
        }
    };
    if config.red_shape_model != ShapeModel::Lgsa
        || !matches!(
            beta_mode,
            SearchMode::Exhaustive | SearchMode::ExhaustiveParallel
        )
    {
        return Err(EstimatorError::Unsupported {
            feature: "proven-pruned zeta search outside exhaustive ADPS16/LGSA",
        });
    }

    let m = explicit_m(params)?;
    let lattice_dimension = config.lattice_dimension.unwrap_or(m).min(m);
    let beta_stop = beta_search_stop(params, config)?;
    let beta_start = MIN_BETA.min(beta_stop);
    let mut best = None::<LatticeCost>;

    for beta in beta_start..beta_stop.max(beta_start.saturating_add(1)) {
        // ADPS16's fixed-beta reduction price is a lower bound on the total
        // attack cost. Once it exceeds the best complete candidate, every
        // larger beta is provably unable to win.
        if best
            .as_ref()
            .is_some_and(|current| adps16_log2_cost(beta, adps16_mode) > cost_order(current.rop))
        {
            break;
        }
        // For fixed beta, LGSA has two dimension regimes. Before the stable
        // dimension its q-ary/GSA prefix grows to the complete profile; after
        // it, added coordinates are unit vectors. In the first regime the
        // modeled attack cost is minimized at the complete-profile endpoint.
        // In the stable regime the Dilithium branch is constant, while the
        // Matzov branch has a single interior maximum in attack cost, so its
        // minimum is at one of the two endpoints. Therefore the stable
        // transition and its immediate predecessor, together with zeta=0 and
        // zeta=1, cover the complete zeta domain for each beta. The predecessor
        // points retain the q-vector transition used by lattice-estimator.
        let stable_dimension = lgsa_stable_dimension(u64::from(params.n), &params.q, beta)?
            .clamp(u64::from(beta), lattice_dimension);
        let before_stable = stable_dimension.saturating_sub(1).max(u64::from(beta));
        let before_full = lattice_dimension.saturating_sub(1).max(u64::from(beta));
        for effective_dimension in [
            before_stable,
            stable_dimension,
            before_full,
            lattice_dimension,
        ] {
            let zeta = lattice_dimension - effective_dimension;
            let candidate = cost_infinity_fixed(beta, params, zeta, config)?;
            if best
                .as_ref()
                .is_none_or(|current| cost_leq(&candidate, current))
            {
                best = Some(candidate);
            }
        }
    }

    best.ok_or_else(empty_range_error("zeta"))
}

fn cost_zeta_with_mode(
    zeta: u64,
    beta_mode: SearchMode,
    params: &SisParameters,
    config: &EstimateConfig,
) -> Result<LatticeCost> {
    match beta_mode {
        SearchMode::PythonLocalMinimum => {
            let stop = beta_search_stop(params, config)?;
            let rough = local_minimum(u64::from(MIN_BETA), u64::from(stop), 2, |beta| {
                let beta = u32::try_from(beta).map_err(|_| EstimatorError::InvalidParameter {
                    field: "beta",
                    reason: "beta search candidate exceeded u32".to_string(),
                })?;
                cost_infinity_fixed(beta, params, zeta, config)
            })?;
            match rough {
                Some(rough) => {
                    let beta = rough.beta.ok_or(EstimatorError::InvalidParameter {
                        field: "beta",
                        reason: "local-minimum beta search returned a cost without beta"
                            .to_string(),
                    })?;
                    // Refinement matches lattice-estimator `it.neighborhood`: half-open
                    // [rough_beta - 2, min(search_stop, rough_beta + 2)), so the largest β
                    // tried is rough_beta + 1, not rough_beta + 2. Do not widen this window
                    // for parity goldens: a strictly cheaper β+1 neighbor can exist but Sage
                    // skips it when coarse search lands two below; exhaustive search may find it.
                    let start = beta.saturating_sub(2).max(MIN_BETA);
                    let stop = beta.saturating_add(2).min(stop);
                    best_in_range(start, stop, |candidate| {
                        cost_infinity_fixed(candidate, params, zeta, config)
                    })?
                    .or(Some(rough))
                    .ok_or_else(|| EstimatorError::InvalidParameter {
                        field: "beta",
                        reason: "beta search range is empty".to_string(),
                    })
                }
                None => Ok(cost_infinity_fixed(MIN_BETA, params, zeta, config)?),
            }
        }
        SearchMode::Exhaustive => {
            let stop = beta_search_stop(params, config)?;
            if stop <= MIN_BETA {
                return cost_infinity_fixed(MIN_BETA, params, zeta, config);
            }
            best_in_range(MIN_BETA, stop, |beta| {
                cost_infinity_fixed(beta, params, zeta, config)
            })?
            .ok_or_else(|| EstimatorError::InvalidParameter {
                field: "beta",
                reason: "beta search range is empty".to_string(),
            })
        }
        SearchMode::ExhaustiveParallel => {
            let stop = beta_search_stop(params, config)?;
            if stop <= MIN_BETA {
                return cost_infinity_fixed(MIN_BETA, params, zeta, config);
            }
            best_in_range_parallel(MIN_BETA, stop, |beta| {
                cost_infinity_fixed(beta, params, zeta, config)
            })?
            .ok_or_else(empty_range_error("beta"))
        }
        SearchMode::ProvenPruned => Err(EstimatorError::Unsupported {
            feature: "proven-pruned beta search",
        }),
    }
}

fn local_minimum<F>(start: u64, stop: u64, precision: u64, mut f: F) -> Result<Option<LatticeCost>>
where
    F: FnMut(u64) -> Result<LatticeCost>,
{
    if stop <= start || precision == 0 {
        return Ok(None);
    }

    let mut search_start = ceil_div(start, precision);
    let mut search_stop = ceil_div(stop, precision);
    if search_stop == 0 {
        return Ok(None);
    }
    search_stop -= 1;
    if search_stop < search_start {
        return Ok(None);
    }

    let initial_low = search_start;
    let initial_high = search_stop;
    let mut direction = -1i8;
    let mut next_x = Some(search_stop);
    let mut best_x = None;
    let mut best = None::<LatticeCost>;
    let mut seen = HashSet::new();

    while let Some(x) = next_search_x(next_x, &seen, initial_low, initial_high) {
        next_x = None;
        let candidate = f(x.saturating_mul(precision))?;
        seen.insert(x);

        let is_equal = best
            .as_ref()
            .is_some_and(|current| cost_order(candidate.rop) == cost_order(current.rop));
        let is_better = best
            .as_ref()
            .is_none_or(|current| cost_leq(&candidate, current));
        if best_x.is_none() {
            best_x = Some(x);
            best = Some(candidate.clone());
        }
        if is_better {
            best_x = Some(x);
            best = Some(candidate);
            if is_equal && direction == -1 {
                // Preserve lattice-estimator's preference for the lowest
                // zeta on an equal-cost plateau, but bisect the plateau rather
                // than walking it one coordinate at a time.
                direction = -2;
                search_stop = x;
                next_x = Some(ceil_div(search_start + search_stop, 2));
            } else if direction.unsigned_abs() != 1 {
                direction = -1;
                next_x = x.checked_sub(1);
            } else if direction == -1 {
                direction = -2;
                search_stop = x;
                next_x = Some(ceil_div(search_start + search_stop, 2));
            } else if direction == 1 {
                direction = 2;
                search_start = x;
                next_x = Some((search_start + search_stop) / 2);
            }
        } else if direction == -1 {
            direction = 1;
            next_x = x.checked_add(2);
        } else if direction == 1 {
            next_x = None;
        } else if direction == -2 {
            search_start = x;
            next_x = Some(ceil_div(search_start + search_stop, 2));
        } else if direction == 2 {
            search_stop = x;
            next_x = Some((search_start + search_stop) / 2);
        }

        if next_x == Some(x) {
            next_x = None;
        }
    }

    Ok(best)
}

fn next_search_x(
    next_x: Option<u64>,
    seen: &HashSet<u64>,
    initial_low: u64,
    initial_high: u64,
) -> Option<u64> {
    let x = next_x?;
    if !seen.contains(&x) && initial_low <= x && x <= initial_high {
        Some(x)
    } else {
        None
    }
}

fn best_in_range<F>(start: u32, stop: u32, mut f: F) -> Result<Option<LatticeCost>>
where
    F: FnMut(u32) -> Result<LatticeCost>,
{
    let mut best = None::<SearchCandidate>;
    for value in start..stop {
        best = Some(select_best(
            best,
            SearchCandidate {
                value,
                cost: f(value)?,
            },
        ));
    }
    Ok(best.map(|candidate| candidate.cost))
}

#[cfg(feature = "parallel")]
fn best_in_range_parallel<F>(start: u32, stop: u32, f: F) -> Result<Option<LatticeCost>>
where
    F: Fn(u32) -> Result<LatticeCost> + Sync,
{
    match (start..stop)
        .into_par_iter()
        .map(|value| f(value).map(|cost| SearchCandidate { value, cost }))
        .try_reduce_with(|best, candidate| Ok(select_best(Some(best), candidate)))
    {
        Some(Ok(candidate)) => Ok(Some(candidate.cost)),
        Some(Err(error)) => Err(error),
        None => Ok(None),
    }
}

#[cfg(not(feature = "parallel"))]
fn best_in_range_parallel<F>(_start: u32, _stop: u32, _f: F) -> Result<Option<LatticeCost>>
where
    F: Fn(u32) -> Result<LatticeCost>,
{
    Err(EstimatorError::Unsupported {
        feature: "parallel exhaustive search",
    })
}

#[derive(Clone, Debug)]
struct SearchCandidate {
    value: u32,
    cost: LatticeCost,
}

fn select_best(best: Option<SearchCandidate>, candidate: SearchCandidate) -> SearchCandidate {
    match best {
        Some(current) if !candidate_leq(&candidate, &current) => current,
        _ => candidate,
    }
}

fn candidate_leq(lhs: &SearchCandidate, rhs: &SearchCandidate) -> bool {
    cost_order(lhs.cost.rop) < cost_order(rhs.cost.rop)
        || (cost_order(lhs.cost.rop) == cost_order(rhs.cost.rop) && lhs.value >= rhs.value)
}

fn empty_range_error(field: &'static str) -> impl FnOnce() -> EstimatorError {
    move || EstimatorError::InvalidParameter {
        field,
        reason: format!("{field} search range is empty"),
    }
}

fn exhaustive_zeta_stop(stop: u64) -> Result<u32> {
    u32::try_from(stop).map_err(|_| EstimatorError::Unsupported {
        feature: "wide exhaustive zeta search",
    })
}

fn explicit_m(params: &SisParameters) -> Result<u64> {
    params.m.ok_or(EstimatorError::InvalidParameter {
        field: "m",
        reason: "optimizer requires an explicit column count m".to_string(),
    })
}

fn zeta_search_stop(params: &SisParameters, config: &EstimateConfig) -> Result<u64> {
    let m = explicit_m(params)?;
    let lattice_dimension = config.lattice_dimension.unwrap_or(m);
    if lattice_dimension == 0 {
        return Err(EstimatorError::InvalidParameter {
            field: "lattice_dimension",
            reason: "zeta search requires a positive lattice dimension".to_string(),
        });
    }
    Ok(lattice_dimension.min(m))
}

fn beta_search_stop(params: &SisParameters, config: &EstimateConfig) -> Result<u32> {
    euclidean_baseline_beta(params, config).map(|beta| beta.saturating_add(1))
}

fn euclidean_baseline_beta(params: &SisParameters, config: &EstimateConfig) -> Result<u32> {
    let m = explicit_m(params)?;
    let d = config
        .lattice_dimension
        .unwrap_or(euclidean_default_dimension(params, m)?);
    let length_bound = euclidean_baseline_length_bound(&params.length_bound)?;
    let target_delta = euclidean_target_delta(params, d, length_bound)?;
    let beta = if target_delta >= 1.0 {
        beta_from_delta(target_delta).map(u64::from).unwrap_or(d)
    } else {
        d
    };
    let capped = beta.min(d).min(u64::from(BETA_SEARCH_MAX));
    u32::try_from(capped).map_err(|_| EstimatorError::InvalidParameter {
        field: "beta",
        reason: "Euclidean baseline beta exceeded u32".to_string(),
    })
}

fn euclidean_default_dimension(params: &SisParameters, m: u64) -> Result<u64> {
    let length_bound = euclidean_baseline_length_bound(&params.length_bound)?;
    let log_bound = log2_positive(length_bound);
    if !log_bound.is_finite() || log_bound == 0.0 {
        return Ok(m);
    }

    let log_q = log2_biguint(&params.q);
    let log_delta = log_bound.powi(2) / (4.0 * params.n as f64 * log_q);
    let d = ((params.n as f64 * log_q / log_delta).sqrt().floor() as u64).max(1);
    Ok(d.min(m))
}

fn euclidean_target_delta(params: &SisParameters, d: u64, length_bound: f64) -> Result<f64> {
    if d <= 1 {
        return Err(EstimatorError::InvalidParameter {
            field: "d",
            reason: "Euclidean baseline dimension must be greater than 1".to_string(),
        });
    }
    let root_volume_log2 = (params.n as f64 / d as f64) * log2_biguint(&params.q);
    let log_delta = (log2_positive(length_bound) - root_volume_log2) / (d as f64 - 1.0);
    Ok(2.0_f64.powf(log_delta))
}

fn euclidean_baseline_length_bound(bound: &Bound) -> Result<f64> {
    let value = match bound {
        Bound::Integer(value) if value.is_one() => 2.0,
        Bound::Integer(value) => value.to_f64().unwrap_or(f64::INFINITY),
        Bound::Float(value) if *value == 1.0 => 2.0,
        Bound::Float(value) => *value,
        Bound::Rational {
            numerator,
            denominator,
        } => numerator.to_f64().unwrap_or(0.0) / denominator.to_f64().unwrap_or(1.0),
        Bound::SqrtInteger(value) => value.to_f64().unwrap_or(f64::INFINITY).sqrt(),
    };
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(EstimatorError::InvalidParameter {
            field: "length_bound",
            reason: "Euclidean baseline requires a finite positive length bound".to_string(),
        })
    }
}

fn ceil_div(numerator: u64, denominator: u64) -> u64 {
    numerator.div_ceil(denominator)
}

fn cost_lt(lhs: &LatticeCost, rhs: &LatticeCost) -> bool {
    cost_order(lhs.rop) < cost_order(rhs.rop)
}

fn cost_leq(lhs: &LatticeCost, rhs: &LatticeCost) -> bool {
    cost_order(lhs.rop) <= cost_order(rhs.rop)
}

fn cost_order(value: CostValue) -> f64 {
    match value {
        CostValue::Finite(cost) | CostValue::ProvenAboveTarget(cost) => cost.log2,
        CostValue::Infinity => f64::INFINITY,
    }
}

fn sage_sanity_check(mut cost: LatticeCost) -> LatticeCost {
    if let CostValue::Finite(value) = cost.rop {
        if value.log2 > SAGE_SANITY_MAX_LOG2 {
            cost.rop = CostValue::ProvenAboveTarget(value);
        }
    }
    cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::Adps16Mode,
        params::{akita_q128, akita_q32, akita_q64, SisNorm},
    };

    fn exhaustive_config(zeta: SearchMode) -> EstimateConfig {
        EstimateConfig {
            red_cost_model: ReductionCostModel::Adps16 {
                mode: Adps16Mode::Quantum,
            },
            red_shape_model: ShapeModel::Lgsa,
            optimizer: OptimizerConfig::OptimizeZeta {
                beta: SearchMode::Exhaustive,
                zeta,
            },
            ..EstimateConfig::default()
        }
    }

    fn local_minimum_test_cost(beta: u64) -> LatticeCost {
        LatticeCost {
            rop: CostValue::finite_log2(1_000.0 - beta as f64),
            red: None,
            sieve: None,
            delta: None,
            beta: Some(u32::try_from(beta).unwrap()),
            eta: None,
            zeta: None,
            d: 0,
            prob: None,
            repetitions: None,
            tag: Default::default(),
        }
    }

    #[test]
    fn local_minimum_checks_only_candidate_in_narrow_strided_range() {
        let best = local_minimum(40, 41, 2, |beta| Ok(local_minimum_test_cost(beta)))
            .unwrap()
            .unwrap();
        assert_eq!(best.beta, Some(40));
    }

    #[test]
    fn local_minimum_checks_last_strided_candidate_below_exclusive_stop() {
        let best = local_minimum(40, 45, 2, |beta| Ok(local_minimum_test_cost(beta)))
            .unwrap()
            .unwrap();
        assert_eq!(best.beta, Some(44));
    }

    #[test]
    fn proven_pruned_matches_finite_exhaustive_domains() {
        let cases = [
            (32, akita_q32(), 384, 2),
            (64, akita_q32(), 768, 127),
            (64, akita_q64(), 768, 32_767),
            (128, akita_q64(), 1_024, 1_048_575),
            (64, akita_q128(), 768, 67_108_863),
            (128, akita_q128(), 1_024, 536_870_911),
        ];
        for (n, q, m, bound) in cases {
            let params =
                SisParameters::try_new(n, q, Some(m), Bound::from_u64(bound), SisNorm::Infinity)
                    .unwrap();
            let exhaustive =
                estimate_infinity(&params, &exhaustive_config(SearchMode::Exhaustive)).unwrap();
            let pruned =
                estimate_infinity(&params, &exhaustive_config(SearchMode::ProvenPruned)).unwrap();
            assert_eq!(
                cost_order(pruned.rop),
                cost_order(exhaustive.rop),
                "n={n} m={m} bound={bound}: pruned={pruned:?}, exhaustive={exhaustive:?}"
            );
        }
    }
}
