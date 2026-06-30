//! Beta and zeta search for infinity-norm SIS estimates.

use std::collections::HashSet;

use crate::{
    config::{EstimateConfig, OptimizerConfig, SearchMode},
    cost::{CostValue, LatticeCost},
    error::{EstimatorError, Result},
    lattice::cost_infinity_fixed,
    math::{log2_biguint, log2_positive},
    params::{Bound, SisParameters},
    reduction::delta::{delta, BETA_SEARCH_MAX},
};
use num_traits::{One, ToPrimitive};

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
    zeta: u32,
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
    match zeta_mode {
        SearchMode::PythonLocalMinimum => {
            let m = explicit_m(params)?;
            let best = local_minimum(0, m, 1, |zeta| {
                cost_zeta_with_mode(zeta, beta_mode, params, config)
            })?;
            let zero = cost_zeta_with_mode(0, beta_mode, params, config)?;
            Ok(match best {
                Some(best) if cost_lt(&zero, &best) => zero,
                Some(best) => best,
                None => zero,
            })
        }
        SearchMode::Exhaustive => {
            let m = explicit_m(params)?;
            best_in_range(0, m, |zeta| {
                cost_zeta_with_mode(zeta, beta_mode, params, config)
            })?
            .ok_or_else(|| EstimatorError::InvalidParameter {
                field: "m",
                reason: "zeta search range is empty".to_string(),
            })
        }
        SearchMode::ExhaustiveParallel => Err(EstimatorError::Unsupported {
            feature: "parallel zeta search",
        }),
        SearchMode::ProvenPruned => Err(EstimatorError::Unsupported {
            feature: "proven-pruned zeta search",
        }),
    }
}

fn cost_zeta_with_mode(
    zeta: u32,
    beta_mode: SearchMode,
    params: &SisParameters,
    config: &EstimateConfig,
) -> Result<LatticeCost> {
    match beta_mode {
        SearchMode::PythonLocalMinimum => {
            let stop = beta_search_stop(params, config)?;
            let rough = local_minimum(MIN_BETA, stop, 2, |beta| {
                cost_infinity_fixed(beta, params, zeta, config)
            })?;
            match rough {
                Some(rough) => {
                    let beta = rough.beta.ok_or(EstimatorError::InvalidParameter {
                        field: "beta",
                        reason: "local-minimum beta search returned a cost without beta"
                            .to_string(),
                    })?;
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
            best_in_range(MIN_BETA, stop, |beta| {
                cost_infinity_fixed(beta, params, zeta, config)
            })?
            .ok_or_else(|| EstimatorError::InvalidParameter {
                field: "beta",
                reason: "beta search range is empty".to_string(),
            })
        }
        SearchMode::ExhaustiveParallel => Err(EstimatorError::Unsupported {
            feature: "parallel beta search",
        }),
        SearchMode::ProvenPruned => Err(EstimatorError::Unsupported {
            feature: "proven-pruned beta search",
        }),
    }
}

fn local_minimum<F>(start: u32, stop: u32, precision: u32, mut f: F) -> Result<Option<LatticeCost>>
where
    F: FnMut(u32) -> Result<LatticeCost>,
{
    if stop < start || precision == 0 {
        return Ok(None);
    }

    let mut search_start = ceil_div(start, precision);
    let mut search_stop = stop / precision;
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
    let mut best = None;
    let mut seen = HashSet::new();

    while let Some(x) = next_search_x(next_x, &seen, initial_low, initial_high) {
        next_x = None;
        let candidate = f(x.saturating_mul(precision))?;
        seen.insert(x);

        let is_better = match &best {
            None => true,
            Some(current) => cost_leq(&candidate, current),
        };
        if best_x.is_none() {
            best_x = Some(x);
            best = Some(candidate.clone());
        }
        if is_better {
            best_x = Some(x);
            best = Some(candidate);
            if direction.unsigned_abs() != 1 {
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
    next_x: Option<u32>,
    seen: &HashSet<u32>,
    initial_low: u32,
    initial_high: u32,
) -> Option<u32> {
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
    let mut best = None;
    for value in start..stop {
        let candidate = f(value)?;
        if best
            .as_ref()
            .is_none_or(|current| cost_leq(&candidate, current))
        {
            best = Some(candidate);
        }
    }
    Ok(best)
}

fn explicit_m(params: &SisParameters) -> Result<u32> {
    params.m.ok_or(EstimatorError::InvalidParameter {
        field: "m",
        reason: "optimizer requires an explicit column count m".to_string(),
    })
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
        beta_from_delta(target_delta).unwrap_or(d)
    } else {
        d
    };
    Ok(beta.min(d))
}

fn euclidean_default_dimension(params: &SisParameters, m: u32) -> Result<u32> {
    let length_bound = euclidean_baseline_length_bound(&params.length_bound)?;
    let log_bound = log2_positive(length_bound);
    if !log_bound.is_finite() || log_bound == 0.0 {
        return Ok(m);
    }

    let log_q = log2_biguint(&params.q);
    let log_delta = log_bound.powi(2) / (4.0 * params.n as f64 * log_q);
    let d = ((params.n as f64 * log_q / log_delta).sqrt().floor() as u32).max(1);
    Ok(d.min(m))
}

fn euclidean_target_delta(params: &SisParameters, d: u32, length_bound: f64) -> Result<f64> {
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

fn beta_from_delta(target_delta: f64) -> Option<u32> {
    if delta(MIN_BETA) < target_delta {
        return Some(MIN_BETA);
    }
    if target_delta < delta(BETA_SEARCH_MAX) {
        return None;
    }

    let mut low = MIN_BETA;
    let mut high = BETA_SEARCH_MAX;
    while low < high {
        let mid = low + (high - low) / 2;
        if delta(mid) <= target_delta {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    Some(low)
}

fn ceil_div(numerator: u32, denominator: u32) -> u32 {
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
        CostValue::Finite(cost) => cost.log2,
        CostValue::Infinity => f64::INFINITY,
    }
}

fn sage_sanity_check(mut cost: LatticeCost) -> LatticeCost {
    if cost_order(cost.rop) > SAGE_SANITY_MAX_LOG2 {
        cost.rop = CostValue::Infinity;
    }
    cost
}
