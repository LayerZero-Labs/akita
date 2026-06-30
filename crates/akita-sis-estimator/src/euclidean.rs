//! Euclidean-norm SIS lattice cost.

use num_bigint::BigUint;
use num_traits::One;

use crate::{
    config::{EstimateConfig, ReductionCostModel},
    cost::{CostValue, EstimateTag, LatticeCost},
    error::{EstimatorError, Result},
    math::log2_biguint,
    params::SisParameters,
    reduction::{bdgl16_log2_cost, beta as beta_from_delta, delta, validate_euclidean_reduction},
};

/// Evaluate the lattice-estimator Euclidean SIS path for the configured model.
pub fn cost_euclidean(params: &SisParameters, config: &EstimateConfig) -> Result<LatticeCost> {
    validate_euclidean_reduction(config.red_cost_model)?;
    if length_bound_trivially_easy(params) {
        return Err(EstimatorError::InvalidParameter {
            field: "length_bound",
            reason: "SIS trivially easy: length_bound must be below (q - 1) / 2".to_string(),
        });
    }

    let m = params.m.ok_or(EstimatorError::InvalidParameter {
        field: "m",
        reason: "Euclidean SIS cost requires an explicit column count m".to_string(),
    })?;
    let log_q = log2_biguint(&params.q);
    let d = match config.lattice_dimension {
        Some(d) => d,
        None => opt_sis_dimension(params, m, log_q)?,
    };
    if d <= 1 {
        return Err(EstimatorError::InvalidParameter {
            field: "d",
            reason: "Euclidean lattice dimension must be at least 2".to_string(),
        });
    }

    let length_log2 = params.length_bound.log2();
    let root_volume_log2 = (params.n as f64 / d as f64) * log_q;
    let log_delta = (length_log2 - root_volume_log2) / (d as f64 - 1.0);
    let delta_value = 2.0_f64.powf(log_delta);

    let required_beta = if delta_value >= 1.0 {
        beta_from_delta(delta_value)
    } else {
        None
    };
    let reduction_possible = required_beta.is_some_and(|beta| u64::from(beta) <= d);
    let beta = required_beta
        .filter(|&beta| u64::from(beta) <= d)
        .unwrap_or_else(|| u32::try_from(d).unwrap_or(u32::MAX));

    let lower_bound_met = length_bound_exceeds_euclidean_lower_bound(params, d, log_q);
    let predicate = reduction_possible && lower_bound_met;
    let cost = if predicate {
        match config.red_cost_model {
            ReductionCostModel::Bdgl16 => CostValue::finite_log2(bdgl16_log2_cost(beta, d)),
            _ => unreachable!("validated above"),
        }
    } else {
        CostValue::Infinity
    };

    Ok(LatticeCost {
        rop: cost,
        red: Some(cost),
        sieve: None,
        delta: Some(delta(beta)),
        beta: Some(beta),
        eta: None,
        zeta: None,
        d,
        prob: None,
        repetitions: None,
        tag: params
            .tag
            .as_ref()
            .map(|value| EstimateTag::new(value.clone()))
            .unwrap_or_default(),
    })
}

fn opt_sis_dimension(params: &SisParameters, m: u64, log_q: f64) -> Result<u64> {
    let log_bound = params.length_bound.log2();
    if !log_bound.is_finite() || log_bound <= 0.0 {
        return Err(EstimatorError::InvalidParameter {
            field: "length_bound",
            reason: "Euclidean dimension optimization requires length_bound > 1".to_string(),
        });
    }
    let log_delta = log_bound.powi(2) / (4.0 * params.n as f64 * log_q);
    let d = ((params.n as f64 * log_q / log_delta).sqrt().floor() as u64).max(1);
    Ok(d.min(m))
}

fn length_bound_trivially_easy(params: &SisParameters) -> bool {
    let half_q_log2 = log2_biguint(&(params.q.clone() - BigUint::one())) - 1.0;
    params.length_bound.log2() >= half_q_log2
}

fn length_bound_exceeds_euclidean_lower_bound(params: &SisParameters, d: u64, log_q: f64) -> bool {
    let ln_q = log_q * std::f64::consts::LN_2;
    let log_a_sq = (params.n as f64 * ln_q).ln();
    let log_b_sq = (d as f64).ln() + 2.0 * (params.n as f64 / d as f64) * ln_q;
    params.length_bound.ln() > 0.5 * log_a_sq.min(log_b_sq)
}

#[cfg(test)]
mod tests {
    use crate::{
        params::{akita_q32, Bound, SisNorm},
        EstimateConfig,
    };

    use super::*;

    #[test]
    fn euclidean_doctest_large_q_returns_infinite_cost_with_beta_40() {
        let params = SisParameters::try_new(
            512,
            BigUint::from(1u32) << 200usize,
            Some(1024),
            Bound::from_u64(1000),
            SisNorm::Euclidean,
        )
        .unwrap();
        let cost = cost_euclidean(
            &params,
            &EstimateConfig {
                red_cost_model: ReductionCostModel::Bdgl16,
                lattice_dimension: Some(40),
                ..EstimateConfig::default()
            },
        )
        .unwrap();
        assert_eq!(cost.rop, CostValue::Infinity);
        assert_eq!(cost.beta, Some(40));
    }

    #[test]
    fn euclidean_doctest_small_q_returns_infinite_cost() {
        let params = SisParameters::try_new(
            32,
            akita_q32(),
            Some(128),
            Bound::from_u64(256),
            SisNorm::Euclidean,
        )
        .unwrap();
        let cost = cost_euclidean(
            &params,
            &EstimateConfig {
                red_cost_model: ReductionCostModel::Bdgl16,
                ..EstimateConfig::default()
            },
        )
        .unwrap();
        assert_eq!(cost.rop, CostValue::Infinity);
    }
}
