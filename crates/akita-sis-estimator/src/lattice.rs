//! Fixed-beta, fixed-zeta infinity-norm SIS lattice cost.

use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};

use crate::{
    config::{Adps16Mode, EstimateConfig, ReductionCostModel, ShapeModel},
    cost::{CostValue, EstimateTag, LatticeCost},
    error::{EstimatorError, Result},
    math::{erf, log2_biguint, log2_positive, sis_trivially_easy},
    params::{Bound, SisParameters},
    probability::log2_amplify,
    reduction::{adps16_log2_cost, adps16_short_vectors, delta, log2_to_cost_value},
    simulator::lgsa_squared_norms,
};

const Q_VECTOR_TOLERANCE: f64 = 1e-8;
const UNIT_VECTOR_TOLERANCE: f64 = 1e-8;
const MIN_SIEVE_LOG2: f64 = -100.0 * std::f64::consts::LOG2_10;
// PR217 computes the sieve floor as Sage RR(1e-100), which overflows to oo
// once repeated past the binary64 exponent range.
const SAGE_RR_MAX_LOG2: f64 = 1024.0;

/// Evaluate fixed-beta, fixed-zeta infinity cost for ADPS16 + LGSA.
pub fn cost_infinity_fixed(
    beta: u32,
    params: &SisParameters,
    zeta: u32,
    config: &EstimateConfig,
) -> Result<LatticeCost> {
    validate_fixed_profile(config)?;
    let length_bound = length_bound_as_f64(&params.length_bound)?;
    if sis_trivially_easy(&params.q, length_bound) {
        return Err(EstimatorError::InvalidParameter {
            field: "length_bound",
            reason: "SIS trivially easy: length_bound must be below (q - 1) / 2".to_string(),
        });
    }
    let m = params.m.ok_or(EstimatorError::InvalidParameter {
        field: "m",
        reason: "fixed infinity cost requires an explicit column count m".to_string(),
    })?;
    let lattice_dimension = config.lattice_dimension.unwrap_or(m);
    let effective_dimension =
        lattice_dimension
            .checked_sub(zeta)
            .ok_or(EstimatorError::InvalidParameter {
                field: "zeta",
                reason: "zeta must not exceed the lattice dimension".to_string(),
            })?;
    if effective_dimension < beta {
        return Ok(infinite_cost(params, beta, zeta, effective_dimension));
    }

    let mode = adps16_mode(config.red_cost_model);
    let identity_vectors = effective_dimension as i64 - params.n as i64;
    let profile = lgsa_squared_norms(effective_dimension, identity_vectors, &params.q, beta)?;
    let short = adps16_short_vectors(beta, effective_dimension, mode);
    let bkz_log2 = adps16_log2_cost(beta, mode);

    let log_trial_prob = infinity_log_trial_probability(
        &params.q,
        length_bound,
        lattice_dimension,
        effective_dimension,
        &profile,
        short.rho,
        short.sieve_dim,
    )?;
    let log_probability = (log_trial_prob + log2_positive(short.count)).min(0.0);
    if !log_probability.is_finite() {
        return Ok(infinite_cost(params, beta, zeta, effective_dimension));
    }

    let repetitions_log2 = log2_amplify(config.success_probability.get(), log_probability);
    if !repetitions_log2.is_finite() {
        return Ok(infinite_cost(params, beta, zeta, effective_dimension));
    }

    let pre_repeat_sieve = pre_repeat_sieve_log2(short.cost_red_log2, bkz_log2);
    let sieve_log2 = pre_repeat_sieve.log2 + repetitions_log2;
    let rop_log2 = short.cost_red_log2 + repetitions_log2;
    let red_log2 = bkz_log2 + repetitions_log2;

    Ok(LatticeCost {
        rop: log2_to_cost_value(rop_log2),
        red: Some(log2_to_cost_value(red_log2)),
        sieve: Some(sieve_cost_value(
            pre_repeat_sieve,
            repetitions_log2,
            sieve_log2,
        )),
        delta: Some(delta(beta)),
        beta: Some(beta),
        eta: Some(short.sieve_dim),
        zeta: Some(zeta),
        d: effective_dimension,
        prob: probability_from_log2(log_probability),
        repetitions: Some(log2_to_cost_value(repetitions_log2)),
        tag: params
            .tag
            .as_ref()
            .map(|value| EstimateTag::new(value.clone()))
            .unwrap_or_default(),
    })
}

fn validate_fixed_profile(config: &EstimateConfig) -> Result<()> {
    let unsupported_cost_model = match config.red_cost_model {
        ReductionCostModel::Adps16 { .. } => None,
        ReductionCostModel::Bdgl16 => Some("red_cost_model::Bdgl16"),
        ReductionCostModel::Matzov { .. } => Some("red_cost_model::Matzov"),
        ReductionCostModel::Gj21 { .. } => Some("red_cost_model::Gj21"),
        ReductionCostModel::Kyber { .. } => Some("red_cost_model::Kyber"),
    };
    if let Some(feature) = unsupported_cost_model {
        return Err(EstimatorError::Unsupported { feature });
    }
    if config.red_shape_model != ShapeModel::Lgsa {
        return Err(EstimatorError::Unsupported {
            feature: "red_shape_model != LGSA",
        });
    }
    Ok(())
}

fn adps16_mode(model: ReductionCostModel) -> Adps16Mode {
    match model {
        ReductionCostModel::Adps16 { mode } => mode,
        _ => Adps16Mode::Classical,
    }
}

fn length_bound_as_f64(bound: &Bound) -> Result<f64> {
    match bound {
        Bound::Integer(value) => {
            if value.is_zero() {
                return Err(EstimatorError::InvalidParameter {
                    field: "length_bound",
                    reason: "integer bound must be positive".to_string(),
                });
            }
            Ok(value.to_f64().unwrap_or(f64::INFINITY))
        }
        Bound::Float(value) => Ok(*value),
        Bound::Rational {
            numerator,
            denominator,
        } => Ok(numerator.to_f64().unwrap_or(0.0) / denominator.to_f64().unwrap_or(1.0)),
    }
}

fn infinity_log_trial_probability(
    q: &BigUint,
    length_bound: f64,
    lattice_dimension: u32,
    effective_dimension: u32,
    profile: &[f64],
    rho: f64,
    sieve_dim: u32,
) -> Result<f64> {
    let d_ = effective_dimension as f64;
    let log_q = log2_biguint(q);
    if ((lattice_dimension as f64).sqrt() * length_bound) <= 2.0_f64.powf(log_q) {
        let vector_length = rho * profile[0].sqrt();
        let sigma = vector_length / d_.sqrt();
        let erf_arg = length_bound / (2.0_f64.sqrt() * sigma);
        Ok(d_ * log2_positive(erf(erf_arg)))
    } else {
        dilithium_log_trial_probability(q, length_bound, profile, sieve_dim)
    }
}

fn dilithium_log_trial_probability(
    q: &BigUint,
    length_bound: f64,
    profile: &[f64],
    sieve_dim: u32,
) -> Result<f64> {
    let log_q = log2_biguint(q);
    let q_f = 2.0_f64.powf(log_q);
    let r0 = profile[0];
    let idx_start = if (r0.sqrt() - q_f).abs() < Q_VECTOR_TOLERANCE {
        profile.iter().position(|value| *value < r0).unwrap_or(0)
    } else {
        0
    };
    let idx_end = profile
        .iter()
        .rposition(|value| value.sqrt() > 1.0 + UNIT_VECTOR_TOLERANCE)
        .map_or(profile.len() - 1, |index| index);
    let vector_length = profile[idx_start].sqrt();
    let gaussian_coords = (idx_end - idx_start + 1).max(sieve_dim as usize) as f64;
    let sigma = vector_length / gaussian_coords.sqrt();
    let erf_arg = length_bound / (2.0_f64.sqrt() * sigma);
    let mut log_trial_prob = log2_positive(erf(erf_arg)) * gaussian_coords;
    log_trial_prob += log2_positive((2.0 * length_bound + 1.0) / q_f) * idx_start as f64;
    Ok(log_trial_prob)
}

#[derive(Clone, Copy, Debug)]
struct PreRepeatSieve {
    log2: f64,
    used_floor: bool,
}

fn pre_repeat_sieve_log2(cost_red_log2: f64, bkz_log2: f64) -> PreRepeatSieve {
    if cost_red_log2 > bkz_log2 {
        PreRepeatSieve {
            log2: cost_red_log2 + log2_positive(1.0 - 2.0_f64.powf(bkz_log2 - cost_red_log2)),
            used_floor: false,
        }
    } else {
        PreRepeatSieve {
            log2: MIN_SIEVE_LOG2,
            used_floor: true,
        }
    }
}

fn sieve_cost_value(
    pre_repeat: PreRepeatSieve,
    repetitions_log2: f64,
    repeated_log2: f64,
) -> CostValue {
    if pre_repeat.used_floor && repetitions_log2 >= SAGE_RR_MAX_LOG2 {
        CostValue::Infinity
    } else {
        log2_to_cost_value(repeated_log2)
    }
}

fn probability_from_log2(log_probability: f64) -> Option<crate::numeric::Probability> {
    let probability = 2.0_f64.powf(log_probability);
    if probability > 0.0 && probability.is_finite() {
        crate::numeric::Probability::new(probability).ok()
    } else {
        None
    }
}

fn infinite_cost(
    params: &SisParameters,
    beta: u32,
    zeta: u32,
    effective_dimension: u32,
) -> LatticeCost {
    LatticeCost {
        rop: CostValue::Infinity,
        red: Some(CostValue::Infinity),
        sieve: Some(CostValue::Infinity),
        delta: Some(delta(beta)),
        beta: Some(beta),
        eta: None,
        zeta: Some(zeta),
        d: effective_dimension,
        prob: None,
        repetitions: None,
        tag: params
            .tag
            .as_ref()
            .map(|value| EstimateTag::new(value.clone()))
            .unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{Adps16Mode, EstimateConfig, ReductionCostModel, ShapeModel},
        params::{akita_q128, akita_q32, SisNorm},
    };

    fn sample_config() -> EstimateConfig {
        EstimateConfig {
            red_cost_model: ReductionCostModel::Adps16 {
                mode: Adps16Mode::Classical,
            },
            red_shape_model: ShapeModel::Lgsa,
            ..EstimateConfig::default()
        }
    }

    #[test]
    fn fixed_infinity_rejects_non_lgsa_shape_model() {
        let params = SisParameters::try_new(
            32,
            akita_q32(),
            Some(64),
            Bound::from_u64(15),
            SisNorm::Infinity,
        )
        .unwrap();
        let mut config = sample_config();
        config.red_shape_model = ShapeModel::Gsa;
        assert!(matches!(
            cost_infinity_fixed(63, &params, 0, &config),
            Err(EstimatorError::Unsupported { .. })
        ));
    }

    #[test]
    fn fixed_infinity_reports_infinite_sieve_for_tiny_probability_goldens() {
        let params = SisParameters::try_new(
            32,
            akita_q128(),
            Some(64),
            Bound::from_u64(15),
            SisNorm::Infinity,
        )
        .unwrap();
        let cost = cost_infinity_fixed(63, &params, 0, &sample_config()).unwrap();
        assert!(matches!(cost.sieve, Some(CostValue::Infinity)));
    }
}
