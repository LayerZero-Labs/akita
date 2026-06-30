//! Fixed-beta, fixed-zeta infinity-norm SIS lattice cost.

use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};

use crate::{
    config::{EstimateConfig, ShapeModel},
    cost::{CostValue, EstimateTag, LatticeCost},
    error::{EstimatorError, Result},
    math::{erf, log2_positive, sis_trivially_easy},
    params::{Bound, SisParameters},
    probability::log2_amplify,
    reduction::{
        log2_bkz_cost, log2_to_cost_value, short_vectors_for, validate_infinity_reduction,
    },
    simulator::{infinity_shape_profile, lgsa_summary, validate_infinity_shape, LgsaSummary},
};

const Q_VECTOR_TOLERANCE: f64 = 1e-8;
const UNIT_VECTOR_TOLERANCE: f64 = 1e-8;
const MIN_SIEVE_LOG2: f64 = -100.0 * std::f64::consts::LOG2_10;
// Pinned lattice-estimator computes the sieve floor as Sage RR(1e-100), which
// overflows to oo once repeated past the binary64 exponent range.
const SAGE_RR_MAX_LOG2: f64 = 1024.0;
const MAX_DENSE_PROFILE_DIM: u64 = 1_000_000;

/// Cached numeric values reused across optimizer probes for one modulus.
#[derive(Clone, Copy, Debug)]
struct EvalScratch {
    log_q: f64,
}

impl EvalScratch {
    fn new(q: &BigUint) -> Self {
        Self {
            log_q: crate::math::log2_biguint(q),
        }
    }
}

/// Evaluate fixed-beta, fixed-zeta infinity cost for the configured profile.
pub fn cost_infinity_fixed(
    beta: u32,
    params: &SisParameters,
    zeta: u64,
    config: &EstimateConfig,
) -> Result<LatticeCost> {
    validate_infinity_profile(config)?;
    let scratch = EvalScratch::new(&params.q);
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
    if effective_dimension < u64::from(beta) {
        return Ok(infinite_cost(params, beta, zeta, effective_dimension));
    }

    let identity_vectors = effective_dimension as i128 - params.n as i128;
    let reduction_dimension = u32::try_from(effective_dimension).unwrap_or(u32::MAX);
    let short = short_vectors_for(config.red_cost_model, beta, reduction_dimension)?;
    let bkz_log2 = log2_bkz_cost(config.red_cost_model, beta, reduction_dimension)?;

    let log_trial_prob = if config.red_shape_model == ShapeModel::Lgsa
        && effective_dimension > MAX_DENSE_PROFILE_DIM
    {
        let summary = lgsa_summary(effective_dimension, identity_vectors, &params.q, beta)?;
        infinity_log_trial_probability_lgsa_summary(
            scratch.log_q,
            length_bound,
            lattice_dimension,
            &summary,
            short.rho,
            short.sieve_dim,
        )?
    } else {
        let effective_dimension_u32 =
            u32::try_from(effective_dimension).map_err(|_| EstimatorError::Unsupported {
                feature: "wide non-compact shape profile",
            })?;
        let identity_vectors_i64 =
            i64::try_from(identity_vectors).map_err(|_| EstimatorError::InvalidParameter {
                field: "d",
                reason: "identity vector count exceeded i64".to_string(),
            })?;
        let profile = infinity_shape_profile(
            config.red_shape_model,
            effective_dimension_u32,
            identity_vectors_i64,
            &params.q,
            beta,
        )?;
        infinity_log_trial_probability(
            scratch.log_q,
            length_bound,
            lattice_dimension,
            effective_dimension,
            profile.squared_norms(),
            short.rho,
            short.sieve_dim,
        )?
    };
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
        delta: Some(crate::reduction::delta(beta)),
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

fn validate_infinity_profile(config: &EstimateConfig) -> Result<()> {
    validate_infinity_reduction(config.red_cost_model)?;
    validate_infinity_shape(config.red_shape_model)
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
    log_q: f64,
    length_bound: f64,
    lattice_dimension: u64,
    effective_dimension: u64,
    profile: &[f64],
    rho: f64,
    sieve_dim: u32,
) -> Result<f64> {
    let d_ = effective_dimension as f64;
    if ((lattice_dimension as f64).sqrt() * length_bound) <= 2.0_f64.powf(log_q) {
        let vector_length = rho * profile[0].sqrt();
        let sigma = vector_length / d_.sqrt();
        let erf_arg = length_bound / (2.0_f64.sqrt() * sigma);
        Ok(d_ * log2_positive(erf(erf_arg)))
    } else {
        dilithium_log_trial_probability(log_q, length_bound, profile, sieve_dim)
    }
}

fn infinity_log_trial_probability_lgsa_summary(
    log_q: f64,
    length_bound: f64,
    lattice_dimension: u64,
    summary: &LgsaSummary,
    rho: f64,
    sieve_dim: u32,
) -> Result<f64> {
    let d_ = summary.effective_dimension as f64;
    if ((lattice_dimension as f64).sqrt() * length_bound) <= 2.0_f64.powf(log_q) {
        let vector_length = rho * summary.first_squared_norm.sqrt();
        let sigma = vector_length / d_.sqrt();
        let erf_arg = length_bound / (2.0_f64.sqrt() * sigma);
        Ok(d_ * log2_positive(erf(erf_arg)))
    } else {
        dilithium_log_trial_probability_lgsa_summary(log_q, length_bound, summary, sieve_dim)
    }
}

fn dilithium_log_trial_probability_lgsa_summary(
    log_q: f64,
    length_bound: f64,
    summary: &LgsaSummary,
    sieve_dim: u32,
) -> Result<f64> {
    let q_f = 2.0_f64.powf(log_q);
    let idx_start = summary.idx_start;
    let idx_end = summary.idx_end.max(idx_start);
    let vector_length = summary.vector_length_at_idx_start;
    let gaussian_coords = (idx_end - idx_start + 1).max(u64::from(sieve_dim)) as f64;
    let sigma = vector_length / gaussian_coords.sqrt();
    let erf_arg = length_bound / (2.0_f64.sqrt() * sigma);
    let mut log_trial_prob = log2_positive(erf(erf_arg)) * gaussian_coords;
    log_trial_prob += log2_positive((2.0 * length_bound + 1.0) / q_f) * idx_start as f64;
    Ok(log_trial_prob)
}

fn dilithium_log_trial_probability(
    log_q: f64,
    length_bound: f64,
    profile: &[f64],
    sieve_dim: u32,
) -> Result<f64> {
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
    zeta: u64,
    effective_dimension: u64,
) -> LatticeCost {
    LatticeCost {
        rop: CostValue::Infinity,
        red: Some(CostValue::Infinity),
        sieve: Some(CostValue::Infinity),
        delta: Some(crate::reduction::delta(beta)),
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
    fn fixed_infinity_rejects_unimplemented_shape_model() {
        let params = SisParameters::try_new(
            32,
            akita_q32(),
            Some(64),
            Bound::from_u64(15),
            SisNorm::Infinity,
        )
        .unwrap();
        let mut config = sample_config();
        config.red_shape_model = ShapeModel::Cn11;
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
