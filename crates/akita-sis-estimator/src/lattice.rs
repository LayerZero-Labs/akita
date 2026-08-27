//! Fixed-beta, fixed-zeta infinity-norm SIS lattice cost.

use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};

use crate::{
    config::{EstimateConfig, OptimizerConfig, SearchMode, ShapeModel},
    cost::{CostValue, EstimateTag, LatticeCost, LogCost},
    error::{EstimatorError, Result},
    math::{erf, log2_positive, sis_trivially_easy},
    params::{Bound, SisParameters},
    probability::log2_amplify,
    reduction::{
        log2_bkz_cost, log2_to_cost_value, short_vectors_for, validate_infinity_reduction,
    },
    simulator::{
        infinity_shape_profile, is_q_vector_length, lgsa_summary, validate_infinity_shape,
        LgsaSummary,
    },
};

const UNIT_VECTOR_TOLERANCE: f64 = 1e-8;
const MIN_SIEVE_LOG2: f64 = -100.0 * std::f64::consts::LOG2_10;
// Pinned lattice-estimator computes the sieve floor as Sage RR(1e-100), which
// overflows to oo once repeated past the binary64 exponent range.
const SAGE_RR_MAX_LOG2: f64 = 1024.0;
// The compact summary is numerically equivalent for the observables consumed
// by the infinity probability path (see the simulator parity test). Keeping
// dense sorting below this threshold preserves lattice-estimator rounding for
// parity fixtures while avoiding wide-profile allocations.
const MAX_DENSE_PROFILE_DIM: u64 = 1_000;
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
    let (lattice_dimension, zeta_stop) = infinity_lattice_domain(params, config)?;
    let lattice_estimator_parity_search = matches!(
        config.optimizer,
        OptimizerConfig::OptimizeZeta {
            zeta: SearchMode::PythonLocalMinimum,
            ..
        }
    );
    if zeta >= zeta_stop && !lattice_estimator_parity_search {
        return Err(EstimatorError::InvalidParameter {
            field: "zeta",
            reason: "zeta must leave an effective lattice dimension greater than n".to_string(),
        });
    }
    let effective_dimension =
        lattice_dimension
            .checked_sub(zeta)
            .ok_or(EstimatorError::InvalidParameter {
                field: "zeta",
                reason: "zeta must not exceed the lattice dimension".to_string(),
            })?;
    let uses_small_box = infinity_uses_small_box(params, effective_dimension)?;
    if effective_dimension < u64::from(beta) {
        return Ok(proven_above_target_cost(
            params,
            beta,
            zeta,
            effective_dimension,
        ));
    }

    let identity_vectors = effective_dimension as i128 - params.n as i128;
    let reduction_dimension = effective_dimension;
    let short = short_vectors_for(config.red_cost_model, beta, reduction_dimension)?;
    let bkz_log2 = log2_bkz_cost(config.red_cost_model, beta, reduction_dimension)?;

    let compact_lgsa_search = matches!(
        config.optimizer,
        OptimizerConfig::OptimizeZeta {
            zeta: SearchMode::Exhaustive
                | SearchMode::ExhaustiveParallel
                | SearchMode::ProvenPruned,
            ..
        }
    );
    let log_trial_prob = if config.red_shape_model == ShapeModel::Lgsa
        && (compact_lgsa_search || effective_dimension > MAX_DENSE_PROFILE_DIM)
    {
        let summary = lgsa_summary(effective_dimension, identity_vectors, &params.q, beta)?;
        infinity_log_trial_probability_lgsa_summary(
            scratch.log_q,
            length_bound,
            uses_small_box,
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
            uses_small_box,
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

/// Return the lattice dimension and exclusive valid zeta bound.
///
/// The q-ary SIS embedding has `d - n` identity vectors. An effective
/// dimension at or below `n` is not a tall SIS lattice and cannot be priced by
/// this attack model. The returned zeta interval is therefore
/// `0..lattice_dimension - n`.
pub(crate) fn infinity_lattice_domain(
    params: &SisParameters,
    config: &EstimateConfig,
) -> Result<(u64, u64)> {
    let m = params.m.ok_or(EstimatorError::InvalidParameter {
        field: "m",
        reason: "infinity lattice estimation requires an explicit column count m".to_string(),
    })?;
    let lattice_dimension = config.lattice_dimension.unwrap_or(m);
    if lattice_dimension > m {
        return Err(EstimatorError::InvalidParameter {
            field: "lattice_dimension",
            reason: "lattice dimension must not exceed m".to_string(),
        });
    }
    let zeta_stop = lattice_dimension
        .checked_sub(u64::from(params.n))
        .filter(|stop| *stop > 0)
        .ok_or(EstimatorError::InvalidParameter {
            field: "lattice_dimension",
            reason: "infinity lattice estimation requires lattice_dimension > n".to_string(),
        })?;
    Ok((lattice_dimension, zeta_stop))
}

/// Return whether the reduced infinity-norm instance uses the small-box
/// probability formula.
pub(crate) fn infinity_uses_small_box(
    params: &SisParameters,
    effective_dimension: u64,
) -> Result<bool> {
    let log_dimension = log2_positive(effective_dimension as f64);
    let log_q_squared = 2.0 * crate::math::log2_biguint(&params.q);
    Ok(match &params.length_bound {
        Bound::Integer(bound) => {
            let log_lhs = log_dimension + 2.0 * crate::math::log2_biguint(bound);
            log_comparison(log_lhs, log_q_squared).unwrap_or_else(|| {
                BigUint::from(effective_dimension) * bound * bound <= &params.q * &params.q
            })
        }
        Bound::Rational {
            numerator,
            denominator,
        } => {
            let log_lhs = log_dimension + 2.0 * crate::math::log2_biguint(numerator);
            let log_rhs = log_q_squared + 2.0 * crate::math::log2_biguint(denominator);
            log_comparison(log_lhs, log_rhs).unwrap_or_else(|| {
                BigUint::from(effective_dimension) * numerator * numerator
                    <= &params.q * &params.q * denominator * denominator
            })
        }
        Bound::SqrtInteger(radicand) => {
            let log_lhs = log_dimension + crate::math::log2_biguint(radicand);
            log_comparison(log_lhs, log_q_squared).unwrap_or_else(|| {
                BigUint::from(effective_dimension) * radicand <= &params.q * &params.q
            })
        }
        Bound::Float(bound) => {
            (effective_dimension as f64).sqrt() * bound
                <= 2.0_f64.powf(crate::math::log2_biguint(&params.q))
        }
    })
}

fn log_comparison(log_lhs: f64, log_rhs: f64) -> Option<bool> {
    const EXACT_FALLBACK_WINDOW: f64 = 1e-8;
    if log_lhs < log_rhs - EXACT_FALLBACK_WINDOW {
        Some(true)
    } else if log_lhs > log_rhs + EXACT_FALLBACK_WINDOW {
        Some(false)
    } else {
        None
    }
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
        Bound::SqrtInteger(value) => Ok(value.to_f64().unwrap_or(f64::INFINITY).sqrt()),
    }
}

fn infinity_log_trial_probability(
    log_q: f64,
    length_bound: f64,
    uses_small_box: bool,
    effective_dimension: u64,
    profile: &[f64],
    rho: f64,
    sieve_dim: u32,
) -> Result<f64> {
    let d_ = effective_dimension as f64;
    // The probability experiment runs after zeta coordinates have been
    // projected away, so its small-box condition uses the active dimension.
    if uses_small_box {
        let log2_sigma =
            log2_positive(rho) + 0.5 * log2_positive(profile[0]) - 0.5 * log2_positive(d_);
        let log2_erf_arg = log2_positive(length_bound) - 0.5 - log2_sigma;
        Ok(d_ * log2_erf_from_log2_arg(log2_erf_arg))
    } else {
        dilithium_log_trial_probability(log_q, length_bound, profile, sieve_dim)
    }
}

fn infinity_log_trial_probability_lgsa_summary(
    log_q: f64,
    length_bound: f64,
    uses_small_box: bool,
    summary: &LgsaSummary,
    rho: f64,
    sieve_dim: u32,
) -> Result<f64> {
    let d_ = summary.effective_dimension as f64;
    if uses_small_box {
        let log2_sigma = log2_positive(rho) + summary.first_log2_norm - 0.5 * log2_positive(d_);
        let log2_erf_arg = log2_positive(length_bound) - 0.5 - log2_sigma;
        Ok(d_ * log2_erf_from_log2_arg(log2_erf_arg))
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
    let gaussian_coords = (idx_end - idx_start + 1).max(u64::from(sieve_dim)) as f64;
    let log2_sigma = summary.log2_vector_length_at_idx_start - 0.5 * log2_positive(gaussian_coords);
    let log2_erf_arg = log2_positive(length_bound) - 0.5 - log2_sigma;
    let mut log_trial_prob = log2_erf_from_log2_arg(log2_erf_arg) * gaussian_coords;
    log_trial_prob += log2_positive((2.0 * length_bound + 1.0) / q_f) * idx_start as f64;
    Ok(log_trial_prob)
}

fn log2_erf_from_log2_arg(log2_arg: f64) -> f64 {
    if log2_arg < -20.0 {
        return log2_arg + log2_positive(2.0 / std::f64::consts::PI.sqrt());
    }
    log2_positive(erf(log2_arg.exp2()))
}

fn dilithium_log_trial_probability(
    log_q: f64,
    length_bound: f64,
    profile: &[f64],
    sieve_dim: u32,
) -> Result<f64> {
    let q_f = 2.0_f64.powf(log_q);
    let idx_start = q_vector_prefix_len(profile, q_f);
    let idx_end = profile
        .iter()
        .rposition(|value| value.sqrt() > 1.0 + UNIT_VECTOR_TOLERANCE)
        .map_or(profile.len() - 1, |index| index);
    let gaussian_coords = (idx_end - idx_start + 1).max(sieve_dim as usize) as f64;
    let log2_sigma = 0.5 * log2_positive(profile[idx_start]) - 0.5 * log2_positive(gaussian_coords);
    let log2_erf_arg = log2_positive(length_bound) - 0.5 - log2_sigma;
    let mut log_trial_prob = log2_erf_from_log2_arg(log2_erf_arg) * gaussian_coords;
    log_trial_prob += log2_positive((2.0 * length_bound + 1.0) / q_f) * idx_start as f64;
    Ok(log_trial_prob)
}

fn q_vector_prefix_len(profile: &[f64], q: f64) -> usize {
    let r0 = profile[0];
    if is_q_vector_length(r0.sqrt(), q) {
        profile.iter().position(|value| *value < r0).unwrap_or(0)
    } else {
        0
    }
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

fn proven_above_target_cost(
    params: &SisParameters,
    beta: u32,
    zeta: u64,
    effective_dimension: u64,
) -> LatticeCost {
    LatticeCost {
        rop: CostValue::ProvenAboveTarget(LogCost::new(f64::INFINITY)),
        red: None,
        sieve: None,
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

    #[test]
    fn infinity_domain_rejects_non_tall_and_oversized_dimensions() {
        let square = SisParameters::try_new(
            64,
            akita_q32(),
            Some(64),
            Bound::from_u64(15),
            SisNorm::Infinity,
        )
        .unwrap();
        assert!(matches!(
            cost_infinity_fixed(40, &square, 0, &sample_config()),
            Err(EstimatorError::InvalidParameter {
                field: "lattice_dimension",
                ..
            })
        ));

        let tall = SisParameters::try_new(
            64,
            akita_q32(),
            Some(128),
            Bound::from_u64(15),
            SisNorm::Infinity,
        )
        .unwrap();
        assert!(matches!(
            cost_infinity_fixed(
                40,
                &tall,
                0,
                &EstimateConfig {
                    lattice_dimension: Some(129),
                    ..sample_config()
                }
            ),
            Err(EstimatorError::InvalidParameter {
                field: "lattice_dimension",
                ..
            })
        ));
    }

    #[test]
    fn infinity_domain_rejects_zeta_that_removes_tallness() {
        let params = SisParameters::try_new(
            64,
            akita_q32(),
            Some(128),
            Bound::from_u64(15),
            SisNorm::Infinity,
        )
        .unwrap();
        let strict_config = EstimateConfig {
            optimizer: OptimizerConfig::OptimizeZeta {
                beta: SearchMode::Exhaustive,
                zeta: SearchMode::Exhaustive,
            },
            ..sample_config()
        };
        assert!(cost_infinity_fixed(40, &params, 63, &strict_config).is_ok());
        assert!(matches!(
            cost_infinity_fixed(40, &params, 64, &strict_config),
            Err(EstimatorError::InvalidParameter { field: "zeta", .. })
        ));
    }

    #[test]
    fn probability_regime_uses_dimension_remaining_after_zeta() {
        let original_dimension = 65_537_u64;
        let zeta = 57_345_u64;
        let effective_dimension = original_dimension - zeta;
        let length_bound = (1_u64 << 24) - 1;
        let q = akita_q32();
        let q_f = q.to_f64().unwrap();

        assert!((original_dimension as f64).sqrt() * length_bound as f64 > q_f);
        assert!((effective_dimension as f64).sqrt() * length_bound as f64 <= q_f);

        let params = SisParameters::try_new(
            1_024,
            q,
            Some(original_dimension),
            Bound::from_u64(length_bound),
            SisNorm::Infinity,
        )
        .unwrap();
        let config = EstimateConfig {
            red_cost_model: ReductionCostModel::Adps16 {
                mode: Adps16Mode::Quantum,
            },
            red_shape_model: ShapeModel::Lgsa,
            ..EstimateConfig::default()
        };
        let cost = cost_infinity_fixed(343, &params, zeta, &config).unwrap();

        assert_eq!(cost.d, effective_dimension);
        assert!((cost.rop.log2().unwrap() - 118.916_112_523_987).abs() < 1e-9);
    }

    #[test]
    fn small_box_comparison_is_exact_at_integer_boundary() {
        let params = SisParameters::try_new(
            1,
            BigUint::from(1_u8) << 32,
            Some(65_537),
            Bound::from_u64(1_u64 << 24),
            SisNorm::Infinity,
        )
        .unwrap();

        assert!(infinity_uses_small_box(&params, 65_536).unwrap());
        assert!(!infinity_uses_small_box(&params, 65_537).unwrap());
    }

    #[test]
    fn log2_erf_stays_finite_for_tiny_arguments() {
        let log2_arg = -1_000.0;
        let log2_erf = log2_erf_from_log2_arg(log2_arg);
        assert!(log2_erf.is_finite());
        assert!(log2_erf < -999.0);
        assert!(log2_erf > -1_001.0);
    }

    #[test]
    fn reconstructed_q64_prefix_uses_scale_aware_detection() {
        let q = 2.0_f64.powi(64);
        let reconstructed_q = q + 28_672.0;
        let r0 = reconstructed_q * reconstructed_q;
        let profile = [r0, r0, (q - 2.0_f64.powi(40)).powi(2), 4.0];

        assert_eq!(q_vector_prefix_len(&profile, q), 2);
        assert!(!is_q_vector_length(profile[2].sqrt(), q));
    }
}
