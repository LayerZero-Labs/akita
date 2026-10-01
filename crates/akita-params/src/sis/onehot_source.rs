//! Kernel-faithful canonical source classes for unit one-hot root folds.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use akita_challenges::SparseChallengeConfig;

/// Number of nonzero canonical coefficients in one source ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct SourceClass {
    pub(super) nonzero_count: usize,
}

impl SourceClass {
    pub(super) fn l1_norm(self) -> Option<u128> {
        Some(self.nonzero_count as u128)
    }

    pub(super) const fn infinity_norm(self) -> u128 {
        if self.nonzero_count != 0 {
            1
        } else {
            0
        }
    }
}

type SourceClassKey = (usize, usize);
type MgfKey = (usize, usize, usize, usize, usize, i32);
type DeterministicKey = (usize, usize, usize, usize);

static SOURCE_CLASSES: LazyLock<Mutex<HashMap<SourceClassKey, Vec<SourceClass>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static MAX_LOG_MGF: LazyLock<Mutex<HashMap<MgfKey, f64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static DETERMINISTIC_CAP: LazyLock<Mutex<HashMap<DeterministicKey, u128>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Maximal canonical source class for one ring.
///
/// `K` is exact. Empty chunks are admitted because the public one-hot type
/// allows them. Distinct chunks occupy distinct canonical coefficients.
pub(super) fn canonical_source_classes(
    ring_dimension: usize,
    chunk_size: usize,
) -> Option<Vec<SourceClass>> {
    let key = (ring_dimension, chunk_size);
    if let Some(cached) = SOURCE_CLASSES.lock().ok()?.get(&key) {
        return Some(cached.clone());
    }
    let classes = compute_canonical_source_classes(ring_dimension, chunk_size)?;
    SOURCE_CLASSES.lock().ok()?.insert(key, classes.clone());
    Some(classes)
}

fn compute_canonical_source_classes(
    ring_dimension: usize,
    chunk_size: usize,
) -> Option<Vec<SourceClass>> {
    if ring_dimension == 0
        || chunk_size == 0
        || !ring_dimension.is_power_of_two()
        || !chunk_size.is_power_of_two()
        || !(ring_dimension.is_multiple_of(chunk_size) || chunk_size.is_multiple_of(ring_dimension))
    {
        return None;
    }
    Some(vec![SourceClass {
        nonzero_count: if chunk_size < ring_dimension {
            ring_dimension / chunk_size
        } else {
            1
        },
    }])
}

/// Rigorous log-MGF upper bound for one fixed output coefficient and challenge.
///
/// Fixed-weight support is sampled without replacement. Hoeffding's comparison
/// theorem bounds it by independent sampling with replacement from the same
/// physical coefficient population. The `+/-1` and `+/-2` challenge supports
/// are disjoint but ordered by the same `|A_s|`; negative association gives
/// the product of their two population means.
pub(super) fn log_mgf_upper(
    class: SourceClass,
    challenge: &SparseChallengeConfig,
    ring_dimension: usize,
    lambda: f64,
) -> Option<f64> {
    let dimension = ring_dimension as f64;
    let cosh_one = cosh_upper(lambda)?;
    let cosh_two = cosh_upper(round_up(2.0 * lambda))?;
    let mean_pm1 = population_mean_upper(class, dimension, round_up(cosh_one - 1.0));
    let mean_pm2 = population_mean_upper(class, dimension, round_up(cosh_two - 1.0));
    let pm1 = round_up(challenge.count_pm1 as f64 * ln_upper(mean_pm1)?);
    let pm2 = round_up(challenge.count_pm2 as f64 * ln_upper(mean_pm2)?);
    Some(round_up(pm1 + pm2))
}

pub(super) fn max_log_mgf_upper(
    ring_dimension: usize,
    chunk_size: usize,
    challenge_dimension: usize,
    challenge: &SparseChallengeConfig,
    lambda_exponent: i32,
) -> Option<f64> {
    let key = (
        ring_dimension,
        chunk_size,
        challenge_dimension,
        challenge.count_pm1,
        challenge.count_pm2,
        lambda_exponent,
    );
    if let Some(cached) = MAX_LOG_MGF.lock().ok()?.get(&key) {
        return Some(*cached);
    }
    let lambda = 2f64.powi(lambda_exponent);
    let value = canonical_source_classes(ring_dimension, chunk_size)?
        .into_iter()
        .filter_map(|class| log_mgf_upper(class, challenge, challenge_dimension, lambda))
        .reduce(f64::max)?;
    MAX_LOG_MGF.lock().ok()?.insert(key, value);
    Some(value)
}

pub(super) fn deterministic_convolution_cap(
    ring_dimension: usize,
    chunk_size: usize,
    challenge: &SparseChallengeConfig,
) -> Option<u128> {
    let key = (
        ring_dimension,
        chunk_size,
        challenge.count_pm1,
        challenge.count_pm2,
    );
    if let Some(cached) = DETERMINISTIC_CAP.lock().ok()?.get(&key) {
        return Some(*cached);
    }
    let challenge_l1 =
        (challenge.count_pm1 as u128).checked_add((challenge.count_pm2 as u128).checked_mul(2)?)?;
    let challenge_linf = challenge.infinity_norm() as u128;
    let value = canonical_source_classes(ring_dimension, chunk_size)?
        .into_iter()
        .filter_map(|class| {
            let l1_route = class.l1_norm()?.checked_mul(challenge_linf)?;
            let linf_route = class.infinity_norm().checked_mul(challenge_l1)?;
            Some(l1_route.min(linf_route))
        })
        .max()?;
    DETERMINISTIC_CAP.lock().ok()?.insert(key, value);
    Some(value)
}

fn population_mean_upper(class: SourceClass, dimension: f64, nonzero_excess: f64) -> f64 {
    let active = round_up(round_up(class.nonzero_count as f64 / dimension) * nonzero_excess);
    round_up(1.0 + active)
}

/// Explicit outward-rounded `cosh` upper bound.
fn cosh_upper(value: f64) -> Option<f64> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    let squared = round_up(value * value);
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 1u32..=128 {
        let denominator = f64::from((2 * n - 1) * (2 * n));
        term = round_up(round_up(term * squared) / denominator);
        sum = round_up(sum + term);
        let next_denominator = f64::from((2 * n + 1) * (2 * n + 2));
        let ratio = round_up(squared / next_denominator);
        if ratio < 1.0 {
            let next = round_up(term * ratio);
            let remainder = round_up(next / round_down(1.0 - ratio));
            if remainder <= sum * f64::EPSILON {
                return Some(round_up(sum + remainder));
            }
        }
    }
    None
}

/// Explicit outward-rounded natural-log upper bound for `value >= 1`.
pub(super) fn ln_upper(value: f64) -> Option<f64> {
    if !value.is_finite() || value < 1.0 {
        return None;
    }
    if value == 1.0 {
        return Some(0.0);
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mantissa = f64::from_bits((bits & ((1u64 << 52) - 1)) | (1023u64 << 52));
    const LN_2_UPPER: f64 = f64::from_bits(0x3fe6_2e42_fefa_39f0);
    if mantissa == 1.0 {
        return Some(round_up(f64::from(exponent) * LN_2_UPPER));
    }
    let z = round_up(round_up(mantissa - 1.0) / round_down(mantissa + 1.0));
    let z_squared = round_up(z * z);
    let mut power = z;
    let mut sum = z;
    let mut denominator = 3u32;
    loop {
        power = round_up(power * z_squared);
        let term = round_up(power / f64::from(denominator));
        sum = round_up(sum + term);
        let next_power = round_up(power * z_squared);
        let next_denominator = denominator + 2;
        let remainder = round_up(
            round_up(next_power / f64::from(next_denominator)) / round_down(1.0 - z_squared),
        );
        if remainder <= sum * f64::EPSILON {
            let mantissa_log = round_up(2.0 * round_up(sum + remainder));
            return Some(round_up(
                round_up(f64::from(exponent) * LN_2_UPPER) + mantissa_log,
            ));
        }
        denominator = next_denominator;
        if denominator > 257 {
            return None;
        }
    }
}

pub(super) fn round_up(value: f64) -> f64 {
    if !value.is_finite() || value == f64::INFINITY {
        return value;
    }
    if value == -0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value >= 0.0 { bits + 1 } else { bits - 1 })
}

fn round_down(value: f64) -> f64 {
    if !value.is_finite() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_classes_cover_all_k_vs_d_cases() {
        assert_eq!(
            canonical_source_classes(256, 256).unwrap(),
            vec![SourceClass { nonzero_count: 1 }]
        );
        assert_eq!(
            canonical_source_classes(256, 512).unwrap(),
            vec![SourceClass { nonzero_count: 1 }]
        );
        assert_eq!(
            canonical_source_classes(256, 16).unwrap(),
            vec![SourceClass { nonzero_count: 16 }]
        );
    }

    #[test]
    fn canonical_classes_match_exhaustive_tables_for_every_small_k_d_case() {
        const RING_DIMENSION: usize = 8;
        for chunk_size in [1, 2, 4, 8, 16] {
            let class = canonical_source_classes(RING_DIMENSION, chunk_size)
                .unwrap()
                .pop()
                .unwrap();
            let mut source = [0i8; RING_DIMENSION];
            for chunk_start in (0..RING_DIMENSION).step_by(chunk_size.min(RING_DIMENSION)) {
                source[chunk_start] = 1;
            }
            assert_eq!(
                class.nonzero_count,
                source.iter().filter(|&&x| x == 1).count()
            );
        }
    }

    #[test]
    fn replacement_mgf_dominates_exact_disjoint_fixed_weight_support() {
        let challenge = SparseChallengeConfig {
            count_pm1: 2,
            count_pm2: 1,
        };
        let class = SourceClass { nonzero_count: 3 };
        let magnitudes = [1.0f64, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let lambda = 0.375;
        let mut exact = 0.0;
        for first in 0..magnitudes.len() {
            for second in first + 1..magnitudes.len() {
                for third in 0..magnitudes.len() {
                    if third == first || third == second {
                        continue;
                    }
                    exact += (lambda * magnitudes[first]).cosh()
                        * (lambda * magnitudes[second]).cosh()
                        * (2.0 * lambda * magnitudes[third]).cosh();
                }
            }
        }
        exact /= 28.0 * 6.0;
        let bound = log_mgf_upper(class, &challenge, 8, lambda).unwrap().exp();
        assert!(bound >= exact, "bound={bound}, exact={exact}");
    }

    #[test]
    fn packing_mgf_uses_the_embedded_residue_class_population() {
        let challenge = SparseChallengeConfig::pm1_only(1);
        let class = SourceClass { nonzero_count: 4 };
        let lambda = 0.5f64;
        let exact = lambda.cosh();
        let subring_bound = log_mgf_upper(class, &challenge, 4, lambda).unwrap().exp();
        let ambient_bound = log_mgf_upper(class, &challenge, 16, lambda).unwrap().exp();
        assert!(subring_bound >= exact);
        assert!(ambient_bound < exact);
    }

    #[test]
    fn outward_series_dominate_standard_library_values() {
        for value in [0.0, 0.125, 0.5, 1.0, 4.0, 16.0] {
            assert!(cosh_upper(value).unwrap() >= value.cosh());
        }
        for value in [1.0, 1.001, 1.5, 2.0, 17.0, 1_000_000.0] {
            assert!(ln_upper(value).unwrap() >= value.ln());
        }
    }
}
