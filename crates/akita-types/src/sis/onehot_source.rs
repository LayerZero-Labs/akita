//! Kernel-faithful physical source classes for unit one-hot root folds.

use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, Mutex};

use akita_challenges::SparseChallengeConfig;

/// Counts of coalesced physical source coefficients by absolute value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct SourceClass {
    pub(super) magnitude_one: usize,
    pub(super) magnitude_two: usize,
}

impl SourceClass {
    pub(super) fn l1_norm(self) -> Option<u128> {
        (self.magnitude_one as u128).checked_add((self.magnitude_two as u128).checked_mul(2)?)
    }

    pub(super) const fn infinity_norm(self) -> u128 {
        if self.magnitude_two != 0 {
            2
        } else if self.magnitude_one != 0 {
            1
        } else {
            0
        }
    }
}

type SourceClassKey = (usize, usize, usize);
type MgfKey = (usize, usize, usize, usize, usize, i32);
type DeterministicKey = (usize, usize, usize, usize, usize);

static SOURCE_CLASSES: LazyLock<Mutex<HashMap<SourceClassKey, Vec<SourceClass>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static MAX_LOG_MGF: LazyLock<Mutex<HashMap<MgfKey, f64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static DETERMINISTIC_CAP: LazyLock<Mutex<HashMap<DeterministicKey, u128>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Attainable maximal physical source classes for one ring.
///
/// `K` is exact. Empty chunks are admitted because the public one-hot type
/// allows them. Coordinate-wise dominated classes are removed because every
/// MGF and deterministic norm used by the policy is monotone in both counts.
pub(super) fn projected_source_classes(
    ring_dimension: usize,
    chunk_size: usize,
    extension_degree: usize,
) -> Option<Vec<SourceClass>> {
    let key = (ring_dimension, chunk_size, extension_degree);
    if let Some(cached) = SOURCE_CLASSES.lock().ok()?.get(&key) {
        return Some(cached.clone());
    }
    let classes = compute_projected_source_classes(ring_dimension, chunk_size, extension_degree)?;
    SOURCE_CLASSES.lock().ok()?.insert(key, classes.clone());
    Some(classes)
}

fn compute_projected_source_classes(
    ring_dimension: usize,
    chunk_size: usize,
    extension_degree: usize,
) -> Option<Vec<SourceClass>> {
    if ring_dimension == 0
        || chunk_size == 0
        || extension_degree == 0
        || !ring_dimension.is_power_of_two()
        || !chunk_size.is_power_of_two()
        || !extension_degree.is_power_of_two()
        || !(ring_dimension.is_multiple_of(chunk_size) || chunk_size.is_multiple_of(ring_dimension))
        || !ring_dimension.is_multiple_of(extension_degree)
    {
        return None;
    }

    if extension_degree == 1 {
        let hot_count = if chunk_size < ring_dimension {
            ring_dimension / chunk_size
        } else {
            1
        };
        return Some(vec![SourceClass {
            magnitude_one: hot_count,
            magnitude_two: 0,
        }]);
    }
    let double_extension = extension_degree.checked_mul(2)?;
    if ring_dimension < double_extension {
        return None;
    }
    if chunk_size >= ring_dimension {
        // One logical hot reaches at most one ring. Coordinate zero emits one
        // monomial and every other coordinate emits two distinct monomials.
        return Some(vec![SourceClass {
            magnitude_one: 2,
            magnitude_two: 0,
        }]);
    }

    let slab_width = chunk_size.max(extension_degree);
    let paired_slab_count = ring_dimension.checked_div(slab_width.checked_mul(2)?)?;
    let configurations = slab_configurations(slab_width, chunk_size)?;
    let mut local = HashSet::new();
    for low in &configurations {
        for upper in &configurations {
            local.insert(classify_paired_slabs(low, upper, extension_degree));
        }
    }
    let mut classes = vec![SourceClass {
        magnitude_one: 0,
        magnitude_two: 0,
    }];
    for _ in 0..paired_slab_count {
        let mut next = HashSet::new();
        for left in &classes {
            for right in &local {
                next.insert(SourceClass {
                    magnitude_one: left.magnitude_one.checked_add(right.magnitude_one)?,
                    magnitude_two: left.magnitude_two.checked_add(right.magnitude_two)?,
                });
            }
        }
        classes = pareto_frontier(next);
    }
    Some(classes)
}

fn slab_configurations(slab_width: usize, chunk_size: usize) -> Option<Vec<Vec<usize>>> {
    let chunk_count = slab_width.checked_div(chunk_size)?;
    let choices_per_chunk = chunk_size.checked_add(1)?;
    let total = choices_per_chunk.checked_pow(chunk_count as u32)?;
    let mut configurations = Vec::with_capacity(total);
    for mut code in 0..total {
        let mut selected = Vec::with_capacity(chunk_count);
        for chunk in 0..chunk_count {
            let choice = code % choices_per_chunk;
            code /= choices_per_chunk;
            if choice != chunk_size {
                selected.push(chunk.checked_mul(chunk_size)?.checked_add(choice)?);
            }
        }
        configurations.push(selected);
    }
    Some(configurations)
}

fn classify_paired_slabs(low: &[usize], upper: &[usize], extension_degree: usize) -> SourceClass {
    let mut coordinate_zero = 0usize;
    let mut low_pair_keys = HashSet::new();
    for &position in low {
        let coordinate = position % extension_degree;
        if coordinate == 0 {
            coordinate_zero += 1;
        } else {
            low_pair_keys.insert((position / extension_degree, coordinate));
        }
    }
    let mut nonzero = low_pair_keys.len();
    let mut collisions = 0usize;
    for &position in upper {
        let coordinate = position % extension_degree;
        if coordinate == 0 {
            coordinate_zero += 1;
            continue;
        }
        nonzero += 1;
        let partner = extension_degree - coordinate;
        if low_pair_keys.contains(&(position / extension_degree, partner)) {
            collisions += 1;
        }
    }
    SourceClass {
        magnitude_one: coordinate_zero + 2 * (nonzero - 2 * collisions),
        magnitude_two: collisions,
    }
}

fn pareto_frontier(classes: HashSet<SourceClass>) -> Vec<SourceClass> {
    let all: Vec<_> = classes.into_iter().collect();
    let mut classes: Vec<_> = all
        .iter()
        .copied()
        .filter(|candidate| {
            !all.iter().any(|other| {
                other != candidate
                    && other.magnitude_one >= candidate.magnitude_one
                    && other.magnitude_two >= candidate.magnitude_two
            })
        })
        .collect();
    classes.sort_unstable_by_key(|class| (class.magnitude_one, class.magnitude_two));
    classes
}

/// Rigorous log-MGF upper bound for one fixed output coefficient and challenge.
///
/// Fixed-weight support is sampled without replacement. Hoeffding's comparison
/// theorem bounds it by independent sampling with replacement from the same
/// physical coefficient population. The `+/-1` and `+/-2` support classes are
/// disjoint but ordered by the same `|A_s|`; negative association gives the
/// product of their two population means. No projected monomial is treated as
/// an independent challenge draw.
pub(super) fn log_mgf_upper(
    class: SourceClass,
    challenge: &SparseChallengeConfig,
    ring_dimension: usize,
    lambda: f64,
) -> Option<f64> {
    let dimension = ring_dimension as f64;
    let cosh_one = cosh_upper(lambda)?;
    let cosh_two = cosh_upper(round_up(2.0 * lambda))?;
    let cosh_four = cosh_upper(round_up(4.0 * lambda))?;
    let mean_pm1 = population_mean_upper(
        class,
        dimension,
        round_up(cosh_one - 1.0),
        round_up(cosh_two - 1.0),
    );
    let mean_pm2 = population_mean_upper(
        class,
        dimension,
        round_up(cosh_two - 1.0),
        round_up(cosh_four - 1.0),
    );
    let pm1 = round_up(challenge.count_pm1 as f64 * ln_upper(mean_pm1)?);
    let pm2 = round_up(challenge.count_pm2 as f64 * ln_upper(mean_pm2)?);
    Some(round_up(pm1 + pm2))
}

pub(super) fn max_log_mgf_upper(
    ring_dimension: usize,
    chunk_size: usize,
    extension_degree: usize,
    challenge: &SparseChallengeConfig,
    lambda_exponent: i32,
) -> Option<f64> {
    let key = (
        ring_dimension,
        chunk_size,
        extension_degree,
        challenge.count_pm1,
        challenge.count_pm2,
        lambda_exponent,
    );
    if let Some(cached) = MAX_LOG_MGF.lock().ok()?.get(&key) {
        return Some(*cached);
    }
    let lambda = 2f64.powi(lambda_exponent);
    let value = projected_source_classes(ring_dimension, chunk_size, extension_degree)?
        .into_iter()
        .filter_map(|class| log_mgf_upper(class, challenge, ring_dimension, lambda))
        .reduce(f64::max)?;
    MAX_LOG_MGF.lock().ok()?.insert(key, value);
    Some(value)
}

pub(super) fn deterministic_convolution_cap(
    ring_dimension: usize,
    chunk_size: usize,
    extension_degree: usize,
    challenge: &SparseChallengeConfig,
) -> Option<u128> {
    let key = (
        ring_dimension,
        chunk_size,
        extension_degree,
        challenge.count_pm1,
        challenge.count_pm2,
    );
    if let Some(cached) = DETERMINISTIC_CAP.lock().ok()?.get(&key) {
        return Some(*cached);
    }
    let challenge_l1 =
        (challenge.count_pm1 as u128).checked_add((challenge.count_pm2 as u128).checked_mul(2)?)?;
    let challenge_linf = challenge.infinity_norm() as u128;
    let value = projected_source_classes(ring_dimension, chunk_size, extension_degree)?
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

fn population_mean_upper(
    class: SourceClass,
    dimension: f64,
    magnitude_one_excess: f64,
    magnitude_two_excess: f64,
) -> f64 {
    let one = round_up(round_up(class.magnitude_one as f64 / dimension) * magnitude_one_excess);
    let two = round_up(round_up(class.magnitude_two as f64 / dimension) * magnitude_two_excess);
    round_up(round_up(1.0 + one) + two)
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

    fn contains(classes: &[SourceClass], one: usize, two: usize) -> bool {
        classes.contains(&SourceClass {
            magnitude_one: one,
            magnitude_two: two,
        })
    }

    fn brute_class(ring_dimension: usize, extension_degree: usize, hots: &[usize]) -> SourceClass {
        let mut source = vec![0i32; ring_dimension];
        if extension_degree == 1 {
            for &hot in hots {
                source[hot] += 1;
            }
        } else {
            let half_stride = ring_dimension / (2 * extension_degree);
            for &hot in hots {
                let coordinate = hot % extension_degree;
                let packed = hot / extension_degree;
                if packed < half_stride {
                    if coordinate == 0 {
                        source[packed] += 1;
                    } else {
                        source[packed + coordinate * half_stride] += 1;
                        source[packed + ring_dimension - coordinate * half_stride] -= 1;
                    }
                } else {
                    let base = ring_dimension / 2 + packed - half_stride;
                    if coordinate == 0 {
                        source[base] += 1;
                    } else {
                        source[base - coordinate * half_stride] += 1;
                        source[base + coordinate * half_stride] += 1;
                    }
                }
            }
        }
        SourceClass {
            magnitude_one: source.iter().filter(|&&value| value.abs() == 1).count(),
            magnitude_two: source.iter().filter(|&&value| value.abs() == 2).count(),
        }
    }

    fn brute_classes(
        ring_dimension: usize,
        chunk_size: usize,
        extension_degree: usize,
    ) -> Vec<SourceClass> {
        let mut classes = HashSet::new();
        if chunk_size >= ring_dimension {
            classes.insert(brute_class(ring_dimension, extension_degree, &[]));
            for hot in 0..ring_dimension {
                classes.insert(brute_class(ring_dimension, extension_degree, &[hot]));
            }
        } else {
            let chunks = ring_dimension / chunk_size;
            let radix = chunk_size + 1;
            for mut code in 0..radix.pow(chunks as u32) {
                let mut hots = Vec::new();
                for chunk in 0..chunks {
                    let choice = code % radix;
                    code /= radix;
                    if choice != chunk_size {
                        hots.push(chunk * chunk_size + choice);
                    }
                }
                classes.insert(brute_class(ring_dimension, extension_degree, &hots));
            }
        }
        pareto_frontier(classes)
    }

    #[test]
    fn direct_classes_cover_all_k_vs_d_cases() {
        assert_eq!(
            projected_source_classes(256, 256, 1).unwrap(),
            vec![SourceClass {
                magnitude_one: 1,
                magnitude_two: 0
            }]
        );
        assert_eq!(
            projected_source_classes(256, 512, 1).unwrap(),
            vec![SourceClass {
                magnitude_one: 1,
                magnitude_two: 0
            }]
        );
        assert_eq!(
            projected_source_classes(256, 16, 1).unwrap(),
            vec![SourceClass {
                magnitude_one: 16,
                magnitude_two: 0
            }]
        );
    }

    #[test]
    fn psi_classes_include_uncollided_and_coalesced_sources() {
        let k16_ext2 = projected_source_classes(256, 16, 2).unwrap();
        assert!(contains(&k16_ext2, 32, 0));
        assert!(contains(&k16_ext2, 0, 8));

        let k16_ext4 = projected_source_classes(512, 16, 4).unwrap();
        assert!(contains(&k16_ext4, 64, 0));
        assert!(contains(&k16_ext4, 0, 16));
    }

    #[test]
    fn psi_k_equal_or_larger_than_d_has_at_most_one_logical_hot() {
        for chunk_size in [256, 512] {
            assert_eq!(
                projected_source_classes(256, chunk_size, 2).unwrap(),
                vec![SourceClass {
                    magnitude_one: 2,
                    magnitude_two: 0
                }]
            );
        }
    }

    #[test]
    fn paired_slab_dp_matches_exhaustive_sources_for_every_small_k_d_case() {
        let ring_dimension = 8;
        for extension_degree in [1, 2, 4] {
            for chunk_size in [1, 2, 4, 8, 16] {
                assert_eq!(
                    projected_source_classes(ring_dimension, chunk_size, extension_degree,)
                        .unwrap(),
                    brute_classes(ring_dimension, chunk_size, extension_degree),
                    "D={ring_dimension}, K={chunk_size}, w={extension_degree}",
                );
            }
        }
    }

    #[test]
    fn replacement_mgf_dominates_exact_disjoint_fixed_weight_support() {
        let challenge = SparseChallengeConfig {
            count_pm1: 2,
            count_pm2: 1,
        };
        let class = SourceClass {
            magnitude_one: 2,
            magnitude_two: 1,
        };
        let magnitudes = [1.0f64, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
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
    fn outward_series_dominate_standard_library_values() {
        for value in [0.0, 0.125, 0.5, 1.0, 4.0, 16.0] {
            assert!(cosh_upper(value).unwrap() >= value.cosh());
        }
        for value in [1.0, 1.001, 1.5, 2.0, 17.0, 1_000_000.0] {
            assert!(ln_upper(value).unwrap() >= value.ln());
        }
    }
}
