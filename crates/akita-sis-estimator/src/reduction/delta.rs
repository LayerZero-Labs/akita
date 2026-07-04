//! Root-Hermite factor helpers from lattice-estimator reduction code.

/// Upper bracket used by beta inversion in lattice-estimator.
pub const BETA_SEARCH_MAX: u32 = 1 << 16;

const SMALL_DELTA: [(u32, f64); 8] = [
    (2, 1.02190),
    (5, 1.01862),
    (10, 1.01616),
    (15, 1.01485),
    (20, 1.01420),
    (25, 1.01342),
    (28, 1.01331),
    (40, 1.01295),
];
const BETA_INVERSION_DELTA_TOLERANCE: f64 = 1e-13;

/// Compute δ from block size β, mirroring `ReductionCost._delta`.
#[must_use]
pub fn delta(beta: u32) -> f64 {
    let beta = beta.max(2);
    if beta <= 2 {
        return 1.0219;
    }
    if beta < 40 {
        for window in SMALL_DELTA.windows(2) {
            if window[1].0 > beta {
                return window[0].1;
            }
        }
        return SMALL_DELTA.last().copied().map_or(1.01295, |(_, d)| d);
    }
    if beta == 40 {
        return SMALL_DELTA.last().copied().map_or(1.01295, |(_, d)| d);
    }
    let beta_f = beta as f64;
    let pi = std::f64::consts::PI;
    let e = std::f64::consts::E;
    (beta_f / (2.0 * pi * e) * (pi * beta_f).powf(1.0 / beta_f)).powf(1.0 / (2.0 * (beta_f - 1.0)))
}

/// Invert a root-Hermite factor to the smallest supported BKZ block size.
///
/// This mirrors lattice-estimator's `ReductionCost._beta_find_root` integer
/// semantics for the SIS Euclidean path: values that would require `β < 40`
/// return `40`, while values beyond the search bracket return `None`.
#[must_use]
pub fn beta(delta_target: f64) -> Option<u32> {
    if !delta_target.is_finite() {
        return None;
    }
    if delta(40) < delta_target {
        return Some(40);
    }
    if delta_target < delta(BETA_SEARCH_MAX) {
        return None;
    }

    let mut low = 40;
    let mut high = BETA_SEARCH_MAX;
    while low < high {
        let mid = low + (high - low) / 2;
        if delta(mid) <= delta_target + BETA_INVERSION_DELTA_TOLERANCE {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    Some(low)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_matches_small_table() {
        assert!((delta(40) - 1.01295).abs() < 1e-5);
    }

    #[test]
    fn beta_inversion_matches_lattice_estimator_doctests() {
        assert_eq!(beta(1.0121), Some(50));
        assert_eq!(beta(1.0093), Some(100));
        assert_eq!(beta(1.0024), Some(808));
        assert_eq!(beta(1.000_000_000_045_374_4), None);
    }
}
