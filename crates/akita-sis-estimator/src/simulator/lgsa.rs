//! LGSA shape model from lattice-estimator.

use crate::{
    error::{EstimatorError, Result},
    math::log2_biguint,
    reduction::delta::delta,
};

/// Compact facts about an LGSA profile needed by the infinity probability path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LgsaSummary {
    /// Effective lattice dimension.
    pub effective_dimension: u64,
    /// First squared Gram-Schmidt norm.
    pub first_squared_norm: f64,
    /// Base-2 logarithm of the first Gram-Schmidt length.
    pub first_log2_norm: f64,
    /// Dilithium-style q-vector prefix length.
    pub idx_start: u64,
    /// Last coordinate whose Gram-Schmidt length is materially above one.
    pub idx_end: u64,
    /// Gram-Schmidt length at `idx_start`.
    pub vector_length_at_idx_start: f64,
    /// Base-2 logarithm of the Gram-Schmidt length at `idx_start`.
    pub log2_vector_length_at_idx_start: f64,
}

/// Smallest dimension at which the LGSA profile has its full GSA prefix.
///
/// For larger dimensions the non-unit part of the profile is unchanged: the
/// extra coordinates are unit vectors. This is the finite transition point
/// used by the proven-pruned zeta optimizer.
pub(crate) fn lgsa_stable_dimension(n: u64, q: &num_bigint::BigUint, beta: u32) -> Result<u64> {
    if beta < 2 {
        return Err(EstimatorError::InvalidParameter {
            field: "beta",
            reason: "LGSA requires beta >= 2".to_string(),
        });
    }

    let log_vol = n as f64 * log2_biguint(q);
    let step = 2.0 * delta(beta).log2();
    let mut count = if log_vol <= 0.0 {
        1
    } else {
        (((1.0 + 8.0 * log_vol / step).sqrt() - 1.0) / 2.0).floor() as u64
    }
    .max(1);
    let profile_sum = |candidate: u64| step * candidate as f64 * (candidate as f64 + 1.0) / 2.0;
    while profile_sum(count) <= log_vol {
        count = count
            .checked_add(1)
            .ok_or(EstimatorError::InvalidParameter {
                field: "n",
                reason: "LGSA stable dimension overflow".to_string(),
            })?;
    }
    while count > 1 && profile_sum(count - 1) > log_vol {
        count -= 1;
    }
    Ok(count)
}

/// Return squared Gram-Schmidt norms for the LGSA profile.
///
/// Mirrors `estimator.simulator.LGSA` with `xi=1` and `tau=False`.
///
/// # Errors
///
/// Returns an error when the requested block size is outside the simulator's
/// supported range.
pub fn lgsa_squared_norms(
    d: u32,
    identity_vectors: i64,
    q: &num_bigint::BigUint,
    beta: u32,
) -> Result<Vec<f64>> {
    if beta < 2 || beta > d {
        return Err(EstimatorError::InvalidParameter {
            field: "beta",
            reason: "LGSA requires 2 <= beta <= effective lattice dimension".to_string(),
        });
    }

    let log_q = log2_biguint(q);
    let log_vol = (d as i64 - identity_vectors) as f64 * log_q;
    let mut r_log = vec![0.0_f64; d as usize];
    let mut profile_log_vol = 0.0_f64;

    let slope = -2.0 * delta(beta).log2();
    let mut log_vec_len = 0.0_f64;
    let mut break_index = 0usize;
    for i in (0..d).rev() {
        log_vec_len -= slope;
        profile_log_vol += log_vec_len;
        r_log[i as usize] += log_vec_len;
        if profile_log_vol > log_vol {
            break_index = i as usize;
            break;
        }
    }

    let num_gsa_vec = d as usize - break_index;
    r_log.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    profile_log_vol = r_log.iter().sum();
    let diff = profile_log_vol - log_vol;
    if num_gsa_vec > 0 {
        let shift = diff / num_gsa_vec as f64;
        for entry in &mut r_log[..num_gsa_vec] {
            *entry -= shift;
        }
    }

    Ok(r_log
        .into_iter()
        .map(|entry| 2.0_f64.powf(2.0 * entry))
        .collect())
}

/// Return compact LGSA profile facts without allocating one entry per dimension.
///
/// Mirrors [`lgsa_squared_norms`] for the values consumed by the infinity-norm
/// probability calculation.
///
/// # Errors
///
/// Returns an error when the requested block size is outside the simulator's
/// supported range.
pub fn lgsa_summary(
    d: u64,
    identity_vectors: i128,
    q: &num_bigint::BigUint,
    beta: u32,
) -> Result<LgsaSummary> {
    if beta < 2 || u64::from(beta) > d {
        return Err(EstimatorError::InvalidParameter {
            field: "beta",
            reason: "LGSA requires 2 <= beta <= effective lattice dimension".to_string(),
        });
    }

    let log_q = log2_biguint(q);
    let n = u64::try_from(d as i128 - identity_vectors).map_err(|_| {
        EstimatorError::InvalidParameter {
            field: "n",
            reason: "LGSA q-ary dimension must be non-negative".to_string(),
        }
    })?;
    let log_vol = n as f64 * log_q;
    let step = 2.0 * delta(beta).log2();
    // The loop in the dense reference implementation sums the arithmetic
    // progression `step * k` until it exceeds `log_vol`. Solve the quadratic
    // directly, then correct the rounded candidate by at most a few steps.
    // This keeps the compact path genuinely compact when `m` is very large.
    let num_gsa_vec = lgsa_stable_dimension(n, q, beta)?.min(d);
    let profile_sum = |count: u64| step * count as f64 * (count as f64 + 1.0) / 2.0;
    let profile_log_vol = profile_sum(num_gsa_vec);

    let shift = if num_gsa_vec > 0 {
        (profile_log_vol - log_vol) / num_gsa_vec as f64
    } else {
        0.0
    };
    let log_norm_at = |index: u64| -> f64 {
        if index < num_gsa_vec {
            (num_gsa_vec - index) as f64 * step - shift
        } else {
            0.0
        }
    };

    let first_log_norm = log_norm_at(0);
    let first_length = 2.0_f64.powf(first_log_norm);
    let q_f = 2.0_f64.powf(log_q);
    let idx_start = if (first_length - q_f).abs() < 1e-8 && num_gsa_vec > 1 {
        1
    } else {
        0
    };
    let unit_threshold = (1.0_f64 + 1e-8).log2();
    let positive_count = if step > 0.0 {
        let first_positive = ((unit_threshold + shift) / step).floor() as u64 + 1;
        if first_positive > num_gsa_vec {
            0
        } else {
            num_gsa_vec - first_positive + 1
        }
    } else {
        0
    };
    let idx_end = positive_count
        .checked_sub(1)
        .unwrap_or_else(|| d.saturating_sub(1));
    let idx_start = idx_start.min(d.saturating_sub(1));
    let log2_vector_length_at_idx_start = log_norm_at(idx_start);
    let vector_length_at_idx_start = 2.0_f64.powf(log2_vector_length_at_idx_start);

    Ok(LgsaSummary {
        effective_dimension: d,
        first_squared_norm: 2.0_f64.powf(2.0 * first_log_norm),
        first_log2_norm: first_log_norm,
        idx_start,
        idx_end,
        vector_length_at_idx_start,
        log2_vector_length_at_idx_start,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;

    #[test]
    fn lgsa_profile_has_expected_length() {
        let q = BigUint::from(31u32);
        let profile = lgsa_squared_norms(12, 6, &q, 3).unwrap();
        assert_eq!(profile.len(), 12);
        assert!(profile
            .iter()
            .all(|value| value.is_finite() && *value > 0.0));
    }

    #[test]
    fn lgsa_summary_matches_dense_profile_observables() {
        let q = BigUint::from(4_294_967_197u64);
        let profile = lgsa_squared_norms(96, 64, &q, 40).unwrap();
        let summary = lgsa_summary(96, 64, &q, 40).unwrap();
        assert!((profile[0] - summary.first_squared_norm).abs() <= 1e-8 * profile[0]);
        assert!(
            (profile[0].sqrt().log2() - summary.first_log2_norm).abs()
                <= 1e-8 * profile[0].sqrt().log2().abs().max(1.0)
        );

        let q_f = log2_biguint(&q).exp2();
        let idx_start = if (profile[0].sqrt() - q_f).abs() < 1e-8 {
            profile
                .iter()
                .position(|value| *value < profile[0])
                .unwrap_or(0)
        } else {
            0
        };
        let idx_end = profile
            .iter()
            .rposition(|value| value.sqrt() > 1.0 + 1e-8)
            .unwrap_or(profile.len() - 1);
        assert_eq!(summary.idx_start, idx_start as u64);
        assert_eq!(summary.idx_end, idx_end as u64);
        assert!(
            (profile[idx_start].sqrt() - summary.vector_length_at_idx_start).abs()
                <= 1e-8 * profile[idx_start].sqrt()
        );
        assert!(
            (profile[idx_start].sqrt().log2() - summary.log2_vector_length_at_idx_start).abs()
                <= 1e-8 * profile[idx_start].sqrt().log2().abs().max(1.0)
        );
    }
}
