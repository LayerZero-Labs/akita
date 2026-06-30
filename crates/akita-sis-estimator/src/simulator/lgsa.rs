//! LGSA shape model from lattice-estimator.

use crate::{
    error::{EstimatorError, Result},
    math::log2_biguint,
    reduction::delta::delta,
};

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
}
