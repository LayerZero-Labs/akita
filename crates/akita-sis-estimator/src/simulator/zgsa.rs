//! ZGSA shape model from lattice-estimator.

use libm::lgamma;
use num_bigint::BigUint;

use crate::{
    error::{EstimatorError, Result},
    math::log2_biguint,
    simulator::zgsa_tables::{GH_CONSTANT, SMALL_SLOPE_T8},
};

const XI: f64 = 1.0;

/// Return squared Gram-Schmidt norms for the ZGSA profile.
///
/// Mirrors `estimator.simulator.ZGSA` with `xi=1` and `tau=False`.
///
/// # Errors
///
/// Returns an error when the requested block size is outside the simulator's
/// supported range.
pub fn zgsa_squared_norms(
    d: u32,
    identity_vectors: i64,
    q: &BigUint,
    beta: u32,
) -> Result<Vec<f64>> {
    if beta < 2 || beta > d {
        return Err(EstimatorError::InvalidParameter {
            field: "beta",
            reason: "ZGSA requires 2 <= beta <= effective lattice dimension".to_string(),
        });
    }

    let n = identity_vectors;
    let d_i = i64::from(d);
    if n < 0 || n > d_i {
        return Err(EstimatorError::InvalidParameter {
            field: "identity_vectors",
            reason: "ZGSA requires 0 <= identity_vectors <= effective lattice dimension"
                .to_string(),
        });
    }

    let log_q = log2_biguint(q) * std::f64::consts::LN_2;
    let log_xi = XI.ln();
    let num_q_vec = d_i - n;
    let mut l_log = vec![0.0_f64; d as usize];
    for entry in &mut l_log[..num_q_vec as usize] {
        *entry = log_q;
    }
    for entry in &mut l_log[num_q_vec as usize..] {
        *entry = log_xi;
    }

    let slope = zgsa_slope(beta);
    let mut diff = slope * 0.5;
    let midpoint = 0.5 * (log_q + log_xi);

    // Every smoothing step must replace one q-vector and one identity vector
    // symmetrically around the midpoint. Iterating past the smaller zone would
    // update only one side and change the profile determinant.
    for i in 0..num_q_vec.min(n) {
        if diff > 0.5 * (log_q - log_xi) {
            break;
        }
        let low = num_q_vec - i - 1;
        let high = num_q_vec + i;
        if low >= 0 {
            l_log[low as usize] = midpoint + diff;
        }
        if high < d_i {
            l_log[high as usize] = midpoint - diff;
        }
        diff += slope;
    }

    l_log.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(l_log.into_iter().map(|entry| (2.0 * entry).exp()).collect())
}

fn zgsa_slope(beta: u32) -> f64 {
    if beta <= 60 {
        return SMALL_SLOPE_T8[(beta - 2) as usize];
    }
    if beta <= 70 {
        let ratio = (70.0 - f64::from(beta)) / 10.0;
        return ratio * SMALL_SLOPE_T8[58] + (1.0 - ratio) * 2.0 * zgsa_delta(70).ln();
    }
    2.0 * zgsa_delta(beta).ln()
}

fn zgsa_delta(k: u32) -> f64 {
    debug_assert!(k >= 60);
    (log_gh(f64::from(k)) / f64::from(k - 1)).exp()
}

fn log_gh(d: f64) -> f64 {
    if d < 49.0 {
        GH_CONSTANT[(d as usize) - 1]
    } else {
        (0.0 - ball_log_vol(d)) / d
    }
}

fn ball_log_vol(n: f64) -> f64 {
    let half = n * 0.5;
    half * std::f64::consts::PI.ln() - lgamma(half + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;

    #[test]
    fn zgsa_profile_matches_lattice_estimator_smoke_cells() {
        let cases = [
            (
                213_u32,
                128_i64,
                BigUint::from(2048_u32),
                40_u32,
                17.512310987622374_f64,
            ),
            (
                64_u32,
                32_i64,
                BigUint::from(4_294_967_197_u64),
                63_u32,
                34.071596469471416_f64,
            ),
        ];
        for (d, n, q, beta, expected_r0_log2) in cases {
            let profile = zgsa_squared_norms(d, n, &q, beta).unwrap();
            assert_eq!(profile.len(), d as usize);
            let r0_log2 = profile[0].log2();
            assert!(
                (r0_log2 - expected_r0_log2).abs() < 1e-9,
                "d={d} beta={beta}: actual={r0_log2}, expected={expected_r0_log2}"
            );
        }
    }

    #[test]
    fn zgsa_q_vector_majority_preserves_volume() {
        let q = BigUint::from(18_446_744_073_709_551_557_u64);
        let profile = zgsa_squared_norms(1_810, 786, &q, 494).unwrap();
        let actual_log2_volume = profile.iter().map(|value| 0.5 * value.log2()).sum::<f64>();
        let expected_log2_volume = 1_024.0 * log2_biguint(&q);

        assert!(
            (actual_log2_volume - expected_log2_volume).abs()
                <= 1e-10 * expected_log2_volume.abs().max(1.0),
            "actual={actual_log2_volume}, expected={expected_log2_volume}"
        );
        let exact_q_prefix = profile
            .iter()
            .take_while(|value| (0.5 * value.log2() - log2_biguint(&q)).abs() < 1e-12)
            .count();
        assert_eq!(exact_q_prefix, 238);
    }

    #[test]
    fn zgsa_preserves_qary_volume_on_supported_profiles() {
        let cases = [
            (64_u32, 32_i64, BigUint::from(4_294_967_197_u64), 63_u32),
            (
                4_096_u32,
                3_072_i64,
                (BigUint::from(1_u8) << 128) - BigUint::from(4_294_944_759_u64),
                651_u32,
            ),
        ];
        for (d, identity_vectors, q, beta) in cases {
            let profile = zgsa_squared_norms(d, identity_vectors, &q, beta).unwrap();
            let actual_log2_volume = profile.iter().map(|value| 0.5 * value.log2()).sum::<f64>();
            let q_vectors = i64::from(d) - identity_vectors;
            let expected_log2_volume = q_vectors as f64 * log2_biguint(&q);
            assert!(
                (actual_log2_volume - expected_log2_volume).abs()
                    <= 1e-10 * expected_log2_volume.abs().max(1.0),
                "d={d} beta={beta}: actual={actual_log2_volume}, expected={expected_log2_volume}"
            );
        }
    }
}
