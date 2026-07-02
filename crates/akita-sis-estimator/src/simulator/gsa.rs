//! GSA shape model from lattice-estimator.

use crate::{
    error::{EstimatorError, Result},
    math::log2_biguint,
    reduction::delta::delta,
};

/// Return squared Gram-Schmidt norms for the GSA profile.
///
/// Mirrors `estimator.simulator.GSA` with `xi=1` and `tau=False`.
///
/// # Errors
///
/// Returns an error when the requested block size is outside the simulator's
/// supported range.
pub fn gsa_squared_norms(
    d: u32,
    identity_vectors: i64,
    q: &num_bigint::BigUint,
    beta: u32,
) -> Result<Vec<f64>> {
    if beta < 2 || beta > d {
        return Err(EstimatorError::InvalidParameter {
            field: "beta",
            reason: "GSA requires 2 <= beta <= effective lattice dimension".to_string(),
        });
    }

    let log_q = log2_biguint(q);
    let log_vol = (d as i64 - identity_vectors) as f64 * log_q;
    let log_delta = delta(beta).log2();
    let d_f = f64::from(d);

    Ok((0..d)
        .map(|i| {
            let r_log = (d_f - 1.0 - 2.0 * f64::from(i)) * log_delta + log_vol / d_f;
            2.0_f64.powf(2.0 * r_log)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;

    #[test]
    fn gsa_profile_matches_lattice_estimator_doctest() {
        let q = BigUint::from(31u32);
        let profile = gsa_squared_norms(12, 6, &q, 3).unwrap();
        let log2_squared: Vec<f64> = profile.iter().map(|value| value.log2()).collect();
        let expected = [
            5.641784870737893,
            5.51676876885589,
            5.391752666973886,
            5.266736565091883,
            5.14172046320988,
            5.016704361327877,
            4.891688259445873,
            4.76667215756387,
            4.6416560556818665,
            4.516639953799864,
            4.39162385191786,
            4.266607750035857,
        ];
        assert_eq!(log2_squared.len(), expected.len());
        for (actual, expected) in log2_squared.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-9,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn gsa_profile_has_expected_length() {
        let q = BigUint::from(31u32);
        let profile = gsa_squared_norms(12, 6, &q, 3).unwrap();
        assert_eq!(profile.len(), 12);
        assert!(profile
            .iter()
            .all(|value| value.is_finite() && *value > 0.0));
    }
}
