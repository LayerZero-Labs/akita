//! Labrador parameter-selection and security checks.

use crate::error::HachiError;
use crate::protocol::labrador::types::{LabradorReductionConfig, LabradorWitness};
use crate::{CanonicalField, FieldCore};

const LABRADOR_LOGQ: f64 = 32.0;
const LABRADOR_N: f64 = 64.0;
const LABRADOR_LOGDELTA: f64 = 0.00639138757765197; // log2(1.00444)
const LABRADOR_T: f64 = 14.0;
const LABRADOR_SLACK: f64 = 2.0;
const LABRADOR_TAU1: f64 = 32.0;
const LABRADOR_TAU2: f64 = 8.0;

/// Module-SIS security check used by the C reference.
///
/// Returns `true` when `log2(norm) < min(LOGQ, 2*sqrt(LOGQ*LOGDELTA*N)*sqrt(rank))`.
pub fn sis_secure(rank: usize, norm: f64) -> bool {
    if rank == 0 || !norm.is_finite() || norm <= 0.0 {
        return false;
    }
    let mut maxlog =
        2.0 * (LABRADOR_LOGQ * LABRADOR_LOGDELTA * LABRADOR_N).sqrt() * (rank as f64).sqrt();
    maxlog = maxlog.min(LABRADOR_LOGQ);
    norm.log2() < maxlog
}

/// Select a linear-only Labrador reduction config.
///
/// This is a simplified, Rust-native port of the C `init_proof` parameter
/// selection path with `quadratic=0` and non-tail mode.
///
/// # Errors
///
/// Returns an error if witness metadata is empty/invalid or if no secure
/// commitment ranks are found within supported bounds.
pub fn select_config<F: FieldCore + CanonicalField, const D: usize>(
    witness: &LabradorWitness<F, D>,
) -> Result<LabradorReductionConfig, HachiError> {
    if witness.rows().is_empty() {
        return Err(HachiError::InvalidInput(
            "cannot select config for empty Labrador witness".to_string(),
        ));
    }

    let row_count = witness.rows().len() as f64;
    let total_len: usize = witness.rows().iter().map(|r| r.len()).sum();
    if total_len == 0 {
        return Err(HachiError::InvalidInput(
            "cannot select config for zero-length Labrador witness".to_string(),
        ));
    }
    let nn = (total_len as f64) / row_count;
    let norm_sum: f64 = witness.norm() as f64;
    let mut varz = norm_sum / ((total_len as f64) * (D as f64));
    varz *= LABRADOR_TAU1 + 4.0 * LABRADOR_TAU2;
    varz = varz.max(1.0);

    let decompose = !sis_secure(
        13,
        6.0 * LABRADOR_T
            * LABRADOR_SLACK
            * (2.0 * (LABRADOR_TAU1 + 4.0 * LABRADOR_TAU2) * varz * nn * (D as f64)).sqrt(),
    ) || 64.0 * varz > (1u64 << 28) as f64;

    let f = if decompose { 2usize } else { 1usize };
    let mut b = if decompose {
        ((12.0f64.log2() + varz.log2()) / 4.0).round() as isize
    } else {
        ((12.0f64.log2() + varz.log2()) / 2.0).round() as isize
    };
    b = b.clamp(1, LABRADOR_LOGQ as isize);

    let fu = (((LABRADOR_LOGQ as usize) + 2 * (b as usize) / 3) / (b as usize)).max(1);
    let bu = (((LABRADOR_LOGQ as usize) + fu / 2) / fu).max(1);

    let rr = witness.rows().len() as f64;
    let mut selected: Option<(usize, usize)> = None;

    for kappa in 1..=32usize {
        let mut normsq = ((2f64.powi(2 * b as i32) / 12.0) * ((f - 1) as f64)
            + varz / 2f64.powi((2 * (f - 1) as isize * b) as i32))
            * nn;
        normsq += ((2f64.powi(2 * bu as i32) * ((fu - 1) as f64)
            + 2f64.powi((2 * ((LABRADOR_LOGQ as usize) - (fu - 1) * bu)) as i32))
            / 12.0)
            * (rr * (kappa as f64) + (rr * rr + rr) / 2.0);
        normsq *= D as f64;

        let inner_ok = sis_secure(
            kappa,
            6.0 * LABRADOR_T
                * LABRADOR_SLACK
                * 2f64.powi(((f - 1) * (b as usize)) as i32)
                * normsq.sqrt(),
        );
        if !inner_ok {
            continue;
        }

        let kappa1 = (1..=32usize).find(|&k1| sis_secure(k1, 2.0 * LABRADOR_SLACK * normsq.sqrt()));
        if let Some(k1) = kappa1 {
            selected = Some((kappa, k1));
            break;
        }
    }

    let (kappa, kappa1) = selected.ok_or_else(|| {
        HachiError::InvalidInput("failed to find secure Labrador commitment ranks".to_string())
    })?;

    Ok(LabradorReductionConfig {
        f,
        b: b as usize,
        fu,
        bu,
        kappa,
        kappa1,
        tail: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::algebra::ring::CyclotomicRing;
    use crate::FromSmallInt;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    fn row(len: usize) -> Vec<CyclotomicRing<F, D>> {
        (0..len)
            .map(|i| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|j| {
                    F::from_i64(((i + j) as i64 % 5) - 2)
                }))
            })
            .collect()
    }

    #[test]
    fn sis_secure_rejects_non_positive_norm() {
        assert!(!sis_secure(4, 0.0));
        assert!(!sis_secure(4, -1.0));
    }

    #[test]
    fn select_config_returns_valid_ranges() {
        let witness = LabradorWitness::new(vec![row(32), row(32), row(32)]);
        let cfg = select_config::<F, D>(&witness).unwrap();
        assert!(cfg.f >= 1 && cfg.f <= 2);
        assert!(cfg.b > 0);
        assert!(cfg.fu > 0);
        assert!(cfg.bu > 0);
        assert!((1..=32).contains(&cfg.kappa));
        assert!((1..=32).contains(&cfg.kappa1));
        assert!(!cfg.tail);
    }
}
