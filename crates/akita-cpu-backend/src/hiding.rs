//! Prover-side hiding randomness for schedules with randomness columns.
//!
//! Extension seam: Whiteout outer hiding randomness (akita#120). Clear
//! schedules declare zero columns and never reach the sampler, so their
//! commitments and proofs stay deterministic.

use akita_error::AkitaError;
use rand_core::CryptoRngCore;
use std::sync::Mutex;

/// The CSPRNG a backend draws hiding randomness from.
///
/// A backend owns one source; commitments and folds draw from it in the
/// order the protocol runs, so a seeded source makes a hiding proof
/// reproducible.
pub(crate) struct HidingRng(Mutex<Option<Box<dyn CryptoRngCore + Send>>>);

impl HidingRng {
    /// The operating system CSPRNG, or no source in a build without the `zk`
    /// feature.
    pub(crate) fn os() -> Self {
        #[cfg(feature = "zk")]
        let rng: Option<Box<dyn CryptoRngCore + Send>> = Some(Box::new(rand_core::OsRng));
        #[cfg(not(feature = "zk"))]
        let rng = None;
        Self(Mutex::new(rng))
    }

    /// No source: drawing any randomness is an error.
    pub(crate) fn sourceless() -> Self {
        Self(Mutex::new(None))
    }

    /// A caller-provided CSPRNG.
    #[cfg(feature = "test-support")]
    pub(crate) fn new(rng: impl CryptoRngCore + Send + 'static) -> Self {
        Self(Mutex::new(Some(Box::new(rng))))
    }

    /// Draw `len` digits uniformly from the balanced alphabet
    /// `[-2^(log_basis-1), 2^(log_basis-1) - 1]`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::UnsupportedSchedule`] when randomness is
    /// requested in a build without the `zk` feature, and an input error for
    /// an alphabet the sampler does not support.
    pub(crate) fn sample(&self, len: usize, log_basis: u32) -> Result<Vec<i8>, AkitaError> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let mut rng = self
            .0
            .lock()
            .map_err(|_| AkitaError::Internal("hiding randomness source is poisoned".into()))?;
        let rng = rng.as_mut().ok_or_else(|| {
            AkitaError::UnsupportedSchedule(
                "schedule declares hiding randomness, which needs the zk feature".into(),
            )
        })?;
        sample_hiding_digits(rng.as_mut(), len, log_basis)
    }
}

#[cfg(feature = "zk")]
fn sample_hiding_digits(
    mut rng: &mut dyn CryptoRngCore,
    len: usize,
    log_basis: u32,
) -> Result<Vec<i8>, AkitaError> {
    Ok(jolt_crypto::lattice::sample_box_digits(
        &mut rng, len, log_basis,
    )?)
}

#[cfg(not(feature = "zk"))]
fn sample_hiding_digits(
    _rng: &mut dyn CryptoRngCore,
    _len: usize,
    _log_basis: u32,
) -> Result<Vec<i8>, AkitaError> {
    Err(AkitaError::UnsupportedSchedule(
        "schedule declares hiding randomness, which needs the zk feature".into(),
    ))
}

#[cfg(all(test, feature = "zk"))]
mod tests {
    use super::*;

    #[test]
    fn samples_cover_exactly_the_balanced_alphabet() {
        let rng = HidingRng::os();
        let digits = rng.sample(4096, 3).unwrap();
        assert_eq!(digits.len(), 4096);
        assert!(digits.iter().all(|digit| (-4..=3).contains(digit)));
        assert!(digits.contains(&-4) && digits.contains(&3));
        assert!(rng.sample(0, 3).unwrap().is_empty());
    }
}
