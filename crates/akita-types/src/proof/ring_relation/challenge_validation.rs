use akita_challenges::{Challenges, SparseChallenge, SparseChallengeConfig};
use akita_error::AkitaError;
#[cfg(feature = "parallel")]
use akita_field::parallel::*;

pub(super) fn validate_packing_challenge_weights(
    challenges: &Challenges,
    config: &SparseChallengeConfig,
) -> Result<(), AkitaError> {
    let validate = |challenge: &SparseChallenge| {
        let mut count_pm1 = 0usize;
        let mut count_pm2 = 0usize;
        for &coefficient in challenge.coeffs.iter() {
            match coefficient.unsigned_abs() {
                1 => count_pm1 += 1,
                2 => count_pm2 += 1,
                _ => {
                    return Err(AkitaError::InvalidInput(
                        "coefficient-packing challenge is outside its audited family".into(),
                    ));
                }
            }
        }
        if count_pm1 != config.count_pm1 || count_pm2 != config.count_pm2 {
            return Err(AkitaError::InvalidInput(
                "coefficient-packing challenge weight disagrees with its audited family".into(),
            ));
        }
        Ok(())
    };
    #[cfg(feature = "parallel")]
    {
        let work = challenges
            .len()
            .checked_mul(config.count_pm1.saturating_add(config.count_pm2))
            .ok_or_else(|| AkitaError::InvalidSetup("challenge validation work overflow".into()))?;
        const PARALLEL_THRESHOLD: usize = 1 << 14;
        if work >= PARALLEL_THRESHOLD {
            return challenges.as_slice().par_iter().try_for_each(validate);
        }
    }
    challenges.as_slice().iter().try_for_each(validate)
}
