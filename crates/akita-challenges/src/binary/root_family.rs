//! The fixed binary root family selected by an ordinary group draw.

use super::{BinaryChallengeProfile, BinaryChallengeSampler, BinaryScalarRing};
use jolt_crypto::challenge::SparseChallenge;
use akita_error::AkitaError;

pub const IDENTITY: &[u8] = b"\x01akita/binary-root/phi243/bounded46/shake256-v1";
pub const WEIGHT: usize = 46;

pub(crate) fn profile() -> Result<BinaryChallengeProfile, AkitaError> {
    BinaryChallengeProfile::bounded_weight(BinaryScalarRing::Cyclotomic243, WEIGHT)
}

pub fn validate(degree: usize, required_bits: u32) -> Result<(), &'static str> {
    if degree != BinaryScalarRing::Cyclotomic243.degree() {
        return Err("binary root challenges require the degree-162 Phi243 scalar ring");
    }
    if !profile().is_ok_and(|profile| profile.meets_budget(1, required_bits)) {
        return Err("binary root family has insufficient per-draw entropy");
    }
    Ok(())
}

pub fn entropy_bits(degree: usize) -> f64 {
    if degree != BinaryScalarRing::Cyclotomic243.degree() {
        return f64::NEG_INFINITY;
    }
    profile().map_or(f64::NEG_INFINITY, |profile| {
        profile
            .cardinality()
            .to_bytes_be()
            .iter()
            .fold(0.0f64, |n, byte| n * 256.0 + f64::from(*byte))
            .log2()
    })
}

pub fn sample(seed: &[u8; 32], count: usize) -> Result<Vec<SparseChallenge>, AkitaError> {
    // Keep verifier-reachable materialization bounded independently of the
    // schedule's own admission. This permits up to one million root columns.
    if count > (1 << 20) {
        return Err(AkitaError::InvalidInput(
            "binary challenge batch exceeds materialization budget".into(),
        ));
    }
    let binary = BinaryChallengeSampler::new(profile()?).sample_from_seed(seed, count)?;
    let mut sampled = Vec::new();
    sampled
        .try_reserve_exact(count)
        .map_err(|_| AkitaError::InvalidInput("binary challenge allocation failed".into()))?;
    for challenge in binary {
        let mut positions = jolt_crypto::challenge::SparseChallengePositions::new();
        let mut coeffs = jolt_crypto::challenge::SparseChallengeCoefficients::new();
        positions
            .try_reserve_exact(challenge.weight())
            .map_err(|_| {
                AkitaError::InvalidInput("binary challenge positions allocation failed".into())
            })?;
        coeffs.try_reserve_exact(challenge.weight()).map_err(|_| {
            AkitaError::InvalidInput("binary challenge coefficients allocation failed".into())
        })?;
        for term in challenge.terms() {
            positions.push(u32::from(term.position));
            coeffs.push(term.coefficient);
        }
        sampled.push(SparseChallenge { positions, coeffs });
    }
    Ok(sampled)
}

#[cfg(test)]
mod tests;
