//! Proof-prime admission: nonzero challenge differences are units.
//!
//! Extraction divides by a difference `s = c - c'` of two distinct fold
//! challenges in the scalar ring `Z[x]/(x^deg + x^(deg/2) + 1)`, reduced
//! modulo the proof prime `P`. That ring is the ring of integers of the
//! cyclotomic field of conductor `n = 3 * deg / 2`, a power of three.
//!
//! Norm argument. The trace of `x^t` is `deg` for `t = 0`, `-deg/2` for
//! `t = +-deg/2`, and zero for every other `|t| < deg`. The squared absolute
//! values of the `deg` embeddings of `s` therefore sum to
//! `deg * (||s||^2 - sum_{a < deg/2} s_a * s_{a + deg/2}) <= 1.5 * deg * ||s||^2`,
//! and the inequality of arithmetic and geometric means gives
//! `|N(s)| <= (1.5 * ||s||^2)^(deg/2)`. Both challenges have squared norm at
//! most `w`, so `||s||^2 <= 4w` and `|N(s)| <= (6w)^(deg/2)`. A prime `P != 3`
//! is unramified, and every prime ideal above it has norm `P^f`, where `f` is
//! the multiplicative order of `P` modulo `n`. A difference that is not a unit
//! modulo `P` lies in one of those ideals, so `P^f` divides the nonzero integer
//! `N(s)`. Hence `P^f > (6w)^(deg/2)` makes every nonzero difference a unit.
//!
//! The check compares bit lengths. That is sufficient and uses exact integers:
//! `P^f >= 2^(f * (bits(P) - 1))` and `(6w)^(deg/2) < 2^((deg/2) * bits(6w))`.

use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing};
use akita_error::AkitaError;

/// Residue degree `f` of `proof_prime` in the scalar ring: its multiplicative
/// order modulo the conductor. `None` when three divides `proof_prime`.
pub fn labinius_proof_prime_residue_degree(
    proof_prime: u128,
    scalar_ring: BinaryScalarRing,
) -> Option<u32> {
    let degree = u32::try_from(scalar_ring.degree()).ok()?;
    let conductor = u128::from(degree).checked_mul(3)? / 2;
    let residue = proof_prime % conductor;
    let mut power = residue;
    // The order divides the group order `deg`; both factors are below `n`.
    for order in 1..=degree {
        if power == 1 {
            return Some(order);
        }
        power = power.checked_mul(residue)? % conductor;
    }
    None
}

/// Admit the prime `proof_prime` for `challenge`: accept only when the bit
/// lengths prove `P^f > (6w)^(deg/2)`, so that every nonzero difference of two
/// challenges is a unit modulo `proof_prime`.
///
/// Primality of `proof_prime` is the caller's premise and is not tested.
pub fn check_labinius_proof_prime_units(
    proof_prime: u128,
    challenge: &BinaryChallengeProfile,
) -> Result<(), AkitaError> {
    let scalar_ring = challenge.scalar_ring();
    let residue_degree =
        labinius_proof_prime_residue_degree(proof_prime, scalar_ring).ok_or_else(|| {
            AkitaError::InvalidSetup("LaBinius proof prime must be coprime to three".into())
        })?;
    let bits = |value: u128| u64::from(u128::BITS - value.leading_zeros());
    let difference_bound = 6 * u128::from(challenge.coefficient_l2_squared_bound());
    let prime_exponent = bits(proof_prime)
        .checked_sub(1)
        .and_then(|log| log.checked_mul(u64::from(residue_degree)));
    let norm_exponent = u64::try_from(scalar_ring.degree() / 2)
        .ok()
        .and_then(|half_degree| half_degree.checked_mul(bits(difference_bound)));
    match prime_exponent.zip(norm_exponent) {
        Some((prime_exponent, norm_exponent)) if prime_exponent >= norm_exponent => Ok(()),
        _ => Err(AkitaError::InvalidSetup(format!(
            "LaBinius proof prime of residue degree {residue_degree} does not make \
             challenge differences units"
        ))),
    }
}

#[cfg(test)]
#[path = "proof_prime_tests.rs"]
mod tests;
