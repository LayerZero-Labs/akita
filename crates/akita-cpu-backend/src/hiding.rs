//! Prover-side hiding randomness for schedules with randomness columns.
//!
//! Extension seam: Whiteout outer hiding randomness (akita#120). Clear
//! schedules declare zero columns and never reach the sampler, so their
//! commitments and proofs stay deterministic. Every buffer that holds sampled
//! digits is `Zeroizing`, so it is wiped when dropped.
//!
//! A commitment reused across proofs (trusted advice, a committed program)
//! may be made by a party other than the prover. [`HidingRandomness`] carries
//! its B randomness out of the committing process, and
//! [`CpuBackend::recommit_hiding`](crate::CpuBackend::recommit_hiding)
//! replays it to rebuild the same commitment.

use akita_error::AkitaError;
use rand_core::CryptoRngCore;
use std::sync::Mutex;
use zeroize::Zeroizing;

/// The B hiding randomness one commitment was made with, exported from its
/// [`CommitmentHandle`](crate::CommitmentHandle) by
/// [`hiding_randomness`](crate::CommitmentHandle::hiding_randomness).
///
/// Whoever holds it and the committed source can rebuild the commitment's
/// opening handle, or check the commitment, with
/// [`CpuBackend::recommit_hiding`](crate::CpuBackend::recommit_hiding). It is
/// as secret as the source: the digits are wiped on drop and never printed.
///
/// WARNING: committing other data under the same randomness would publish
/// the commitment to the difference of the two messages. The only consumer,
/// `recommit_hiding`, therefore accepts it only when the rebuilt commitment
/// equals an expected one, so it never yields a second commitment under
/// this randomness.
pub struct HidingRandomness {
    digits: Zeroizing<Vec<i8>>,
}

impl HidingRandomness {
    const MAGIC: [u8; 4] = *b"AKHR";
    const VERSION: u8 = 1;
    const HEADER_BYTES: usize = Self::MAGIC.len() + 1 + size_of::<u64>();

    /// Wraps the randomness a hiding commitment drew; empty for a clear one.
    pub(crate) fn from_digits(digits: &[i8]) -> Result<Self, AkitaError> {
        if digits.is_empty() {
            return Err(AkitaError::InvalidInput(
                "a clear commitment has no hiding randomness".into(),
            ));
        }
        Ok(Self {
            digits: Zeroizing::new(digits.to_vec()),
        })
    }

    /// The versioned wire form: `b"AKHR"`, version `1`, the coefficient
    /// count as a little-endian `u64`, then one two's-complement byte per
    /// coefficient.
    pub fn to_bytes(&self) -> Zeroizing<Vec<u8>> {
        let mut bytes = Zeroizing::new(Vec::with_capacity(Self::HEADER_BYTES + self.digits.len()));
        bytes.extend_from_slice(&Self::MAGIC);
        bytes.push(Self::VERSION);
        bytes.extend_from_slice(&(self.digits.len() as u64).to_le_bytes());
        bytes.extend(self.digits.iter().map(|&digit| digit.to_le_bytes()[0]));
        bytes
    }

    /// Decodes [`to_bytes`](Self::to_bytes)'s form.
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidInput`] for another magic or version, a count
    /// that disagrees with the payload length, or no coefficients. The
    /// digits' alphabet is checked against the commitment's profile when
    /// they are replayed.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AkitaError> {
        let invalid = |reason: &str| {
            AkitaError::InvalidInput(format!("hiding randomness encoding: {reason}"))
        };
        let (magic, rest) = bytes
            .split_first_chunk::<4>()
            .ok_or_else(|| invalid("truncated header"))?;
        if *magic != Self::MAGIC {
            return Err(invalid("unknown magic"));
        }
        let (&version, rest) = rest
            .split_first()
            .ok_or_else(|| invalid("truncated header"))?;
        if version != Self::VERSION {
            return Err(invalid("unsupported version"));
        }
        let (count, payload) = rest
            .split_first_chunk::<8>()
            .ok_or_else(|| invalid("truncated header"))?;
        if u64::from_le_bytes(*count) != payload.len() as u64 {
            return Err(invalid("coefficient count disagrees with the payload"));
        }
        let digits = Zeroizing::new(
            payload
                .iter()
                .map(|&byte| i8::from_le_bytes([byte]))
                .collect::<Vec<_>>(),
        );
        if digits.is_empty() {
            return Err(invalid("no coefficients"));
        }
        Ok(Self { digits })
    }
}

impl core::fmt::Debug for HidingRandomness {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HidingRandomness")
            .field("len", &self.digits.len())
            .finish_non_exhaustive()
    }
}

/// Where a backend draws hiding randomness from.
enum Draw {
    /// No source: drawing any randomness is an error.
    None,
    /// A CSPRNG; only the `zk` and `test-support` features install one.
    #[cfg_attr(not(any(feature = "zk", feature = "test-support")), allow(dead_code))]
    Rng(Box<dyn CryptoRngCore + Send>),
    /// Exported randomness, replayed once and in full; `None` once drawn.
    Replay(Option<Zeroizing<Vec<i8>>>),
}

/// The CSPRNG a backend draws hiding randomness from.
///
/// A backend owns one source; commitments and folds draw from it in the
/// order the protocol runs, so a seeded source makes a hiding proof
/// reproducible.
pub(crate) struct HidingRng(Mutex<Draw>);

impl HidingRng {
    /// The operating system CSPRNG, or no source in a build without the `zk`
    /// feature.
    pub(crate) fn os() -> Self {
        #[cfg(feature = "zk")]
        let draw = Draw::Rng(Box::new(rand_core::OsRng));
        #[cfg(not(feature = "zk"))]
        let draw = Draw::None;
        Self(Mutex::new(draw))
    }

    /// No source: drawing any randomness is an error.
    pub(crate) fn sourceless() -> Self {
        Self(Mutex::new(Draw::None))
    }

    /// A caller-provided CSPRNG.
    #[cfg(feature = "test-support")]
    pub(crate) fn new(rng: impl CryptoRngCore + Send + 'static) -> Self {
        Self(Mutex::new(Draw::Rng(Box::new(rng))))
    }

    /// A source that yields `randomness` as its one draw.
    pub(crate) fn replay(randomness: &HidingRandomness) -> Self {
        Self(Mutex::new(Draw::Replay(Some(randomness.digits.clone()))))
    }

    /// Draw `len` digits uniformly from the balanced alphabet
    /// `[-2^(log_basis-1), 2^(log_basis-1) - 1]`; a replaying source returns
    /// its digits instead.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::UnsupportedSchedule`] when randomness is
    /// requested in a build without the `zk` feature, and an input error for
    /// an alphabet the sampler does not support, or for replayed randomness
    /// of another length or outside the alphabet, or drawn twice.
    pub(crate) fn sample(
        &self,
        len: usize,
        log_basis: u32,
    ) -> Result<Zeroizing<Vec<i8>>, AkitaError> {
        if len == 0 {
            return Ok(Zeroizing::new(Vec::new()));
        }
        let mut draw = self
            .0
            .lock()
            .map_err(|_| AkitaError::Internal("hiding randomness source is poisoned".into()))?;
        match &mut *draw {
            Draw::None => Err(AkitaError::UnsupportedSchedule(
                "schedule declares hiding randomness, which needs the zk feature".into(),
            )),
            Draw::Rng(rng) => sample_hiding_digits(rng.as_mut(), len, log_basis),
            Draw::Replay(digits) => {
                let digits = digits.take().ok_or_else(|| {
                    AkitaError::InvalidInput("replayed hiding randomness drawn twice".into())
                })?;
                if digits.len() != len {
                    return Err(AkitaError::InvalidInput(format!(
                        "hiding randomness has {} coefficients; the commitment's profile draws {len}",
                        digits.len()
                    )));
                }
                if !(1..=8).contains(&log_basis) {
                    return Err(AkitaError::InvalidInput(format!(
                        "hiding randomness needs log_basis in 1..=8, got {log_basis}"
                    )));
                }
                let half = 1i16 << (log_basis - 1);
                if digits
                    .iter()
                    .any(|&digit| !(-half..half).contains(&i16::from(digit)))
                {
                    return Err(AkitaError::InvalidInput(
                        "hiding randomness lies outside the profile's digit alphabet".into(),
                    ));
                }
                Ok(digits)
            }
        }
    }

    /// Fails unless a replaying source was drawn, so replayed randomness
    /// never goes unused by a clear profile.
    pub(crate) fn ensure_replayed(&self) -> Result<(), AkitaError> {
        let draw = self
            .0
            .lock()
            .map_err(|_| AkitaError::Internal("hiding randomness source is poisoned".into()))?;
        match &*draw {
            Draw::Replay(Some(_)) => Err(AkitaError::InvalidInput(
                "the commitment's profile draws no hiding randomness".into(),
            )),
            Draw::None | Draw::Rng(_) | Draw::Replay(None) => Ok(()),
        }
    }
}

#[cfg(feature = "zk")]
fn sample_hiding_digits(
    mut rng: &mut dyn CryptoRngCore,
    len: usize,
    log_basis: u32,
) -> Result<Zeroizing<Vec<i8>>, AkitaError> {
    Ok(jolt_crypto::lattice::sample_box_digits(
        &mut rng, len, log_basis,
    )?)
}

#[cfg(not(feature = "zk"))]
fn sample_hiding_digits(
    _rng: &mut dyn CryptoRngCore,
    _len: usize,
    _log_basis: u32,
) -> Result<Zeroizing<Vec<i8>>, AkitaError> {
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
