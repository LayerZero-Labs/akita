//! Reusable transport for a statement-bound nested Akita opening.
//!
//! The caller binds its complete statement and Akita instance descriptor before
//! deriving the session. Scalar and grouped openings share this framing.

use akita_config::{policy_of, CommitmentConfig, ResolvedScheduleRow};
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::channel::ClearChannel;
use akita_params::ScheduleLookupKey;
use akita_serialization::AkitaSerialize;

pub(crate) fn canonical_bytes<T: AkitaSerialize>(value: &T) -> Result<Vec<u8>, AkitaError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(value.compressed_size())
        .map_err(|_| AkitaError::InvalidProof)?;
    value
        .serialize_compressed(&mut bytes)
        .map_err(|_| AkitaError::InvalidProof)?;
    Ok(bytes)
}

/// Absorb `u64_le(byte_length(bytes)) || bytes` as public data.
pub fn public_length_prefixed<S: ClearChannel>(
    channel: &mut S,
    bytes: &[u8],
) -> Result<(), AkitaError> {
    let len = u64::try_from(bytes.len())
        .map_err(|_| AkitaError::InvalidInput("public byte length exceeds u64".into()))?;
    channel.public(&len.to_le_bytes())?;
    channel.public(bytes)
}

/// A parent-state-derived session with an expanded-schedule parser bound.
///
/// This carries no channel ownership. Each call sends or receives one complete
/// inner proof through the parent, leaving EOF and later challenges to it.
pub struct NestedOpeningSession {
    session: Vec<u8>,
    proof_bound: usize,
}

impl NestedOpeningSession {
    /// Bind the session domain and draw exactly one parent challenge block.
    ///
    /// The supplied row must already be resolved from the caller's trusted
    /// catalog, and the complete statement and instance must already be public.
    /// The bound uses the exact expanded schedule, never a planner estimate.
    pub fn derive<Cfg: CommitmentConfig, S: ClearChannel>(
        domain: &[u8],
        row: &ResolvedScheduleRow,
        channel: &mut S,
    ) -> Result<Self, AkitaError> {
        let profiles = row.profiles();
        let mut precommitteds = Vec::new();
        precommitteds
            .try_reserve_exact(profiles.precommitteds.len())
            .map_err(|_| AkitaError::InvalidSetup("opening-key allocation failed".into()))?;
        precommitteds.extend_from_slice(&profiles.precommitteds);
        let key = ScheduleLookupKey {
            final_group: profiles.final_group.group,
            precommitteds,
        };
        let proof_bound = akita_schedules::expanded_schedule_proof_bound(
            &key,
            row.schedule(),
            &policy_of::<Cfg>(),
        )?;
        let capacity = checked::sum([domain.len(), 32])
            .ok_or_else(|| AkitaError::InvalidInput("session length overflow".into()))?;
        let mut session = Vec::new();
        session
            .try_reserve_exact(capacity)
            .map_err(|_| AkitaError::InvalidInput("session allocation failed".into()))?;
        public_length_prefixed(channel, domain)?;
        let seed = channel.challenge_block()?;
        session.extend_from_slice(domain);
        session.extend_from_slice(&seed);
        Ok(Self {
            session,
            proof_bound,
        })
    }

    /// Conservative maximum inner proof length for this resolved row.
    pub fn proof_bound(&self) -> usize {
        self.proof_bound
    }

    /// Prove under the nested session and send its bounded length and payload.
    pub fn prove<S: ClearChannel>(
        &self,
        channel: &mut S,
        prove: impl FnOnce(&[u8]) -> Result<Vec<u8>, AkitaError>,
    ) -> Result<(), AkitaError> {
        let mut proof = prove(&self.session)?;
        if proof.is_empty() || proof.len() > self.proof_bound {
            return Err(AkitaError::InvalidProof);
        }
        let mut length = u64::try_from(proof.len())
            .map_err(|_| AkitaError::InvalidProof)?
            .to_le_bytes();
        channel.message(&mut length)?;
        channel.message(&mut proof)
    }

    /// Read a checked length, allocate fallibly, and verify the exact payload.
    ///
    /// Framing failures return `InvalidProof`. The inner verifier's error is
    /// preserved so setup and catalog mistakes retain their native diagnosis.
    pub fn verify<S: ClearChannel>(
        &self,
        channel: &mut S,
        verify: impl FnOnce(&[u8], &[u8]) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        let mut length = [0u8; 8];
        channel
            .message(&mut length)
            .map_err(|_| AkitaError::InvalidProof)?;
        let length =
            usize::try_from(u64::from_le_bytes(length)).map_err(|_| AkitaError::InvalidProof)?;
        if length == 0 || length > self.proof_bound {
            return Err(AkitaError::InvalidProof);
        }
        let mut proof = Vec::new();
        proof
            .try_reserve_exact(length)
            .map_err(|_| AkitaError::InvalidProof)?;
        proof.resize(length, 0);
        channel
            .message(&mut proof)
            .map_err(|_| AkitaError::InvalidProof)?;
        verify(&proof, &self.session)
    }
}
