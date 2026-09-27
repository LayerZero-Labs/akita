//! Versioned public identities for callers that bind Akita inputs early.

use crate::{derive_transcript_grinding_plan, CommitmentConfig};
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::{
    digest_descriptor_bytes, digest_serializable, AkitaVerifierSetup, AlgebraSection, FoldSchedule,
    OpeningClaimsLayout, OpeningScheduleSelection, SetupSection,
};
use jolt_field::CanonicalEncoding;

/// Identify the complete public commitment key, including installed prefix commitments.
///
/// Returns Blake2b-256 of the ASCII domain `akita/public-setup/v1`, followed by
/// canonical uncompressed `AlgebraSection` and `SetupSection` bytes, followed by
/// Blake2b-256 of the canonical uncompressed verifier setup. The latter includes
/// provisioning bounds, the derivation tag and seed, materialized matrix, and
/// ordered prefix registry. Prepared caches are excluded. This identifies the
/// supplied key; it does not validate its provenance.
///
/// Callers may absorb the returned 32 bytes before their first challenge.
pub fn public_setup_identity<Cfg: CommitmentConfig>(
    setup: &AkitaVerifierSetup<Cfg::Field>,
) -> Result<[u8; 32], AkitaError>
where
    Cfg::Field: CanonicalEncoding + AkitaSerialize,
{
    let mut bytes = b"akita/public-setup/v1".to_vec();
    let encode_error = |error| AkitaError::InvalidSetup(format!("public setup identity: {error}"));
    AlgebraSection::for_fields::<Cfg::Field, Cfg::ExtField>()?
        .serialize_uncompressed(&mut bytes)
        .map_err(encode_error)?;
    SetupSection::from_parts(
        Cfg::decomposition(),
        Cfg::sis_modulus_profile(),
        &setup.expanded().descriptor.setup_seed,
    )
    .map_err(encode_error)?
    .serialize_uncompressed(&mut bytes)
    .map_err(encode_error)?;
    bytes.extend_from_slice(&digest_serializable(setup).map_err(encode_error)?);
    Ok(digest_descriptor_bytes(&bytes))
}

/// Identify a selected schedule and its canonical, call-specific grinding plan.
///
/// Returns Blake2b-256 of ASCII `akita/public-schedule/v1`, canonical uncompressed
/// `AlgebraSection`, the 32-byte selected row digest, then two length-prefixed
/// blobs: `FoldSchedule::canonical_descriptor_bytes()` and
/// `GrindingPlan::canonical_bytes()`. Each length is a little-endian `u64`.
/// The plan is derived from `layout` and includes its policy version and exact
/// field cardinality. No runtime randomness, cache state, or Rust layout enters
/// this identity. This does not establish catalog admission for `selection`.
pub fn public_schedule_identity<Cfg: CommitmentConfig>(
    selection: OpeningScheduleSelection,
    schedule: &FoldSchedule,
    layout: &OpeningClaimsLayout,
) -> Result<[u8; 32], AkitaError>
where
    Cfg::Field: CanonicalEncoding,
{
    for fold in std::iter::once(&schedule.root).chain(&schedule.recursive_folds) {
        fold.params.witness_chunk.validate()?;
    }
    let plan = derive_transcript_grinding_plan::<Cfg>(schedule, layout)?;
    let mut bytes = b"akita/public-schedule/v1".to_vec();
    AlgebraSection::for_fields::<Cfg::Field, Cfg::ExtField>()?
        .serialize_uncompressed(&mut bytes)
        .map_err(|error| AkitaError::InvalidSetup(format!("schedule identity: {error}")))?;
    bytes.extend_from_slice(selection.row_digest.as_bytes());
    for encoded in [
        schedule.canonical_descriptor_bytes(),
        plan.canonical_bytes()?,
    ] {
        let len = u64::try_from(encoded.len())
            .map_err(|_| AkitaError::InvalidSetup("identity length exceeds u64".into()))?;
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(&encoded);
    }
    Ok(digest_descriptor_bytes(&bytes))
}

#[cfg(test)]
#[path = "public_identity_tests.rs"]
mod tests;
