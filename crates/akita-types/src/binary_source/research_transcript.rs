//! Bind every source, owner, terminal and layout before the joint D barrier.

use akita_error::AkitaError;
use akita_params::GroupCommitPhaseParams;
use akita_transcript::{ProofChannel, ProtocolSiteId, SITE_FAMILY_ROOT_STATEMENT};
use jolt_field::{CanonicalEncoding, ExtField, Field};

use crate::opening_claims::GroupClaimKind;
use crate::OpeningClaims;

/// Explicit research wire identity: direct setup, joint retry, base S11a points.
/// The caller sends the concatenated rank-sized D payload only after this call.
/// The sole response nonce follows that complete payload on both sides.
pub fn research_batch_statement<F, E, C>(
    state: &mut impl ProofChannel,
    claims: &OpeningClaims<'_, E, C>,
    profiles: &[GroupCommitPhaseParams],
    payload: impl for<'a> Fn(&'a C) -> &'a [F],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    if claims.num_groups() != 8 || profiles.len() != 8 {
        return Err(AkitaError::InvalidProof);
    }
    akita_params::binary_setup::BinaryFieldProfile::P64
        .check_field_pair(akita_params::field_modulus::<F>()?, E::DEGREE)?;
    akita_transcript::public_bytes(
        state,
        ProtocolSiteId { family: SITE_FAMILY_ROOT_STATEMENT, stage: 20,
            ..ProtocolSiteId::default() },
        b"akita/research/route1/4+4/direct/joint/base-s11a/v1",
    )?;
    for (index, (group, profile)) in claims.groups().iter().zip(profiles).enumerate() {
        let mut identity = profile.canonical_descriptor_bytes();
        identity.extend_from_slice(
            &u64::try_from(index).map_err(|_| AkitaError::InvalidProof)?.to_le_bytes(),
        );
        let bytes = payload(group.commitment());
        let dimension = match group.kind() {
            GroupClaimKind::Binary { manifest, claim: None, prime } if index < 4 => {
                if bytes.len() != manifest.payload_coefficients()? {
                    return Err(AkitaError::InvalidProof);
                }
                identity.extend_from_slice(&manifest.canonical_bytes());
                identity.extend_from_slice(&prime.canonical_bytes());
                manifest.b_ring_dimension()
            }
            GroupClaimKind::ResearchOrdinary { owners, width } if index >= 4 => {
                let compression = profile.outer_compression_plan()?;
                if bytes.len() != compression.terminal_coefficients()
                    || group.evaluations().len() != owners.len()
                    || group.point().len() != profile.group.num_vars()
                {
                    return Err(AkitaError::InvalidProof);
                }
                identity.extend_from_slice(&width.bits().to_le_bytes());
                for owner in owners { identity.extend_from_slice(owner); }
                compression.maps().last().ok_or(AkitaError::InvalidProof)?.ring_dimension()
            }
            GroupClaimKind::Ordinary | GroupClaimKind::Binary { .. }
            | GroupClaimKind::ResearchOrdinary { .. } => return Err(AkitaError::InvalidProof),
        };
        akita_transcript::public_bytes(
            state,
            ProtocolSiteId { family: SITE_FAMILY_ROOT_STATEMENT, stage: 21,
                ..ProtocolSiteId::default() },
            &identity,
        )?;
        let lifted = bytes.iter().copied().map(E::lift_base).collect::<Vec<_>>();
        akita_transcript::public_extensions::<F, E, _>(
            state,
            crate::FoldSite::RootCommitment {
                group: index,
                ring_dimension: dimension,
            }.id()?,
            &lifted,
        )?;
        akita_transcript::public_extensions::<F, E, _>(
            state, crate::FoldSite::RootPoint { group: index }.id()?, group.point(),
        )?;
        akita_transcript::public_extensions::<F, E, _>(
            state, crate::FoldSite::Openings { level: 0 }.id()?, group.evaluations(),
        )?;
    }
    Ok(())
}
