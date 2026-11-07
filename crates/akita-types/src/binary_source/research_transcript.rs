//! Bind every source, owner, terminal and layout before the joint D barrier.

use akita_error::AkitaError;
use akita_params::GroupCommitPhaseParams;
use akita_transcript::{ProofChannel, ProtocolSiteId, SITE_FAMILY_ROOT_STATEMENT};
use jolt_field::{CanonicalEncoding, ExtField, Field};

use crate::opening_claims::GroupClaimKind;
use crate::OpeningClaims;

/// Explicit research wire identity: direct setup, joint retry and S11a reading mode.
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
    let mode = claims
        .groups()
        .first()
        .and_then(|group| group.research_terminals())
        .ok_or(AkitaError::InvalidProof)?
        .mode;
    let identity: &[u8] = match mode {
        super::ResearchTerminalMode::BaseS11a => {
            b"akita/research/route1/4+4/direct/joint/base-s11a/v2"
        }
        super::ResearchTerminalMode::E64TwoPointS11a => {
            b"akita/research/route1/4+4/direct/joint/e64-two-point-s11a/v3"
        }
    };
    akita_transcript::public_bytes(
        state,
        ProtocolSiteId {
            family: SITE_FAMILY_ROOT_STATEMENT,
            stage: 20,
            ..ProtocolSiteId::default()
        },
        identity,
    )?;
    for (index, (group, profile)) in claims.groups().iter().zip(profiles).enumerate() {
        let terminals = group.research_terminals().ok_or_else(|| {
            AkitaError::InvalidInput("research group lacks typed terminal admission".into())
        })?;
        terminals.validate::<F>()?;
        if terminals.mode != mode {
            return Err(AkitaError::InvalidProof);
        }
        let mut identity = profile.canonical_descriptor_bytes();
        identity.extend_from_slice(
            &u64::try_from(index)
                .map_err(|_| AkitaError::InvalidProof)?
                .to_le_bytes(),
        );
        let bytes = payload(group.commitment());
        let dimension = match group.kind() {
            GroupClaimKind::Binary {
                manifest,
                claim,
                prime,
            } if index < 4 && (index == 0 || claim.is_none()) => {
                if bytes.len() != manifest.payload_coefficients()? {
                    return Err(AkitaError::InvalidProof);
                }
                identity.extend_from_slice(&manifest.canonical_bytes());
                identity.push(u8::from(claim.is_some()));
                if let Some(claim) = claim {
                    claim.validate(manifest.descriptor())?;
                    identity.extend_from_slice(&claim.canonical_bytes());
                }
                identity.extend_from_slice(&prime.canonical_bytes());
                manifest.b_ring_dimension()
            }
            GroupClaimKind::ResearchOrdinary {
                owners,
                width,
                mode: selected,
            } if index >= 4 => {
                let compression = profile.outer_compression_plan()?;
                if *selected != mode
                    || bytes.len() != compression.terminal_coefficients()
                    || group.evaluations().len() != owners.len()
                    || group.point().len() != profile.group.num_vars()
                {
                    return Err(AkitaError::InvalidProof);
                }
                identity.extend_from_slice(&width.bits().to_le_bytes());
                for owner in owners {
                    identity.extend_from_slice(owner);
                }
                compression
                    .maps()
                    .last()
                    .ok_or(AkitaError::InvalidProof)?
                    .ring_dimension()
            }
            GroupClaimKind::Ordinary
            | GroupClaimKind::Binary { .. }
            | GroupClaimKind::ResearchOrdinary { .. } => return Err(AkitaError::InvalidProof),
        };
        akita_transcript::public_bytes(
            state,
            ProtocolSiteId {
                family: SITE_FAMILY_ROOT_STATEMENT,
                stage: 21,
                ..ProtocolSiteId::default()
            },
            &identity,
        )?;
        let lifted = bytes.iter().copied().map(E::lift_base).collect::<Vec<_>>();
        akita_transcript::public_extensions::<F, E, _>(
            state,
            crate::FoldSite::RootCommitment {
                group: index,
                ring_dimension: dimension,
            }
            .id()?,
            &lifted,
        )?;
        // Base mode retains its original wire bytes. E64 binds both source-batched passes.
        let readings: Vec<(&[E], &[E])> = match mode {
            super::ResearchTerminalMode::BaseS11a => vec![(group.point(), group.evaluations())],
            super::ResearchTerminalMode::E64TwoPointS11a => terminals
                .readings
                .iter()
                .map(|terminal| (terminal.point.as_slice(), terminal.evaluations.as_slice()))
                .collect(),
        };
        for (point, evaluations) in readings {
            akita_transcript::public_extensions::<F, E, _>(
                state,
                crate::FoldSite::RootPoint { group: index }.id()?,
                point,
            )?;
            akita_transcript::public_extensions::<F, E, _>(
                state,
                crate::FoldSite::Openings { level: 0 }.id()?,
                evaluations,
            )?;
        }
    }
    Ok(())
}
