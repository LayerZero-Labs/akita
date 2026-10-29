//! Shared transcript atoms of an explicitly selected binary root.

use super::{BinaryCommittedGroupManifest, BinaryWordClaim};
use akita_challenges::{Challenges, FoldChallengeDrawDomain, FoldChallengeFamily, FoldDraw};
use akita_error::AkitaError;
use akita_transcript::{ProofChannel, ProtocolSiteId, SITE_FAMILY_ROOT_STATEMENT};
use jolt_field::{CanonicalEncoding, ExtField, Field};

/// Extension seam: bind the complete order-zero binary and mixed statement.
/// <https://github.com/LayerZero-Labs/akita/issues/285>
///
/// Bind the source contract and caller's logical word claim before the field
/// switch produces its internal carrier terminal.
#[allow(clippy::type_complexity)]
pub fn binary_root_statement<F, E, S>(
    state: &mut S,
    manifest: &BinaryCommittedGroupManifest,
    claim: &BinaryWordClaim,
    binary_commitment: &[F],
    ordinary: Option<(usize, &[F], &[E], &[E])>,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    S: ProofChannel + ?Sized,
{
    if E::DEGREE != 1 {
        return Err(AkitaError::InvalidSetup(
            "binary root statement requires extension degree one".into(),
        ));
    }
    manifest.validate()?;
    claim.validate(manifest.descriptor())?;
    for (stage, bytes) in [
        (3, manifest.canonical_bytes()),
        (4, claim.canonical_bytes()),
    ] {
        akita_transcript::public_bytes(
            state,
            ProtocolSiteId {
                family: SITE_FAMILY_ROOT_STATEMENT,
                stage,
                ..ProtocolSiteId::default()
            },
            &bytes,
        )?;
    }
    // The admitted binary root has extension degree one, so lifting base
    // payloads preserves their canonical field-atom encoding on both sides.
    let lift = |payload: &[F]| -> Result<Vec<E>, AkitaError> {
        let mut lifted = Vec::new();
        lifted
            .try_reserve_exact(payload.len())
            .map_err(|_| AkitaError::InvalidProof)?;
        lifted.extend(payload.iter().copied().map(E::lift_base));
        Ok(lifted)
    };
    akita_transcript::public_extensions::<F, E, _>(
        state,
        crate::FoldSite::RootCommitment {
            group: 0,
            ring_dimension: manifest.b_ring_dimension(),
        }
        .id()?,
        &lift(binary_commitment)?,
    )?;
    if let Some((dimension, payload, point, evaluations)) = ordinary {
        akita_transcript::public_extensions::<F, E, _>(
            state,
            crate::FoldSite::RootCommitment {
                group: 1,
                ring_dimension: dimension,
            }
            .id()?,
            &lift(payload)?,
        )?;
        akita_transcript::public_extensions::<F, E, _>(
            state,
            crate::FoldSite::RootPoint { group: 1 }.id()?,
            point,
        )?;
        akita_transcript::public_extensions::<F, E, _>(
            state,
            crate::FoldSite::Openings { level: 0 }.id()?,
            evaluations,
        )?;
    }
    Ok(())
}

/// Draw the one root response family on the current transcript or its nonce preview.
pub fn binary_root_challenges(
    draw: &mut impl FoldDraw,
    columns: usize,
) -> Result<Challenges, AkitaError> {
    draw.draw_folding_challenges_with_rejection(
        FoldChallengeDrawDomain::EvaluationTrace,
        162,
        0,
        columns,
        1,
        &FoldChallengeFamily::BinaryRoot,
        None,
    )
}
