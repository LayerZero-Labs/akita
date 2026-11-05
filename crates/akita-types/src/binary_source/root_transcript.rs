//! Shared transcript atoms of an explicitly selected binary root.

use super::{BinaryCommittedGroupManifest, BinaryWordClaim};
use crate::FoldChallengeDrawDomain;
use crate::FoldChallengeFamily;
use crate::GrindingReplay;
use crate::{FoldChallengeDraw, FoldDraw};
use akita_error::AkitaError;
use akita_params::FoldChallengeDistribution;
use akita_params::GrindingSite;
use akita_transcript::{
    ext_challenges, ChallengeCodec, ProofChannel, ProtocolSiteId, SITE_FAMILY_ROOT_STATEMENT,
};
use jolt_algebra::{CanonicalEncoding, ExtField, Field};
use jolt_crypto::challenge::Challenges;

/// Extension seam: bind the complete order-zero binary and mixed statement.
/// <https://github.com/LayerZero-Labs/akita/issues/285>
///
/// Bind the source contract and caller's logical word claim before the field
/// switch produces its internal carrier terminal.
#[allow(clippy::type_complexity)]
pub fn binary_root_statement<F, E, S>(
    state: &mut S,
    manifest: &BinaryCommittedGroupManifest,
    claim: Option<&BinaryWordClaim>,
    prime: &super::PrimePlaneManifest,
    binary_commitment: &[F],
    ordinary: Option<(usize, &[F], &[E], &[E])>,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    S: ProofChannel,
{
    if E::DEGREE != 1 {
        return Err(AkitaError::InvalidSetup(
            "binary root statement requires extension degree one".into(),
        ));
    }
    manifest.validate()?;
    if let Some(claim) = claim {
        claim.validate(manifest.descriptor())?;
    }
    super::PrimePlaneManifest::new(manifest.descriptor(), prime.planes())?;
    let mut messages = vec![(3, manifest.canonical_bytes())];
    if let Some(claim) = claim {
        messages.push((4, claim.canonical_bytes()));
    }
    if !prime.planes().is_empty() {
        messages.push((5, prime.canonical_bytes()));
    }
    for (stage, bytes) in messages {
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

/// Elements per Tau1 challenge message. `jolt-transcript` caps one challenge
/// message at 16 MiB, and a mixed root draws millions of coefficients.
const BATCH_CHALLENGE_CHUNK: usize = 1 << 18;

/// Run the root's Tau1 proof of work, then draw its `count` independent batch
/// coefficients as consecutive challenge messages of at most
/// [`BATCH_CHALLENGE_CHUNK`] elements at that site.
pub fn binary_root_batch_challenges<E, G>(
    grinding: &mut G,
    count: usize,
) -> Result<Vec<E>, AkitaError>
where
    E: ChallengeCodec,
    G: GrindingReplay,
{
    let site = GrindingSite::Tau1Point { level: 0 };
    grinding.grind_query(site)?;
    let mut challenges = Vec::new();
    let result = challenges
        .try_reserve_exact(count)
        .map_err(|_| AkitaError::InvalidInput("batch challenge allocation failed".into()))
        .and_then(|()| {
            let mut remaining = count;
            loop {
                let chunk = remaining.min(BATCH_CHALLENGE_CHUNK);
                challenges.extend(ext_challenges::<E, _>(
                    grinding.state_mut(),
                    site.site_id(0),
                    chunk,
                )?);
                remaining = remaining.saturating_sub(chunk);
                if remaining == 0 {
                    return Ok(());
                }
            }
        });
    if result.is_err() {
        grinding.poison();
    }
    result.map(|()| challenges)
}

/// The fold challenge family of the binary root's single fold.
pub(crate) fn binary_root_family() -> Result<FoldChallengeFamily, AkitaError> {
    FoldChallengeFamily::new(
        FoldChallengeDrawDomain::EvaluationTrace,
        162,
        FoldChallengeDistribution::BinaryRoot,
        None,
    )
}

/// Draw the one root response family on the current transcript or its nonce preview.
pub fn binary_root_challenges(
    draw: &mut impl FoldDraw,
    columns: usize,
    polynomials: usize,
) -> Result<Challenges, AkitaError> {
    draw.draw_fold_challenges(&binary_root_family()?, columns, polynomials)?
        .ok_or_else(|| AkitaError::Internal("binary root family has no rejection".into()))
}
