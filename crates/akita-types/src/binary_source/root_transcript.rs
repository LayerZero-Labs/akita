//! Shared transcript atoms of an explicitly selected binary root.

use super::{BinaryCommittedGroupManifest, BinaryTerminalClaim};
use akita_challenges::{Challenges, FoldChallengeDrawDomain, FoldChallengeFamily, FoldDraw};
use akita_error::AkitaError;
use akita_transcript::{ProofChannel, ProtocolSiteId, SITE_FAMILY_ROOT_STATEMENT};

/// Bind the complete source contract and the terminal supplied by the frontend.
/// The future field switch produces this same terminal claim.
pub fn binary_root_statement<S: ProofChannel + ?Sized>(
    state: &mut S,
    manifest: &BinaryCommittedGroupManifest,
    claim: &BinaryTerminalClaim,
) -> Result<(), AkitaError> {
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
