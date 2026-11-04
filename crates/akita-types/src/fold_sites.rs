//! Proof-stream sites that bind fold statements, payloads, and terminal atoms.
//!
//! Most of these atoms are exchanged through each role's own transcript call,
//! so this module owns their site identity: both roles name a site through
//! [`FoldSite`] and convert it with [`FoldSite::id`]. A level's scalar
//! openings are bound once for both roles by [`bind_fold_openings`].

use crate::GrindingReplay;
use akita_error::AkitaError;
use akita_transcript::{
    public_extensions, PrivateProofChannel, ProtocolSiteId, SITE_FAMILY_FOLD_BINDING,
    SITE_FAMILY_NEXT_WITNESS, SITE_FAMILY_OPENING_PAYLOAD, SITE_FAMILY_ROOT_STATEMENT,
    SITE_FAMILY_TERMINAL,
};
use jolt_algebra::narrowing::usize_to_u32;
use jolt_algebra::{CanonicalEncoding, ExtField, Field};

/// One fold-owned proof-stream site.
///
/// Group indices and ring dimensions are taken as `usize` and converted in
/// [`Self::id`], so callers never narrow them just to fill a site field.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FoldSite {
    /// Root commitment digest for one opening group.
    RootCommitment { group: usize, ring_dimension: usize },
    /// Root opening point for one opening group.
    RootPoint { group: usize },
    /// Protocol opening point for one group at a fold level.
    GroupPoint { level: u32, group: usize },
    /// Scalar openings bound before a non-terminal fold's relation.
    Openings { level: u32 },
    /// Terminal scalar opening bound when no extension reduction runs.
    TerminalOpening { level: u32 },
    /// Terminal protocol point.
    TerminalPoint { level: u32 },
    /// Opening payload `D/H` rows at a fold level.
    OpeningPayload { level: u32, ring_dimension: usize },
    /// Outer commitment payload for the next fold's witness.
    NextWitnessPayload { level: u32 },
    /// Inner `t` state handed to the suffix-terminal fold, stopping before
    /// the last block.
    NextWitnessInnerState { level: u32 },
    /// Digest of the last inner `t` block, which the terminal recomputes.
    NextWitnessInnerStateDigest { level: u32 },
    /// Terminal `e` fields, stopping before the last block.
    TerminalEFields { level: u32 },
    /// Digest of the last terminal `e` block, which the terminal recomputes.
    TerminalEDigest { level: u32 },
    /// Terminal `z` response payload.
    TerminalZPayload { level: u32 },
}

impl FoldSite {
    /// Site identity recorded for this atom.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidInput`] when a group index or ring
    /// dimension does not fit a `u32` site field.
    pub fn id(self) -> Result<ProtocolSiteId, AkitaError> {
        let site = |family, level, stage| ProtocolSiteId {
            family,
            level,
            stage,
            ..ProtocolSiteId::default()
        };
        Ok(match self {
            Self::RootCommitment {
                group,
                ring_dimension,
            } => ProtocolSiteId {
                group: usize_to_u32(group, "fold site group index")?,
                detail: usize_to_u32(ring_dimension, "fold site ring dimension")?,
                ..site(SITE_FAMILY_ROOT_STATEMENT, 0, 1)
            },
            Self::RootPoint { group } => ProtocolSiteId {
                group: usize_to_u32(group, "fold site group index")?,
                ..site(SITE_FAMILY_ROOT_STATEMENT, 0, 2)
            },
            Self::GroupPoint { level, group } => ProtocolSiteId {
                group: usize_to_u32(group, "fold site group index")?,
                ..site(SITE_FAMILY_FOLD_BINDING, level, 1)
            },
            Self::Openings { level } => site(SITE_FAMILY_FOLD_BINDING, level, 2),
            Self::TerminalOpening { level } => site(SITE_FAMILY_FOLD_BINDING, level, 4),
            Self::TerminalPoint { level } => site(SITE_FAMILY_FOLD_BINDING, level, 5),
            Self::OpeningPayload {
                level,
                ring_dimension,
            } => ProtocolSiteId {
                detail: usize_to_u32(ring_dimension, "fold site ring dimension")?,
                ..site(SITE_FAMILY_OPENING_PAYLOAD, level, 0)
            },
            Self::NextWitnessPayload { level } => site(SITE_FAMILY_NEXT_WITNESS, level, 1),
            Self::NextWitnessInnerState { level } => site(SITE_FAMILY_NEXT_WITNESS, level, 2),
            Self::NextWitnessInnerStateDigest { level } => site(SITE_FAMILY_NEXT_WITNESS, level, 3),
            Self::TerminalEFields { level } => site(SITE_FAMILY_TERMINAL, level, 1),
            Self::TerminalEDigest { level } => site(SITE_FAMILY_TERMINAL, level, 3),
            Self::TerminalZPayload { level } => ProtocolSiteId {
                round: 3,
                ..site(SITE_FAMILY_TERMINAL, level, 0)
            },
        })
    }
}

/// Binds a fold level's scalar openings, in group order, at its
/// [`FoldSite::Openings`] site.
///
/// `prefix` is the setup-prefix evaluation (group 0), present when the level
/// consumes a setup prefix: the evaluation of a public setup slot at a public
/// point, so it stays public. `derived` holds every other opening, the
/// statement claims (level 0) or the predecessor's witness evaluation, which
/// the verifier derives from earlier messages. A clear channel absorbs all of
/// them as one public message, in group order; a masked channel absorbs only
/// the prefix, since the masked messages the derived claims come from already
/// bind them (`specs/whiteout-paper-deltas.md` D25).
///
/// # Errors
///
/// The channel's error.
pub fn bind_fold_openings<F, E, G>(
    grinding: &mut G,
    level: u32,
    prefix: Option<&E>,
    derived: &[E],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    G: GrindingReplay<State: PrivateProofChannel<F>>,
{
    let state = grinding.state_mut();
    let mut openings: Vec<E> = prefix.into_iter().copied().collect();
    if !state.is_masked() {
        openings.extend_from_slice(derived);
    }
    public_extensions::<F, E, _>(state, FoldSite::Openings { level }.id()?, &openings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_site_narrowings_name_the_bad_argument() {
        let Some(too_large) = (u32::MAX as usize).checked_add(1) else {
            return;
        };
        for site in [
            FoldSite::RootCommitment {
                group: too_large,
                ring_dimension: 1,
            },
            FoldSite::RootPoint { group: too_large },
            FoldSite::GroupPoint {
                level: 1,
                group: too_large,
            },
        ] {
            assert!(matches!(
                site.id(),
                Err(AkitaError::InvalidInput(message)) if message.contains("group index")
            ));
        }
        for site in [
            FoldSite::RootCommitment {
                group: 0,
                ring_dimension: too_large,
            },
            FoldSite::OpeningPayload {
                level: 1,
                ring_dimension: too_large,
            },
        ] {
            assert!(matches!(
                site.id(),
                Err(AkitaError::InvalidInput(message)) if message.contains("ring dimension")
            ));
        }
    }
}
