//! Proof-stream sites that bind fold statements, payloads, and terminal atoms.
//!
//! The prover and verifier each exchange these atoms through their own
//! role-specific transcript call, so this module owns only the site identity.
//! Both roles name a site through [`FoldSite`] and convert it with
//! [`FoldSite::id`].

use akita_error::{narrowing::usize_to_u32, AkitaError};
use akita_transcript::{
    ProtocolSiteId, SITE_FAMILY_FOLD_BINDING, SITE_FAMILY_NEXT_WITNESS,
    SITE_FAMILY_OPENING_PAYLOAD, SITE_FAMILY_ROOT_STATEMENT, SITE_FAMILY_TERMINAL,
};

/// One fold-owned proof-stream site.
///
/// Group indices and ring dimensions are taken as `usize` and converted in
/// [`Self::id`], so callers never narrow them just to fill a site field.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FoldSite {
    /// Root commitment rows for one opening group.
    RootCommitment { group: usize, ring_dimension: usize },
    /// Root opening point for one opening group.
    RootPoint { group: usize },
    /// Protocol opening point for one group at a fold level.
    GroupPoint { level: u32, group: usize },
    /// Scalar openings bound before a non-terminal fold's relation.
    Openings { level: u32 },
    /// Committed witness rows bound before the terminal fold.
    WitnessCommitment { level: u32, ring_dimension: usize },
    /// Terminal scalar opening bound when no extension reduction runs.
    TerminalOpening { level: u32 },
    /// Terminal protocol point.
    TerminalPoint { level: u32 },
    /// Opening payload `D/H` rows at a fold level.
    OpeningPayload { level: u32, ring_dimension: usize },
    /// Outer commitment payload for the next fold's witness.
    NextWitnessPayload { level: u32 },
    /// Inner `t` state handed to the suffix-terminal fold.
    NextWitnessInnerState { level: u32 },
    /// Terminal `e` fields.
    TerminalEFields { level: u32 },
    /// Terminal public `t` fields.
    TerminalTFields { level: u32 },
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
            Self::WitnessCommitment {
                level,
                ring_dimension,
            } => ProtocolSiteId {
                detail: usize_to_u32(ring_dimension, "fold site ring dimension")?,
                ..site(SITE_FAMILY_FOLD_BINDING, level, 3)
            },
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
            Self::TerminalEFields { level } => site(SITE_FAMILY_TERMINAL, level, 1),
            Self::TerminalTFields { level } => site(SITE_FAMILY_TERMINAL, level, 2),
            Self::TerminalZPayload { level } => ProtocolSiteId {
                round: 3,
                ..site(SITE_FAMILY_TERMINAL, level, 0)
            },
        })
    }
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
            FoldSite::WitnessCommitment {
                level: 1,
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
