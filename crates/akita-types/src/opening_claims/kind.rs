//! Relation kinds carried by canonical grouped opening claims.

/// Extension seam: identify a group's relation independently of its commitment payload.
///
/// Tracking issue: <https://github.com/LayerZero-Labs/akita/issues/285>.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupClaimKind {
    /// The ordinary negacyclic-ring opening relation.
    Ordinary,
    /// Explicit research signed-source owners in commitment slot order.
    #[cfg(feature = "labinius-trinomial")]
    ResearchOrdinary {
        owners: Vec<[u8; 32]>,
        width: akita_params::ordinary_source::OrdinarySourceWidth,
        mode: crate::binary_source::ResearchTerminalMode,
    },
    /// A binary word claim supplied by the caller.
    #[cfg(feature = "labinius-trinomial")]
    Binary {
        /// Frozen binary A/B contract, retained when mapping commitment payloads.
        manifest: Box<crate::binary_source::BinaryCommittedGroupManifest>,
        /// The caller's logical word MLE evaluation over GF(2^128).
        claim: Option<crate::binary_source::BinaryWordClaim>,
        /// Checked base or extension claims, in point/polynomial/coordinate order.
        prime: crate::binary_source::PrimePlaneManifest,
    },
}

impl<F: Clone, C> super::OpeningClaims<'_, F, C> {
    /// Extension seam: validate group kinds and cross-group claim constraints.
    /// Tracking issue: <https://github.com/LayerZero-Labs/akita/issues/285>.
    pub(super) fn check_group_kinds(
        groups: &[super::PolynomialGroupClaims<'_, F, C>],
    ) -> Result<(), akita_error::AkitaError> {
        for group in groups {
            if group.evaluations.is_empty() {
                return Err(akita_error::AkitaError::InvalidInput(
                    "opening claim group evaluations must be nonempty".into(),
                ));
            }
        }
        #[cfg(feature = "labinius-trinomial")]
        if groups.iter().any(|group| {
            group.research_terminals.is_some()
                || matches!(&group.kind, GroupClaimKind::ResearchOrdinary { .. })
        }) {
            if groups.len() != 8 {
                return Err(akita_error::AkitaError::InvalidInput(
                    "research terminals require the complete four-plus-four roster".into(),
                ));
            }
            let mode = groups
                .first()
                .and_then(|group| group.research_terminals.as_ref())
                .map(|terminals| terminals.mode);
            for (index, group) in groups.iter().enumerate() {
                let correct_kind = if index < 4 {
                    matches!(&group.kind, GroupClaimKind::Binary { claim, .. }
                        if index == 0 || claim.is_none())
                } else {
                    matches!(&group.kind, GroupClaimKind::ResearchOrdinary { .. })
                };
                if !correct_kind
                    || mode.is_none()
                    || group
                        .research_terminals
                        .as_ref()
                        .map(|terminals| terminals.mode)
                        != mode
                {
                    return Err(akita_error::AkitaError::InvalidInput(
                        "research terminals cannot enter another root protocol".into(),
                    ));
                }
            }
        }
        #[cfg(feature = "labinius-trinomial")]
        for (position, group) in groups.iter().enumerate() {
            if let GroupClaimKind::Binary {
                manifest,
                claim,
                prime,
            } = &group.kind
            {
                prime.validate(manifest.descriptor())?;
                if !prime.planes().is_empty() && prime.field_profile() != manifest.field_profile() {
                    return Err(akita_error::AkitaError::InvalidInput(
                        "prime plane profile mismatch".into(),
                    ));
                }
                let Some(claim) = claim else {
                    if prime.planes().is_empty() {
                        return Err(akita_error::AkitaError::InvalidInput(
                            "binary group has no claims".into(),
                        ));
                    }
                    continue;
                };
                claim.validate(manifest.descriptor())?;
                if groups.iter().take(position).any(|previous| {
                    matches!(&previous.kind, GroupClaimKind::Binary { claim: Some(earlier), .. }
                        if earlier.polynomial() == claim.polynomial())
                }) {
                    return Err(akita_error::AkitaError::InvalidInput(
                        "one binary word claim per polynomial; \
                         reduce multiple points before opening"
                            .into(),
                    ));
                }
            }
        }
        Ok(())
    }
}
