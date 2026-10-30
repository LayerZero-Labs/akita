//! Relation kinds carried by canonical grouped opening claims.

/// Extension seam: identify a group's relation independently of its commitment payload.
///
/// Tracking issue: <https://github.com/LayerZero-Labs/akita/issues/285>.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupClaimKind {
    /// The ordinary negacyclic-ring opening relation.
    Ordinary,
    /// A binary word claim supplied by the caller.
    #[cfg(feature = "labinius-trinomial")]
    Binary {
        /// Frozen binary A/B contract, retained when mapping commitment payloads.
        manifest: Box<crate::binary_source::BinaryCommittedGroupManifest>,
        /// The caller's logical word MLE evaluation over GF(2^128).
        claim: crate::binary_source::BinaryWordClaim,
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
        for (position, group) in groups.iter().enumerate() {
            if let GroupClaimKind::Binary { manifest, claim } = &group.kind {
                claim.validate(manifest.descriptor())?;
                if groups.iter().take(position).any(|previous| {
                    matches!(&previous.kind, GroupClaimKind::Binary { claim: earlier, .. }
                        if earlier.polynomial() == claim.polynomial())
                }) {
                    return Err(akita_error::AkitaError::InvalidInput(
                        "one binary word claim per polynomial; reduce multiple points before opening"
                            .into(),
                    ));
                }
            }
        }
        Ok(())
    }
}
