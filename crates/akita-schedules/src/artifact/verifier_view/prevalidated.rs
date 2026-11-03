use super::*;

/// A selected-row verifier view with the values that
/// [`ValidatedScheduleCatalog::from_verifier_view`] derived from it when it
/// was loaded ahead of time (for example while building a guest image).
///
/// This is trusted verifier configuration: [`ValidatedScheduleCatalog::from_prevalidated_verifier_view`]
/// uses it only for bytes equal to `view`, and the bytes keep the provenance
/// their caller gives them.
#[cfg(feature = "prevalidated-catalog")]
#[derive(Clone, Copy, Debug)]
pub struct PrevalidatedVerifierView<'a> {
    /// The exact view bytes that were validated.
    view: &'a [u8],
    /// The runtime policy digest the view was validated under.
    policy_digest: [u8; 32],
    /// The catalog digest reproduced from the view's identity list.
    catalog_digest: [u8; 32],
    /// The row digest of every carried row, in the view's row order.
    carried_row_digests: &'a [[u8; 32]],
}

/// What [`ValidatedScheduleCatalog::prevalidate_verifier_view`] records.
#[cfg(feature = "prevalidated-catalog")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrevalidatedVerifierRecord {
    view: Vec<u8>,
    /// The runtime policy digest.
    policy_digest: [u8; 32],
    /// The catalog digest.
    catalog_digest: [u8; 32],
    /// The carried rows' digests, in the view's row order.
    carried_row_digests: Vec<[u8; 32]>,
}

#[cfg(feature = "prevalidated-catalog")]
impl ValidatedScheduleCatalog {
    /// Load `bytes` through [`Self::from_verifier_view`] and record the values
    /// it derived. The record is accepted only if
    /// [`Self::from_prevalidated_verifier_view`] rebuilds a catalog equal to
    /// the loaded one (compared by its complete `Debug` form).
    ///
    /// # Errors
    ///
    /// Returns the loading error, or [`AkitaError::InvalidSetup`] when the
    /// rebuilt catalog differs.
    pub fn prevalidate_verifier_view(
        bytes: &[u8],
        expected_family_name: &str,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    ) -> Result<PrevalidatedVerifierRecord, AkitaError> {
        let catalog =
            Self::from_verifier_view(bytes, expected_family_name, policy, ring_challenge_config)?;
        let record = PrevalidatedVerifierRecord {
            view: bytes.to_vec(),
            policy_digest: catalog.policy_digest,
            catalog_digest: catalog.catalog_digest,
            carried_row_digests: catalog
                .rows()
                .map(|row| *row.selection().row_digest.as_bytes())
                .collect(),
        };
        let prevalidated = PrevalidatedVerifierView {
            view: bytes,
            policy_digest: record.policy_digest,
            catalog_digest: record.catalog_digest,
            carried_row_digests: &record.carried_row_digests,
        };
        let rebuilt = Self::from_prevalidated_verifier_view(
            bytes,
            &prevalidated,
            expected_family_name,
            policy,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup("prevalidated verifier view was not used".into())
        })?;
        if format!("{rebuilt:?}") != format!("{catalog:?}") {
            return Err(AkitaError::InvalidSetup(
                "prevalidated verifier view rebuilds a different catalog".into(),
            ));
        }
        Ok(record)
    }

    /// Rebuild the catalog that [`Self::from_verifier_view`] returns for
    /// `bytes`, from a [`PrevalidatedVerifierView`] of the same bytes, without
    /// repeating the row audit, the catalog digest or the canonical re-encode.
    ///
    /// In this order it requires: `bytes` has `prevalidated.view`'s length,
    /// `bytes` equals `prevalidated.view` byte for byte, and the policy digest
    /// of `policy` equals `prevalidated.policy_digest`; then that the decoded
    /// view names `expected_family_name` and that policy digest, carries
    /// exactly the recorded number of rows, and that every carried row passes
    /// its topology check. It returns `None` on any miss, and the caller then
    /// loads the bytes with [`Self::from_verifier_view`]. Never inlined: one copy serves every caller.
    #[inline(never)]
    pub fn from_prevalidated_verifier_view(
        bytes: &[u8],
        prevalidated: &PrevalidatedVerifierView<'_>,
        expected_family_name: &str,
        policy: &PlannerPolicy,
    ) -> Option<Self> {
        if bytes.len() > MAX_VERIFIER_VIEW_BYTES
            || bytes.len() != prevalidated.view.len()
            || bytes != prevalidated.view
            || policy_digest(policy) != prevalidated.policy_digest
        {
            return None;
        }
        let (artifact, rest): (BinaryVerifierCatalog, &[u8]) =
            postcard::take_from_bytes(bytes).ok()?;
        if !rest.is_empty()
            || artifact.magic != VERIFIER_MAGIC
            || artifact.protocol_epoch != AKITA_INSTANCE_DESCRIPTOR_VERSION
            || artifact.family_name != expected_family_name
            || artifact.policy_digest != prevalidated.policy_digest
            || artifact.rows.len() != prevalidated.carried_row_digests.len()
            || !artifact.row_digests.len().is_multiple_of(32)
        {
            return None;
        }
        let mut resolved = Vec::with_capacity(artifact.rows.len());
        for (row, digest) in artifact
            .rows
            .into_iter()
            .zip(prevalidated.carried_row_digests)
        {
            let (profiles, schedule) = row.into_profile_and_schedule().ok()?;
            resolved.push(crate::resolve::ResolvedScheduleRow::from_prevalidated(
                OpeningScheduleSelection {
                    row_digest: ScheduleRowDigest::from_bytes(*digest),
                },
                profiles,
                schedule,
            ));
        }
        resolved.sort_by_key(|row| row.selection().row_digest);
        let mut rows_by_key = resolved
            .iter()
            .enumerate()
            .map(|(index, row)| {
                (
                    super::super::key_for_profiles(
                        row.profiles(), row.schedule().root_accumulated_setup_prefix_len,
                    ).canonical_order_key(),
                    index,
                )
            })
            .collect::<Vec<_>>();
        rows_by_key.sort_by(|(left_key, left_index), (right_key, right_index)| {
            left_key.cmp(right_key).then_with(|| {
                resolved[*left_index]
                    .selection()
                    .row_digest
                    .cmp(&resolved[*right_index].selection().row_digest)
            })
        });
        #[cfg(not(feature = "digest-nozero"))]
        let row_digests = {
            let mut digests = vec![[0u8; 32]; artifact.row_digests.len() / 32];
            digests.as_flattened_mut().copy_from_slice(&artifact.row_digests);
            digests
        };
        #[cfg(feature = "digest-nozero")]
        let row_digests = artifact.row_digests.as_chunks::<32>().0.to_vec();
        Some(Self {
            policy: *policy,
            family_name: artifact.family_name,
            policy_digest: prevalidated.policy_digest,
            catalog_digest: prevalidated.catalog_digest,
            coverage: CatalogCoverage::Selected { row_digests },
            rows_by_digest: resolved,
            rows_by_key,
        })
    }
}


impl PrevalidatedVerifierRecord {
    /// Borrow the immutable bytes and identities established by validation.
    pub fn as_view(&self) -> PrevalidatedVerifierView<'_> {
        PrevalidatedVerifierView {
            view: &self.view,
            policy_digest: self.policy_digest,
            catalog_digest: self.catalog_digest,
            carried_row_digests: &self.carried_row_digests,
        }
    }

    /// Policy identity established by validation.
    pub const fn policy_digest(&self) -> [u8; 32] {
        self.policy_digest
    }

    /// Catalog identity established by validation.
    pub const fn catalog_digest(&self) -> [u8; 32] {
        self.catalog_digest
    }

    /// Carried row identities established by validation.
    pub fn carried_row_digests(&self) -> &[[u8; 32]] {
        &self.carried_row_digests
    }
}

impl<'a> PrevalidatedVerifierView<'a> {
    /// Restore a record embedded by a trusted offline generator.
    ///
    /// # Safety
    ///
    /// All arguments must be copied from one successful call to
    /// `prevalidate_verifier_view` in this protocol version. `view` must be
    /// exactly the validated bytes. These values must never come from a proof
    /// or unaudited caller input: this constructor bypasses semantic admission.
    pub const unsafe fn from_trusted_parts(
        view: &'a [u8],
        policy_digest: [u8; 32],
        catalog_digest: [u8; 32],
        carried_row_digests: &'a [[u8; 32]],
    ) -> Self {
        Self {
            view,
            policy_digest,
            catalog_digest,
            carried_row_digests,
        }
    }
}
