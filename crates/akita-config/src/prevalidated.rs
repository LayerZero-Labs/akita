use super::*;

#[cfg(feature = "prevalidated-catalog")]
impl<Cfg: CommitmentConfig> TrustedScheduleCatalog<Cfg> {
    /// Load a selected-row verifier view as [`Self::from_verifier_view`] does
    /// and record what loading derived, for
    /// [`Self::from_prevalidated_verifier_view`] (see
    /// [`ValidatedScheduleCatalog::prevalidate_verifier_view`]).
    ///
    /// # Errors
    ///
    /// Returns the loading error, or [`AkitaError::InvalidSetup`] when the
    /// record does not rebuild the loaded catalog.
    pub fn prevalidate_verifier_view(
        bytes: &[u8],
        chunking: ChunkedWitnessCfg,
    ) -> Result<akita_schedules::PrevalidatedVerifierRecord, AkitaError> {
        validate_config_policy::<Cfg>(chunking)?;
        ValidatedScheduleCatalog::prevalidate_verifier_view(
            bytes,
            Cfg::schedule_family_name(),
            &policy_of::<Cfg>(chunking),
            Cfg::ring_challenge_config,
        )
    }

    /// The catalog [`Self::from_verifier_view`] returns for `bytes`, rebuilt
    /// from a prevalidated view of the same bytes. `None` when the config
    /// policy is invalid or the view does not match (see
    /// [`ValidatedScheduleCatalog::from_prevalidated_verifier_view`]); the
    /// caller then loads `bytes` with [`Self::from_verifier_view`]. Never inlined.
    #[inline(never)]
    pub fn from_prevalidated_verifier_view(
        bytes: &[u8],
        prevalidated: &akita_schedules::PrevalidatedVerifierView<'_>,
        chunking: ChunkedWitnessCfg,
    ) -> Option<Self> {
        validate_config_policy::<Cfg>(chunking).ok()?;
        ValidatedScheduleCatalog::from_prevalidated_verifier_view(
            bytes,
            prevalidated,
            Cfg::schedule_family_name(),
            &policy_of::<Cfg>(chunking),
        )
        .map(Self::from_validated)
    }
}
