//! Explicit trusted admission for an experimental native-root transition.
//!
//! This validates the ordinary suffix, not the native root's cryptographic
//! contract. The caller supplies that contract and explicitly selects this
//! experimental admission path. Native rows are never ordinary artifacts.

use crate::artifact::validate_schedule_challenge_hooks;
use crate::resolve::RootTransitionValidation;
use crate::{PlannerPolicy, ResolvedScheduleRow, ValidatedScheduleCatalog};
use akita_challenges::FoldChallengeDistribution;
use akita_error::{checked, AkitaError};
use akita_params::{CommittedGroupBatchProfile, FoldParams, FoldSchedule, OpeningClaimsLayout};

/// The source contract authenticated by a grouped native row.
/// The caller obtains the identity from its validated grouped plan. The schedule
/// layer validates frozen profiles and signed widths; the opening layer must
/// reconstruct that plan from the actual manifests and compare its identity.
#[derive(Clone, Debug)]
pub struct NativeRootSources {
    profiles: CommittedGroupBatchProfile,
    signed_widths: [akita_params::ordinary_source::OrdinarySourceWidth; 4],
    identity: Vec<u8>,
}

impl NativeRootSources {
    pub fn new(
        profiles: CommittedGroupBatchProfile,
        signed_widths: [akita_params::ordinary_source::OrdinarySourceWidth; 4],
        identity: Vec<u8>,
    ) -> Result<Self, AkitaError> {
        profiles.validate(64)?;
        if profiles.precommitteds.len() != 7 || identity.is_empty() || identity.len() > 1 << 20 {
            return Err(AkitaError::InvalidSetup("invalid grouped source inventory".into()));
        }
        for (profile, width) in profiles.precommitteds.iter().skip(4)
            .chain(std::iter::once(&profiles.final_group)).zip(signed_widths)
        {
            width.validate_profile(profile)?;
        }
        for profile in profiles.precommitteds.iter().chain(std::iter::once(&profiles.final_group)) {
            if profile.inner.matrix.sis_modulus_profile()
                != akita_params::SisModulusProfileId::Q64Offset59
                || profile.outer.matrix.sis_modulus_profile()
                    != akita_params::SisModulusProfileId::Q64Offset59
            {
                return Err(AkitaError::InvalidSetup("grouped source field mismatch".into()));
            }
        }
        Ok(Self { profiles, signed_widths, identity })
    }

    pub fn profiles(&self) -> &CommittedGroupBatchProfile {
        &self.profiles
    }

    pub fn signed_widths(&self) -> &[akita_params::ordinary_source::OrdinarySourceWidth; 4] {
        &self.signed_widths
    }

    pub fn identity(&self) -> &[u8] {
        &self.identity
    }
}

/// Trusted geometry of the native root's balanced-digit output and typed A prefix.
#[derive(Clone, Debug)]
pub struct NativeRootTransition {
    physical_cells: usize,
    log_digit_bound: u32,
    setup_prefix_natural_cells: usize,
    sources: Option<NativeRootSources>,
}

impl NativeRootTransition {
    /// Check geometry before planning or admitting an experimental native row.
    ///
    /// The current binary root emits balanced base-eight digits, and requires
    /// a typed A prefix. Its physical live span may have virtual zero tails;
    /// its commitment domain is derived independently from that live span.
    pub fn new(
        physical_cells: usize,
        log_digit_bound: u32,
        setup_prefix_natural_cells: usize,
    ) -> Result<Self, AkitaError> {
        if physical_cells == 0 || log_digit_bound != 3 || setup_prefix_natural_cells == 0 {
            return Err(AkitaError::InvalidSetup(
                "native root requires a nonempty base-eight witness and typed A prefix".into(),
            ));
        }
        for cells in [physical_cells, setup_prefix_natural_cells] {
            checked::ceil_log2(cells)
                .and_then(checked::pow2)
                .ok_or_else(|| AkitaError::InvalidSetup("native root domain overflow".into()))?;
        }
        Ok(Self {
            physical_cells,
            log_digit_bound,
            setup_prefix_natural_cells,
            sources: None,
        })
    }

    /// Physical live span, including virtual tails between witness segments.
    pub const fn physical_cells(&self) -> usize {
        self.physical_cells
    }
    /// Logarithm of the balanced digit alphabet.
    pub const fn log_digit_bound(&self) -> u32 {
        self.log_digit_bound
    }
    /// Natural cells of the typed A product's padded rank/coefficient tensor.
    pub const fn setup_prefix_natural_cells(&self) -> usize {
        self.setup_prefix_natural_cells
    }

    /// Bind a trusted source inventory without changing the physical transition.
    pub fn with_sources(mut self, sources: NativeRootSources) -> Self {
        self.sources = Some(sources);
        self
    }

    /// Re-audit an explicitly selected native transition and every ordinary suffix field.
    pub fn resolve(
        &self,
        profiles: CommittedGroupBatchProfile,
        schedule: FoldSchedule,
        policy: &PlannerPolicy,
    ) -> Result<ResolvedScheduleRow, AkitaError> {
        ResolvedScheduleRow::with_root_transition(profiles, schedule, policy, self)
    }
}

impl RootTransitionValidation for NativeRootTransition {
    fn grouped_root(&self) -> Option<&NativeRootSources> {
        self.sources.as_ref()
    }

    fn validate_root(
        &self,
        _root_layout: &OpeningClaimsLayout,
        schedule: &FoldSchedule,
        _policy: &PlannerPolicy,
    ) -> Result<usize, AkitaError> {
        Self::new(
            self.physical_cells,
            self.log_digit_bound,
            self.setup_prefix_natural_cells,
        )?;
        if schedule.root.input_witness_len
            != akita_params::root_input_witness_len(&schedule.root.params)
            || !schedule.root.params.witness_chunk_ends.is_empty()
            || schedule.root.output_witness_len != self.physical_cells
            || schedule.root_accumulated_setup_prefix_len != 0
            || schedule.recursive_folds.is_empty()
        {
            return Err(AkitaError::InvalidSetup(
                "native root transition disagrees with its trusted geometry".into(),
            ));
        }
        Ok(self.physical_cells)
    }

    fn validate_first_successor(
        &self,
        _producer: &FoldParams,
        _producer_opening: &OpeningClaimsLayout,
        step: &FoldParams,
        expected_input: usize,
        _policy: &PlannerPolicy,
    ) -> Result<(), AkitaError> {
        let prefix = step.params.setup_prefix().ok_or_else(|| {
            AkitaError::InvalidSetup("native root requires a typed A prefix successor".into())
        })?;
        let prefix_domain = checked::ceil_log2(self.setup_prefix_natural_cells)
            .and_then(checked::pow2)
            .ok_or(AkitaError::InvalidProof)?;
        if expected_input != self.physical_cells
            || step.params.witness_chunk.num_chunks != 1
            || !step.params.witness_chunk_ends.is_empty()
            || step.params.inner().digits.log_basis != self.log_digit_bound
            || step.params.inner().digits.num_digits != 1
            || prefix.setup_natural_len != Some(self.setup_prefix_natural_cells)
            || prefix.n_prefix()? != prefix_domain
        {
            return Err(AkitaError::InvalidSetup("native root successor does not preserve its exact digit, prefix, or ownership geometry".into()));
        }
        Ok(())
    }

    fn is_ordinary(&self) -> bool {
        false
    }
}

impl ValidatedScheduleCatalog {
    /// Explicitly admit trusted native rows beside ordinary producer rows.
    ///
    /// This experimental path does not certify native security. The caller
    /// owns provenance and must reconstruct admission in each process from
    /// trusted native geometry; ordinary artifact decoders never accept it.
    pub fn try_new_native_root(
        family_name: impl Into<String>,
        ordinary_rows: impl IntoIterator<Item = (CommittedGroupBatchProfile, FoldSchedule)>,
        native_rows: impl IntoIterator<
            Item = (
                CommittedGroupBatchProfile,
                FoldSchedule,
                NativeRootTransition,
            ),
        >,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<FoldChallengeDistribution, AkitaError>,
    ) -> Result<Self, AkitaError> {
        let mut resolved = Vec::new();
        for (profiles, schedule) in ordinary_rows
            .into_iter()
            .take(crate::artifact::MAX_TRUSTED_CATALOG_ROWS + 1)
        {
            resolved.push(ResolvedScheduleRow::try_new(profiles, schedule, policy)?);
        }
        for (profiles, schedule, transition) in native_rows
            .into_iter()
            .take(crate::artifact::MAX_TRUSTED_CATALOG_ROWS + 1)
        {
            if resolved.len() >= crate::artifact::MAX_TRUSTED_CATALOG_ROWS {
                return Err(AkitaError::InvalidSetup(
                    "trusted native catalog row count exceeds limit".into(),
                ));
            }
            resolved.push(transition.resolve(profiles, schedule, policy)?);
        }
        for row in &resolved {
            validate_schedule_challenge_hooks(row.schedule(), &ring_challenge_config)?;
        }
        Self::from_resolved_rows(family_name.into(), resolved, policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_wrong_alphabet_and_overflowing_geometry() {
        for (cells, basis, prefix) in [
            (0, 3, 1024),
            (1024, 0, 1024),
            (1024, 4, 1024),
            (1024, 3, 0),
            (usize::MAX, 3, 1024),
            (1024, 3, usize::MAX),
        ] {
            assert!(NativeRootTransition::new(cells, basis, prefix).is_err());
        }
        let transition = NativeRootTransition::new(25_956_096, 3, 16_777_216).unwrap();
        assert_eq!(transition.physical_cells(), 25_956_096);
        assert_eq!(transition.setup_prefix_natural_cells(), 16_777_216);
    }
}
