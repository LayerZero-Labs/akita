//! Checked packing catalogs for root and setup-prefixed successor roundtrips.
use akita_challenges::SparseChallengeConfig;
use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_params::ScheduleLookupKey;
use akita_params::{
    CommittedGroupBatchProfile, DecompositionParams, GroupCommitPhaseParams, SetupMatrixCapacity,
    SisModulusProfileId,
};
use std::marker::PhantomData;

mod geometry;
mod prefix;
mod root;
mod successor;
mod unsupported;

use geometry::retarget_synthetic_terminal;

/// Test-only catalog adapter that replaces the root opening with reduced-width
/// coefficient packing over the smallest production challenge subring.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RootCoefficientPackingConfig<Base>(PhantomData<fn() -> Base>);

/// Test-only adapter for rejected evaluation-trace rows. `LEVEL=0` mutates
/// the canonical extension-field root; `LEVEL=1` mutates the first recursive
/// witness and its incoming prefix and makes that fold multi-chunk.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct UnsupportedEvaluationTraceConfig<Base, const LEVEL: usize>(
    PhantomData<fn() -> Base>,
);

impl<Base> RootCoefficientPackingConfig<Base>
where
    Base: CommitmentConfig + 'static,
{
    pub(crate) fn setup_matrix_capacity(
        catalog: &akita_config::TrustedScheduleCatalog<Base>,
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixCapacity, AkitaError> {
        let base = akita_config::SetupRequirements::from_catalog::<Base>(
            catalog,
            max_num_vars,
            max_num_batched_polys,
        )
        .map(|requirements| requirements.matrix_capacity())?;
        Ok(SetupMatrixCapacity {
            num_field_elements: base.num_field_elements.checked_mul(16).ok_or_else(|| {
                AkitaError::InvalidSetup("coefficient-packing test setup capacity overflow".into())
            })?,
        })
    }

    pub(crate) fn derive_catalog_row(
        catalog: &akita_config::ValidatedScheduleCatalog,
        key: &ScheduleLookupKey,
        challenge_subring_dimension: usize,
        prefixed_fold_count: usize,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        if !key.precommitteds.is_empty() {
            return Err(AkitaError::UnsupportedSchedule(
                "the coefficient-packing test catalog supports one root group".into(),
            ));
        }
        let base = catalog.resolve_key(key)?;
        // Folds 0 and 1 always use quotient lift with packing, so the row's
        // own first recursive fold is a packing-capable quotient successor for
        // Stage 3. It follows this root, so its inner digits chain from the
        // root's opening digits.
        let successor_template = base
            .schedule()
            .recursive_folds
            .first()
            .cloned()
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "packing Stage 3 test requires a recursive successor template".into(),
                )
            })?;
        let policy = catalog.policy();
        let mut schedule = base.schedule().clone();
        schedule.root = root::build_root(&schedule.root, key, challenge_subring_dimension, policy)?;
        schedule.recursive_folds.clear();
        let mut opening_batch = key.opening_layout()?;
        for level in 1..=prefixed_fold_count {
            let predecessor = schedule.recursive_folds.last().unwrap_or(&schedule.root);
            let successor = successor::build_successor(
                &successor_template,
                predecessor,
                &opening_batch,
                level,
                challenge_subring_dimension,
                policy,
            )?;
            opening_batch = akita_params::suffix_opening_layout(
                successor.input_witness_len,
                successor
                    .incoming_setup_prefix()
                    .and_then(|prefix| prefix.setup_natural_len),
            )?;
            schedule.recursive_folds.push(successor);
        }

        retarget_synthetic_terminal::<Self>(&mut schedule, policy)?;

        schedule.validate_nonterminal_opening_execution(Self::EXT_DEGREE)?;
        let root = &schedule.root.params;
        let profiles = CommittedGroupBatchProfile {
            final_group: GroupCommitPhaseParams::try_from_params(key.final_group, root)?,
            precommitteds: Vec::new(),
        };
        akita_config::ResolvedScheduleRow::try_new(profiles, schedule, policy)
    }
}

impl<Base> CommitmentConfig for RootCoefficientPackingConfig<Base>
where
    Base: CommitmentConfig + 'static,
{
    type Field = Base::Field;
    type ExtField = Base::ExtField;

    const EXT_DEGREE: usize = Base::EXT_DEGREE;
    const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
        Base::RING_DIMENSION_SCHEDULE;

    fn schedule_family_name() -> &'static str {
        Base::schedule_family_name()
    }

    fn decomposition() -> DecompositionParams {
        Base::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Base::ring_challenge_config(d)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Base::sis_modulus_profile()
    }

    fn opening_basis_range() -> (u32, u32) {
        Base::opening_basis_range()
    }

    fn inner_basis_range() -> (u32, u32) {
        Base::inner_basis_range()
    }

    fn committed_source_class() -> akita_params::sis::CommittedSourceClass {
        Base::committed_source_class()
    }

    fn recursive_setup_planning() -> bool {
        Base::recursive_setup_planning()
    }

    fn selection_policy() -> akita_schedules::SelectionPolicyId {
        Base::selection_policy()
    }
}

impl<Base, const LEVEL: usize> CommitmentConfig for UnsupportedEvaluationTraceConfig<Base, LEVEL>
where
    Base: CommitmentConfig + 'static,
{
    type Field = Base::Field;
    type ExtField = Base::ExtField;

    const EXT_DEGREE: usize = Base::EXT_DEGREE;
    const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
        Base::RING_DIMENSION_SCHEDULE;

    fn schedule_family_name() -> &'static str {
        Base::schedule_family_name()
    }

    fn decomposition() -> DecompositionParams {
        Base::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Base::ring_challenge_config(d)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Base::sis_modulus_profile()
    }

    fn opening_basis_range() -> (u32, u32) {
        Base::opening_basis_range()
    }

    fn inner_basis_range() -> (u32, u32) {
        Base::inner_basis_range()
    }

    fn committed_source_class() -> akita_params::sis::CommittedSourceClass {
        Base::committed_source_class()
    }

    fn recursive_setup_planning() -> bool {
        Base::recursive_setup_planning()
    }

    fn selection_policy() -> akita_schedules::SelectionPolicyId {
        Base::selection_policy()
    }
}
