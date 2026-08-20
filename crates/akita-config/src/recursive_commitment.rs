//! Recursive setup-offloading config adapter.

use crate::CommitmentConfig;
use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_types::{
    ChunkedWitnessCfg, DecompositionParams, SetupMatrixCapacity, SisModulusProfileId,
};
#[cfg(any(
    feature = "schedules-fp128-onehot-recursive",
    feature = "schedules-fp128-onehot-recursive-multi-chunk-w8r2"
))]
use std::any::TypeId;
use std::marker::PhantomData;

/// Config adapter that enables recursion-aware setup offloading schedules.
#[derive(Clone, Copy, Debug, Default)]
pub struct RecursiveCommitmentConfig<Cfg>(PhantomData<Cfg>);

impl<Cfg: CommitmentConfig> CommitmentConfig for RecursiveCommitmentConfig<Cfg> {
    type Field = Cfg::Field;
    type ExtField = Cfg::ExtField;

    const RING_DIMENSION_SCHEDULE_MODE: akita_schedules::RingDimensionScheduleMode =
        Cfg::RING_DIMENSION_SCHEDULE_MODE;
    fn decomposition() -> DecompositionParams {
        Cfg::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Cfg::ring_challenge_config(d)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Cfg::sis_modulus_profile()
    }

    fn setup_matrix_capacity(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixCapacity, AkitaError> {
        crate::proof_optimized::proof_optimized_setup_matrix_capacity::<Self>(
            max_num_vars,
            max_num_batched_polys,
        )
    }

    fn opening_basis_range() -> (u32, u32) {
        Cfg::opening_basis_range()
    }

    fn inner_basis_range() -> (u32, u32) {
        Cfg::inner_basis_range()
    }

    fn committed_source_class() -> akita_types::sis::CommittedSourceClass {
        Cfg::committed_source_class()
    }

    fn chunked_witness_cfg() -> ChunkedWitnessCfg {
        Cfg::chunked_witness_cfg()
    }

    fn recursive_setup_planning() -> bool {
        true
    }

    fn schedule_catalog() -> Option<akita_schedules::GeneratedScheduleTable> {
        #[cfg(feature = "schedules-fp128-onehot-recursive")]
        {
            if TypeId::of::<Cfg>() == TypeId::of::<crate::proof_optimized::fp128::OneHot>() {
                return Some(akita_schedules::fp128_onehot_recursive_table());
            }
        }
        #[cfg(feature = "schedules-fp128-onehot-recursive-multi-chunk-w8r2")]
        {
            if TypeId::of::<Cfg>()
                == TypeId::of::<crate::proof_optimized::fp128::OneHotMultiChunk>()
            {
                return Some(akita_schedules::fp128_onehot_recursive_multi_chunk_w8r2_table());
            }
        }
        None
    }
}

#[cfg(all(
    test,
    any(
        feature = "schedules-fp128-onehot-recursive",
        feature = "schedules-fp128-onehot-recursive-multi-chunk-w8r2"
    )
))]
mod base_precommit_tests {
    use super::*;
    use crate::proof_optimized::fp128;
    use akita_types::PolynomialGroupLayout;

    /// A recursive companion ships no row without precommitted groups at a precommit
    /// layout, so the caller precommits under `Cfg` and proves the grouped
    /// root under `RecursiveCommitmentConfig<Cfg>`. Assert both halves of that
    /// split: the base config produces the profile, and the recursive one
    /// rejects the same request.
    fn assert_nv16_precommit_needs_the_base_config<Cfg: CommitmentConfig>() {
        let group = PolynomialGroupLayout::new(16, 1);
        Cfg::profile_without_precommitted_groups(group)
            .expect("base catalog must expose its independent profile");
        RecursiveCommitmentConfig::<Cfg>::profile_without_precommitted_groups(group)
            .expect_err("a recursive companion must not self-derive a precommit profile");
    }

    #[cfg(feature = "schedules-fp128-onehot-recursive")]
    #[test]
    fn onehot_recursive_precommit_comes_from_the_base_catalog() {
        assert_nv16_precommit_needs_the_base_config::<fp128::OneHot>();
    }

    #[cfg(feature = "schedules-fp128-onehot-recursive-multi-chunk-w8r2")]
    #[test]
    fn onehot_recursive_w8r2_precommit_comes_from_the_base_catalog() {
        assert_nv16_precommit_needs_the_base_config::<fp128::OneHotMultiChunk>();
    }
}

#[cfg(all(test, feature = "schedules-fp128-onehot-recursive"))]
mod tests {
    use super::*;
    use crate::proof_optimized::fp128;
    use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};

    #[test]
    fn recursive_grouped_row_freezes_the_base_configs_precommit_profile() {
        type Cfg = RecursiveCommitmentConfig<fp128::OneHot>;

        let precommitted = PolynomialGroupLayout::new(16, 1);
        let final_group = PolynomialGroupLayout::new(32, 2);
        let expected = fp128::OneHot::profile_without_precommitted_groups(precommitted)
            .expect("independent profile");
        let schedule = Cfg::resolve_catalog_row_for_key(&AkitaScheduleLookupKey {
            final_group,
            precommitteds: vec![expected, expected],
        })
        .expect("recursive schedule");

        assert_eq!(
            schedule.schedule().root.params.precommitted_groups().len(),
            2
        );
        assert!(schedule
            .schedule()
            .root
            .params
            .precommitted_groups
            .iter()
            .all(|group| group.profile == expected));
    }

    #[test]
    fn scalar_recursive_profile_uses_offloaded_catalog_row() {
        type Cfg = RecursiveCommitmentConfig<fp128::OneHot>;
        let schedule = Cfg::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::new(36, 1),
        ))
        .expect("scalar recursive schedule");

        assert!(schedule
            .schedule()
            .root
            .params
            .precommitted_groups
            .is_empty());
        assert!(schedule
            .schedule()
            .recursive_folds
            .iter()
            .any(|fold| fold.params.setup_prefix().is_some()));
    }
}
