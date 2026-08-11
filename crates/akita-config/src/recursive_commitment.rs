//! Recursive setup-offloading config adapter.

use crate::CommitmentConfig;
use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_types::{
    ChunkedWitnessCfg, DecompositionParams, OpeningClaimsLayout, SetupMatrixCapacity,
    SisModulusProfileId,
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

    const D: usize = Cfg::D;
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

    fn ring_subfield_embedding_norm_bound() -> u32 {
        Cfg::ring_subfield_embedding_norm_bound()
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

    fn root_honest_fold_policy() -> akita_types::sis::HonestFoldPolicySpec {
        Cfg::root_honest_fold_policy()
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

    fn select_schedule_for_key(
        key: &akita_types::AkitaScheduleLookupKey,
    ) -> Result<akita_schedules::ResolvedScheduleRow, AkitaError> {
        akita_schedules::select_generated_schedule_row(
            key,
            &crate::policy_of::<Self>(),
            Self::ring_challenge_config,
            Self::schedule_catalog(),
        )
    }

    fn select_schedule_for_opening(
        layout: &OpeningClaimsLayout,
    ) -> Result<akita_schedules::ResolvedScheduleRow, AkitaError> {
        Self::select_schedule_for_key(&crate::proof_optimized::proof_optimized_schedule_key(
            layout,
        )?)
    }
}

#[cfg(all(
    test,
    any(
        feature = "schedules-fp128-onehot-recursive",
        feature = "schedules-fp128-onehot-recursive-multi-chunk-w8r2"
    )
))]
mod adaptive_precommit_tests {
    use super::*;
    use crate::proof_optimized::fp128;
    use akita_types::OpeningClaimsLayout;

    fn assert_nv16_precommit<Cfg: CommitmentConfig>() {
        Cfg::select_schedule_for_opening(&OpeningClaimsLayout::new(16, 1).expect("opening layout"))
            .expect("base catalog must expose its independent profile");
    }

    #[cfg(feature = "schedules-fp128-onehot-recursive")]
    #[test]
    fn onehot_recursive_catalog_exposes_base_precommit_profiles() {
        assert_nv16_precommit::<fp128::OneHot>();
    }

    #[cfg(feature = "schedules-fp128-onehot-recursive-multi-chunk-w8r2")]
    #[test]
    fn onehot_recursive_w8r2_catalog_exposes_base_precommit_profiles() {
        assert_nv16_precommit::<fp128::OneHotMultiChunk>();
    }
}

#[cfg(all(test, feature = "schedules-fp128-onehot-recursive"))]
mod tests {
    use super::*;
    use crate::proof_optimized::fp128;
    use akita_types::{AkitaScheduleLookupKey, OpeningClaimsLayout, PolynomialGroupLayout};

    #[test]
    fn prove_layout_uses_the_recursive_catalogs_frozen_precommit_params() {
        type Cfg = RecursiveCommitmentConfig<fp128::OneHot>;

        let precommitted = PolynomialGroupLayout::new(16, 1);
        let final_group = PolynomialGroupLayout::new(32, 2);
        let expected = fp128::OneHot::select_schedule_for_opening(
            &OpeningClaimsLayout::new(precommitted.num_vars(), precommitted.num_polynomials())
                .expect("opening layout"),
        )
        .expect("independent schedule")
        .profiles()
        .final_group;
        let schedule = Cfg::select_schedule_for_key(&AkitaScheduleLookupKey {
            final_group,
            prior_group_profiles: vec![expected, expected],
        })
        .expect("recursive schedule");

        assert_eq!(schedule.schedule().root.params.precommitted_groups.len(), 2);
        assert!(schedule
            .schedule()
            .root
            .params
            .precommitted_groups
            .iter()
            .all(|group| group.descriptor == expected));
    }

    #[test]
    fn scalar_recursive_profile_uses_offloaded_catalog_row() {
        type Cfg = RecursiveCommitmentConfig<fp128::OneHot>;
        let schedule = Cfg::select_schedule_for_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(36),
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
            .any(|fold| fold.params.incoming_setup_prefix.is_some()));
    }
}
