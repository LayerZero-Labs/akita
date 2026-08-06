//! Recursive setup-offloading config adapter.

use crate::CommitmentConfig;
use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::{
    AkitaScheduleInputs, ChunkedWitnessCfg, DecompositionParams, FoldSchedule, OpeningClaimsLayout,
    SetupMatrixCapacity, SisModulusProfileId,
};
#[cfg(any(
    feature = "schedules-fp128-d64-onehot-recursive",
    feature = "schedules-fp128-d64-onehot-recursive-multi-chunk-w8r2"
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

    fn fold_challenge_shape_at_level(inputs: AkitaScheduleInputs) -> TensorChallengeShape {
        Cfg::fold_challenge_shape_at_level(inputs)
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

    fn basis_range() -> (u32, u32) {
        Cfg::basis_range()
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
        #[cfg(feature = "schedules-fp128-d64-onehot-recursive")]
        {
            if TypeId::of::<Cfg>() == TypeId::of::<crate::proof_optimized::fp128::D64OneHot>() {
                return Some(akita_schedules::fp128_d64_onehot_recursive_table());
            }
        }
        #[cfg(feature = "schedules-fp128-d64-onehot-recursive-multi-chunk-w8r2")]
        {
            if TypeId::of::<Cfg>()
                == TypeId::of::<crate::proof_optimized::fp128::D64OneHotMultiChunk>()
            {
                return Some(akita_schedules::fp128_d64_onehot_recursive_multi_chunk_w8r2_table());
            }
        }
        None
    }

    fn runtime_schedule(
        key: akita_types::AkitaScheduleLookupKey,
    ) -> Result<akita_types::FoldSchedule, AkitaError> {
        if key.precommitteds.is_empty() {
            return Cfg::runtime_schedule(key);
        }
        akita_schedules::resolve_group_batch_schedule(
            &key,
            &crate::policy_of::<Self>(),
            Self::ring_challenge_config,
            Self::fold_challenge_shape_at_level,
            Self::schedule_catalog(),
        )
    }

    fn get_params_for_prove(layout: &OpeningClaimsLayout) -> Result<FoldSchedule, AkitaError> {
        Self::runtime_schedule(crate::proof_optimized::proof_optimized_schedule_key(
            layout,
        )?)
    }
}

#[cfg(all(test, feature = "schedules-fp128-d64-onehot-recursive"))]
mod tests {
    use super::*;
    use crate::proof_optimized::fp128;
    use crate::PrecommittedCommitmentConfig;
    use akita_field::Prime128OffsetA7F7;
    use akita_types::{
        r_decomp_levels, shared_setup_fold_gadget, AkitaScheduleLookupKey, CommittedGroupProfile,
        PolynomialGroupLayout, PreparedRelationAddress, RelationAddressGeometry,
        SetupContributionGroupInputs, SetupContributionPlan, WitnessLayout,
    };

    fn scalar(value: u128) -> Prime128OffsetA7F7 {
        Prime128OffsetA7F7::from_canonical_u128(value)
    }

    fn eq_at_index(point: &[Prime128OffsetA7F7], index: usize) -> Prime128OffsetA7F7 {
        point.iter().copied().enumerate().fold(
            Prime128OffsetA7F7::one(),
            |weight, (bit, challenge)| {
                if (index >> bit) & 1 == 1 {
                    weight * challenge
                } else {
                    weight * (Prime128OffsetA7F7::one() - challenge)
                }
            },
        )
    }

    fn profiling_schedule() -> (FoldSchedule, OpeningClaimsLayout) {
        type Cfg = RecursiveCommitmentConfig<fp128::D64OneHot>;

        let precommitted = PolynomialGroupLayout::new(16, 1);
        let singleton = OpeningClaimsLayout::new(16, 1).expect("singleton precommit layout");
        let params =
            PrecommittedCommitmentConfig::<Cfg>::get_params_for_batched_commitment(&singleton)
                .expect("recursive-catalog precommit params");
        let descriptor = CommittedGroupProfile::from_params(precommitted, &params);
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(32, 2),
            precommitteds: vec![descriptor, descriptor],
        };
        let layout = key.opening_layout().expect("profile opening layout");
        let schedule = Cfg::runtime_schedule(key).expect("recursive profile schedule");
        (schedule, layout)
    }

    #[test]
    fn prove_layout_uses_the_recursive_catalogs_frozen_precommit_params() {
        type Cfg = RecursiveCommitmentConfig<fp128::D64OneHot>;

        let precommitted = PolynomialGroupLayout::new(16, 1);
        let final_group = PolynomialGroupLayout::new(32, 2);
        let singleton = OpeningClaimsLayout::new(16, 1).expect("singleton precommit layout");
        let params =
            PrecommittedCommitmentConfig::<Cfg>::get_params_for_batched_commitment(&singleton)
                .expect("recursive-catalog precommit params");
        let expected = CommittedGroupProfile::from_params(precommitted, &params);
        let schedule = Cfg::runtime_schedule(AkitaScheduleLookupKey {
            final_group,
            precommitteds: vec![expected, expected],
        })
        .expect("recursive schedule");

        assert_eq!(schedule.root.params.precommitted_groups.len(), 2);
        assert!(schedule
            .root
            .params
            .precommitted_groups
            .iter()
            .all(|group| group.descriptor == expected));
    }

    #[test]
    fn profiling_schedule_tensor_setup_weights_match_dense_materialization() {
        let (schedule, opening_batch) = profiling_schedule();
        let params = &schedule.root.params.final_group.commitment;
        let witness_layout = WitnessLayout::new(
            params,
            &opening_batch,
            params.witness_chunk.num_chunks,
            r_decomp_levels::<Prime128OffsetA7F7>(params.log_basis_open),
        )
        .expect("root witness layout");
        let rows = witness_layout.r_rows().len();
        let order = opening_batch.root_group_order().expect("relation order");
        let groups = order
            .iter()
            .map(|&group_id| {
                let group_params = params
                    .group_params(&opening_batch, group_id)
                    .expect("group params");
                let group_layout = opening_batch.group_layout(group_id).expect("group layout");
                SetupContributionGroupInputs {
                    group_id,
                    num_claims: group_layout.num_polynomials(),
                    depth_fold: group_params.num_digits_fold(),
                    a_row_start: params
                        .a_row_range(&opening_batch, group_id)
                        .expect("A rows")
                        .start,
                    b_row_start: params
                        .commitment_row_range(&opening_batch, group_id)
                        .expect("B rows")
                        .start,
                }
            })
            .collect::<Vec<_>>();
        let next_d = schedule.recursive_folds[0].params.witness.d_a();
        let address_geometry = RelationAddressGeometry::new(
            params.role_dims(),
            next_d,
            witness_layout.live_coeff_len(),
        )
        .expect("relation address geometry");
        let address_point = (0..address_geometry.relation_lane_variable_count())
            .map(|index| scalar(101 + index as u128))
            .collect::<Vec<_>>();
        let eq_tau1 = (0..rows)
            .map(|index| scalar(211 + index as u128))
            .collect::<Vec<_>>();
        let alpha = scalar(3);
        let fold_gadget =
            shared_setup_fold_gadget::<Prime128OffsetA7F7>(params, &opening_batch, &groups);
        let plan = SetupContributionPlan::prepare::<Prime128OffsetA7F7>(
            params,
            &opening_batch,
            eq_tau1.into(),
            &witness_layout,
            &groups,
            PreparedRelationAddress::new(&address_point).expect("relation address"),
            fold_gadget.as_deref(),
            address_geometry,
        )
        .expect("setup tensor plan");
        let setup_idx_bits = plan.required().next_power_of_two().trailing_zeros() as usize;
        let rho = (0..setup_idx_bits)
            .map(|index| scalar(401 + index as u128))
            .collect::<Vec<_>>();
        let dense_mle = plan
            .materialize_setup_index_weights(alpha)
            .expect("dense setup weights")
            .into_iter()
            .enumerate()
            .fold(Prime128OffsetA7F7::zero(), |acc, (index, weight)| {
                acc + eq_at_index(&rho, index) * weight
            });
        assert_eq!(
            plan.evaluate_setup_index_weight_mle(&rho, alpha)
                .expect("tensor setup weight MLE"),
            dense_mle
        );
    }
}
