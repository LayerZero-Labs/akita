//! Recursive setup-offloading config adapter.

use crate::{CommitmentConfig, PrecommittedCommitmentConfig};
use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::{
    AkitaScheduleInputs, AkitaScheduleLookupKey, ChunkedWitnessCfg, DecompositionParams,
    FoldSchedule, OpeningClaimsLayout, PrecommittedGroupDescriptor, SetupMatrixEnvelope,
    SisModulusProfileId, SETUP_OFFLOAD_D_SETUP,
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

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixEnvelope, AkitaError> {
        crate::proof_optimized::proof_optimized_max_setup_matrix_size::<Self>(
            max_num_vars,
            max_num_batched_polys,
        )
    }

    fn basis_range() -> (u32, u32) {
        Cfg::basis_range()
    }

    fn onehot_chunk_size() -> usize {
        Cfg::onehot_chunk_size()
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
        if Cfg::D != SETUP_OFFLOAD_D_SETUP {
            return Err(AkitaError::InvalidSetup(
                "recursive setup planning requires D64".to_string(),
            ));
        }
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
        Self::runtime_schedule(recursive_schedule_key::<Self>(layout)?)
    }
}

fn recursive_schedule_key<Cfg: CommitmentConfig>(
    layout: &OpeningClaimsLayout,
) -> Result<AkitaScheduleLookupKey, AkitaError> {
    layout.check()?;
    let final_group = layout.root_final_group_layout()?;
    if layout.num_groups() == 1 {
        return Ok(AkitaScheduleLookupKey::single(final_group));
    }
    let precommitteds = layout
        .root_precommitted_group_layouts()?
        .iter()
        .copied()
        .map(|group| {
            group.validate()?;
            let singleton =
                OpeningClaimsLayout::new(group.num_vars(), group.num_polynomials())?;
            let params = <PrecommittedCommitmentConfig<Cfg> as CommitmentConfig>::get_params_for_batched_commitment(
                &singleton,
            )?;
            Ok(PrecommittedGroupDescriptor::from_params(group, &params))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let key = AkitaScheduleLookupKey {
        final_group,
        precommitteds,
    };
    key.validate()?;
    Ok(key)
}

#[cfg(all(
    test,
    feature = "schedules-fp128-d64-onehot",
    feature = "schedules-fp128-d64-onehot-recursive"
))]
mod tests {
    use super::*;
    use crate::proof_optimized::fp128;
    use akita_field::Prime128OffsetA7F7;
    use akita_types::{
        r_decomp_levels, shared_setup_fold_gadget, PolynomialGroupLayout, PreparedRelationAddress,
        RelationAddressGeometry, SetupContributionGroupInputs, SetupContributionPlan,
        WitnessLayout,
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
        let descriptor = PrecommittedGroupDescriptor::from_params(precommitted, &params);
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
        let layout =
            OpeningClaimsLayout::from_root_groups(&[precommitted, precommitted], final_group)
                .expect("multi-group layout");
        let schedule = Cfg::get_params_for_prove(&layout).expect("recursive schedule");

        let singleton = OpeningClaimsLayout::new(16, 1).expect("singleton precommit layout");
        let params =
            PrecommittedCommitmentConfig::<Cfg>::get_params_for_batched_commitment(&singleton)
                .expect("recursive-catalog precommit params");
        let expected = PrecommittedGroupDescriptor::from_params(precommitted, &params);

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
                    depth_fold: params
                        .num_digits_fold_for_params(
                            group_params,
                            group_layout.num_polynomials(),
                            params.field_bits_for_cache(),
                        )
                        .expect("group fold depth"),
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
