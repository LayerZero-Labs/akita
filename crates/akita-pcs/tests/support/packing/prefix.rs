//! Build the complete frozen setup-prefix group consumed by one packing fold.
use super::geometry::{universal_fold_digit_depth, FoldDigitInputs};
use akita_error::AkitaError;
use akita_params::{CommittedGroupParams, GroupOpenPhaseParams};
use akita_schedules::PlannerPolicy;

pub(super) fn build_prefix(
    predecessor: &CommittedGroupParams,
    consumer: &CommittedGroupParams,
    predecessor_setup_natural_len: usize,
    prefix_ring_slots: usize,
    policy: &PlannerPolicy,
) -> Result<GroupOpenPhaseParams, AkitaError> {
    let outer_slices = consumer.outer_slice_count().get();
    let max_prefix_positions = prefix_ring_slots
        .checked_div(outer_slices)
        .filter(|positions| positions.is_power_of_two())
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "packing setup prefix cannot be partitioned across its outer slices".into(),
            )
        })?;
    let prefix_positions = max_prefix_positions.min(256);
    let prefix_blocks = prefix_ring_slots
        .checked_div(prefix_positions)
        .filter(|blocks| *blocks >= outer_slices)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "packing setup prefix has no balanced split: slots={prefix_ring_slots}, positions={prefix_positions}, outer_slices={outer_slices}",
            ))
        })?;
    let mut prefix_source_params = consumer.clone();
    prefix_source_params.set_setup_prefix(None)?;
    prefix_source_params
        .own_group_mut()
        .profile
        .inner
        .digits
        .log_basis = predecessor.inner().digits.log_basis;
    prefix_source_params
        .own_group_mut()
        .profile
        .inner
        .digits
        .num_digits = akita_params::sis::compute_num_digits_field_width(
        policy.decomposition.field_bits(),
        predecessor.inner().digits.log_basis,
    );
    let prefix_inner_width = prefix_positions
        .checked_mul(prefix_source_params.inner().digits.num_digits)
        .ok_or_else(|| AkitaError::InvalidSetup("packing prefix A width overflow".into()))?;
    prefix_source_params.own_group_mut().profile.inner.matrix =
        akita_params::InnerCommitMatrixParams::try_new_with_min_rank(
            prefix_source_params
                .inner()
                .matrix
                .sis_table_key()
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("packing prefix requires a L-infinity A matrix".into())
                })?,
            prefix_inner_width,
        )?;
    let prefix_outer_width = akita_params::CommitmentSliceGeometry::try_new(
        prefix_source_params.outer_slice_count(),
        prefix_blocks,
        1,
        prefix_source_params.inner().matrix.output_rank(),
        prefix_source_params.outer().digits.num_digits,
        prefix_source_params.d_a(),
        prefix_source_params.role_dims().d_b(),
    )?
    .physical_input_width();
    prefix_source_params.own_group_mut().profile.outer.matrix =
        akita_params::OuterCommitMatrixParams::try_new_with_min_rank(
            prefix_source_params.outer().matrix.sis_table_key(),
            prefix_outer_width,
        )?;
    let mut prefix_params = akita_params::setup_prefix_precommitted_params(
        &prefix_source_params,
        akita_params::padded_setup_prefix_len(predecessor_setup_natural_len),
    )?;
    if prefix_params.profile.inner.matrix.ring_dimension() != consumer.d_a() {
        return Err(AkitaError::InvalidSetup(
            "packing setup prefix left the base row's planned commitment class".into(),
        ));
    }
    prefix_params.opening.opening_method = consumer.opening_method();
    prefix_params.opening.fold_challenge_config = consumer.fold_challenge_config();
    prefix_params.opening.num_digits_fold = universal_fold_digit_depth(
        FoldDigitInputs::of_group(&prefix_params),
        policy.decomposition.field_bits(),
        1,
        consumer.witness_chunk.num_chunks,
    )?;
    let prefix_a_bound = akita_params::sis::rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        prefix_params.profile.inner.matrix.ring_dimension(),
        prefix_params.opening.log_basis_open,
        &prefix_params.opening.fold_challenge_config,
        prefix_params.opening.num_digits_fold,
        consumer.witness_chunk.num_chunks,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("packing prefix has no audited A bound".into()))?;
    let mut prefix_a_key = prefix_params
        .profile
        .inner
        .matrix
        .sis_table_key()
        .ok_or_else(|| AkitaError::InvalidSetup("packing prefix requires Linf A".into()))?;
    prefix_a_key.coeff_linf_bound = prefix_a_bound;
    prefix_params.profile.inner.matrix =
        akita_params::InnerCommitMatrixParams::try_new_with_min_rank(
            prefix_a_key,
            prefix_params.profile.inner.matrix.input_width(),
        )?;
    let prefix_outer_width = akita_params::CommitmentSliceGeometry::try_new(
        prefix_params.profile.outer_slice_count,
        prefix_params.profile.blocks.live_blocks,
        1,
        prefix_params.profile.inner.matrix.output_rank(),
        prefix_params.profile.outer.digits.num_digits,
        prefix_params.profile.inner.matrix.ring_dimension(),
        prefix_params.profile.outer.matrix.ring_dimension(),
    )?
    .physical_input_width();
    prefix_params.profile.outer.matrix =
        akita_params::OuterCommitMatrixParams::try_new_with_min_rank(
            prefix_params.profile.outer.matrix.sis_table_key(),
            prefix_outer_width,
        )?;
    Ok(akita_params::scheduled_setup_prefix(
        predecessor_setup_natural_len,
        prefix_params,
    ))
}
