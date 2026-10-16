//! Build one fully sized successor, including its setup-prefix group and D input.
use super::geometry::{rebuild_group_output_matrices, universal_fold_digit_depth, FoldDigitInputs};
use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_params::{FoldParams, OpeningClaimsLayout};
use akita_schedules::PlannerPolicy;

pub(super) fn build_successor(
    template: &FoldParams,
    predecessor: &FoldParams,
    predecessor_opening_batch: &OpeningClaimsLayout,
    level: usize,
    challenge_subring_dimension: usize,
    policy: &PlannerPolicy,
) -> Result<FoldParams, AkitaError> {
    let mut successor = template.clone();
    successor.input_witness_len = predecessor.output_witness_len;
    let successor_witness = &mut successor.params;
    if level >= 2 {
        successor_witness.own_group_mut().profile.outer_slice_count =
            akita_params::CommitmentSliceCount::ONE;
    }
    // Recursive witness coefficients are the predecessor's opening digits.
    // A reused template may use a different opening basis at the next fold.
    successor_witness.own_group_mut().profile.inner.digits =
        akita_params::GadgetDigits::new(predecessor.params.open().digits.log_basis, 1);
    successor_witness.own_group_mut().opening.opening_method =
        akita_params::OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        };
    successor_witness.source_encoding =
        akita_params::CommittedSourceEncoding::CanonicalCoefficientTable;
    successor_witness
        .own_group_mut()
        .opening
        .fold_challenge_config = SparseChallengeConfig::production_for_ring_dim(
        challenge_subring_dimension,
    )
    .ok_or_else(|| {
        AkitaError::InvalidSetup("successor packing subring is not in the production ladder".into())
    })?;
    successor_witness
        .own_group_mut()
        .profile
        .blocks
        .live_ring_elements_per_claim = predecessor
        .output_witness_len
        .div_ceil(successor_witness.d_a());
    let successor_field_len = akita_error::checked::product([
        successor_witness.blocks().live_ring_elements_per_claim,
        successor_witness.d_a(),
    ])
    .ok_or_else(|| AkitaError::InvalidSetup("packing successor input overflow".into()))?;
    let successor_num_vars = akita_error::checked::ceil_log2(successor_field_len)
        .ok_or_else(|| AkitaError::InvalidSetup("packing successor input is empty".into()))?;
    successor_witness.own_group_mut().profile.group =
        akita_params::PolynomialGroupLayout::singleton(successor_num_vars);

    let predecessor_setup_natural_len =
        akita_params::active_setup_field_len(&predecessor.params, predecessor_opening_batch)?;
    let predecessor_setup_prefix_len =
        akita_params::padded_setup_prefix_len(predecessor_setup_natural_len);
    let prefix_ring_slots = predecessor_setup_prefix_len
        .checked_div(successor_witness.d_a())
        .filter(|slots| {
            *slots != 0 && predecessor_setup_prefix_len.is_multiple_of(successor_witness.d_a())
        })
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "packing setup prefix does not align to the successor A ring".into(),
            )
        })?;
    successor_witness
        .own_group_mut()
        .profile
        .blocks
        .positions_per_block = prefix_ring_slots.next_power_of_two();
    successor_witness.own_group_mut().profile.blocks.live_blocks = successor_witness
        .blocks()
        .live_ring_elements_per_claim
        .div_ceil(successor_witness.blocks().positions_per_block);
    successor_witness.own_group_mut().opening.num_digits_fold = universal_fold_digit_depth(
        FoldDigitInputs::of_fold(successor_witness),
        policy.decomposition.field_bits(),
        1,
        successor_witness.witness_chunk.num_chunks,
    )?;
    let successor_a_width = successor_witness
        .blocks()
        .positions_per_block
        .checked_mul(successor_witness.inner().digits.num_digits)
        .ok_or_else(|| AkitaError::InvalidSetup("packing successor A width overflow".into()))?;
    let mut successor_a_key = successor_witness
        .inner()
        .matrix
        .sis_table_key()
        .ok_or_else(|| {
            AkitaError::InvalidSetup("packing successor requires a L-infinity A matrix".into())
        })?;
    successor_a_key.coeff_linf_bound = akita_params::sis::rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        successor_witness.d_a(),
        successor_witness.open().digits.log_basis,
        &successor_witness.fold_challenge_config(),
        successor_witness.num_digits_fold(),
        successor_witness.witness_chunk.num_chunks,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("packing successor has no audited A bound".into()))?;
    successor_witness.own_group_mut().profile.inner.matrix =
        akita_params::InnerCommitMatrixParams::try_new_with_min_rank(
            successor_a_key,
            successor_a_width,
        )?;
    rebuild_group_output_matrices(successor_witness, 1, policy.claim_ext_degree)?;

    let incoming_setup_prefix = super::prefix::build_prefix(
        &predecessor.params,
        successor_witness,
        predecessor_setup_natural_len,
        prefix_ring_slots,
        policy,
    )?;
    successor_witness.set_setup_prefix(Some(incoming_setup_prefix))?;
    let successor_d_width = successor_witness
        .open()
        .matrix
        .input_width()
        .checked_add(
            incoming_setup_prefix
                .d_segment_width(policy.claim_ext_degree, successor_witness.role_dims().d_d())?,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("packing successor D width overflow".into()))?;
    successor_witness.open_matrix = akita_params::OpenCommitMatrixParams::try_new_with_min_rank(
        successor_witness.open_matrix.sis_table_key(),
        successor_d_width,
    )?;
    let successor_opening_batch = akita_params::suffix_opening_layout(
        predecessor.output_witness_len,
        Some(predecessor_setup_natural_len),
    )?;
    successor.output_witness_len = successor_witness.output_witness_len_for_field_bits(
        policy.decomposition.field_bits(),
        policy.claim_ext_degree,
        &successor_opening_batch,
    )?;
    Ok(successor)
}
