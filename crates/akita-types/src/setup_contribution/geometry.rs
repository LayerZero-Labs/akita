//! Challenge-free setup product geometry: footprint sizing and envelope guards.
//!
//! [`setup_required_for_inputs`] derives the packed-scan footprint (`required`)
//! without fold challenges so NTT sizing, prefix offload, and envelope checks
//! do not depend on `tau1`.

use akita_error::AkitaError;
use jolt_field::FieldCore;

use crate::layout::RelationMatrixRowLayout;
use crate::proof::AkitaExpandedSetup;
use crate::schedule::Schedule;

use super::SetupContributionPlanInputs;

/// Required setup ring rows for one level (challenge-free).
///
/// # Errors
///
/// Returns an error when layout parameters are inconsistent with the canonical
/// M-row packing used by setup sumcheck.
pub fn setup_required_for_inputs<E: FieldCore>(
    inputs: &SetupContributionPlanInputs<E>,
) -> Result<usize, AkitaError> {
    if inputs.num_blocks == 0 || !inputs.num_blocks.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "num_blocks must be a non-zero power of two".into(),
        ));
    }
    if inputs.block_len == 0
        || inputs.depth_open == 0
        || inputs.depth_commit == 0
        || inputs.depth_fold == 0
    {
        return Err(AkitaError::InvalidSetup(
            "setup evaluator layout has zero width".into(),
        ));
    }
    if inputs.num_polys_per_group.len() != inputs.num_groups {
        return Err(AkitaError::InvalidSize {
            expected: inputs.num_groups,
            actual: inputs.num_polys_per_group.len(),
        });
    }

    let z_range = inputs.inner_width;
    let expected_z_range = inputs
        .block_len
        .checked_mul(inputs.depth_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("Z width overflow".into()))?;
    if z_range != expected_z_range {
        return Err(AkitaError::InvalidSize {
            expected: expected_z_range,
            actual: z_range,
        });
    }

    let n_d_active = match inputs.relation_matrix_row_layout {
        RelationMatrixRowLayout::WithDBlock => inputs.n_d,
        RelationMatrixRowLayout::WithoutDBlock => 0,
    };
    // Canonical row layout: consistency (1) | A | B | D.
    let b_rows_total = inputs
        .n_b
        .checked_mul(inputs.num_groups)
        .ok_or_else(|| AkitaError::InvalidSetup("B row count overflow".into()))?;
    let b_row_start = 1usize
        .checked_add(inputs.n_a)
        .ok_or_else(|| AkitaError::InvalidSetup("B row start overflow".into()))?;
    let d_row_start = b_row_start
        .checked_add(b_rows_total)
        .ok_or_else(|| AkitaError::InvalidSetup("D row start overflow".into()))?;
    let a_end = d_row_start
        .checked_add(n_d_active)
        .ok_or_else(|| AkitaError::InvalidSetup("D row end overflow".into()))?;
    if a_end > inputs.rows {
        return Err(AkitaError::InvalidSetup(
            "relation-matrix row weights are inconsistent with setup evaluator layout".into(),
        ));
    }

    let b_per_claim_e = inputs
        .num_blocks
        .checked_mul(inputs.depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("e-hat claim width overflow".into()))?;
    let n_cols_e = inputs
        .num_claims
        .checked_mul(b_per_claim_e)
        .ok_or_else(|| AkitaError::InvalidSetup("e-hat column width overflow".into()))?;
    let max_group_poly_count = inputs
        .num_polys_per_group
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let t_stride = inputs
        .n_a
        .checked_mul(inputs.depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("T stride overflow".into()))?;
    let t_polynomial_width = t_stride
        .checked_mul(inputs.num_blocks)
        .ok_or_else(|| AkitaError::InvalidSetup("T polynomial width overflow".into()))?;
    let n_cols_t = max_group_poly_count
        .checked_mul(t_polynomial_width)
        .ok_or_else(|| AkitaError::InvalidSetup("T column width overflow".into()))?;

    let d_required = n_d_active
        .checked_mul(n_cols_e)
        .ok_or_else(|| AkitaError::InvalidSetup("D setup footprint overflow".into()))?;
    let a_required = inputs
        .n_a
        .checked_mul(z_range)
        .ok_or_else(|| AkitaError::InvalidSetup("A setup footprint overflow".into()))?;
    let b_required = inputs
        .n_b
        .checked_mul(n_cols_t)
        .ok_or_else(|| AkitaError::InvalidSetup("B setup footprint overflow".into()))?;
    let required = d_required.max(b_required).max(a_required);
    if required == 0 {
        return Err(AkitaError::InvalidSetup(
            "setup evaluator requires a non-empty packed footprint".into(),
        ));
    }

    Ok(required)
}

/// Fail-closed envelope guard: `required` inner (`d_a`) rows must fit the shared
/// matrix prefix at `fold_ring_d`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `required` exceeds the envelope.
pub fn ensure_setup_envelope<F: FieldCore>(
    expanded: &AkitaExpandedSetup<F>,
    required: usize,
    fold_ring_d: usize,
) -> Result<(), AkitaError> {
    let setup_len = expanded
        .shared_matrix()
        .total_ring_elements_at_dyn(fold_ring_d)?;
    if required > setup_len {
        return Err(AkitaError::InvalidSetup(
            "shared matrix is too small for selected setup product".into(),
        ));
    }
    Ok(())
}

/// Flat coefficient count for stage-3 prefix offload (`natural_field_len`).
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] on overflow.
pub fn stage3_offload_natural_field_len(
    required: usize,
    d_setup: usize,
) -> Result<usize, AkitaError> {
    required.checked_mul(d_setup).ok_or_else(|| {
        AkitaError::InvalidSetup("setup product natural field length overflow".into())
    })
}

/// Active inner (`d_a`) setup ring rows for one fold, fail-closed on envelope overflow.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `required` exceeds the shared matrix
/// prefix available at `fold_ring_d`.
pub fn setup_active_ring_elems_for_fold<F: FieldCore, E: FieldCore>(
    expanded: &AkitaExpandedSetup<F>,
    inputs: &SetupContributionPlanInputs<E>,
    fold_ring_d: usize,
) -> Result<usize, AkitaError> {
    let required = setup_required_for_inputs(inputs)?;
    ensure_setup_envelope(expanded, required, fold_ring_d)?;
    Ok(required)
}

/// Active inner (`d_a`) setup ring rows at `level`, fail-closed on envelope overflow.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `required` exceeds the shared matrix
/// prefix available at the fold ring dimension.
pub fn setup_active_ring_elems_at<F: FieldCore, E: FieldCore>(
    level: usize,
    schedule: &Schedule,
    expanded: &AkitaExpandedSetup<F>,
    inputs: &SetupContributionPlanInputs<E>,
) -> Result<usize, AkitaError> {
    let exec = schedule.get_execution_schedule(level)?;
    setup_active_ring_elems_for_fold(expanded, inputs, exec.params.d_a())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gadget_row_scalars, RelationMatrixRowLayout, SetupContributionGroupInputs,
        SetupContributionPlan,
    };
    use jolt_field::Prime128OffsetA7F7;

    type F = Prime128OffsetA7F7;

    fn test_scalar(value: u128) -> F {
        F::from_canonical_u128(value)
    }

    fn single_chunk_layout(
        num_blocks: usize,
        offset_z: usize,
        z_len: usize,
        offset_e: usize,
        offset_t: usize,
        offset_r: usize,
    ) -> crate::WitnessLayout {
        crate::WitnessLayout {
            blocks_per_chunk: num_blocks,
            chunks: vec![crate::WitnessChunkLayout {
                offset_z,
                offset_e,
                offset_t,
                offset_r: Some(offset_r),
                global_block_base: 0,
            }],
            chunk_lengths: vec![crate::WitnessChunkLengths {
                z_len,
                e_len: 0,
                t_len: 0,
                r_len: Some(0),
            }],
        }
    }

    fn prepare_single_group_plan(
        inputs: &SetupContributionPlanInputs<F>,
        full_vec_randomness: &[F],
        fold_gadget: &[F],
        chunk_layout: &crate::WitnessLayout,
    ) -> Result<SetupContributionPlan<F>, AkitaError> {
        let single_group =
            SetupContributionGroupInputs::single_group_layout(inputs, chunk_layout, 0)?;
        let groups = std::slice::from_ref(&single_group.group);
        let static_plan = SetupContributionPlan::prepare_static(
            inputs,
            groups,
            single_group.d_row_start,
            single_group.d_rows,
            single_group.d_physical_cols,
        )?;
        SetupContributionPlan::finish_plan::<F>(
            &static_plan,
            full_vec_randomness,
            None,
            None,
            Some(fold_gadget),
            groups,
        )
    }

    #[test]
    fn setup_required_for_inputs_matches_prepare_required() {
        let block_len = 12;
        let depth_commit = 3;
        let depth_fold = 2;
        let num_points = 1;
        let z_range = block_len * depth_commit;
        let offset_z = 0;
        let full_vec_randomness = (0..9)
            .map(|idx| test_scalar(101 + idx as u128))
            .collect::<Vec<_>>();
        let fold_gadget = gadget_row_scalars::<F>(depth_fold, 4);
        let inputs = SetupContributionPlanInputs::<F> {
            relation_matrix_row_layout: RelationMatrixRowLayout::WithoutDBlock,
            rows: 2,
            n_a: 1,
            n_b: 0,
            n_d: 0,
            num_groups: num_points,
            num_polys_per_group: vec![0],
            num_t_vectors: 0,
            num_claims: 1,
            num_blocks: 4,
            block_len,
            depth_open: 16,
            depth_commit,
            depth_fold,
            inner_width: z_range,
            eq_tau1: vec![test_scalar(11), test_scalar(12)].into(),
        };
        let required = setup_required_for_inputs(&inputs).expect("required");
        let chunk_layout = single_chunk_layout(4, offset_z, z_range, 0, 64, 0);
        let plan =
            prepare_single_group_plan(&inputs, &full_vec_randomness, &fold_gadget, &chunk_layout)
                .expect("plan");
        assert_eq!(required, plan.required().unwrap());
    }

    #[test]
    fn setup_required_for_inputs_is_challenge_free() {
        let block_len = 12;
        let depth_commit = 3;
        let depth_fold = 2;
        let z_range = block_len * depth_commit;
        let inputs = SetupContributionPlanInputs::<F> {
            relation_matrix_row_layout: RelationMatrixRowLayout::WithoutDBlock,
            rows: 2,
            n_a: 1,
            n_b: 0,
            n_d: 0,
            num_groups: 1,
            num_polys_per_group: vec![2],
            num_t_vectors: 2,
            num_claims: 1,
            num_blocks: 4,
            block_len,
            depth_open: 16,
            depth_commit,
            depth_fold,
            inner_width: z_range,
            eq_tau1: vec![test_scalar(11), test_scalar(12)].into(),
        };
        let required = setup_required_for_inputs(&inputs).expect("required");
        assert!(required > 0);

        let fold_gadget = gadget_row_scalars::<F>(depth_fold, 4);
        let mut inputs_a = inputs.clone();
        let chunk_layout = single_chunk_layout(4, 0, z_range, 0, 64, 0);
        let plan_a = prepare_single_group_plan(
            &inputs_a,
            &[test_scalar(99), test_scalar(100)],
            &fold_gadget,
            &chunk_layout,
        )
        .expect("plan a");
        inputs_a.eq_tau1 = vec![test_scalar(1); 8].into();
        let plan_b = prepare_single_group_plan(
            &inputs_a,
            &[test_scalar(77), test_scalar(88)],
            &fold_gadget,
            &chunk_layout,
        )
        .expect("plan b");
        assert_eq!(required, plan_a.required().unwrap());
        assert_eq!(plan_a.required().unwrap(), plan_b.required().unwrap());
    }

    #[test]
    fn ensure_setup_envelope_rejects_undersized_matrix() {
        let inputs = SetupContributionPlanInputs::<F> {
            relation_matrix_row_layout: RelationMatrixRowLayout::WithDBlock,
            rows: 8,
            n_a: 2,
            n_b: 2,
            n_d: 1,
            num_groups: 1,
            num_polys_per_group: vec![1],
            num_t_vectors: 1,
            num_claims: 1,
            num_blocks: 4,
            block_len: 16,
            depth_open: 8,
            depth_commit: 2,
            depth_fold: 3,
            inner_width: 32,
            eq_tau1: vec![].into(),
        };
        let required = setup_required_for_inputs(&inputs).expect("required");
        let seed = crate::AkitaSetupSeed {
            max_num_vars: 32,
            max_num_batched_polys: 1,
            gen_ring_dim: 32,
            max_setup_len: 1,
            public_matrix_seed: [1u8; 32],
        };
        let shared = crate::derive_public_matrix_flat::<F, 32>(1, &seed.public_matrix_seed);
        let expanded =
            crate::AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, shared);
        let err = ensure_setup_envelope(&expanded, required, 32).expect_err("undersized");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn setup_required_for_inputs_rejects_non_pow2_num_blocks() {
        let inputs = SetupContributionPlanInputs::<F> {
            relation_matrix_row_layout: RelationMatrixRowLayout::WithDBlock,
            rows: 8,
            n_a: 2,
            n_b: 2,
            n_d: 1,
            num_groups: 1,
            num_polys_per_group: vec![1],
            num_t_vectors: 1,
            num_claims: 1,
            num_blocks: 3,
            block_len: 16,
            depth_open: 8,
            depth_commit: 2,
            depth_fold: 3,
            inner_width: 32,
            eq_tau1: vec![].into(),
        };
        let err = setup_required_for_inputs(&inputs).expect_err("non-pow2 blocks");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn stage3_offload_natural_field_len_uses_d_setup() {
        let required = 128usize;
        let d_setup = crate::SETUP_OFFLOAD_D_SETUP;
        assert_eq!(
            stage3_offload_natural_field_len(required, d_setup).expect("len"),
            required * d_setup
        );
    }
}
