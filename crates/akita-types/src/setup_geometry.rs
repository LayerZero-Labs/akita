//! Challenge-free setup product geometry shared by prover and verifier.
//!
//! [`compute_setup_layout`] factors the row-layout footprint (`required`) out of
//! [`crate::SetupContributionPlan::prepare`] so NTT sizing, prefix offload, and
//! NTT envelope checks do not depend on `tau1` / fold challenges.

use akita_field::{AkitaError, FieldCore};

use crate::layout::MRowLayout;
use crate::proof::AkitaExpandedSetup;
use crate::schedule::Schedule;
use crate::setup_contribution::SetupContributionPlanInputs;

/// Full row-layout footprint used by weight materialization and geometry tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetupLayoutFootprint {
    pub required: usize,
    pub d_required: usize,
    pub b_required: usize,
    pub a_required: usize,
    pub b_inner_required: usize,
    pub f_required: usize,
    pub a_end: usize,
    pub d_start: usize,
    pub f_start: usize,
    pub b_start: usize,
    pub b_inner_start: usize,
    pub a_start: usize,
    pub n_d_active: usize,
    pub tiered: bool,
    pub b_inner_stride: usize,
    pub f_stride: usize,
    pub n_cols_e: usize,
    pub n_cols_t: usize,
    pub z_range: usize,
    pub stride_t: usize,
    pub cols_per_poly_t: usize,
    pub b_per_claim_e: usize,
}

/// Pure, challenge-free row-layout footprint for a setup level.
///
/// # Errors
///
/// Returns an error when layout parameters are inconsistent with the canonical
/// M-row packing used by setup sumcheck.
pub fn compute_setup_layout<E: FieldCore>(
    inputs: &SetupContributionPlanInputs<E>,
) -> Result<SetupLayoutFootprint, AkitaError> {
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
    if inputs.num_polys_per_segment.len() != inputs.num_segments {
        return Err(AkitaError::InvalidSize {
            expected: inputs.num_segments,
            actual: inputs.num_polys_per_segment.len(),
        });
    }

    let z_range = inputs.inner_width;
    let expected_z_range = checked_mul(inputs.block_len, inputs.depth_commit, "Z width")?;
    if z_range != expected_z_range {
        return Err(AkitaError::InvalidSize {
            expected: expected_z_range,
            actual: z_range,
        });
    }

    let n_d_active = match inputs.m_row_layout {
        MRowLayout::WithDBlock => inputs.n_d,
        MRowLayout::WithoutDBlock => 0,
    };
    // Canonical row layout: consistency (1) | A | B | D.
    let a_start = 1usize;
    let b_start = checked_add(a_start, inputs.n_a, "B row start")?;
    let b_rows_total = checked_mul(inputs.n_b, inputs.num_segments, "B row count")?;
    let d_start = checked_add(b_start, b_rows_total, "D row start")?;
    let a_end = checked_add(d_start, n_d_active, "D row end")?;
    if a_end > inputs.rows {
        return Err(AkitaError::InvalidSetup(
            "M-row weights are inconsistent with setup evaluator layout".into(),
        ));
    }

    let stride_t = checked_mul(inputs.n_a, inputs.depth_open, "T stride")?;
    let cols_per_poly_t = checked_mul(stride_t, inputs.num_blocks, "T polynomial width")?;
    let b_per_claim_e = checked_mul(inputs.num_blocks, inputs.depth_open, "e-hat claim width")?;
    let n_cols_e = checked_mul(inputs.num_claims, b_per_claim_e, "e-hat column width")?;
    let max_group_poly_count = inputs
        .num_polys_per_segment
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let n_cols_t = checked_mul(max_group_poly_count, cols_per_poly_t, "T column width")?;

    let d_required = checked_mul(n_d_active, n_cols_e, "D setup footprint")?;
    let a_required = checked_mul(inputs.n_a, z_range, "A setup footprint")?;
    let b_required = checked_mul(inputs.n_b, n_cols_t, "B setup footprint")?;
    let required = d_required.max(b_required).max(a_required);
    if required == 0 {
        return Err(AkitaError::InvalidSetup(
            "setup evaluator requires a non-empty packed footprint".into(),
        ));
    }

    Ok(SetupLayoutFootprint {
        required,
        d_required,
        b_required,
        a_required,
        b_inner_required: 0,
        f_required: 0,
        a_end,
        d_start,
        f_start: b_start,
        b_start,
        b_inner_start: 0,
        a_start,
        n_d_active,
        tiered: false,
        b_inner_stride: 0,
        f_stride: 0,
        n_cols_e,
        n_cols_t,
        z_range,
        stride_t,
        cols_per_poly_t,
        b_per_claim_e,
    })
}

/// Required setup ring rows for one level (challenge-free).
///
/// # Errors
///
/// Returns an error when layout parameters are inconsistent.
pub fn setup_required_for_inputs<E: FieldCore>(
    inputs: &SetupContributionPlanInputs<E>,
) -> Result<usize, AkitaError> {
    Ok(compute_setup_layout(inputs)?.required)
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

#[inline]
fn checked_add(lhs: usize, rhs: usize, name: &'static str) -> Result<usize, AkitaError> {
    lhs.checked_add(rhs)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{name} overflow")))
}

#[inline]
fn checked_mul(lhs: usize, rhs: usize, name: &'static str) -> Result<usize, AkitaError> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{name} overflow")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gadget_row_scalars, MRowLayout, SetupContributionPlan};
    use akita_field::Prime128OffsetA7F7;

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
                offset_u: None,
                offset_r: Some(offset_r),
                global_block_base: 0,
            }],
            chunk_lengths: vec![crate::WitnessChunkLengths {
                z_len,
                e_len: 0,
                t_len: 0,
                u_len: None,
                r_len: Some(0),
            }],
            quotient_layout: crate::WitnessLayout::empty_quotient_layout(),
        }
    }

    #[test]
    fn compute_setup_layout_matches_prepare_required() {
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
            eq_tau1: vec![test_scalar(11), test_scalar(12)],
            num_t_vectors: 0,
            num_blocks: 4,
            num_claims: 1,
            depth_open: 16,
            depth_commit,
            depth_fold,
            block_len,
            inner_width: z_range,
            n_a: 1,
            n_d: 0,
            m_row_layout: MRowLayout::WithoutDBlock,
            n_b: 0,
            num_segments: num_points,
            rows: 2,
            num_polys_per_segment: vec![0],
        };
        let layout = compute_setup_layout(&inputs).expect("layout");
        let chunk_layout = single_chunk_layout(4, offset_z, z_range, 0, 64, 0);
        let plan = SetupContributionPlan::prepare::<F>(
            &inputs,
            &full_vec_randomness,
            None,
            None,
            &fold_gadget,
            &chunk_layout,
        )
        .expect("plan");
        assert_eq!(layout.required, plan.required());
    }

    #[test]
    fn setup_required_for_inputs_is_challenge_free() {
        let block_len = 12;
        let depth_commit = 3;
        let depth_fold = 2;
        let z_range = block_len * depth_commit;
        let inputs = SetupContributionPlanInputs::<F> {
            eq_tau1: vec![test_scalar(11), test_scalar(12)],
            num_t_vectors: 2,
            num_blocks: 4,
            num_claims: 1,
            depth_open: 16,
            depth_commit,
            depth_fold,
            block_len,
            inner_width: z_range,
            n_a: 1,
            n_d: 0,
            m_row_layout: MRowLayout::WithoutDBlock,
            n_b: 0,
            num_segments: 1,
            rows: 2,
            num_polys_per_segment: vec![2],
        };
        let required = setup_required_for_inputs(&inputs).expect("required");
        assert!(required > 0);

        let fold_gadget = gadget_row_scalars::<F>(depth_fold, 4);
        let mut inputs_a = inputs.clone();
        let chunk_layout = single_chunk_layout(4, 0, z_range, 0, 64, 0);
        let plan_a = SetupContributionPlan::prepare::<F>(
            &inputs_a,
            &[test_scalar(99), test_scalar(100)],
            None,
            None,
            &fold_gadget,
            &chunk_layout,
        )
        .expect("plan a");
        inputs_a.eq_tau1 = vec![test_scalar(1); 8];
        let plan_b = SetupContributionPlan::prepare::<F>(
            &inputs_a,
            &[test_scalar(77), test_scalar(88)],
            None,
            None,
            &fold_gadget,
            &chunk_layout,
        )
        .expect("plan b");
        assert_eq!(required, plan_a.required());
        assert_eq!(plan_a.required(), plan_b.required());
    }

    #[test]
    fn ensure_setup_envelope_rejects_undersized_matrix() {
        let inputs = SetupContributionPlanInputs::<F> {
            eq_tau1: vec![],
            num_t_vectors: 1,
            num_blocks: 4,
            num_claims: 1,
            depth_open: 8,
            depth_commit: 2,
            depth_fold: 3,
            block_len: 16,
            inner_width: 32,
            n_a: 2,
            n_d: 1,
            m_row_layout: MRowLayout::WithDBlock,
            n_b: 2,
            num_segments: 1,
            rows: 8,
            num_polys_per_segment: vec![1],
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
    fn compute_setup_layout_rejects_non_pow2_num_blocks() {
        let inputs = SetupContributionPlanInputs::<F> {
            eq_tau1: vec![],
            num_t_vectors: 1,
            num_blocks: 3,
            num_claims: 1,
            depth_open: 8,
            depth_commit: 2,
            depth_fold: 3,
            block_len: 16,
            inner_width: 32,
            n_a: 2,
            n_d: 1,
            m_row_layout: MRowLayout::WithDBlock,
            n_b: 2,
            num_segments: 1,
            rows: 8,
            num_polys_per_segment: vec![1],
        };
        let err = compute_setup_layout(&inputs).expect_err("non-pow2 blocks");
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
