//! Challenge-free setup product geometry shared by prover and verifier.
//!
//! [`setup_required_for_inputs`] derives the packed-scan footprint (`required`)
//! without fold challenges so NTT sizing, prefix offload, and envelope checks
//! do not depend on `tau1`.

use akita_field::{AkitaError, FieldCore};

use crate::layout::MRowLayout;
use crate::proof::AkitaExpandedSetup;
use crate::schedule::Schedule;
use crate::setup_contribution::SetupContributionPlanInputs;

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
    let b_rows_total = checked_mul(inputs.n_b, inputs.num_groups, "B row count")?;
    let a_end = checked_add(
        checked_add(1, inputs.n_a, "B row start")?
            .checked_add(b_rows_total)
            .ok_or_else(|| AkitaError::InvalidSetup("D row start overflow".into()))?,
        n_d_active,
        "D row end",
    )?;
    if a_end > inputs.rows {
        return Err(AkitaError::InvalidSetup(
            "M-row weights are inconsistent with setup evaluator layout".into(),
        ));
    }

    let b_per_claim_e = checked_mul(inputs.num_blocks, inputs.depth_open, "e-hat claim width")?;
    let n_cols_e = checked_mul(inputs.num_claims, b_per_claim_e, "e-hat column width")?;
    let max_group_poly_count = inputs
        .num_polys_per_group
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let n_cols_t = checked_mul(
        max_group_poly_count,
        checked_mul(
            checked_mul(inputs.n_a, inputs.depth_open, "T stride")?,
            inputs.num_blocks,
            "T polynomial width",
        )?,
        "T column width",
    )?;

    let d_required = checked_mul(n_d_active, n_cols_e, "D setup footprint")?;
    let a_required = checked_mul(inputs.n_a, z_range, "A setup footprint")?;
    let b_required = checked_mul(inputs.n_b, n_cols_t, "B setup footprint")?;
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
        }
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
            num_groups: num_points,
            rows: 2,
            num_polys_per_group: vec![0],
        };
        let required = setup_required_for_inputs(&inputs).expect("required");
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
        assert_eq!(required, plan.required());
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
            num_groups: 1,
            rows: 2,
            num_polys_per_group: vec![2],
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
            num_groups: 1,
            rows: 8,
            num_polys_per_group: vec![1],
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
            num_groups: 1,
            rows: 8,
            num_polys_per_group: vec![1],
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
