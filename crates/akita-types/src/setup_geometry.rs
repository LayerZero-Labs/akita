//! Challenge-free setup product geometry shared by prover and verifier.
//!
//! [`compute_setup_layout`] factors the row-layout footprint (`required`) out of
//! [`crate::SetupContributionPlan::prepare`] so NTT sizing, prefix offload, and
//! NTT envelope checks do not depend on `tau1` / fold challenges.

use akita_field::{AkitaError, FieldCore};

use crate::layout::{LevelParams, MRowLayout};
use crate::proof::{AkitaExpandedSetup, OpeningBatchShape};
use crate::schedule::Schedule;
use crate::setup_contribution::SetupContributionPlanInputs;

/// Shape projection for one setup-contribution level (no challenges, no weights).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetupRelationShape {
    pub num_t_vectors: usize,
    pub num_blocks: usize,
    pub num_claims: usize,
    pub depth_open: usize,
    pub depth_commit: usize,
    pub depth_fold: usize,
    pub block_len: usize,
    pub inner_width: usize,
    pub n_a: usize,
    pub n_d: usize,
    pub m_row_layout: MRowLayout,
    pub n_b: usize,
    pub num_segments: usize,
    pub rows: usize,
    pub num_polys_per_segment: Vec<usize>,
    pub num_public_rows: usize,
    pub tier_split: usize,
    pub n_f: usize,
}

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

impl SetupRelationShape {
    /// Build the setup-contribution layout shape from per-level params.
    ///
    /// Mirrors the prover's `create_setup_contribution_inputs` field derivation
    /// without materializing `eq_tau1`.
    ///
    /// # Errors
    ///
    /// Returns an error when level layout parameters are inconsistent.
    pub fn from_level_params(
        lp: &LevelParams,
        num_polynomials: usize,
        m_row_layout: MRowLayout,
        depth_fold: usize,
    ) -> Result<Self, AkitaError> {
        let depth_commit = lp.num_digits_commit;
        let depth_open = lp.num_digits_open;
        if lp.num_blocks == 0 || !lp.num_blocks.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "num_blocks must be a non-zero power of two".into(),
            ));
        }
        if lp.block_len == 0 || depth_commit == 0 || depth_open == 0 || depth_fold == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup evaluator layout has zero width".into(),
            ));
        }
        let inner_width = lp
            .block_len
            .checked_mul(depth_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("inner width overflow".into()))?;
        if lp.a_key.col_len() < inner_width {
            return Err(AkitaError::InvalidSetup(
                "A-key column width is too small for setup contribution layout".into(),
            ));
        }
        let num_public_rows = 0usize;
        let rows = lp.m_row_count_for(1, num_public_rows, m_row_layout)?;
        Ok(Self {
            num_t_vectors: num_polynomials,
            num_blocks: lp.num_blocks,
            num_claims: num_polynomials,
            depth_open,
            depth_commit,
            depth_fold,
            block_len: lp.block_len,
            inner_width,
            n_a: lp.a_key.row_len(),
            n_d: lp.d_key.row_len(),
            m_row_layout,
            n_b: lp.b_key.row_len(),
            num_segments: 1,
            rows,
            num_polys_per_segment: vec![num_polynomials],
            num_public_rows,
            tier_split: lp.tier_split,
            n_f: lp.f_key.as_ref().map_or(0, |fk| fk.row_len()),
        })
    }
}

impl<E: FieldCore> From<&SetupContributionPlanInputs<E>> for SetupRelationShape {
    fn from(inputs: &SetupContributionPlanInputs<E>) -> Self {
        Self {
            num_t_vectors: inputs.num_t_vectors,
            num_blocks: inputs.num_blocks,
            num_claims: inputs.num_claims,
            depth_open: inputs.depth_open,
            depth_commit: inputs.depth_commit,
            depth_fold: inputs.depth_fold,
            block_len: inputs.block_len,
            inner_width: inputs.inner_width,
            n_a: inputs.n_a,
            n_d: inputs.n_d,
            m_row_layout: inputs.m_row_layout,
            n_b: inputs.n_b,
            num_segments: inputs.num_segments,
            rows: inputs.rows,
            num_polys_per_segment: inputs.num_polys_per_segment.clone(),
            num_public_rows: inputs.num_public_rows,
            tier_split: inputs.tier_split,
            n_f: inputs.n_f,
        }
    }
}

/// Pure, challenge-free row-layout footprint for a setup level.
///
/// # Errors
///
/// Returns an error when layout parameters are inconsistent with the canonical
/// M-row packing used by setup sumcheck.
pub fn compute_setup_layout(
    shape: &SetupRelationShape,
) -> Result<SetupLayoutFootprint, AkitaError> {
    if shape.num_blocks == 0 || !shape.num_blocks.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "num_blocks must be a non-zero power of two".into(),
        ));
    }
    if shape.block_len == 0
        || shape.depth_open == 0
        || shape.depth_commit == 0
        || shape.depth_fold == 0
    {
        return Err(AkitaError::InvalidSetup(
            "setup evaluator layout has zero width".into(),
        ));
    }
    if shape.num_polys_per_segment.len() != shape.num_segments {
        return Err(AkitaError::InvalidSize {
            expected: shape.num_segments,
            actual: shape.num_polys_per_segment.len(),
        });
    }

    let z_range = shape.inner_width;
    let expected_z_range = checked_mul(shape.block_len, shape.depth_commit, "Z width")?;
    if z_range != expected_z_range {
        return Err(AkitaError::InvalidSize {
            expected: expected_z_range,
            actual: z_range,
        });
    }

    let tiered = shape.tier_split > 1;
    if tiered && (shape.n_f == 0 || shape.num_segments != 1) {
        return Err(AkitaError::InvalidSetup(
            "tiered setup contribution requires n_f > 0 and a single commitment bundle".into(),
        ));
    }
    let n_d_active = match shape.m_row_layout {
        MRowLayout::WithDBlock => shape.n_d,
        MRowLayout::WithoutDBlock => 0,
    };
    let d_start = checked_add(1, shape.num_public_rows, "D row start")?;
    let f_start = checked_add(d_start, n_d_active, "COMMIT row start")?;
    let commit_rows_pg = if tiered { shape.n_f } else { shape.n_b };
    let b_inner_rows_pg = if tiered {
        checked_mul(shape.tier_split, shape.n_b, "B_inner rows")?
    } else {
        0
    };
    let commit_rows = checked_mul(commit_rows_pg, shape.num_segments, "COMMIT row count")?;
    let b_inner_start = checked_add(f_start, commit_rows, "B_inner row start")?;
    let b_inner_rows_total = checked_mul(b_inner_rows_pg, shape.num_segments, "B_inner row count")?;
    let a_start = checked_add(b_inner_start, b_inner_rows_total, "A row start")?;
    let a_end = checked_add(a_start, shape.n_a, "A row end")?;
    let b_start = f_start;
    if a_end > shape.rows {
        return Err(AkitaError::InvalidSetup(
            "M-row weights are inconsistent with setup evaluator layout".into(),
        ));
    }

    let stride_t = checked_mul(shape.n_a, shape.depth_open, "T stride")?;
    let cols_per_poly_t = checked_mul(stride_t, shape.num_blocks, "T polynomial width")?;
    let b_per_claim_e = checked_mul(shape.num_blocks, shape.depth_open, "e-hat claim width")?;
    let n_cols_e = checked_mul(shape.num_claims, b_per_claim_e, "e-hat column width")?;
    let max_group_poly_count = shape
        .num_polys_per_segment
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let n_cols_t = checked_mul(max_group_poly_count, cols_per_poly_t, "T column width")?;

    let d_required = checked_mul(n_d_active, n_cols_e, "D setup footprint")?;
    let a_required = checked_mul(shape.n_a, z_range, "A setup footprint")?;
    let b_required = if tiered {
        0
    } else {
        checked_mul(shape.n_b, n_cols_t, "B setup footprint")?
    };
    let (b_inner_stride, b_inner_required, f_stride, f_required) = if tiered {
        if n_cols_t == 0 || !n_cols_t.is_multiple_of(shape.tier_split) {
            return Err(AkitaError::InvalidSetup(
                "tiered B' width does not divide the per-group T width".into(),
            ));
        }
        let b_inner_stride = n_cols_t / shape.tier_split;
        let b_inner_required = checked_mul(shape.n_b, b_inner_stride, "B_inner setup footprint")?;
        let f_stride = checked_mul(
            checked_mul(shape.tier_split, shape.n_b, "F width")?,
            shape.depth_open,
            "F width",
        )?;
        let f_required = checked_mul(shape.n_f, f_stride, "F setup footprint")?;
        (b_inner_stride, b_inner_required, f_stride, f_required)
    } else {
        (0, 0, 0, 0)
    };
    let required = d_required
        .max(b_required)
        .max(a_required)
        .max(b_inner_required)
        .max(f_required);
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
        b_inner_required,
        f_required,
        a_end,
        d_start,
        f_start,
        b_start,
        b_inner_start,
        a_start,
        n_d_active,
        tiered,
        b_inner_stride,
        f_stride,
        n_cols_e,
        n_cols_t,
        z_range,
        stride_t,
        cols_per_poly_t,
        b_per_claim_e,
    })
}

/// Required setup ring rows for one level shape (challenge-free).
///
/// # Errors
///
/// Returns an error when layout parameters are inconsistent.
pub fn setup_required_for_shape(relation_shape: &SetupRelationShape) -> Result<usize, AkitaError> {
    Ok(compute_setup_layout(relation_shape)?.required)
}

/// Flat coefficient count for setup-prefix sizing at offload ring `d_setup`.
///
/// Uses the same [`compute_setup_layout`] footprint as setup sumcheck and
/// [`setup_active_ring_elems_at`].
///
/// # Errors
///
/// Returns an error when layout parameters are inconsistent or the product overflows.
pub fn active_setup_field_len(
    level_params: &LevelParams,
    opening_batch: &OpeningBatchShape,
    m_row_layout: MRowLayout,
    depth_fold: usize,
    d_setup: usize,
) -> Result<usize, AkitaError> {
    let shape = SetupRelationShape::from_level_params(
        level_params,
        opening_batch.num_polynomials(),
        m_row_layout,
        depth_fold,
    )?;
    let required = setup_required_for_shape(&shape)?;
    required
        .checked_mul(d_setup)
        .ok_or_else(|| AkitaError::InvalidSetup("active setup field length overflow".into()))
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
/// Uses `d_setup` (setup-prefix slot ring dimension, today [`crate::SETUP_OFFLOAD_D_SETUP`]),
/// not fold `ring_d`. Prefix slots are minted at `d_setup`; see `specs/setup-prefix-ladder.md`.
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
/// Used by [`RingDimPlan::context_at`](crate::RingDimPlan::context_at) and schedule
/// validation. Stage-3 prove/verify should derive `required` from
/// [`crate::SetupContributionPlan::prepare`] and call [`ensure_setup_envelope`] separately.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `required` exceeds the shared matrix
/// prefix available at `fold_ring_d`.
pub fn setup_active_ring_elems_for_fold<F: FieldCore>(
    expanded: &AkitaExpandedSetup<F>,
    relation_shape: &SetupRelationShape,
    fold_ring_d: usize,
) -> Result<usize, AkitaError> {
    let required = setup_required_for_shape(relation_shape)?;
    ensure_setup_envelope(expanded, required, fold_ring_d)?;
    Ok(required)
}

/// Active inner (`d_a`) setup ring rows at `level`, fail-closed on envelope overflow.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `required` exceeds the shared matrix
/// prefix available at the fold ring dimension.
pub fn setup_active_ring_elems_at<F: FieldCore>(
    level: usize,
    schedule: &Schedule,
    expanded: &AkitaExpandedSetup<F>,
    relation_shape: &SetupRelationShape,
) -> Result<usize, AkitaError> {
    let exec = schedule.get_execution_schedule(level)?;
    setup_active_ring_elems_for_fold(expanded, relation_shape, exec.params.ring_dimension)
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
        let inputs = SetupContributionPlanInputs {
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
            num_public_rows: 0,
            tier_split: 1,
            n_f: 0,
        };
        let shape = SetupRelationShape::from(&inputs);
        let layout = compute_setup_layout(&shape).expect("layout");
        let plan = SetupContributionPlan::prepare::<F>(
            &inputs,
            &full_vec_randomness,
            None,
            None,
            &fold_gadget,
            0,
            64,
            offset_z,
            0,
            None,
            None,
        )
        .expect("plan");
        assert_eq!(layout.required, plan.required());
    }

    #[test]
    fn setup_required_for_shape_is_challenge_free() {
        let block_len = 12;
        let depth_commit = 3;
        let depth_fold = 2;
        let z_range = block_len * depth_commit;
        let shape = SetupRelationShape {
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
            num_public_rows: 0,
            tier_split: 1,
            n_f: 0,
        };
        let required = setup_required_for_shape(&shape).expect("required");
        assert!(required > 0);

        let fold_gadget = gadget_row_scalars::<F>(depth_fold, 4);
        let mut inputs_a = SetupContributionPlanInputs {
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
            num_public_rows: 0,
            tier_split: 1,
            n_f: 0,
        };
        let plan_a = SetupContributionPlan::prepare::<F>(
            &inputs_a,
            &[test_scalar(99), test_scalar(100)],
            None,
            None,
            &fold_gadget,
            0,
            64,
            0,
            0,
            None,
            None,
        )
        .expect("plan a");
        inputs_a.eq_tau1 = vec![test_scalar(1); 8];
        let plan_b = SetupContributionPlan::prepare::<F>(
            &inputs_a,
            &[test_scalar(77), test_scalar(88)],
            None,
            None,
            &fold_gadget,
            0,
            64,
            0,
            0,
            None,
            None,
        )
        .expect("plan b");
        assert_eq!(required, plan_a.required());
        assert_eq!(plan_a.required(), plan_b.required());
    }

    #[test]
    fn ensure_setup_envelope_rejects_undersized_matrix() {
        let shape = SetupRelationShape {
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
            num_public_rows: 1,
            tier_split: 1,
            n_f: 0,
        };
        let required = setup_required_for_shape(&shape).expect("required");
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
        let shape = SetupRelationShape {
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
            num_public_rows: 0,
            tier_split: 1,
            n_f: 0,
        };
        let err = compute_setup_layout(&shape).expect_err("non-pow2 blocks");
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
