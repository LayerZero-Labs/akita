//! Challenge-free setup product geometry: footprint sizing and envelope guards.
//!
//! [`setup_required_for_inputs`] derives the packed-scan footprint (`required`)
//! without fold challenges so NTT sizing, prefix offload, and envelope checks
//! do not depend on `tau1`.

use akita_algebra::offset_eq::MAX_COMPACT_STRIDE_TERMS;
use akita_field::{AkitaError, FieldCore};

use super::SetupContributionPlanInputs;
use crate::layout::{validate_role_dims, CommitmentRingDims, RelationMatrixRowLayout};
use crate::proof::AkitaExpandedSetup;
use crate::schedule::Schedule;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SetupProjectionGroupGeometry {
    pub(crate) a_rows: usize,
    pub(crate) a_cols: usize,
    pub(crate) b_rows: usize,
    pub(crate) b_cols: usize,
    pub(crate) d_active_cols: usize,
    pub(crate) ownership_units: usize,
    pub(crate) depth_fold: usize,
}

/// Checked common-base geometry for the Stage 3 setup projection.
///
/// Physical A, B, and D matrices retain their native role dimensions. Stage 3
/// views their flat coefficients as rings over `base_ring_dim = min(d_a,d_b,d_d)`.
/// The projection ratios expand each native role footprint into that common
/// base without changing its flat coefficient count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetupProjectionGeometry {
    role_dims: CommitmentRingDims,
    base_ring_dim: usize,
    a_ratio: usize,
    b_ratio: usize,
    d_ratio: usize,
    a_projection_width: usize,
    b_projection_width: usize,
    d_projection_width: usize,
    required: usize,
    setup_index_len: usize,
    ring_bits: usize,
    rounds: usize,
    natural_field_len: usize,
    evaluation_terms: usize,
}

impl SetupProjectionGeometry {
    pub(crate) fn from_groups(
        role_dims: CommitmentRingDims,
        d_rows: usize,
        d_physical_cols: usize,
        groups: &[SetupProjectionGroupGeometry],
    ) -> Result<Self, AkitaError> {
        if groups.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "setup projection requires at least one group".into(),
            ));
        }
        let d_footprint = d_rows
            .checked_mul(d_physical_cols)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D footprint overflow".into()))?;
        let mut a_footprint = 0usize;
        let mut b_footprint = 0usize;
        for group in groups {
            a_footprint =
                a_footprint.max(group.a_rows.checked_mul(group.a_cols).ok_or_else(|| {
                    AkitaError::InvalidSetup("setup A footprint overflow".into())
                })?);
            b_footprint =
                b_footprint.max(group.b_rows.checked_mul(group.b_cols).ok_or_else(|| {
                    AkitaError::InvalidSetup("setup B footprint overflow".into())
                })?);
        }
        let mut geometry =
            Self::from_role_footprints(role_dims, a_footprint, b_footprint, d_footprint)?;
        let mut evaluation_terms = 0usize;
        for group in groups {
            let d_terms = group
                .d_active_cols
                .checked_mul(d_rows)
                .and_then(|terms| terms.checked_mul(geometry.d_ratio))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("setup D evaluation work overflow".into())
                })?;
            let b_terms = group
                .b_cols
                .checked_mul(group.b_rows)
                .and_then(|terms| terms.checked_mul(geometry.b_ratio))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("setup B evaluation work overflow".into())
                })?;
            let a_terms = group
                .a_cols
                .checked_mul(group.a_rows)
                .and_then(|terms| terms.checked_mul(geometry.a_ratio))
                .and_then(|terms| terms.checked_mul(group.ownership_units))
                .and_then(|terms| terms.checked_mul(group.depth_fold))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("setup A evaluation work overflow".into())
                })?;
            evaluation_terms = evaluation_terms
                .checked_add(d_terms)
                .and_then(|terms| terms.checked_add(b_terms))
                .and_then(|terms| terms.checked_add(a_terms))
                .ok_or_else(|| AkitaError::InvalidSetup("setup evaluation work overflow".into()))?;
        }
        geometry.evaluation_terms = evaluation_terms;
        Ok(geometry)
    }

    pub(crate) fn from_role_footprints(
        role_dims: CommitmentRingDims,
        a_footprint: usize,
        b_footprint: usize,
        d_footprint: usize,
    ) -> Result<Self, AkitaError> {
        let (base_ring_dim, a_ratio, b_ratio, d_ratio) = checked_role_ratios(role_dims)?;
        let a_projection_width = a_footprint
            .checked_mul(a_ratio)
            .ok_or_else(|| AkitaError::InvalidSetup("setup A projection width overflow".into()))?;
        let b_projection_width = b_footprint
            .checked_mul(b_ratio)
            .ok_or_else(|| AkitaError::InvalidSetup("setup B projection width overflow".into()))?;
        let d_projection_width = d_footprint
            .checked_mul(d_ratio)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D projection width overflow".into()))?;
        let required = a_projection_width
            .max(b_projection_width)
            .max(d_projection_width);
        if required == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup projection requires a non-empty footprint".into(),
            ));
        }
        let setup_index_len = required
            .checked_next_power_of_two()
            .ok_or_else(|| AkitaError::InvalidSetup("setup index domain overflow".into()))?;
        let ring_bits = base_ring_dim.trailing_zeros() as usize;
        let rounds = ring_bits
            .checked_add(setup_index_len.trailing_zeros() as usize)
            .ok_or_else(|| AkitaError::InvalidSetup("setup round count overflow".into()))?;
        let natural_field_len = required.checked_mul(base_ring_dim).ok_or_else(|| {
            AkitaError::InvalidSetup("setup product natural field length overflow".into())
        })?;
        Ok(Self {
            role_dims,
            base_ring_dim,
            a_ratio,
            b_ratio,
            d_ratio,
            a_projection_width,
            b_projection_width,
            d_projection_width,
            required,
            setup_index_len,
            ring_bits,
            rounds,
            natural_field_len,
            evaluation_terms: 0,
        })
    }

    /// Number of native B- and D-role subcolumns in one A-role witness column.
    pub(crate) fn witness_subcolumn_ratios(
        role_dims: CommitmentRingDims,
    ) -> Result<(usize, usize), AkitaError> {
        let (_, a_ratio, b_ratio, d_ratio) = checked_role_ratios(role_dims)?;
        let b_subcolumns = a_ratio
            .checked_div(b_ratio)
            .filter(|ratio| *ratio != 0)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("B role does not divide the A-role witness width".into())
            })?;
        let d_subcolumns = a_ratio
            .checked_div(d_ratio)
            .filter(|ratio| *ratio != 0)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("D role does not divide the A-role witness width".into())
            })?;
        if !b_subcolumns.is_power_of_two() || !d_subcolumns.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "relation role projection ratios must be powers of two".into(),
            ));
        }
        Ok((b_subcolumns, d_subcolumns))
    }

    #[must_use]
    pub const fn role_dims(self) -> CommitmentRingDims {
        self.role_dims
    }

    #[must_use]
    pub const fn base_ring_dim(self) -> usize {
        self.base_ring_dim
    }

    #[must_use]
    pub const fn a_ratio(self) -> usize {
        self.a_ratio
    }

    #[must_use]
    pub const fn b_ratio(self) -> usize {
        self.b_ratio
    }

    #[must_use]
    pub const fn d_ratio(self) -> usize {
        self.d_ratio
    }

    #[must_use]
    pub const fn a_projection_width(self) -> usize {
        self.a_projection_width
    }

    #[must_use]
    pub const fn b_projection_width(self) -> usize {
        self.b_projection_width
    }

    #[must_use]
    pub const fn d_projection_width(self) -> usize {
        self.d_projection_width
    }

    #[must_use]
    pub const fn required(self) -> usize {
        self.required
    }

    #[must_use]
    pub const fn setup_index_len(self) -> usize {
        self.setup_index_len
    }

    #[must_use]
    pub const fn ring_bits(self) -> usize {
        self.ring_bits
    }

    #[must_use]
    pub const fn rounds(self) -> usize {
        self.rounds
    }

    #[must_use]
    pub const fn alpha_power_len(self) -> usize {
        self.base_ring_dim
    }

    #[must_use]
    pub const fn natural_field_len(self) -> usize {
        self.natural_field_len
    }

    #[must_use]
    pub const fn evaluation_terms(self) -> usize {
        self.evaluation_terms
    }

    pub fn ensure_evaluation_budget(self) -> Result<(), AkitaError> {
        if self.evaluation_terms > MAX_COMPACT_STRIDE_TERMS {
            return Err(AkitaError::InvalidSize {
                expected: MAX_COMPACT_STRIDE_TERMS,
                actual: self.evaluation_terms,
            });
        }
        Ok(())
    }

    pub(crate) fn validate_alpha_power_lengths(
        self,
        a_len: usize,
        b_len: usize,
        d_len: usize,
    ) -> Result<(), AkitaError> {
        for (role, expected, actual) in [
            ("A", self.role_dims.d_a(), a_len),
            ("B", self.role_dims.d_b(), b_len),
            ("D", self.role_dims.d_d(), d_len),
        ] {
            if actual != expected {
                return Err(AkitaError::InvalidSize { expected, actual });
            }
            if actual < self.base_ring_dim {
                return Err(AkitaError::InvalidSetup(format!(
                    "{role} alpha powers are shorter than the Stage 3 base"
                )));
            }
        }
        Ok(())
    }
}

fn checked_role_ratios(
    role_dims: CommitmentRingDims,
) -> Result<(usize, usize, usize, usize), AkitaError> {
    validate_role_dims(role_dims)?;
    let base_ring_dim = role_dims.d_a().min(role_dims.d_b()).min(role_dims.d_d());
    let ratio = |role: &'static str, dimension: usize| {
        if !dimension.is_multiple_of(base_ring_dim) {
            return Err(AkitaError::InvalidSetup(format!(
                "{role} ring dimension does not decompose over the Stage 3 base"
            )));
        }
        let ratio = dimension / base_ring_dim;
        if ratio == 0 || !ratio.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(format!(
                "{role} Stage 3 projection ratio must be a non-zero power of two"
            )));
        }
        Ok(ratio)
    };
    Ok((
        base_ring_dim,
        ratio("A", role_dims.d_a())?,
        ratio("B", role_dims.d_b())?,
        ratio("D", role_dims.d_d())?,
    ))
}

/// Required setup ring rows for one level (challenge-free).
///
/// # Errors
///
/// Returns an error when layout parameters are inconsistent with the canonical
/// M-row packing used by setup sumcheck.
pub fn setup_required_for_inputs<E: FieldCore>(
    inputs: &SetupContributionPlanInputs<E>,
    role_dims: CommitmentRingDims,
) -> Result<usize, AkitaError> {
    if inputs.live_fold_count == 0 {
        return Err(AkitaError::InvalidSetup(
            "live_fold_count must be positive".into(),
        ));
    }
    if inputs.fold_position_count == 0
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
        .fold_position_count
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
        .live_fold_count
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
        .checked_mul(inputs.live_fold_count)
        .ok_or_else(|| AkitaError::InvalidSetup("T polynomial width overflow".into()))?;
    let n_cols_t = max_group_poly_count
        .checked_mul(t_polynomial_width)
        .ok_or_else(|| AkitaError::InvalidSetup("T column width overflow".into()))?;

    let d_footprint = n_d_active
        .checked_mul(n_cols_e)
        .ok_or_else(|| AkitaError::InvalidSetup("D setup footprint overflow".into()))?;
    let a_footprint = inputs
        .n_a
        .checked_mul(z_range)
        .ok_or_else(|| AkitaError::InvalidSetup("A setup footprint overflow".into()))?;
    let b_footprint = inputs
        .n_b
        .checked_mul(n_cols_t)
        .ok_or_else(|| AkitaError::InvalidSetup("B setup footprint overflow".into()))?;
    Ok(SetupProjectionGeometry::from_role_footprints(
        role_dims,
        a_footprint,
        b_footprint,
        d_footprint,
    )?
    .required())
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

/// Active base-ring setup rows for one fold, fail-closed on envelope overflow.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `required` exceeds the shared matrix
/// prefix available at `fold_ring_d`.
pub fn setup_active_ring_elems_for_fold<F: FieldCore, E: FieldCore>(
    expanded: &AkitaExpandedSetup<F>,
    inputs: &SetupContributionPlanInputs<E>,
    role_dims: CommitmentRingDims,
) -> Result<usize, AkitaError> {
    let required = setup_required_for_inputs(inputs, role_dims)?;
    ensure_setup_envelope(
        expanded,
        required,
        role_dims.d_a().min(role_dims.d_b()).min(role_dims.d_d()),
    )?;
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
    setup_active_ring_elems_for_fold(expanded, inputs, exec.params.role_dims())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gadget_row_scalars, RelationMatrixRowLayout, SetupContributionGroupInputs,
        SetupContributionLayout, SetupContributionPlan, WitnessLayout, WitnessUnitLayout,
    };
    use akita_field::Prime128OffsetA7F7;

    type F = Prime128OffsetA7F7;

    fn test_scalar(value: u128) -> F {
        F::from_canonical_u128(value)
    }

    fn single_chunk_layout_from_inputs(inputs: &SetupContributionPlanInputs<F>) -> WitnessLayout {
        let z_len = inputs.fold_position_count * inputs.depth_commit * inputs.depth_fold;
        let e_len = inputs.num_claims * inputs.live_fold_count * inputs.depth_open;
        let t_len = e_len * inputs.n_a;
        let unit = WitnessUnitLayout::new_for_test(
            0,
            0,
            0,
            inputs.live_fold_count,
            0..z_len,
            z_len..z_len + e_len,
            z_len + e_len..z_len + e_len + t_len,
        );
        let cursor = z_len + e_len + t_len;
        WitnessLayout::new_for_test(vec![unit], cursor..cursor + inputs.depth_fold.max(1))
    }

    fn prepare_single_group_plan(
        inputs: &SetupContributionPlanInputs<F>,
        full_vec_randomness: &[F],
        fold_gadget: &[F],
        layout: &WitnessLayout,
    ) -> Result<SetupContributionPlan<F>, AkitaError> {
        let group = SetupContributionGroupInputs::from_single_group(inputs, 0)?;
        let layout = SetupContributionLayout::new(
            std::sync::Arc::new(layout.clone()),
            layout.total_len(),
            vec![group],
        )?;
        let static_plan = SetupContributionPlan::prepare_static(inputs, &layout)?;
        SetupContributionPlan::finish_plan::<F>(
            &static_plan,
            full_vec_randomness,
            None,
            None,
            Some(fold_gadget),
            &layout,
            CommitmentRingDims::uniform(64),
        )
    }

    #[test]
    fn setup_required_for_inputs_matches_prepare_required() {
        let fold_position_count = 12;
        let depth_commit = 3;
        let depth_fold = 2;
        let num_points = 1;
        let z_range = fold_position_count * depth_commit;
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
            live_fold_count: 4,
            fold_position_count,
            depth_open: 16,
            depth_commit,
            depth_fold,
            inner_width: z_range,
            eq_tau1: vec![test_scalar(11), test_scalar(12)].into(),
        };
        let required =
            setup_required_for_inputs(&inputs, CommitmentRingDims::uniform(64)).expect("required");
        let layout = single_chunk_layout_from_inputs(&inputs);
        let plan = prepare_single_group_plan(&inputs, &full_vec_randomness, &fold_gadget, &layout)
            .expect("plan");
        assert_eq!(required, plan.required());
    }

    #[test]
    fn setup_required_for_inputs_is_challenge_free() {
        let fold_position_count = 12;
        let depth_commit = 3;
        let depth_fold = 2;
        let z_range = fold_position_count * depth_commit;
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
            live_fold_count: 4,
            fold_position_count,
            depth_open: 16,
            depth_commit,
            depth_fold,
            inner_width: z_range,
            eq_tau1: vec![test_scalar(11), test_scalar(12)].into(),
        };
        let required =
            setup_required_for_inputs(&inputs, CommitmentRingDims::uniform(64)).expect("required");
        assert!(required > 0);

        let fold_gadget = gadget_row_scalars::<F>(depth_fold, 4);
        let mut inputs_a = inputs.clone();
        let layout = single_chunk_layout_from_inputs(&inputs_a);
        let plan_a = prepare_single_group_plan(
            &inputs_a,
            &[test_scalar(99), test_scalar(100)],
            &fold_gadget,
            &layout,
        )
        .expect("plan a");
        inputs_a.eq_tau1 = vec![test_scalar(1); 8].into();
        let plan_b = prepare_single_group_plan(
            &inputs_a,
            &[test_scalar(77), test_scalar(88)],
            &fold_gadget,
            &layout,
        )
        .expect("plan b");
        assert_eq!(required, plan_a.required());
        assert_eq!(plan_a.required(), plan_b.required());
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
            live_fold_count: 4,
            fold_position_count: 16,
            depth_open: 8,
            depth_commit: 2,
            depth_fold: 3,
            inner_width: 32,
            eq_tau1: vec![].into(),
        };
        let required =
            setup_required_for_inputs(&inputs, CommitmentRingDims::uniform(64)).expect("required");
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
    fn setup_required_for_inputs_accepts_exact_non_pow2_fold_count() {
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
            live_fold_count: 3,
            fold_position_count: 16,
            depth_open: 8,
            depth_commit: 2,
            depth_fold: 3,
            inner_width: 32,
            eq_tau1: vec![].into(),
        };
        assert!(setup_required_for_inputs(&inputs, CommitmentRingDims::uniform(64)).is_ok());
    }

    #[test]
    fn projection_geometry_uses_nested_common_base() {
        let geometry = SetupProjectionGeometry::from_role_footprints(
            CommitmentRingDims {
                inner: 64,
                outer: 32,
                opening: 32,
            },
            7,
            11,
            13,
        )
        .expect("nested geometry");
        assert_eq!(geometry.base_ring_dim(), 32);
        assert_eq!(geometry.a_ratio(), 2);
        assert_eq!(geometry.b_ratio(), 1);
        assert_eq!(geometry.d_ratio(), 1);
        assert_eq!(geometry.required(), 14);
        assert_eq!(geometry.alpha_power_len(), 32);
        assert_eq!(geometry.natural_field_len(), 14 * 32);
    }

    #[test]
    fn projection_geometry_rejects_non_nested_roles() {
        let err = SetupProjectionGeometry::from_role_footprints(
            CommitmentRingDims {
                inner: 64,
                outer: 16,
                opening: 32,
            },
            1,
            1,
            1,
        )
        .expect_err("non-nested roles");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn evaluation_budget_accepts_cap_and_rejects_next_term() {
        let geometry_at_cap = SetupProjectionGeometry::from_groups(
            CommitmentRingDims::uniform(64),
            0,
            0,
            &[SetupProjectionGroupGeometry {
                a_rows: 1,
                a_cols: MAX_COMPACT_STRIDE_TERMS,
                b_rows: 0,
                b_cols: 0,
                d_active_cols: 0,
                ownership_units: 1,
                depth_fold: 1,
            }],
        )
        .expect("geometry at cap");
        assert_eq!(geometry_at_cap.evaluation_terms(), MAX_COMPACT_STRIDE_TERMS);
        geometry_at_cap
            .ensure_evaluation_budget()
            .expect("cap accepted");

        let geometry_above_cap = SetupProjectionGeometry::from_groups(
            CommitmentRingDims::uniform(64),
            0,
            0,
            &[SetupProjectionGroupGeometry {
                a_rows: 1,
                a_cols: MAX_COMPACT_STRIDE_TERMS + 1,
                b_rows: 0,
                b_cols: 0,
                d_active_cols: 0,
                ownership_units: 1,
                depth_fold: 1,
            }],
        )
        .expect("geometry above cap");
        assert!(matches!(
            geometry_above_cap.ensure_evaluation_budget(),
            Err(AkitaError::InvalidSize { .. })
        ));
    }
}
