//! Shared fixtures for packed setup inner-product equivalence tests.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::{eval_ring_at_pows, scalar_powers};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, Prime128OffsetA7F7};
use akita_types::{
    gadget_row_scalars, AkitaExpandedSetup, AkitaSetupSeed, FlatMatrix, MRowLayout,
    RingRelationSegmentLayout, SetupContributionPlan,
};

use crate::protocol::ring_switch::{PreparedChallengeEvals, RingSwitchDeferredRowEval};

type TestField = Prime128OffsetA7F7;
const TEST_RING_DIM: usize = 32;

fn test_scalar(value: u128) -> TestField {
    TestField::from_canonical_u128_reduced(value)
}

pub(super) struct SetupContributionFixture {
    prepared: RingSwitchDeferredRowEval<TestField>,
    setup: AkitaExpandedSetup<TestField>,
    full_vec_randomness: Vec<TestField>,
    eq_low: Vec<TestField>,
    z_block_low_eq: Vec<TestField>,
    alpha_pows: Vec<TestField>,
    fold_gadget: Vec<TestField>,
}

pub(super) struct SetupContributionShape {
    num_blocks: usize,
    depth_open: usize,
    depth_commit: usize,
    depth_fold: usize,
    block_len: usize,
    log_basis: u32,
    n_a: usize,
    n_d: usize,
    n_b: usize,
    num_polys_per_point: Vec<usize>,
    m_row_layout: MRowLayout,
    z_first: bool,
    claim_to_point_poly: Vec<(usize, usize)>,
    /// Tiered split factor `f` (`1` = single-tier).
    tier_split: usize,
    /// Second-tier `F` rank (`0` = single-tier).
    n_f: usize,
}

impl SetupContributionShape {
    pub(super) fn root_single_point() -> Self {
        Self {
            num_blocks: 4,
            depth_open: 8,
            depth_commit: 2,
            depth_fold: 3,
            block_len: 16,
            log_basis: 4,
            n_a: 2,
            n_d: 1,
            n_b: 2,
            num_polys_per_point: vec![1],
            m_row_layout: MRowLayout::WithDBlock,
            z_first: false,
            claim_to_point_poly: vec![(0, 0)],
            tier_split: 1,
            n_f: 0,
        }
    }

    /// Tiered single-point root: first-tier `B'` reused across `tier_split`
    /// column-slices plus the second-tier `F` (COMMIT) block, exercising the
    /// tiered `bar_omega` / direct-scan equivalence. Tiering requires a single
    /// commitment group (`num_points == 1`) and `n_f > 0`; `tier_split` divides
    /// the per-group `n_cols_t = n_a · depth_open · num_blocks` (here 64).
    pub(super) fn tiered_root_single_point() -> Self {
        let mut shape = Self::root_single_point();
        shape.tier_split = 4;
        shape.n_f = 1;
        shape
    }

    pub(super) fn recursive_multigroup() -> Self {
        Self {
            num_blocks: 8,
            depth_open: 26,
            depth_commit: 1,
            depth_fold: 4,
            block_len: 512,
            log_basis: 5,
            n_a: 2,
            n_d: 2,
            n_b: 2,
            num_polys_per_point: vec![2, 1],
            m_row_layout: MRowLayout::WithDBlock,
            z_first: false,
            claim_to_point_poly: vec![(0, 1), (1, 0), (0, 0)],
            tier_split: 1,
            n_f: 0,
        }
    }

    pub(super) fn terminal_relation_only() -> Self {
        let mut shape = Self::root_single_point();
        shape.m_row_layout = MRowLayout::WithoutDBlock;
        shape
    }

    pub(super) fn dense_non_pow2_z() -> Self {
        let mut shape = Self::root_single_point();
        shape.block_len = 12;
        shape.depth_commit = 3;
        shape.depth_fold = 2;
        shape
    }

    pub(super) fn batched_root() -> Self {
        let mut shape = Self::root_single_point();
        shape.claim_to_point_poly = vec![(0, 0), (0, 0), (0, 0), (0, 0)];
        shape
    }

    pub(super) fn z_first_e_t_offset_carry() -> Self {
        let mut shape = Self::root_single_point();
        shape.num_blocks = 8;
        shape.block_len = 10;
        shape.depth_commit = 3;
        shape.depth_fold = 2;
        shape.z_first = true;
        shape
    }

    pub(super) fn pow2_z_offset_carry() -> Self {
        let mut shape = Self::root_single_point();
        shape.block_len = 64;
        shape
    }
}

fn recursive_inner_product(
    plan: &SetupContributionPlan<TestField>,
    setup: &AkitaExpandedSetup<TestField>,
    alpha_pows: &[TestField],
) -> Result<TestField, AkitaError> {
    let bar_omega = plan.materialize_bar_omega();
    let setup_len = setup
        .shared_matrix()
        .total_ring_elements_at::<TEST_RING_DIM>()?;
    if setup_len < bar_omega.len() {
        return Err(AkitaError::InvalidSize {
            expected: bar_omega.len(),
            actual: setup_len,
        });
    }
    let setup_view = setup
        .shared_matrix()
        .ring_view::<TEST_RING_DIM>(1, setup_len)?;
    Ok(setup_view
        .as_slice()
        .iter()
        .zip(bar_omega)
        .map(|(ring, weight)| eval_ring_at_pows(ring, alpha_pows) * weight)
        .sum())
}

impl SetupContributionFixture {
    pub(super) fn from_shape(shape: &SetupContributionShape) -> Self {
        let num_points = shape.num_polys_per_point.len();
        let num_public_rows = num_points;
        let num_claims = shape.claim_to_point_poly.len();
        let num_t_vectors = shape.num_polys_per_point.iter().sum();
        let total_blocks = shape.num_blocks * num_claims;
        let inner_width = shape.block_len * shape.depth_commit;

        let tiered = shape.tier_split > 1;
        // Canonical M-row layout: consistency | public | D | COMMIT | B_inner | A.
        // COMMIT is the `F` block when tiered (`n_f` rows/point), else the full
        // `B` block (`n_b` rows/point); B_inner (`tier_split·n_b` rows/point) is
        // tiered-only.
        let commit_rows = if tiered { shape.n_f } else { shape.n_b } * num_points;
        let b_inner_rows = if tiered {
            shape.tier_split * shape.n_b * num_points
        } else {
            0
        };
        let rows = 1 + num_public_rows + shape.n_d + commit_rows + b_inner_rows + shape.n_a;

        let stride_t = shape.n_a * shape.depth_open;
        let cols_per_poly_t = stride_t * shape.num_blocks;
        let n_cols_e = num_claims * shape.num_blocks * shape.depth_open;
        let n_cols_t = shape.num_polys_per_point.iter().copied().max().unwrap() * cols_per_poly_t;
        // Tiered footprints: stored `B'` is `n_cols_t / tier_split` wide, `F`
        // commits `tier_split·n_b·depth_open` decomposed digits.
        let f_stride = shape.tier_split * shape.n_b * shape.depth_open;
        let b_inner_stride = if tiered {
            n_cols_t / shape.tier_split
        } else {
            0
        };
        let b_inner_required = shape.n_b * b_inner_stride;
        let f_required = shape.n_f * f_stride;

        let e_len = shape.depth_open * total_blocks;
        let t_len = shape.depth_open * shape.n_a * shape.num_blocks * num_t_vectors;
        let z_len = shape.depth_fold * shape.depth_commit * num_points * shape.block_len;
        // û_concat witness segment (tiered only): `num_points · f_stride` planes
        // placed between `t` and `z`. For single-tier `u_len == 0` and the layout
        // is unchanged.
        let u_len = if tiered { num_points * f_stride } else { 0 };
        let (offset_e, offset_t, offset_u, offset_z, total_len) = if shape.z_first {
            (
                z_len,
                z_len + e_len,
                z_len + e_len + t_len,
                0usize,
                z_len + e_len + t_len + u_len,
            )
        } else {
            (
                0usize,
                e_len,
                e_len + t_len,
                e_len + t_len + u_len,
                e_len + t_len + u_len + z_len,
            )
        };
        let offset_r: usize = total_len;
        let bits = total_len.next_power_of_two().trailing_zeros() as usize;

        let max_setup_len = (shape.n_d * n_cols_e)
            .max(shape.n_a * inner_width)
            .max(if tiered {
                b_inner_required.max(f_required)
            } else {
                shape.n_b * n_cols_t
            });

        let matrix_entries: Vec<CyclotomicRing<TestField, TEST_RING_DIM>> = (0..max_setup_len)
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|coeff| {
                    test_scalar(1_000 + (idx * TEST_RING_DIM + coeff) as u128)
                }))
            })
            .collect();
        let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupSeed {
                max_num_vars: 32,
                max_num_batched_polys: shape.num_polys_per_point.iter().sum(),
                max_num_points: num_points,
                gen_ring_dim: TEST_RING_DIM,
                max_setup_len,
                #[cfg(feature = "zk")]
                max_zk_b_len: 1,
                #[cfg(feature = "zk")]
                max_zk_d_len: 1,
                public_matrix_seed: [7u8; 32],
            },
            FlatMatrix::from_ring_slice::<TEST_RING_DIM>(&matrix_entries),
            #[cfg(feature = "zk")]
            FlatMatrix::from_flat_data(vec![TestField::zero(); TEST_RING_DIM], TEST_RING_DIM),
            #[cfg(feature = "zk")]
            FlatMatrix::from_flat_data(vec![TestField::zero(); TEST_RING_DIM], TEST_RING_DIM),
        );

        let eq_tau1: Vec<TestField> = (0..rows.next_power_of_two())
            .map(|idx| test_scalar(11 + idx as u128))
            .collect();
        let t_vector_offsets: Vec<usize> = shape
            .num_polys_per_point
            .iter()
            .scan(0usize, |acc, &count| {
                let offset = *acc;
                *acc += count;
                Some(offset)
            })
            .collect();
        let claim_to_t_vector: Vec<usize> = shape
            .claim_to_point_poly
            .iter()
            .map(|&(point_idx, poly_idx)| t_vector_offsets[point_idx] + poly_idx)
            .collect();
        let claim_to_point: Vec<usize> = shape
            .claim_to_point_poly
            .iter()
            .map(|&(point_idx, _)| point_idx)
            .collect();
        let prepared = RingSwitchDeferredRowEval {
            c_alphas: PreparedChallengeEvals::Flat {
                evals: (0..total_blocks)
                    .map(|idx| test_scalar(41 + idx as u128))
                    .collect(),
                num_claims,
                num_blocks: shape.num_blocks,
            },
            eq_tau1,
            num_t_vectors,
            num_blocks: shape.num_blocks,
            num_claims,
            depth_open: shape.depth_open,
            depth_commit: shape.depth_commit,
            depth_fold: shape.depth_fold,
            #[cfg(feature = "zk")]
            d_blinding_segment_len: 0,
            #[cfg(feature = "zk")]
            b_blinding_digit_planes_per_point: 0,
            #[cfg(feature = "zk")]
            b_blinding_segment_len: 0,
            block_len: shape.block_len,
            inner_width,
            log_basis: shape.log_basis,
            n_a: shape.n_a,
            n_d: shape.n_d,
            m_row_layout: shape.m_row_layout,
            n_b: shape.n_b,
            tier_split: shape.tier_split,
            n_f: shape.n_f,
            num_points,
            rows,
            claim_to_t_vector,
            num_polys_per_commitment_group: shape.num_polys_per_point.clone(),
            num_public_rows,
            gamma: vec![TestField::one(); num_claims],
            claim_to_point,
            witness_segment_layout: RingRelationSegmentLayout {
                offset_e,
                offset_t,
                offset_u,
                offset_z,
                offset_r,
                #[cfg(feature = "zk")]
                b_blinding_offset: offset_u,
                #[cfg(feature = "zk")]
                d_blinding_offset: offset_u,
            },
        };

        let full_vec_randomness: Vec<TestField> = (0..bits)
            .map(|idx| test_scalar(101 + idx as u128))
            .collect();
        let alpha = test_scalar(19);
        let alpha_pows = scalar_powers(alpha, TEST_RING_DIM);
        let fold_gadget = gadget_row_scalars::<TestField>(shape.depth_fold, shape.log_basis);
        let block_bits = shape.num_blocks.trailing_zeros() as usize;
        let eq_low = EqPolynomial::evals(&full_vec_randomness[..block_bits]).unwrap();
        let z_offset_low_bits = shape.block_len.trailing_zeros() as usize;
        let z_block_low_eq = if z_offset_low_bits == 0 {
            vec![TestField::one()]
        } else {
            EqPolynomial::evals(&full_vec_randomness[..z_offset_low_bits]).unwrap()
        };

        Self {
            prepared,
            setup,
            full_vec_randomness,
            eq_low,
            z_block_low_eq,
            alpha_pows,
            fold_gadget,
        }
    }

    fn prepare_plan(
        &self,
        eq_low: Option<&[TestField]>,
        z_block_low_eq: Option<&[TestField]>,
    ) -> SetupContributionPlan<TestField> {
        self.prepared
            .prepare_setup_contribution_plan::<TestField>(
                &self.full_vec_randomness,
                eq_low,
                z_block_low_eq,
                &self.fold_gadget,
            )
            .unwrap()
    }

    pub(super) fn assert_direct_matches_recursive(&self) {
        let got = self
            .prepare_plan(Some(&self.eq_low), Some(&self.z_block_low_eq))
            .evaluate_direct::<TestField, TEST_RING_DIM>(&self.setup, &self.alpha_pows)
            .unwrap();
        let recursive = recursive_inner_product(
            &self.prepare_plan(None, None),
            &self.setup,
            &self.alpha_pows,
        )
        .unwrap();
        assert_eq!(
            got, recursive,
            "packed setup contribution must equal recursive setup contribution"
        );
    }

    /// `evaluate_bar_omega_with_eq` (the const-generic `bar_omega_segment_eval`
    /// kernel) must equal the eq-weighted materialized `bar_omega` (the generic
    /// `weight_at` path). Cross-checks the two `bar_omega` implementations agree,
    /// including the tiered `B'`/`F` blocks.
    pub(super) fn assert_eq_eval_matches_materialized(&self) {
        let plan = self.prepare_plan(Some(&self.eq_low), Some(&self.z_block_low_eq));
        let bar_omega = plan.materialize_bar_omega();
        let lambda_len = plan.required().checked_next_power_of_two().unwrap();
        let eq_lambda: Vec<TestField> = (0..lambda_len)
            .map(|idx| test_scalar(7 + idx as u128 * 13))
            .collect();
        let expected: TestField = bar_omega
            .iter()
            .enumerate()
            .map(|(lambda, weight)| eq_lambda[lambda] * *weight)
            .sum();
        let got = plan.evaluate_bar_omega_with_eq(&eq_lambda).unwrap();
        assert_eq!(
            got, expected,
            "evaluate_bar_omega_with_eq must equal the eq-weighted materialized bar_omega"
        );
    }
}
