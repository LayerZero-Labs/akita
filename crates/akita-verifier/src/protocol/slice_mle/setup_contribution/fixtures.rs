//! Shared fixtures for packed setup inner-product equivalence tests.

#![allow(unreachable_pub)]

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::{eval_ring_at_pows, scalar_powers};
use akita_algebra::CyclotomicRing;
use akita_field::{CanonicalField, Prime128OffsetA7F7};
use akita_types::{
    gadget_row_scalars, AkitaExpandedSetup, AkitaSetupSeed, CommitmentRingDims, FlatMatrix,
    RelationMatrixRowLayout, SetupContributionPlan, SetupContributionPlanInputs,
    WitnessChunkLayout, WitnessChunkLengths, WitnessLayout,
};

use super::evaluate_setup_contribution_direct;
use crate::protocol::ring_switch::{
    build_setup_contribution_groups, PreparedChallengeEvals, RelationMatrixEvaluator,
    RelationMatrixGroupEvaluator,
};

pub(crate) type TestField = Prime128OffsetA7F7;
pub(crate) const TEST_RING_DIM: usize = 64;

pub(crate) fn test_scalar(value: u128) -> TestField {
    TestField::from_canonical_u128_reduced(value)
}

pub(crate) struct SetupContributionFixture {
    pub relation_matrix_evaluator: RelationMatrixEvaluator<TestField>,
    pub setup: AkitaExpandedSetup<TestField>,
    pub full_vec_randomness: Vec<TestField>,
    pub eq_low: Vec<TestField>,
    pub z_block_low_eq: Vec<TestField>,
    pub alpha_pows: Vec<TestField>,
    pub fold_gadget: Vec<TestField>,
}

pub(crate) struct SetupContributionShape {
    pub num_blocks: usize,
    pub num_claims: usize,
    pub depth_open: usize,
    pub depth_commit: usize,
    pub depth_fold: usize,
    pub block_len: usize,
    pub log_basis: u32,
    pub n_a: usize,
    pub n_d: usize,
    pub n_b: usize,
    pub num_polys_per_group: Vec<usize>,
    pub relation_matrix_row_layout: RelationMatrixRowLayout,
}

impl SetupContributionShape {
    pub fn root_single_point() -> Self {
        Self {
            num_blocks: 4,
            num_claims: 1,
            depth_open: 8,
            depth_commit: 2,
            depth_fold: 3,
            block_len: 16,
            log_basis: 4,
            n_a: 2,
            n_d: 1,
            n_b: 2,
            num_polys_per_group: vec![1],
            relation_matrix_row_layout: RelationMatrixRowLayout::WithDBlock,
        }
    }

    pub fn recursive_multi_group() -> Self {
        Self {
            num_blocks: 8,
            num_claims: 3,
            depth_open: 26,
            depth_commit: 1,
            depth_fold: 4,
            block_len: 512,
            log_basis: 5,
            n_a: 2,
            n_d: 2,
            n_b: 2,
            num_polys_per_group: vec![3],
            relation_matrix_row_layout: RelationMatrixRowLayout::WithDBlock,
        }
    }

    pub fn terminal_relation_only() -> Self {
        let mut shape = Self::root_single_point();
        shape.relation_matrix_row_layout = RelationMatrixRowLayout::WithoutDBlock;
        shape
    }

    pub fn dense_non_pow2_z() -> Self {
        let mut shape = Self::root_single_point();
        shape.block_len = 12;
        shape.depth_commit = 3;
        shape.depth_fold = 2;
        shape
    }

    pub fn batched_root() -> Self {
        let mut shape = Self::root_single_point();
        shape.num_claims = 4;
        shape.num_polys_per_group = vec![4];
        shape
    }

    pub fn e_t_offset_carry() -> Self {
        let mut shape = Self::root_single_point();
        shape.num_blocks = 8;
        shape.block_len = 10;
        shape.depth_commit = 3;
        shape.depth_fold = 2;
        shape
    }

    pub fn pow2_z_offset_carry() -> Self {
        let mut shape = Self::root_single_point();
        shape.block_len = 64;
        shape
    }
}

impl SetupContributionFixture {
    pub fn from_shape(shape: &SetupContributionShape) -> Self {
        let num_points = shape.num_polys_per_group.len();
        let num_t_vectors = shape.num_polys_per_group.iter().sum();
        let total_blocks = shape.num_blocks * shape.num_claims;
        let inner_width = shape.block_len * shape.depth_commit;

        // Canonical relation-matrix row layout: consistency | A | B | D.
        let rows = 1 + shape.n_a + shape.n_b * num_points + shape.n_d;

        let stride_t = shape.n_a * shape.depth_open;
        let cols_per_poly_t = stride_t * shape.num_blocks;
        let n_cols_e = shape.num_claims * shape.num_blocks * shape.depth_open;
        let n_cols_t = shape.num_polys_per_group.iter().copied().max().unwrap() * cols_per_poly_t;

        let e_len = shape.depth_open * total_blocks;
        let t_len = shape.depth_open * shape.n_a * shape.num_blocks * num_t_vectors;
        let z_len = shape.depth_fold * shape.depth_commit * num_points * shape.block_len;
        let offset_z = 0usize;
        let offset_e = z_len;
        let offset_t = z_len + e_len;
        let total_len = z_len + e_len + t_len;
        let offset_r: usize = total_len;
        let bits = total_len.next_power_of_two().trailing_zeros() as usize;

        let max_setup_len = (shape.n_d * n_cols_e)
            .max(shape.n_a * inner_width)
            .max(shape.n_b * n_cols_t);

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
                max_num_batched_polys: shape.num_polys_per_group.iter().sum(),
                gen_ring_dim: TEST_RING_DIM,
                max_setup_len,
                public_matrix_seed: [7u8; 32],
            },
            FlatMatrix::from_ring_slice::<TEST_RING_DIM>(&matrix_entries),
        );

        let setup_contribution_inputs = SetupContributionPlanInputs {
            relation_matrix_row_layout: shape.relation_matrix_row_layout,
            rows,
            n_a: shape.n_a,
            n_b: shape.n_b,
            n_d: shape.n_d,
            num_groups: 1,
            num_polys_per_group: shape.num_polys_per_group.clone(),
            num_t_vectors,
            num_claims: shape.num_claims,
            num_blocks: shape.num_blocks,
            block_len: shape.block_len,
            depth_open: shape.depth_open,
            depth_commit: shape.depth_commit,
            depth_fold: shape.depth_fold,
            inner_width,
            eq_tau1: (0..rows.next_power_of_two())
                .map(|idx| test_scalar(11 + idx as u128))
                .collect(),
        };
        let chunk_layout = WitnessLayout {
            blocks_per_chunk: shape.num_blocks,
            chunks: vec![WitnessChunkLayout {
                offset_z,
                offset_e,
                offset_t,
                offset_r: Some(offset_r),
                global_block_base: 0,
            }],
            chunk_lengths: vec![WitnessChunkLengths {
                z_len,
                e_len,
                t_len,
                r_len: Some(0),
            }],
        };
        let n_d_active = match shape.relation_matrix_row_layout {
            RelationMatrixRowLayout::WithDBlock => shape.n_d,
            RelationMatrixRowLayout::WithoutDBlock => 0,
        };
        let groups = vec![RelationMatrixGroupEvaluator {
            c_alphas: PreparedChallengeEvals::Flat(
                (0..total_blocks)
                    .map(|idx| test_scalar(41 + idx as u128))
                    .collect(),
            ),
            a_evals: (0..shape.block_len)
                .map(|idx| test_scalar(501 + idx as u128))
                .collect(),
            chunk_range: 0..chunk_layout.chunks.len(),
            e_col_offset: 0,
            num_claims: shape.num_claims,
            num_blocks: shape.num_blocks,
            block_len: shape.block_len,
            depth_open: shape.depth_open,
            depth_commit: shape.depth_commit,
            depth_fold: shape.depth_fold,
            log_basis: shape.log_basis,
            n_a: shape.n_a,
            n_b: shape.n_b,
            t_cols_per_vector: shape.n_a * shape.depth_open * shape.num_blocks,
            a_row_start: 1,
            b_row_start: 1 + shape.n_a,
        }];
        let setup_contribution_groups =
            build_setup_contribution_groups(&chunk_layout, &groups).unwrap();
        let setup_contribution_static = SetupContributionPlan::prepare_static(
            &setup_contribution_inputs,
            &setup_contribution_groups,
            rows - n_d_active,
            n_d_active,
            n_cols_e,
        )
        .unwrap();
        let relation_matrix_evaluator = RelationMatrixEvaluator {
            role_dims: CommitmentRingDims::uniform(TEST_RING_DIM),
            groups,
            depth_fold: shape.depth_fold,
            log_basis: shape.log_basis,
            chunk_layout,
            setup_contribution_groups,
            setup_contribution_inputs,
            setup_contribution_static,
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
            relation_matrix_evaluator,
            setup,
            full_vec_randomness,
            eq_low,
            z_block_low_eq,
            alpha_pows,
            fold_gadget,
        }
    }

    pub fn compute_contribution(&self) -> TestField {
        evaluate_setup_contribution_direct::<TestField, TestField, TEST_RING_DIM>(
            &self.relation_matrix_evaluator,
            &self.full_vec_randomness,
            Some(&self.eq_low),
            Some(&self.z_block_low_eq),
            &self.alpha_pows,
            &self.alpha_pows,
            &self.alpha_pows,
            &self.fold_gadget,
            &self.setup,
        )
        .unwrap()
    }

    pub fn materialized_contribution(&self) -> TestField {
        let plan = SetupContributionPlan::finish_plan::<TestField>(
            &self.relation_matrix_evaluator.setup_contribution_static,
            &self.full_vec_randomness,
            Some(&self.eq_low),
            Some(&self.z_block_low_eq),
            Some(&self.fold_gadget),
            &self.relation_matrix_evaluator.setup_contribution_groups,
        )
        .unwrap();
        let bar_omega = plan.materialize_bar_omega().unwrap();
        let setup_len = self
            .setup
            .shared_matrix()
            .total_ring_elements_at::<TEST_RING_DIM>()
            .unwrap();
        assert!(
            setup_len >= bar_omega.len(),
            "fixture setup must cover materialized setup weights"
        );
        let setup_view = self
            .setup
            .shared_matrix()
            .ring_view::<TEST_RING_DIM>(1, setup_len)
            .unwrap();
        setup_view
            .as_slice()
            .iter()
            .zip(bar_omega)
            .map(|(ring, weight)| eval_ring_at_pows(ring, &self.alpha_pows) * weight)
            .sum()
    }

    pub fn assert_direct_matches_materialized(&self) {
        let got = self.compute_contribution();
        let expected = self.materialized_contribution();
        assert_eq!(
            got, expected,
            "packed setup contribution must equal materialized setup contribution"
        );
    }

    /// `evaluate_bar_omega_with_eq` (the const-generic `bar_omega_segment_eval`
    /// kernel) must equal the eq-weighted materialized `bar_omega` (the generic
    /// `weight_at` path). Cross-checks the two `bar_omega` implementations agree.
    pub fn assert_eq_eval_matches_materialized(&self) {
        let plan = SetupContributionPlan::finish_plan::<TestField>(
            &self.relation_matrix_evaluator.setup_contribution_static,
            &self.full_vec_randomness,
            Some(&self.eq_low),
            Some(&self.z_block_low_eq),
            Some(&self.fold_gadget),
            &self.relation_matrix_evaluator.setup_contribution_groups,
        )
        .unwrap();
        let bar_omega = plan.materialize_bar_omega().unwrap();
        let setup_idx_len = plan
            .required()
            .unwrap()
            .checked_next_power_of_two()
            .unwrap();
        let eq_setup_idx: Vec<TestField> = (0..setup_idx_len)
            .map(|idx| test_scalar(7 + idx as u128 * 13))
            .collect();
        let expected: TestField = bar_omega
            .iter()
            .enumerate()
            .map(|(setup_idx, weight)| eq_setup_idx[setup_idx] * *weight)
            .sum();
        let got = plan.evaluate_bar_omega_with_eq(&eq_setup_idx).unwrap();
        assert_eq!(
            got, expected,
            "evaluate_bar_omega_with_eq must equal the eq-weighted materialized bar_omega"
        );
    }
}
