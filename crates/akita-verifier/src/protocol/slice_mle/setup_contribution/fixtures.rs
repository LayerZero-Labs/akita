//! Shared fixtures for packed setup inner-product equivalence tests.

#![allow(unreachable_pub)]

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::{eval_ring_at_pows, scalar_powers};
use akita_algebra::CyclotomicRing;
use akita_field::{CanonicalField, Prime128OffsetA7F7};
use akita_types::{
    gadget_row_scalars, AkitaExpandedSetup, AkitaSetupSeed, CommitmentRingDims, FlatMatrix,
    LevelParams, OpeningClaimsLayout, RelationMatrixRowLayout, SetupContributionPlan,
    SisModulusProfileId, WitnessLayout,
};

use super::evaluate_setup_contribution_direct;
use crate::protocol::ring_switch::{
    build_setup_contribution_layout, PreparedChallengeEvals, RelationMatrixEvaluator,
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

#[derive(Clone)]
pub(crate) struct SetupContributionShape {
    pub num_live_blocks: usize,
    pub num_claims: usize,
    pub depth_open: usize,
    pub depth_commit: usize,
    pub depth_fold: usize,
    pub num_positions_per_block: usize,
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
            num_live_blocks: 4,
            num_claims: 1,
            depth_open: 8,
            depth_commit: 2,
            depth_fold: 3,
            num_positions_per_block: 16,
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
            num_live_blocks: 8,
            num_claims: 3,
            depth_open: 26,
            depth_commit: 1,
            depth_fold: 4,
            num_positions_per_block: 512,
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
        shape.num_positions_per_block = 16;
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
        shape.num_live_blocks = 8;
        shape.num_positions_per_block = 16;
        shape.depth_commit = 3;
        shape.depth_fold = 2;
        shape
    }

    pub fn pow2_z_offset_carry() -> Self {
        let mut shape = Self::root_single_point();
        shape.num_positions_per_block = 64;
        shape
    }
}

impl SetupContributionFixture {
    pub fn from_shape(shape: &SetupContributionShape) -> Self {
        let mut shape = shape.clone();
        let num_points = shape.num_polys_per_group.len();
        let total_blocks = shape.num_live_blocks * shape.num_claims;
        let inner_width = shape.num_positions_per_block * shape.depth_commit;

        // Canonical relation-matrix row layout: consistency | A | B | D.
        let rows = 1 + shape.n_a + shape.n_b * num_points + shape.n_d;

        let stride_t = shape.n_a * shape.depth_open;
        let cols_per_poly_t = stride_t * shape.num_live_blocks;
        let n_cols_e = shape.num_claims * shape.num_live_blocks * shape.depth_open;
        let n_cols_t = shape.num_polys_per_group.iter().copied().max().unwrap() * cols_per_poly_t;

        let lp = LevelParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            TEST_RING_DIM,
            shape.log_basis,
            shape.n_a,
            shape.n_b,
            shape.n_d,
            akita_challenges::SparseChallengeConfig::pm1_only(1),
        )
        .with_decomp(
            shape.num_positions_per_block,
            shape.num_live_blocks * shape.num_positions_per_block,
            shape.depth_commit,
            shape.depth_open,
        )
        .expect("setup contribution fixture params");
        shape.depth_fold = lp
            .num_digits_fold(shape.num_claims, lp.field_bits_for_cache())
            .expect("setup contribution fixture fold depth");
        let opening_batch = OpeningClaimsLayout::from_group_sizes(0, &shape.num_polys_per_group)
            .expect("setup contribution fixture opening batch");
        let layout = WitnessLayout::new(&lp, &opening_batch, 1, 0, 1)
            .expect("setup contribution fixture layout");
        let offset_r = layout.r_range().start;
        let total_len = offset_r;
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

        let eq_tau1 = (0..rows.next_power_of_two())
            .map(|idx| test_scalar(11 + idx as u128))
            .collect::<Vec<_>>()
            .into();
        let groups = vec![RelationMatrixGroupEvaluator {
            c_alphas: PreparedChallengeEvals::Flat(
                (0..total_blocks)
                    .map(|idx| test_scalar(41 + idx as u128))
                    .collect(),
            ),
            opening_a_evals: (0..shape.num_positions_per_block)
                .map(|idx| test_scalar(501 + idx as u128))
                .collect(),
            group_id: 0,
            num_claims: shape.num_claims,
            num_live_blocks: shape.num_live_blocks,
            depth_open: shape.depth_open,
            depth_commit: shape.depth_commit,
            depth_fold: shape.depth_fold,
            log_basis: shape.log_basis,
            n_a: shape.n_a,
            a_row_start: 1,
            b_row_start: 1 + shape.n_a,
        }];
        let opening_source_len = layout.total_len();
        let layout = std::sync::Arc::new(layout);
        let setup_contribution_layout = build_setup_contribution_layout(
            &lp,
            &opening_batch,
            shape.relation_matrix_row_layout,
            layout.clone(),
            opening_source_len,
            &groups,
        )
        .unwrap();
        let setup_contribution_static = SetupContributionPlan::prepare_static(
            &lp,
            &opening_batch,
            shape.relation_matrix_row_layout,
            eq_tau1,
            &setup_contribution_layout,
        )
        .unwrap();
        let relation_matrix_evaluator = RelationMatrixEvaluator {
            role_dims: CommitmentRingDims::uniform(TEST_RING_DIM),
            groups,
            log_basis: shape.log_basis,
            setup_contribution_layout,
            setup_contribution_static,
            flat_context: None,
        };

        let full_vec_randomness: Vec<TestField> = (0..bits)
            .map(|idx| test_scalar(101 + idx as u128))
            .collect();
        let alpha = test_scalar(19);
        let alpha_pows = scalar_powers(alpha, TEST_RING_DIM);
        let fold_gadget = gadget_row_scalars::<TestField>(shape.depth_fold, shape.log_basis);
        let block_index_bits = shape.num_live_blocks.trailing_zeros() as usize;
        let eq_low = EqPolynomial::evals(&full_vec_randomness[..block_index_bits]).unwrap();
        let z_offset_low_bits = shape.num_positions_per_block.trailing_zeros() as usize;
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
            &self.relation_matrix_evaluator.setup_contribution_layout,
            self.relation_matrix_evaluator.role_dims,
        )
        .unwrap();
        let alpha = self.alpha_pows[1];
        let setup_index_weight = plan.materialize_setup_index_weights(alpha).unwrap();
        let setup_len = self
            .setup
            .shared_matrix()
            .total_ring_elements_at::<TEST_RING_DIM>()
            .unwrap();
        assert!(
            setup_len >= setup_index_weight.len(),
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
            .zip(setup_index_weight)
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

    /// Direct MLE evaluation of the setup-index weight must equal an
    /// eq-weighted materialized `setup_index_weight`.
    pub fn assert_setup_index_weight_mle_matches_materialized(&self) {
        let plan = SetupContributionPlan::finish_plan::<TestField>(
            &self.relation_matrix_evaluator.setup_contribution_static,
            &self.full_vec_randomness,
            Some(&self.eq_low),
            Some(&self.z_block_low_eq),
            Some(&self.fold_gadget),
            &self.relation_matrix_evaluator.setup_contribution_layout,
            self.relation_matrix_evaluator.role_dims,
        )
        .unwrap();
        let alpha = self.alpha_pows[1];
        let setup_index_weight = plan.materialize_setup_index_weights(alpha).unwrap();
        let setup_idx_len = plan.required().checked_next_power_of_two().unwrap();
        let setup_idx_bits = setup_idx_len.trailing_zeros() as usize;
        let rho_setup_idx: Vec<TestField> = (0..setup_idx_bits)
            .map(|idx| test_scalar(7 + idx as u128 * 13))
            .collect();
        let eq_setup_idx = EqPolynomial::evals(&rho_setup_idx).unwrap();
        let expected: TestField = setup_index_weight
            .iter()
            .enumerate()
            .map(|(setup_idx, weight)| eq_setup_idx[setup_idx] * *weight)
            .sum();
        let got = plan
            .evaluate_setup_index_weight_mle(&rho_setup_idx, alpha)
            .unwrap();
        assert_eq!(
            got, expected,
            "setup-index weight MLE must equal the eq-weighted materialized setup-index weight"
        );
    }
}
