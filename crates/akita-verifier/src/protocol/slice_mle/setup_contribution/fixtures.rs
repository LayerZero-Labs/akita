//! Shared fixtures for packed setup inner-product equivalence tests.

#![allow(unreachable_pub)]

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::scalar_powers;
use akita_algebra::CyclotomicRing;
use akita_field::{CanonicalField, Prime128OffsetA7F7};
use akita_types::{
    gadget_row_scalars, outer_consistency_row_start, AkitaExpandedSetup, AkitaSetupSeed,
    FlatMatrix, MRowLayout, SetupContributionPlanInputs, WitnessChunkLayout, WitnessChunkLengths,
    WitnessLayout,
};

use super::{SetupEvaluation, SetupEvaluator, SetupEvaluatorMode};
use crate::protocol::ring_switch::{PreparedChallengeEvals, RingSwitchDeferredRowEval};

pub(crate) type TestField = Prime128OffsetA7F7;
pub(crate) const TEST_RING_DIM: usize = 32;

pub(crate) fn test_scalar(value: u128) -> TestField {
    TestField::from_canonical_u128_reduced(value)
}

pub(crate) struct SetupContributionFixture {
    pub prepared: RingSwitchDeferredRowEval<TestField>,
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
    pub num_polys_per_segment: Vec<usize>,
    pub m_row_layout: MRowLayout,
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
            num_polys_per_segment: vec![1],
            m_row_layout: MRowLayout::WithDBlock,
        }
    }

    pub fn recursive_multigroup() -> Self {
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
            num_polys_per_segment: vec![3],
            m_row_layout: MRowLayout::WithDBlock,
        }
    }

    pub fn terminal_relation_only() -> Self {
        let mut shape = Self::root_single_point();
        shape.m_row_layout = MRowLayout::WithoutDBlock;
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
        shape.num_polys_per_segment = vec![4];
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
        let num_points = shape.num_polys_per_segment.len();
        let num_t_vectors = shape.num_polys_per_segment.iter().sum();
        let total_blocks = shape.num_blocks * shape.num_claims;
        let inner_width = shape.block_len * shape.depth_commit;

        // Canonical M-row layout: EvaluationTrace | FoldEvaluation | FoldConsistency | B | D.
        let rows = outer_consistency_row_start(shape.n_a) + shape.n_b * num_points + shape.n_d;

        let stride_t = shape.n_a * shape.depth_open;
        let cols_per_poly_t = stride_t * shape.num_blocks;
        let n_cols_e = shape.num_claims * shape.num_blocks * shape.depth_open;
        let n_cols_t = shape.num_polys_per_segment.iter().copied().max().unwrap() * cols_per_poly_t;

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
                max_num_batched_polys: shape.num_polys_per_segment.iter().sum(),
                gen_ring_dim: TEST_RING_DIM,
                max_setup_len,
                public_matrix_seed: [7u8; 32],
            },
            FlatMatrix::from_ring_slice::<TEST_RING_DIM>(&matrix_entries),
        );

        let eq_tau1: Vec<TestField> = (0..rows.next_power_of_two())
            .map(|idx| test_scalar(11 + idx as u128))
            .collect();
        let setup_contribution_inputs = SetupContributionPlanInputs {
            eq_tau1: eq_tau1.clone(),
            num_t_vectors,
            num_blocks: shape.num_blocks,
            num_claims: shape.num_claims,
            depth_open: shape.depth_open,
            depth_commit: shape.depth_commit,
            depth_fold: shape.depth_fold,
            block_len: shape.block_len,
            inner_width,
            n_a: shape.n_a,
            n_d: shape.n_d,
            m_row_layout: shape.m_row_layout,
            n_b: shape.n_b,
            num_segments: 1,
            rows,
            num_polys_per_segment: shape.num_polys_per_segment.clone(),
        };
        let prepared = RingSwitchDeferredRowEval {
            c_alphas: PreparedChallengeEvals::Flat(
                (0..total_blocks)
                    .map(|idx| test_scalar(41 + idx as u128))
                    .collect(),
            ),
            eq_tau1,
            num_t_vectors,
            num_blocks: shape.num_blocks,
            num_claims: shape.num_claims,
            depth_open: shape.depth_open,
            depth_commit: shape.depth_commit,
            depth_fold: shape.depth_fold,
            block_len: shape.block_len,
            log_basis: shape.log_basis,
            n_a: shape.n_a,
            chunk_layout: WitnessLayout {
                blocks_per_chunk: shape.num_blocks,
                chunks: vec![WitnessChunkLayout {
                    offset_z,
                    offset_e,
                    offset_t,
                    offset_u: None,
                    offset_r: Some(offset_r),
                    global_block_base: 0,
                }],
                chunk_lengths: vec![WitnessChunkLengths {
                    z_len,
                    e_len,
                    t_len,
                    u_len: None,
                    r_len: Some(0),
                }],
                quotient_layout: WitnessLayout::empty_quotient_layout(),
            },
            setup_contribution_inputs,
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

    pub fn compute_contribution(&self) -> TestField {
        let setup_contribution = self.prepared.create_setup_contribution_inputs();
        let evaluator = SetupEvaluator::new(
            &setup_contribution,
            &self.full_vec_randomness,
            Some(&self.eq_low),
            Some(&self.z_block_low_eq),
            &self.alpha_pows,
            &self.fold_gadget,
            &self.prepared.chunk_layout,
        );
        match evaluator
            .evaluate::<TEST_RING_DIM>(SetupEvaluatorMode::Direct { setup: &self.setup })
            .unwrap()
        {
            SetupEvaluation::Direct(value) => value,
            SetupEvaluation::Recursive(_) => {
                panic!("setup evaluator returned recursive output for direct mode")
            }
        }
    }

    pub fn recursive_contribution(&self) -> TestField {
        let setup_contribution = self.prepared.create_setup_contribution_inputs();
        let evaluator = SetupEvaluator::new(
            &setup_contribution,
            &self.full_vec_randomness,
            None,
            None,
            &self.alpha_pows,
            &self.fold_gadget,
            &self.prepared.chunk_layout,
        );
        match evaluator
            .evaluate::<TEST_RING_DIM>(SetupEvaluatorMode::Recursive { setup: &self.setup })
            .unwrap()
        {
            SetupEvaluation::Recursive(value) => value,
            SetupEvaluation::Direct(_) => {
                panic!("setup evaluator returned direct output for recursive mode")
            }
        }
    }

    pub fn assert_direct_matches_recursive(&self) {
        let got = self.compute_contribution();
        let recursive = self.recursive_contribution();
        assert_eq!(
            got, recursive,
            "packed setup contribution must equal recursive setup contribution"
        );
    }

    /// `evaluate_bar_omega_with_eq` (the const-generic `bar_omega_segment_eval`
    /// kernel) must equal the eq-weighted materialized `bar_omega` (the generic
    /// `weight_at` path). Cross-checks the two `bar_omega` implementations agree.
    pub fn assert_eq_eval_matches_materialized(&self) {
        let setup_contribution = self.prepared.create_setup_contribution_inputs();
        let evaluator = SetupEvaluator::new(
            &setup_contribution,
            &self.full_vec_randomness,
            Some(&self.eq_low),
            Some(&self.z_block_low_eq),
            &self.alpha_pows,
            &self.fold_gadget,
            &self.prepared.chunk_layout,
        );
        let plan = evaluator.prepare().unwrap();
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
