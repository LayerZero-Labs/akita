//! Fast-verifier presets — small-proof / tensor-shaped Akita configurations.
//!
//! Sibling of [`crate::proof_optimized`]. The proof-optimized presets target
//! the smallest proof at the canonical flat (per-block) stage-1 challenge
//! sampling. This module is the home for presets that opt one or more fold
//! levels into a **tensor-shaped** challenge sampler instead, which
//!
//! 1. shrinks the fold-section bytes of the proof (the root fold's challenge
//!    sampling collapses from `O(num_blocks)` to `O(√num_blocks)` per claim),
//!    and
//! 2. lets the verifier evaluate the fold-round challenge at a random ring
//!    point via [`akita_challenges::Challenges::evals_at_pows`]'s factored
//!    aggregate, in `O(√num_blocks · D)` per claim rather than
//!    `O(num_blocks)`.
//!
//! Tensor-shaped challenges propagate through the protocol exclusively via
//! the [`akita_challenges::Challenges`] enum (`Sparse` vs `Tensor`).
//! Prover and verifier code only ever calls the enum's methods
//! (`evals_at_pows`, `accumulate_high_half`, `select_claims`,
//! `decompose_fold` on the poly backend); the per-variant dispatch lives
//! inside those methods. Adding another fast-verifier preset is a matter of
//! defining the preset struct and overriding
//! [`CommitmentConfig::fold_challenge_shape_at_level`].
//!
//! Current shipping preset:
//!
//! - [`fp128::D64OneHotTensor`] — fp128 ring `D=64`, binary onehot data,
//!   tensor-shaped root fold, flat recursive folds. Uses the
//!   `fp128_d64_onehot_tensor` generated schedule table.

use crate::CommitmentConfig;

/// fp128 presets that activate tensor-shaped fold challenges.
pub mod fp128 {
    use super::CommitmentConfig;
    use crate::proof_optimized::fp128::Field;

    /// Binary onehot `D=64` preset that samples a tensor-shaped stage-1 fold
    /// challenge at the root level (recursive levels remain flat). Uses the
    /// dedicated `fp128_d64_onehot_tensor` generated schedule table.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D64OneHotTensor;

    impl akita_types::ScheduleProvider for D64OneHotTensor {
        fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
            Some(akita_types::generated::fp128_d64_onehot_tensor_table())
        }

        fn schedule_key(key: akita_types::AkitaScheduleLookupKey) -> String {
            crate::proof_optimized::proof_optimized_schedule_key::<Self>(key)
        }

        fn schedule_plan(
            key: akita_types::AkitaScheduleLookupKey,
        ) -> Result<Option<akita_types::AkitaSchedulePlan>, akita_field::AkitaError> {
            crate::proof_optimized::proof_optimized_schedule_plan::<Self>(key)
        }
    }

    impl CommitmentConfig for D64OneHotTensor {
        type Field = Field;
        type ClaimField = Field;
        type ChallengeField = Field;
        const D: usize = 64;

        fn decomposition() -> akita_types::DecompositionParams {
            crate::proof_optimized::fp128_decomposition(1, 3)
        }

        fn stage1_challenge_config(d: usize) -> akita_challenges::SparseChallengeConfig {
            crate::proof_optimized::fp128_stage1_challenge_config(d)
        }

        fn fold_challenge_shape_at_level(
            inputs: akita_types::AkitaScheduleInputs,
        ) -> akita_challenges::TensorChallengeShape {
            // Root fold (level 0) is tensor-shaped; recursive levels stay
            // flat because their layout is much smaller and the tensor
            // win does not amortize.
            if inputs.level == 0 {
                akita_challenges::TensorChallengeShape::Tensor
            } else {
                akita_challenges::TensorChallengeShape::Flat
            }
        }

        fn sis_modulus_family() -> akita_types::SisModulusFamily {
            akita_types::SisModulusFamily::Q128
        }

        fn audited_root_rank(role: akita_types::AjtaiRole, max_num_vars: usize) -> usize {
            crate::proof_optimized::fp128_audited_root_rank::<Self>(role, max_num_vars)
        }

        fn envelope(max_num_vars: usize) -> akita_types::CommitmentEnvelope {
            crate::proof_optimized::proof_optimized_envelope::<Self>(max_num_vars)
        }

        fn max_setup_matrix_size(
            max_num_vars: usize,
            max_num_batched_polys: usize,
            max_num_points: usize,
        ) -> Result<(usize, usize), akita_field::AkitaError> {
            crate::proof_optimized::proof_optimized_max_setup_matrix_size::<Self>(
                max_num_vars,
                max_num_batched_polys,
                max_num_points,
            )
        }

        fn level_params_with_log_basis(
            inputs: akita_types::AkitaScheduleInputs,
            log_basis: u32,
        ) -> akita_types::LevelParams {
            crate::proof_optimized::proof_optimized_level_params_with_log_basis::<Self>(
                inputs, log_basis,
            )
        }

        fn root_level_params_for_layout_with_log_basis(
            inputs: akita_types::AkitaScheduleInputs,
            lp: &akita_types::LevelParams,
        ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
            crate::proof_optimized::proof_optimized_root_level_params_for_layout_with_log_basis::<
                Self,
            >(inputs, lp)
        }

        fn root_level_layout_with_log_basis(
            inputs: akita_types::AkitaScheduleInputs,
            log_basis: u32,
        ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
            crate::proof_optimized::proof_optimized_root_level_layout_with_log_basis::<Self>(
                inputs, log_basis,
            )
        }

        fn log_basis_at_level(inputs: akita_types::AkitaScheduleInputs) -> u32 {
            crate::proof_optimized::proof_optimized_log_basis_at_level::<Self>(inputs)
        }

        fn log_basis_search_range(_inputs: akita_types::AkitaScheduleInputs) -> (u32, u32) {
            crate::proof_optimized::proof_optimized_log_basis_search_range()
        }
    }

    #[cfg(feature = "planner")]
    impl akita_planner::PlannerConfig for D64OneHotTensor {
        type PlannerField = Field;
        const PLANNER_D: usize = 64;

        fn planner_field_bits() -> u32 {
            <Self as CommitmentConfig>::decomposition().field_bits()
        }

        fn planner_challenge_field_bits() -> u32 {
            <Self as CommitmentConfig>::decomposition().field_bits()
                * (<Self as CommitmentConfig>::CHAL_EXT_DEGREE as u32)
        }

        fn planner_extension_opening_width() -> usize {
            <Self as CommitmentConfig>::CLAIM_EXT_DEGREE
        }

        fn planner_recursive_witness_expansion() -> usize {
            1
        }

        fn planner_recursive_public_rows() -> usize {
            1
        }

        fn planner_sis_modulus_family() -> akita_types::SisModulusFamily {
            <Self as CommitmentConfig>::sis_modulus_family()
        }

        fn planner_stage1_challenge_config(d: usize) -> akita_challenges::SparseChallengeConfig {
            <Self as CommitmentConfig>::stage1_challenge_config(d)
        }

        fn planner_schedule_plan(
            key: akita_types::AkitaScheduleLookupKey,
        ) -> Result<Option<akita_types::AkitaSchedulePlan>, akita_field::AkitaError> {
            <Self as akita_types::ScheduleProvider>::schedule_plan(key)
        }

        fn planner_root_level_layout_with_log_basis(
            inputs: akita_types::AkitaScheduleInputs,
            log_basis: u32,
        ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
            <Self as CommitmentConfig>::root_level_layout_with_log_basis(inputs, log_basis)
        }

        fn planner_current_level_layout_with_log_basis(
            inputs: akita_types::AkitaScheduleInputs,
            log_basis: u32,
        ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
            crate::current_level_layout_with_log_basis::<Self>(inputs, log_basis)
        }

        fn planner_direct_level_params_with_log_basis(
            inputs: akita_types::AkitaScheduleInputs,
            log_basis: u32,
        ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
            crate::schedule_policy::direct_level_params_with_log_basis::<Self>(inputs, log_basis)
        }

        fn planner_root_level_params_for_layout_with_log_basis(
            inputs: akita_types::AkitaScheduleInputs,
            lp: &akita_types::LevelParams,
        ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
            <Self as CommitmentConfig>::root_level_params_for_layout_with_log_basis(inputs, lp)
        }

        fn planner_log_basis_search_range(inputs: akita_types::AkitaScheduleInputs) -> (u32, u32) {
            <Self as CommitmentConfig>::log_basis_search_range(inputs)
        }
    }
}
