//! Fast-verifier presets — small-proof / tensor-shaped Akita configurations.
//!
//! Sibling of [`crate::proof_optimized`]. Proof-optimized presets target the
//! smallest proof at the canonical flat (per-block) stage-1 challenge
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
//! inside those methods.
//!
//! Each fast-verifier preset's `schedule_plan` impl returns the standard
//! `proof_optimized_schedule_plan` output post-processed to set
//! [`akita_types::LevelParams::fold_challenge_shape`] on the levels that
//! the preset wants to sample tensor-shaped.

use crate::CommitmentConfig;

/// fp128 presets that activate tensor-shaped fold challenges.
pub mod fp128 {
    use super::CommitmentConfig;
    use akita_challenges::TensorChallengeShape;
    use akita_field::Prime128OffsetA7F7;
    use akita_types::generated::GeneratedScheduleTable;
    use akita_types::{
        AjtaiRole, AkitaPlannedStep, AkitaScheduleInputs, AkitaScheduleLookupKey,
        AkitaSchedulePlan, CommitmentEnvelope, DecompositionParams, SisModulusFamily,
    };

    /// Base field for the fp128 fast-verifier presets.
    pub type Field = Prime128OffsetA7F7;

    /// Binary onehot `D=64` preset that samples a tensor-shaped stage-1 fold
    /// challenge at the root level (recursive levels remain flat). Uses the
    /// dedicated `fp128_d64_onehot_tensor` generated schedule table.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D64OneHotTensor;

    impl CommitmentConfig for D64OneHotTensor {
        type Field = Field;
        type ClaimField = Field;
        type ChallengeField = Field;
        const D: usize = 64;

        fn decomposition() -> DecompositionParams {
            // Mirror the fp128 onehot preset: `log_basis = 3`,
            // `log_commit_bound = 1`, `log_open_bound = Some(128)`.
            DecompositionParams {
                log_basis: 3,
                log_commit_bound: 1,
                log_open_bound: Some(128),
            }
        }

        fn stage1_challenge_config(
            d: usize,
        ) -> Result<akita_challenges::SparseChallengeConfig, akita_field::AkitaError> {
            // Same exact-shell family as `fp128::D64OneHot`.
            match d {
                64 => Ok(akita_challenges::SparseChallengeConfig::ExactShell {
                    count_mag1: 30,
                    count_mag2: 12,
                }),
                _ => Err(akita_field::AkitaError::InvalidSetup(format!(
                    "unsupported D={d} for D64OneHotTensor"
                ))),
            }
        }

        fn sis_modulus_family() -> SisModulusFamily {
            SisModulusFamily::Q128
        }

        fn schedule_table() -> Option<GeneratedScheduleTable> {
            Some(akita_types::generated::fp128_d64_onehot_tensor_table())
        }

        fn schedule_plan(
            key: AkitaScheduleLookupKey,
        ) -> Result<Option<AkitaSchedulePlan>, akita_field::AkitaError> {
            let envelope = <Self as CommitmentConfig>::envelope(key.num_vars);
            let plan =
                crate::proof_optimized::proof_optimized_schedule_plan::<Self>(key, envelope)?;
            Ok(plan.map(apply_tensor_root_fold_shape))
        }

        fn audited_root_rank(role: AjtaiRole, max_num_vars: usize) -> usize {
            // Same `(D=64, log_commit_bound=1)` cell as `D64OneHot` (outer
            // escalates to 2 from `max_num_vars >= 38`).
            let threshold: Option<usize> = match role {
                AjtaiRole::Inner => None,
                AjtaiRole::Outer => Some(38),
            };
            1 + usize::from(threshold.is_some_and(|t| max_num_vars >= t))
        }

        fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
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

        fn log_basis_search_range(_inputs: AkitaScheduleInputs) -> (u32, u32) {
            (
                crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MIN,
                crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MAX,
            )
        }
    }

    /// Walk a planned schedule and set every fold step's
    /// `fold_challenge_shape` per the tensor-root policy: tensor at level 0,
    /// flat elsewhere. Used by [`D64OneHotTensor::schedule_plan`].
    fn apply_tensor_root_fold_shape(mut plan: AkitaSchedulePlan) -> AkitaSchedulePlan {
        for step in &mut plan.steps {
            if let AkitaPlannedStep::Fold(level) = step {
                level.lp.fold_challenge_shape = if level.inputs.level == 0 {
                    TensorChallengeShape::Tensor
                } else {
                    TensorChallengeShape::Flat
                };
            }
        }
        plan
    }
}
