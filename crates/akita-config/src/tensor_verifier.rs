//! Tensor-verifier presets that trade a tensor-shaped fold challenge for cheaper
//! verifier-side challenge evaluation.

use crate::CommitmentConfig;

/// fp128 presets that activate tensor-shaped fold challenges.
pub mod fp128 {
    use super::CommitmentConfig;
    use akita_challenges::TensorChallengeShape;
    use akita_field::Prime128OffsetA7F7;
    use akita_types::{
        AkitaScheduleInputs, DecompositionParams, OpeningClaimsLayout, Schedule,
        SisModulusProfileId,
    };

    /// Base field for the fp128 tensor-verifier presets.
    pub type Field = Prime128OffsetA7F7;

    /// Binary onehot `D=64` preset that samples a tensor-shaped stage-1 fold
    /// challenge at the root level (recursive levels remain flat). Uses the
    /// dedicated `fp128_d64_onehot_tensor` generated schedule table.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D64OneHotTensor;

    impl CommitmentConfig for D64OneHotTensor {
        type Field = Field;
        type ExtField = Field;
        const D: usize = 64;

        fn decomposition() -> DecompositionParams {
            DecompositionParams {
                log_basis: 3,
                log_commit_bound: 1,
                log_open_bound: Some(128),
            }
        }

        fn ring_challenge_config(
            d: usize,
        ) -> Result<akita_challenges::SparseChallengeConfig, akita_field::AkitaError> {
            crate::proof_optimized::proof_optimized_ring_challenge_config(d)
        }

        /// Tensor at the root level (`level == 0`), flat at every recursive
        /// level. The schedule materializer reads this hook *before* deriving
        /// the fold digit count and the `(m_vars, r_vars)` split, so the root
        /// step's `LevelParams` are dimensioned for `omega^2`.
        fn fold_challenge_shape_at_level(inputs: AkitaScheduleInputs) -> TensorChallengeShape {
            if inputs.level == 0 {
                TensorChallengeShape::Tensor
            } else {
                TensorChallengeShape::Flat
            }
        }

        fn sis_modulus_profile() -> SisModulusProfileId {
            SisModulusProfileId::Q128OffsetA7F7
        }

        fn max_setup_matrix_size(
            max_num_vars: usize,
            max_num_batched_polys: usize,
        ) -> Result<akita_types::SetupMatrixEnvelope, akita_field::AkitaError> {
            crate::proof_optimized::proof_optimized_max_setup_matrix_size::<Self>(
                max_num_vars,
                max_num_batched_polys,
            )
        }

        fn basis_range() -> (u32, u32) {
            (
                crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MIN,
                crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MAX,
            )
        }

        fn onehot_chunk_size() -> usize {
            256
        }

        fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
            #[cfg(feature = "schedules-fp128-d64-onehot-tensor")]
            {
                Some(akita_schedules::fp128_d64_onehot_tensor_table())
            }
            #[cfg(not(feature = "schedules-fp128-d64-onehot-tensor"))]
            {
                None
            }
        }

        fn get_params_for_prove(
            layout: &OpeningClaimsLayout,
        ) -> Result<Schedule, akita_field::AkitaError> {
            Self::runtime_schedule(
                crate::proof_optimized::proof_optimized_schedule_key::<Self>(layout)?,
            )
        }
    }
}
