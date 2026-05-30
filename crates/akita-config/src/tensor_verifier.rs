//! Tensor-verifier presets that trade a tensor-shaped fold challenge for cheaper
//! verifier-side challenge evaluation.

use crate::CommitmentConfig;

/// fp128 presets that activate tensor-shaped fold challenges.
pub mod fp128 {
    use super::CommitmentConfig;
    use akita_challenges::TensorChallengeShape;
    use akita_field::Prime128OffsetA7F7;
    use akita_types::generated::GeneratedScheduleTable;
    use akita_types::{
        AkitaScheduleInputs, AkitaScheduleLookupKey, AkitaSchedulePlan, DecompositionParams,
        SisModulusFamily,
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
        type ClaimField = Field;
        type ChallengeField = Field;
        const D: usize = 64;

        fn decomposition() -> DecompositionParams {
            DecompositionParams {
                log_basis: 3,
                log_commit_bound: 1,
                log_open_bound: Some(128),
            }
        }

        fn stage1_challenge_config(
            d: usize,
        ) -> Result<akita_challenges::SparseChallengeConfig, akita_field::AkitaError> {
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

        fn sis_modulus_family() -> SisModulusFamily {
            SisModulusFamily::Q128
        }

        fn schedule_table() -> Option<GeneratedScheduleTable> {
            Some(akita_types::generated::fp128_d64_onehot_tensor_table())
        }

        fn schedule_plan(
            key: AkitaScheduleLookupKey,
        ) -> Result<Option<AkitaSchedulePlan>, akita_field::AkitaError> {
            crate::proof_optimized::proof_optimized_schedule_plan::<Self>(key)
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

        fn basis_range() -> (u32, u32) {
            (
                crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MIN,
                crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MAX,
            )
        }
    }
}
