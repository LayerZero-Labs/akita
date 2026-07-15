//! Shared batched schedule selection for prove and verify entry points.

use crate::CommitmentConfig;
use akita_field::{AkitaError, FieldCore};
use akita_types::{
    dispatch_for_field, folded_root_supports_opening_shape, root_direct_schedule,
    root_tensor_projection_enabled, schedule_is_root_direct, schedule_root_fold_step,
    FpExtEncoding, OpeningClaimsLayout, Schedule,
};

/// Select the effective runtime schedule for a batched opening, including the
/// root-direct rewrite when the folded-root opening geometry is unsupported.
///
/// Prove and verify must call this helper so fold-vs-direct decisions dispatch
/// on the schedule root `ring_dimension`, not a caller-supplied stack `D`.
///
/// # Errors
///
/// Returns an error when schedule lookup fails or an unsupported ring dimension
/// is encountered during dispatch.
pub fn effective_batched_schedule<Cfg>(
    opening_batch: &OpeningClaimsLayout,
    opening_point: &[Cfg::ExtField],
) -> Result<Schedule, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
{
    let num_vars = opening_batch.max_num_vars();
    let root_direct_witness_len = opening_batch.root_direct_witness_len()?;
    let mut schedule = Cfg::get_params_for_prove(opening_batch)?;
    if opening_batch.num_groups() > 1 && schedule_is_root_direct(&schedule) {
        return Err(AkitaError::InvalidSetup(
            "multi-group openings require a folded root".to_string(),
        ));
    }
    if let Some(root_step) = schedule_root_fold_step(&schedule) {
        let alpha_bits = root_step.params.ring_dimension.trailing_zeros() as usize;
        let supports_opening_shape = dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            Cfg::Field,
            root_step.params.ring_dimension,
            |D| Ok(folded_root_supports_opening_shape::<
                Cfg::Field,
                Cfg::ExtField,
                D,
            >(
                std::slice::from_ref(&opening_point),
                &root_step.params,
                alpha_bits,
            ))
        )?;
        let tensor_projection_enabled = root_tensor_projection_enabled::<Cfg::Field, Cfg::ExtField>(
            root_step.params.ring_dimension,
            num_vars,
        );
        let needs_root_direct_rewrite = !supports_opening_shape && !tensor_projection_enabled;

        if opening_batch.num_groups() > 1 {
            if Cfg::EXT_DEGREE != 1 {
                return Err(AkitaError::InvalidSetup(
                    "multi-group extension openings cannot use root-direct rewrite".to_string(),
                ));
            }
            if needs_root_direct_rewrite {
                return Err(AkitaError::InvalidSetup(
                    "multi-group folded-root opening geometry is unsupported".to_string(),
                ));
            }
        } else if needs_root_direct_rewrite {
            let commit_params = Cfg::get_params_for_batched_commitment(opening_batch)?;
            schedule = root_direct_schedule(root_direct_witness_len, commit_params)?;
        }
    }

    schedule.validate_structure()?;
    Ok(schedule)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::{ExtField, Fp32, FpExt4};
    use akita_types::{
        AkitaScheduleLookupKey, CleartextWitnessShape, DirectStep, FoldStep, LevelParams,
        PolynomialGroupLayout, SetupMatrixEnvelope, SisModulusProfileId, Step,
    };

    type Base = Fp32<251>;
    type BaseExt = FpExt4<Base>;

    #[derive(Clone)]
    struct GroupedExtensionCfg;

    fn multi_group_extension_params() -> Result<LevelParams, AkitaError> {
        Ok(LevelParams::params_only(
            SisModulusProfileId::Q32Offset99,
            GroupedExtensionCfg::D,
            3,
            1,
            1,
            1,
            GroupedExtensionCfg::ring_challenge_config(GroupedExtensionCfg::D)?,
        ))
    }

    impl CommitmentConfig for GroupedExtensionCfg {
        type Field = Base;
        type ExtField = BaseExt;

        const D: usize = 8;

        fn decomposition() -> akita_types::DecompositionParams {
            akita_types::DecompositionParams {
                log_basis: 3,
                log_commit_bound: 8,
                log_open_bound: Some(8),
            }
        }

        fn ring_challenge_config(_d: usize) -> Result<SparseChallengeConfig, AkitaError> {
            Ok(SparseChallengeConfig::pm1_only(1))
        }

        fn sis_modulus_profile() -> SisModulusProfileId {
            SisModulusProfileId::Q32Offset99
        }

        fn max_setup_matrix_size(
            _max_num_vars: usize,
            _max_num_batched_polys: usize,
        ) -> Result<SetupMatrixEnvelope, AkitaError> {
            Ok(SetupMatrixEnvelope { max_setup_len: 1 })
        }

        fn basis_range() -> (u32, u32) {
            (3, 3)
        }

        fn runtime_schedule(_key: AkitaScheduleLookupKey) -> Result<Schedule, AkitaError> {
            let params = multi_group_extension_params()?;
            Ok(Schedule {
                steps: vec![
                    Step::Fold(FoldStep {
                        params,
                        current_w_len: 1 << 8,
                        next_w_len: 1 << 5,
                        level_bytes: 0,
                    }),
                    Step::Direct(DirectStep {
                        current_w_len: 1 << 5,
                        witness_shape: CleartextWitnessShape::FieldElements(1 << 5),
                        direct_bytes: 0,
                        params: None,
                    }),
                ],
                total_bytes: 0,
            })
        }

        fn get_params_for_prove(_layout: &OpeningClaimsLayout) -> Result<Schedule, AkitaError> {
            Self::runtime_schedule(AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(
                4, 1,
            )))
        }

        fn get_params_for_batched_commitment(
            _layout: &OpeningClaimsLayout,
        ) -> Result<LevelParams, AkitaError> {
            multi_group_extension_params()
        }
    }

    #[test]
    fn multi_group_extension_openings_reject_root_direct_rewrite() {
        let opening_batch = OpeningClaimsLayout::from_groups(vec![
            PolynomialGroupLayout::new(2, 1),
            PolynomialGroupLayout::new(4, 1),
        ])
        .expect("multi-group opening batch");
        let point = vec![
            BaseExt::from_base_slice(&[
                Base::from_u64(0),
                Base::from_u64(1),
                Base::from_u64(2),
                Base::from_u64(3),
            ]);
            4
        ];

        let err = effective_batched_schedule::<GroupedExtensionCfg>(&opening_batch, &point)
            .expect_err("multi-group extension openings must not rewrite to root direct");

        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[derive(Clone)]
    struct MultiGroupScalarDirectCfg;

    impl CommitmentConfig for MultiGroupScalarDirectCfg {
        type Field = Base;
        type ExtField = Base;

        const D: usize = 8;

        fn decomposition() -> akita_types::DecompositionParams {
            akita_types::DecompositionParams {
                log_basis: 3,
                log_commit_bound: 8,
                log_open_bound: Some(8),
            }
        }

        fn ring_challenge_config(_d: usize) -> Result<SparseChallengeConfig, AkitaError> {
            Ok(SparseChallengeConfig::pm1_only(1))
        }

        fn sis_modulus_profile() -> SisModulusProfileId {
            SisModulusProfileId::Q32Offset99
        }

        fn max_setup_matrix_size(
            _max_num_vars: usize,
            _max_num_batched_polys: usize,
        ) -> Result<SetupMatrixEnvelope, AkitaError> {
            Ok(SetupMatrixEnvelope { max_setup_len: 1 })
        }

        fn basis_range() -> (u32, u32) {
            (3, 3)
        }

        fn runtime_schedule(_key: AkitaScheduleLookupKey) -> Result<Schedule, AkitaError> {
            let params = LevelParams::params_only(
                SisModulusProfileId::Q32Offset99,
                Self::D,
                3,
                1,
                1,
                1,
                Self::ring_challenge_config(Self::D)?,
            );
            Ok(Schedule {
                steps: vec![Step::Direct(DirectStep {
                    current_w_len: 1 << 4,
                    witness_shape: CleartextWitnessShape::FieldElements(1 << 4),
                    direct_bytes: 0,
                    params: Some(params),
                })],
                total_bytes: 0,
            })
        }

        fn get_params_for_prove(_layout: &OpeningClaimsLayout) -> Result<Schedule, AkitaError> {
            Self::runtime_schedule(AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(
                4, 1,
            )))
        }

        fn get_params_for_batched_commitment(
            _layout: &OpeningClaimsLayout,
        ) -> Result<LevelParams, AkitaError> {
            Ok(LevelParams::params_only(
                SisModulusProfileId::Q32Offset99,
                Self::D,
                3,
                1,
                1,
                1,
                Self::ring_challenge_config(Self::D)?,
            ))
        }
    }

    #[test]
    fn multi_group_openings_reject_preselected_scalar_root_direct() {
        let opening_batch = OpeningClaimsLayout::from_groups(vec![
            PolynomialGroupLayout::new(2, 1),
            PolynomialGroupLayout::new(4, 1),
        ])
        .expect("multi-group opening batch");
        let point = vec![Base::from_u64(1); 4];

        let err = effective_batched_schedule::<MultiGroupScalarDirectCfg>(&opening_batch, &point)
            .expect_err("multi-group openings must reject a preselected root-direct schedule");

        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }
}
