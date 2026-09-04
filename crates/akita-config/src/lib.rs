//! [`CommitmentConfig`] — the single `<Cfg>` parameter used by
//! `akita-prover`, `akita-verifier`, `akita-pcs`, and `akita-setup`.
//!
//! Schedule rows are supplied through an owned [`TrustedScheduleCatalog`].
//! Runtime resolution is strict: missing catalog rows reject instead of invoking
//! planner search.

use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_schedules::PlannerPolicy;
use akita_serialization::Valid;
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
#[cfg(test)]
use akita_types::{AkitaScheduleLookupKey, OpeningClaimsLayout, PolynomialGroupLayout};
use akita_types::{ChunkedWitnessCfg, DecompositionParams, SisModulusProfileId};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};

/// Define a multi-chunk companion preset that delegates every layout-affecting
/// parameter to a base `Cfg` and overrides only the multi-chunk witness config
/// and external schedule family identity.
///
/// The companion shares the base's field, dimension schedule, decomposition,
/// challenge config, and SIS family, so its `_multi_chunk` table enumerates the
/// same `(num_vars, num_polynomials)` keys as its sibling; the schedules differ
/// only because `policy_of` picks up the chunked `ChunkedWitnessCfg`.
macro_rules! impl_multi_chunk_companion {
    ($cfg:ty, $base:ty, $profile:expr, $family:literal) => {
        impl $crate::CommitmentConfig for $cfg {
            type Field = <$base as $crate::CommitmentConfig>::Field;
            type ExtField = <$base as $crate::CommitmentConfig>::ExtField;
            fn schedule_family_name() -> &'static str {
                $family
            }
            const RING_DIMENSION_SCHEDULE_MODE: akita_schedules::RingDimensionScheduleMode =
                <$base as $crate::CommitmentConfig>::RING_DIMENSION_SCHEDULE_MODE;
            const EXT_DEGREE: usize = <$base as $crate::CommitmentConfig>::EXT_DEGREE;
            fn decomposition() -> akita_types::DecompositionParams {
                <$base as $crate::CommitmentConfig>::decomposition()
            }
            fn ring_challenge_config(
                d: usize,
            ) -> Result<akita_challenges::SparseChallengeConfig, akita_error::AkitaError> {
                <$base as $crate::CommitmentConfig>::ring_challenge_config(d)
            }
            fn sis_modulus_profile() -> akita_types::SisModulusProfileId {
                <$base as $crate::CommitmentConfig>::sis_modulus_profile()
            }
            fn opening_basis_range() -> (u32, u32) {
                <$base as $crate::CommitmentConfig>::opening_basis_range()
            }
            fn inner_basis_range() -> (u32, u32) {
                <$base as $crate::CommitmentConfig>::inner_basis_range()
            }
            fn committed_source_class() -> akita_types::sis::CommittedSourceClass {
                <$base as $crate::CommitmentConfig>::committed_source_class()
            }
            fn recursive_setup_planning() -> bool {
                <$base as $crate::CommitmentConfig>::recursive_setup_planning()
            }
            fn chunked_witness_cfg() -> akita_types::ChunkedWitnessCfg {
                $profile.cfg()
            }
        }
    };
}

pub mod proof_optimized;
pub mod recursive_commitment;
#[cfg(test)]
mod schedule_artifact_tests;
mod setup_prefix_slots;
mod setup_requirements;
pub use setup_requirements::{validate_setup_capacity_metadata, SetupRequirements};
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
mod transcript_binding;
mod transcript_grinding_plan;
pub use akita_schedules::ResolvedScheduleRow;
pub use akita_schedules::RingDimensionScheduleMode;
pub use akita_schedules::{
    TrustedScheduleCatalog, MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES,
    MAX_TRUSTED_SCHEDULE_ARTIFACT_ROW_BYTES,
};
pub use proof_optimized::{
    ensure_prover_schedule_fits_setup, ensure_verifier_schedule_fits_setup,
    setup_level_params_from_schedule,
};
pub use recursive_commitment::RecursiveCommitmentConfig;
pub use transcript_binding::bind_transcript_instance_descriptor;
pub use transcript_grinding_plan::derive_transcript_grinding_plan;

/// Derive the runtime schedule policy from a preset.
///
/// Every validation input is *derived* from the `Cfg` impl, so the `Cfg` impl
/// stays the one source of truth for each preset's `(dimension schedule, decomposition,
/// sis_modulus_profile, ...)`.
pub fn policy_of<Cfg: CommitmentConfig>() -> PlannerPolicy {
    let recursive_setup_planning = Cfg::recursive_setup_planning();
    PlannerPolicy {
        cost_model: akita_schedules::PlannerCostModelId::ExactPayloadAndSetupEnvelope,
        selective_l2_response_model:
            akita_schedules::SelectiveL2ResponseModelId::TypedProtocolMomentsV1,
        selection_policy: Cfg::selection_policy(),
        recursive_split_search_policy:
            akita_schedules::RecursiveSplitSearchPolicy::BoundedBalancedExtremesV1,
        recursive_setup_search_policy: if recursive_setup_planning {
            akita_schedules::RecursiveSetupSearchPolicy::RootAndFirstChildV1
        } else {
            akita_schedules::RecursiveSetupSearchPolicy::Exhaustive
        },
        setup_field_budget: None,
        min_offloaded_witness_contraction: 3,
        ring_dimension_schedule_mode: Cfg::RING_DIMENSION_SCHEDULE_MODE,
        decomposition: Cfg::decomposition(),
        sis_modulus_profile: Cfg::sis_modulus_profile(),
        sis_security_policy: akita_types::DEFAULT_SIS_SECURITY_POLICY,
        sis_table_digest: akita_types::sis::SisTableDigest::CURRENT,
        sis_l2_table_digest: akita_types::SisL2TableDigest::CURRENT,
        claim_ext_degree: Cfg::EXT_DEGREE,
        chal_ext_degree: Cfg::EXT_DEGREE,
        inner_basis_range: Cfg::inner_basis_range(),
        opening_basis_range: Cfg::opening_basis_range(),
        witness_chunk: Cfg::chunked_witness_cfg(),
        recursive_setup_planning,
    }
}

/// Decode and validate a trusted schedule artifact for one concrete config.
///
/// This is an application or preprocessing boundary. Proof parsing never
/// calls it and no proof field contains these artifact bytes.
pub fn trusted_schedule_catalog_from_bytes<Cfg: CommitmentConfig>(
    bytes: &[u8],
) -> Result<TrustedScheduleCatalog, AkitaError> {
    validate_config_policy::<Cfg>()?;
    TrustedScheduleCatalog::from_artifact_bytes(
        bytes,
        Cfg::schedule_family_name(),
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )
}

/// Check that an already validated catalog belongs to one concrete config.
pub fn validate_trusted_schedule_catalog<Cfg: CommitmentConfig>(
    catalog: &TrustedScheduleCatalog,
) -> Result<(), AkitaError> {
    validate_config_policy::<Cfg>()?;
    catalog.validate_binding(
        Cfg::schedule_family_name(),
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )
}

/// Validate a config's schedule policy and concrete extension-field tower.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when the policy domains, declared
/// extension degree, or concrete field tower disagree.
pub fn validate_config_policy<Cfg: CommitmentConfig>() -> Result<(), AkitaError> {
    Cfg::validate_sis_modulus_profile()?;
    let policy = policy_of::<Cfg>();
    akita_schedules::validate_policy(&policy)?;
    let actual_degree = <Cfg::ExtField as ExtField<Cfg::Field>>::DEGREE;
    if Cfg::EXT_DEGREE != actual_degree
        || policy.claim_ext_degree != actual_degree
        || policy.chal_ext_degree != actual_degree
    {
        return Err(AkitaError::InvalidSetup(format!(
            "config extension degree does not match concrete field tower degree {actual_degree}"
        )));
    }
    if !matches!(actual_degree, 1 | 2 | 4 | 8) {
        return Err(AkitaError::InvalidSetup(format!(
            "unsupported extension-field degree {actual_degree}"
        )));
    }
    Ok(())
}

/// Root group's source-specific policy for offline schedule generation.
pub fn honest_fold_policy_of<Cfg: CommitmentConfig>() -> akita_types::sis::HonestFoldPolicySpec {
    Cfg::committed_source_class()
        .honest_fold_policy(<Cfg::Field as CanonicalEncoding>::MODULUS_BITS)
}

/// Return the config-owned chunk size for a unit-one-hot committed source.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `Cfg` does not declare a unit-one-hot source.
pub fn unit_onehot_source_chunk_size<Cfg: CommitmentConfig>() -> Result<usize, AkitaError> {
    match Cfg::committed_source_class() {
        akita_types::sis::CommittedSourceClass::UnitOneHot { source_chunk_size } => {
            Ok(source_chunk_size)
        }
        source_class => Err(AkitaError::InvalidSetup(format!(
            "unit-one-hot fixture requires a unit-one-hot committed source, got {source_class:?}"
        ))),
    }
}

/// Commitment-config trait for the ring-native commitment core (§4.1–§4.2).
///
/// Two field roles, both extending `Field`:
/// - `Field` — base ring / SIS scalar.
/// - `ExtField` — public opening points, claimed evaluations, proof scalars,
///   and Fiat-Shamir challenges.
///
/// The degree-one specialization `Field = ExtField` is the production fp128
/// path. For fp32/fp64 presets, extension-opening reduction still aligns the
/// extension opening with base-field committed witnesses internally.
pub trait CommitmentConfig: Clone + Send + Sync + 'static {
    /// Base field used by ring commitments, setup matrices, and SIS bounds.
    type Field: CanonicalEncoding + Field;

    /// Field used by public openings and all proof scalars.
    type ExtField: ExtField<Self::Field> + MulBaseUnreduced<Self::Field> + Valid;

    /// Extension degree `K = [ExtField : Field]`.
    ///
    /// This is the `K` consumed by [`field_reduction::psi_embed`] and
    /// [`field_reduction::embed_subfield`] in `akita-types`, and the `K` that
    /// validates `SubfieldParams<D, K>`. Default body delegates to
    /// `<ExtField as ExtField<Field>>::DEGREE`; presets should not
    /// override unless they have a reason to disagree with that.
    ///
    /// [`field_reduction::psi_embed`]: akita_types::field_reduction::psi_embed
    /// [`field_reduction::embed_subfield`]: akita_types::field_reduction::embed_subfield
    const EXT_DEGREE: usize = <Self::ExtField as ExtField<Self::Field>>::DEGREE;

    /// Absorb an extension-field element into a base-field transcript.
    fn append_extension_field<T: Transcript<Self::Field>>(
        transcript: &mut T,
        label: &[u8],
        x: &Self::ExtField,
    ) {
        append_ext_field::<Self::Field, Self::ExtField, T>(transcript, label, x);
    }

    /// Squeeze an extension-field element from a base-field transcript.
    fn sample_extension_field<T: Transcript<Self::Field>>(
        transcript: &mut T,
        label: &[u8],
    ) -> Self::ExtField {
        sample_ext_challenge::<Self::Field, Self::ExtField, T>(transcript, label)
    }

    /// Uniform or bounded-adaptive ring-dimension schedule policy.
    const RING_DIMENSION_SCHEDULE_MODE: RingDimensionScheduleMode;

    /// Gadget base + coefficient bounds.
    fn decomposition() -> DecompositionParams;

    /// Short ring challenge family for ring dimension `d`.
    ///
    /// This is the short ring element `c(X)` that folds the committed witness
    /// (the weak-binding challenge). It is sampled before the stage-1 sumcheck,
    /// so it is not itself a sumcheck-stage challenge. "Short" means bounded
    /// norm, not sparse: larger protocol degrees use sparse fixed-weight families.
    ///
    /// # Errors
    ///
    /// `InvalidSetup` if `d` is not supported.
    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError>;

    /// Exact SIS modulus profile used by security-floor lookups.
    fn sis_modulus_profile() -> SisModulusProfileId;

    /// Prove that the concrete base field has exactly the modulus named by
    /// the SIS profile. Runtime callers use this before table lookup so a
    /// synthetic or miswired field cannot silently inherit a nearby profile.
    fn validate_sis_modulus_profile() -> Result<(), AkitaError> {
        let modulus = (-Self::Field::from_u64(1))
            .to_u128_checked()
            .ok_or_else(|| {
                AkitaError::InvalidSetup("SIS field modulus does not fit in u128".to_string())
            })?
            .checked_add(1)
            .ok_or_else(|| AkitaError::InvalidSetup("SIS field modulus overflow".to_string()))?;
        if Self::sis_modulus_profile().matches_modulus(modulus) {
            Ok(())
        } else {
            Err(AkitaError::InvalidSetup(format!(
                "SIS modulus profile {:?} does not match field modulus {modulus}",
                Self::sis_modulus_profile()
            )))
        }
    }

    /// Inclusive `(min, max)` B/D opening and folded-response basis range.
    #[doc(hidden)]
    fn opening_basis_range() -> (u32, u32);

    /// Inclusive `(min, max)` A/source decomposition basis range.
    #[doc(hidden)]
    fn inner_basis_range() -> (u32, u32) {
        Self::opening_basis_range()
    }

    /// Declared committed-source class: the canonical source representation.
    fn committed_source_class() -> akita_types::sis::CommittedSourceClass;

    /// This config's validated producer contract: declared class plus bound.
    fn committed_source_contract() -> Result<akita_types::sis::CommittedSourceContract, AkitaError>
    {
        akita_types::sis::CommittedSourceContract::try_new(
            Self::committed_source_class(),
            Self::decomposition(),
        )
    }

    /// Multi-chunk witness layout parameters for schedule planning and (future)
    /// prover orchestration.
    ///
    /// Default is single-chunk ([`ChunkedWitnessCfg::default`]), which leaves
    /// every schedule byte-identical to the historical layout. Distributed-prover
    /// presets override this to price the chunked witness layout.
    fn chunked_witness_cfg() -> ChunkedWitnessCfg {
        ChunkedWitnessCfg::default()
    }

    /// Whether schedule planning may emit recursive setup-contribution edges.
    ///
    /// Ordinary configs are direct-only. Config adapters that opt into recursive
    /// setup offloading override this and use a separate external family artifact.
    fn recursive_setup_planning() -> bool {
        false
    }

    /// Catalog-bound schedule selection objective.
    ///
    /// Uniform/direct presets minimize proof payload. Adaptive-dimension and
    /// recursive setup presets minimize the first remaining direct setup
    /// footprint before payload. The policy is part of catalog identity.
    fn selection_policy() -> akita_schedules::SelectionPolicyId {
        akita_schedules::SelectionPolicyId::for_policy(
            Self::recursive_setup_planning(),
            Self::RING_DIMENSION_SCHEDULE_MODE,
        )
    }

    /// Stable trusted schedule family identity for external artifact loading.
    fn schedule_family_name() -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_transcript::{
        append_ext_field, labels, sample_ext_challenge, AkitaTranscript, Transcript,
    };
    use jolt_field::{Fp32, FpExt4};

    type Base = Fp32<251>;
    type BaseExt = FpExt4<Base>;

    #[derive(Clone)]
    struct SingleExtensionConfig;

    #[derive(Clone)]
    struct WrongDeclaredExtensionDegree;

    impl CommitmentConfig for SingleExtensionConfig {
        type Field = Base;
        type ExtField = BaseExt;

        fn schedule_family_name() -> &'static str {
            "test_single_extension"
        }

        const RING_DIMENSION_SCHEDULE_MODE: RingDimensionScheduleMode =
            RingDimensionScheduleMode::UniformDimension { ring_dimension: 64 };

        fn decomposition() -> DecompositionParams {
            DecompositionParams {
                log_basis: 3,
                log_commit_bound: 8,
                log_open_bound: Some(8),
            }
        }

        fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
            if d != 64 {
                return Err(AkitaError::InvalidSetup(format!(
                    "unsupported D={d} for SingleExtensionConfig (expected 64)"
                )));
            }
            Ok(SparseChallengeConfig::pm1_only(1))
        }

        fn sis_modulus_profile() -> SisModulusProfileId {
            SisModulusProfileId::Q32Offset99
        }

        fn opening_basis_range() -> (u32, u32) {
            (3, 3)
        }

        fn committed_source_class() -> akita_types::sis::CommittedSourceClass {
            akita_types::sis::CommittedSourceClass::BalancedSignedDigit
        }
    }

    impl CommitmentConfig for WrongDeclaredExtensionDegree {
        type Field = crate::proof_optimized::fp32::Field;
        type ExtField = crate::proof_optimized::fp32::ExtensionField;

        fn schedule_family_name() -> &'static str {
            "test_wrong_declared_extension_degree"
        }

        const EXT_DEGREE: usize = 2;
        const RING_DIMENSION_SCHEDULE_MODE: RingDimensionScheduleMode =
            SingleExtensionConfig::RING_DIMENSION_SCHEDULE_MODE;

        fn decomposition() -> DecompositionParams {
            SingleExtensionConfig::decomposition()
        }

        fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
            SingleExtensionConfig::ring_challenge_config(d)
        }

        fn sis_modulus_profile() -> SisModulusProfileId {
            SingleExtensionConfig::sis_modulus_profile()
        }

        fn opening_basis_range() -> (u32, u32) {
            SingleExtensionConfig::opening_basis_range()
        }

        fn committed_source_class() -> akita_types::sis::CommittedSourceClass {
            SingleExtensionConfig::committed_source_class()
        }
    }

    #[test]
    fn config_samples_extension_challenge() {
        let mut t1 = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
        let mut t2 = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);

        let c1 =
            SingleExtensionConfig::sample_extension_field(&mut t1, labels::CHALLENGE_RING_SWITCH);
        let c2 = sample_ext_challenge::<Base, BaseExt, _>(&mut t2, labels::CHALLENGE_RING_SWITCH);
        assert_eq!(c1, c2);
    }

    #[test]
    fn ext_degree_default_matches_ext_field_degree() {
        assert_eq!(
            SingleExtensionConfig::EXT_DEGREE,
            <BaseExt as ExtField<Base>>::DEGREE
        );
        assert_eq!(SingleExtensionConfig::EXT_DEGREE, 4);
    }

    #[test]
    fn config_appends_extension_opening() {
        let opening = BaseExt::from_base_slice(&[
            Base::from_u64(9),
            Base::from_u64(10),
            Base::from_u64(11),
            Base::from_u64(12),
        ]);

        let mut t1 = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
        let mut t2 = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);

        SingleExtensionConfig::append_extension_field(
            &mut t1,
            labels::ABSORB_EVALUATION_CLAIMS,
            &opening,
        );
        append_ext_field::<Base, BaseExt, _>(&mut t2, labels::ABSORB_EVALUATION_CLAIMS, &opening);

        let c1 = t1.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
        let c2 = t2.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
        assert_eq!(c1, c2);
    }

    #[test]
    fn config_policy_rejects_a_declared_degree_that_disagrees_with_the_tower() {
        let error = validate_config_policy::<WrongDeclaredExtensionDegree>()
            .expect_err("declared extension degree must match the concrete tower");
        assert!(error
            .to_string()
            .contains("does not match concrete field tower degree 4"));
    }
}

#[cfg(test)]
mod sis_schedule_width_audit {
    use akita_types::sis::{min_secure_l2_rank, min_secure_rank, InnerCommitSecurityRoute};

    pub(super) fn assert_schedule_stays_within_audited_sis_widths(
        schedule: &akita_types::FoldSchedule,
        num_vars: usize,
    ) {
        for (level_idx, lp) in std::iter::once(&schedule.root.params)
            .chain(schedule.recursive_folds.iter().map(|step| &step.params))
            .enumerate()
        {
            let d = u32::try_from(lp.d_a()).expect("ring dimension fits in u32");

            let width = u64::try_from(lp.inner_width()).expect("inner width should fit in u64");
            let a_rank = match lp.inner().matrix.security_route() {
                InnerCommitSecurityRoute::Linf(key) => min_secure_rank(key, width),
                InnerCommitSecurityRoute::L2 { table_key, .. } => {
                    min_secure_l2_rank(table_key, width)
                }
            }
            .unwrap_or_else(|| {
                panic!(
                    "missing audited A-row SIS width for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, width={}",
                    lp.inner().digits.log_basis,
                    lp.inner_width()
                )
            });
            assert!(
                a_rank <= lp.inner().matrix.output_rank(),
                "A-row SIS audit failed for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, width={}, required_rank={a_rank}, actual_rank={}",
                lp.inner().digits.log_basis,
                lp.inner_width(),
                lp.inner().matrix.output_rank(),
            );

            let b_rank = min_secure_rank(
                lp.outer().matrix.sis_table_key(),
                u64::try_from(lp.outer_width()).expect("outer width should fit in u64"),
            )
            .unwrap_or_else(|| {
                panic!(
                    "missing audited B-row SIS width for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, width={}",
                    lp.outer().digits.log_basis,
                    lp.outer_width()
                )
            });
            assert!(
                b_rank <= lp.outer().matrix.output_rank(),
                "B-row SIS audit failed for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, width={}, required_rank={b_rank}, actual_rank={}",
                lp.outer().digits.log_basis,
                lp.outer_width(),
                lp.outer().matrix.output_rank(),
            );

            let d_rank = min_secure_rank(
                lp.open().matrix.sis_table_key(),
                u64::try_from(lp.d_matrix_width()).expect("d-matrix width should fit in u64"),
            )
            .unwrap_or_else(|| {
                panic!(
                    "missing audited D-row SIS width for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, width={}",
                    lp.open().digits.log_basis,
                    lp.d_matrix_width()
                )
            });
            assert!(
                d_rank <= lp.open().matrix.output_rank(),
                "D-row SIS audit failed for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, width={}, required_rank={d_rank}, actual_rank={}",
                lp.open().digits.log_basis,
                lp.d_matrix_width(),
                lp.open().matrix.output_rank(),
            );
        }
    }
}

#[cfg(test)]
mod fp128_policy_tests {
    use super::proof_optimized::fp128;
    use super::schedule_artifact_tests::mutated_row_admission_error;
    use super::sis_schedule_width_audit::assert_schedule_stays_within_audited_sis_widths;
    use super::*;

    fn assert_cfg_schedule_stays_within_audited_sis_widths<Cfg: CommitmentConfig>(
        num_vars_values: &[usize],
    ) {
        let catalog = crate::test_support::workspace_schedule_catalog::<Cfg>()
            .expect("workspace schedule catalog");
        let catalog_max = catalog
            .rows()
            .map(|row| row.profiles().final_group.group.num_vars())
            .max()
            .expect("nonempty workspace schedule catalog");
        assert!(
            num_vars_values.contains(&catalog_max),
            "SIS-width spot checks must include catalog maximum nv={catalog_max}"
        );
        for &num_vars in num_vars_values {
            let group = match crate::honest_fold_policy_of::<Cfg>() {
                akita_types::sis::HonestFoldPolicySpec::BalancedSignedDigit(_) => {
                    PolynomialGroupLayout::singleton(num_vars)
                }
                akita_types::sis::HonestFoldPolicySpec::UnitOneHot(_) => {
                    PolynomialGroupLayout::new(num_vars, 1)
                }
            };
            let schedule = catalog
                .resolve_key(&AkitaScheduleLookupKey::single(group))
                .unwrap()
                .schedule()
                .clone();
            assert_schedule_stays_within_audited_sis_widths(&schedule, num_vars);
        }
    }

    /// Spot-check keys aligned with `specs/archive/2026-Q2/sis-euclidean-estimator.md` plus each catalog maximum.
    const CI_DENSE_SIS_WIDTH_NUM_VARS: &[usize] = &[14, 16, 28, 30, 32];
    const CI_ONEHOT_SIS_WIDTH_NUM_VARS: &[usize] = &[14, 16, 28, 30, 44, 50];

    #[test]
    fn fp128_onehot_uses_adaptive_schedule_policy() {
        assert!(matches!(
            fp128::OneHot::RING_DIMENSION_SCHEDULE_MODE,
            RingDimensionScheduleMode::AdaptiveDimension { .. }
        ));
        assert!(matches!(
            <fp128::OneHot as CommitmentConfig>::RING_DIMENSION_SCHEDULE_MODE,
            RingDimensionScheduleMode::AdaptiveDimension {
                num_search_levels: 2,
                suffix_dimensions: &[64],
                ..
            }
        ));
    }

    #[test]
    fn fp128_dense_uses_adaptive_schedule_policy() {
        assert!(matches!(
            fp128::Dense::RING_DIMENSION_SCHEDULE_MODE,
            RingDimensionScheduleMode::AdaptiveDimension { .. }
        ));
        assert!(matches!(
            <fp128::Dense as CommitmentConfig>::RING_DIMENSION_SCHEDULE_MODE,
            RingDimensionScheduleMode::AdaptiveDimension {
                num_search_levels: 2,
                suffix_dimensions: &[64],
                ..
            }
        ));
    }

    #[test]
    fn current_dense_schedule_stays_within_audited_sis_widths() {
        assert_cfg_schedule_stays_within_audited_sis_widths::<fp128::Dense>(
            CI_DENSE_SIS_WIDTH_NUM_VARS,
        );
    }

    #[test]
    fn current_adaptive_dense_schedule_stays_within_audited_sis_widths() {
        assert_cfg_schedule_stays_within_audited_sis_widths::<fp128::Dense>(
            CI_DENSE_SIS_WIDTH_NUM_VARS,
        );
    }

    #[test]
    fn current_onehot_schedule_stays_within_audited_sis_widths() {
        assert_cfg_schedule_stays_within_audited_sis_widths::<fp128::OneHot>(
            CI_ONEHOT_SIS_WIDTH_NUM_VARS,
        );
    }

    #[test]
    fn fp128_generated_singleton_plans_resolve() {
        let dense_catalog = crate::test_support::workspace_schedule_catalog::<fp128::Dense>()
            .expect("dense schedule catalog");
        let onehot_catalog = crate::test_support::workspace_schedule_catalog::<fp128::OneHot>()
            .expect("one-hot schedule catalog");
        let dense_key = PolynomialGroupLayout::singleton(32);
        let onehot_key = PolynomialGroupLayout::new(32, 1);

        let dense = dense_catalog
            .resolve_key(&AkitaScheduleLookupKey::single(dense_key))
            .expect("adaptive dense schedule")
            .schedule()
            .clone();
        let onehot = onehot_catalog
            .resolve_key(&AkitaScheduleLookupKey::single(onehot_key))
            .expect("adaptive onehot schedule")
            .schedule()
            .clone();

        assert_eq!(dense.initial_witness_len(), 1usize << 32);
        assert_eq!(onehot.initial_witness_len(), 1usize << 32);
    }

    #[test]
    fn fp128_adaptive_onehot_supports_batched_keys() {
        let catalog = crate::test_support::workspace_schedule_catalog::<fp128::OneHot>()
            .expect("one-hot schedule catalog");
        let key = PolynomialGroupLayout::new(30, 4);

        let schedule = catalog
            .resolve_key(&AkitaScheduleLookupKey::single(key))
            .expect("adaptive batched onehot schedule")
            .schedule()
            .clone();

        assert_eq!(schedule.initial_witness_len(), 1usize << 30);
    }

    #[test]
    fn row_admission_rederives_root_and_terminal_transitions() {
        let catalog = crate::test_support::workspace_schedule_catalog::<fp128::OneHot>()
            .expect("one-hot schedule catalog");
        let opening_batch = OpeningClaimsLayout::new(14, 1).expect("opening layout");
        let row = catalog
            .resolve_key(&AkitaScheduleLookupKey::single(
                opening_batch.root_final_group_layout().expect("root group"),
            ))
            .expect("generated row");

        let root_input_error = mutated_row_admission_error::<fp128::OneHot>(row, |schedule| {
            schedule.root.input_witness_len /= 2;
        });
        assert!(!root_input_error.to_string().is_empty());
        let transition_error = mutated_row_admission_error::<fp128::OneHot>(row, |schedule| {
            let alignment = schedule.root.params.d_a().max(schedule.terminal.d_a());
            schedule.root.output_witness_len -= alignment;
            if let Some(next) = schedule.recursive_folds.first_mut() {
                next.input_witness_len -= alignment;
            } else {
                schedule.terminal.input_witness_len -= alignment;
            }
        });
        assert!(
            transition_error.to_string().contains("canonical"),
            "unexpected transition error: {transition_error}"
        );
    }

    #[test]
    fn row_admission_rederives_recursive_transitions() {
        let catalog = crate::test_support::workspace_schedule_catalog::<fp128::OneHot>()
            .expect("one-hot schedule catalog");
        let row = (14..=50)
            .find_map(|num_vars| {
                let layout = OpeningClaimsLayout::new(num_vars, 1).ok()?;
                let row = catalog
                    .resolve_key(&AkitaScheduleLookupKey::single(
                        layout.root_final_group_layout().ok()?,
                    ))
                    .ok()?;
                (!row.schedule().recursive_folds.is_empty()).then_some(row)
            })
            .expect("recursive generated row");
        let error = mutated_row_admission_error::<fp128::OneHot>(row, |schedule| {
            let alignment = schedule.recursive_folds[0].params.d_a().max(
                if schedule.recursive_folds.len() > 1 {
                    schedule.recursive_folds[1].params.d_a()
                } else {
                    schedule.terminal.d_a()
                },
            );
            schedule.recursive_folds[0].output_witness_len -= alignment;
            if schedule.recursive_folds.len() > 1 {
                schedule.recursive_folds[1].input_witness_len -= alignment;
            } else {
                schedule.terminal.input_witness_len -= alignment;
            }
        });
        assert!(
            error.to_string().contains("canonical"),
            "unexpected recursive transition error: {error}"
        );
    }

    #[test]
    fn row_admission_rejects_overpadded_setup_prefix() {
        type RecursiveOneHot = crate::RecursiveCommitmentConfig<fp128::OneHot>;
        let catalog = crate::test_support::workspace_schedule_catalog::<RecursiveOneHot>()
            .expect("recursive one-hot schedule catalog");

        let row = (14..=50)
            .find_map(|num_vars| {
                let layout = OpeningClaimsLayout::new(num_vars, 1).ok()?;
                let row = catalog
                    .resolve_key(&AkitaScheduleLookupKey::single(
                        layout.root_final_group_layout().ok()?,
                    ))
                    .ok()?;
                row.schedule()
                    .recursive_folds
                    .iter()
                    .any(|fold| fold.params.setup_prefix().is_some())
                    .then_some(row)
            })
            .expect("generated row with a setup prefix");
        let error = mutated_row_admission_error::<RecursiveOneHot>(row, |schedule| {
            let step = schedule
                .recursive_folds
                .iter_mut()
                .find(|fold| fold.params.setup_prefix().is_some())
                .expect("recursive setup-prefix fold");
            let mut prefix = *step.params.setup_prefix().expect("setup-prefix group");
            prefix.profile.group = PolynomialGroupLayout::new(
                prefix.profile.group.num_vars() + 1,
                prefix.profile.group.num_polynomials(),
            );
            step.params
                .set_setup_prefix(Some(prefix))
                .expect("valid prefix topology");
        });
        assert!(
            error.to_string().contains("setup-prefix geometry"),
            "unexpected setup-prefix error: {error}"
        );
    }

    #[test]
    fn row_admission_rejects_setup_prefix_that_omits_real_tail_rings() {
        type RecursiveOneHot = crate::RecursiveCommitmentConfig<fp128::OneHot>;
        let catalog = crate::test_support::workspace_schedule_catalog::<RecursiveOneHot>()
            .expect("recursive one-hot schedule catalog");

        let row = (14..=50)
            .find_map(|num_vars| {
                let layout = OpeningClaimsLayout::new(num_vars, 1).ok()?;
                let row = catalog
                    .resolve_key(&AkitaScheduleLookupKey::single(
                        layout.root_final_group_layout().ok()?,
                    ))
                    .ok()?;
                row.schedule()
                    .recursive_folds
                    .iter()
                    .any(|fold| fold.params.setup_prefix().is_some())
                    .then_some(row)
            })
            .expect("generated row with a setup prefix");
        let error = mutated_row_admission_error::<RecursiveOneHot>(row, |schedule| {
            let step = schedule
                .recursive_folds
                .iter_mut()
                .find(|fold| fold.params.setup_prefix().is_some())
                .expect("recursive setup-prefix fold");
            let mut prefix = *step.params.setup_prefix().expect("setup-prefix group");
            let blocks = prefix.profile.blocks;
            let omitted_tail_rings = blocks.live_ring_elements_per_claim - 1;
            prefix.profile.blocks = akita_types::BlockGeometry::new(
                omitted_tail_rings,
                blocks.positions_per_block,
                omitted_tail_rings.div_ceil(blocks.positions_per_block),
            );
            step.params
                .set_setup_prefix(Some(prefix))
                .expect("valid prefix topology");
        });
        assert!(
            error.to_string().contains("must commit all"),
            "unexpected setup-prefix error: {error}"
        );
    }
}

#[cfg(test)]
mod independent_commitment_tests {
    use super::proof_optimized::fp128;
    use super::*;

    #[test]
    fn independent_profile_comes_from_the_scalar_row() {
        let catalog = crate::test_support::workspace_schedule_catalog::<fp128::OneHot>()
            .expect("one-hot schedule catalog");
        let group = PolynomialGroupLayout::new(16, 1);
        group.validate().expect("group layout");
        let scalar_row = catalog
            .resolve_key(&AkitaScheduleLookupKey::single(group))
            .expect("generated scalar row");
        let profile = scalar_row.profiles().final_group;
        assert_eq!(profile, scalar_row.profiles().final_group);
        assert_eq!(profile.inner.matrix.ring_dimension(), 256);
        assert_eq!(profile.outer.matrix.ring_dimension(), 64);
        assert_eq!(profile.inner.digits.log_basis, 3);
        assert_eq!(profile.outer.digits.log_basis, 3);
    }
}
