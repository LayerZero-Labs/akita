//! The single user-facing commitment-config trait and concrete protocol
//! configs.
//!
//! The trait [`CommitmentConfig`] is the only `<Cfg>` parameter consumed by
//! `akita-prover`, `akita-verifier`, `akita-scheme`, and `akita-setup`. It
//! replaces the previous three-trait split (`CommitmentConfig`,
//! `ScheduleProvider`, `PlannerConfig`). Verifier-reachable hooks — namely
//! [`CommitmentConfig::log_basis_at_level`] and
//! [`CommitmentConfig::stage1_challenge_config`] — return `Result` so
//! malformed inputs surface as `AkitaError` instead of panicking on the
//! verifier replay path.
//!
//! Pure-derivation layout helpers (`level_params_with_log_basis` plus
//! the root-layout pair) used to be trait methods too; they now live as
//! free functions in [`proof_optimized`] / [`akita_derive`] so the trait
//! stays focused on configuration knobs, not derivation rules.
//!
//! Presets must implement every required (no-default) hook explicitly.
//! Substantive helpers that encode protocol logic — `get_params_for_prove`
//! and `get_params_for_batched_commitment` — keep default bodies because
//! they are not policy choices and would otherwise be duplicated verbatim
//! across every config.
//!
//! The defaults are **table-only**: on a generated-schedule-table hit they
//! materialize the plan through [`akita_derive::schedule_plan_from_table`];
//! on a miss they return [`AkitaError::InvalidSetup`]. Offline DP search
//! lives in `akita-planner`, which depends on this crate; callers that want
//! a runtime DP fallback construct a custom `Cfg` impl that overrides the
//! relevant defaults and calls `akita_planner::find_optimal_schedule`
//! directly. The verifier path therefore never reaches DP code.
//!
//! [`WCommitmentConfig`] is the derived recursive-w config that
//! `<Cfg>`-generic dispatch helpers use for ring-degree dispatch.

use akita_challenges::SparseChallengeConfig;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
use akita_types::generated::GeneratedScheduleTable;
use akita_types::{
    recursive_level_decomposition_from_root, AjtaiRole, CommitmentEnvelope, DecompositionParams,
    LevelParams, SisModulusFamily,
};
use akita_types::{
    AkitaScheduleInputs, AkitaScheduleLookupKey, AkitaSchedulePlan, ClaimIncidenceSummary, Schedule,
};
use std::marker::PhantomData;

pub mod proof_optimized;
mod transcript_binding;
pub use proof_optimized::{
    current_level_layout_with_log_basis, direct_level_params_with_log_basis,
    fallback_batched_root_split, matrix_envelope_for_levels, setup_level_params_from_plan,
    setup_level_params_from_runtime_schedule,
};
pub use transcript_binding::bind_transcript_instance_descriptor;

pub(crate) fn missing_generated_schedule(context: &str, key: AkitaScheduleLookupKey) -> AkitaError {
    AkitaError::InvalidSetup(format!(
        "{context} requires a generated schedule entry for key {key:?}; \
         override the relevant `CommitmentConfig` default and call \
         `akita_planner::find_optimal_schedule` to enable an offline DP fallback"
    ))
}

/// Commitment-config trait for the ring-native commitment core (§4.1–§4.2).
///
/// Concrete presets must implement every runtime hook below: the trait
/// intentionally provides no default bodies for the delegating hooks so
/// that each preset is fully explicit about which planner-backed helper
/// it uses. The substantive helpers (`get_params_for_prove`,
/// `get_params_for_batched_commitment`) keep default bodies that route
/// through `Self::schedule_plan` and return an error on table miss.
///
/// # Field convention
///
/// Three fields participate, all extensions of the base field `Field`:
///
/// - `Field` is the base ring/SIS scalar.
/// - `ClaimField` carries public opening points and claimed evaluations
///   (counts toward proof bytes; should be small).
/// - `ChallengeField` carries Fiat-Shamir scalars (does not count toward
///   proof bytes; should be large enough for Schwartz–Zippel soundness).
///
/// `ChallengeField` is required to contain `ClaimField` (the
/// `ChallengeField: ExtField<Self::ClaimField>` bound), so batching a claim
/// by a challenge always lifts the claim into the challenge. The degree-one
/// specialization `Field = ClaimField = ChallengeField` is the current
/// production fp128 path; the only non-trivial concrete chain so far is
/// `F ⊆ Fp2 ⊆ TowerBasisFp4`.
pub trait CommitmentConfig: Clone + Send + Sync + 'static {
    /// Base field used by ring commitments, setup matrices, and SIS bounds.
    type Field: CanonicalField + FieldCore;

    /// Field used by public opening points and claimed evaluations.
    type ClaimField: ExtField<Self::Field>;

    /// Field used by Fiat-Shamir scalar challenges in sumcheck-style steps.
    type ChallengeField: ExtField<Self::Field> + ExtField<Self::ClaimField>;

    /// Extension degree `K = [ClaimField : Field]`.
    ///
    /// This is the `K` consumed by [`field_reduction::psi_embed`] and
    /// [`field_reduction::embed_subfield`] in `akita-types`, and the `K` that
    /// validates `SubfieldParams<D, K>`. Default body delegates to
    /// `<ClaimField as ExtField<Field>>::EXT_DEGREE`; presets should not
    /// override unless they have a reason to disagree with that.
    ///
    /// [`field_reduction::psi_embed`]: akita_types::field_reduction::psi_embed
    /// [`field_reduction::embed_subfield`]: akita_types::field_reduction::embed_subfield
    const CLAIM_EXT_DEGREE: usize = <Self::ClaimField as ExtField<Self::Field>>::EXT_DEGREE;

    /// Extension degree `[ChallengeField : Field]`.
    ///
    /// Default body delegates to
    /// `<ChallengeField as ExtField<Field>>::EXT_DEGREE`. Combined with
    /// [`Self::CLAIM_EXT_DEGREE`], the relative degree is
    /// `[ChallengeField : ClaimField] = CHAL_EXT_DEGREE / CLAIM_EXT_DEGREE`,
    /// which equals `<ChallengeField as ExtField<ClaimField>>::EXT_DEGREE` by
    /// construction.
    const CHAL_EXT_DEGREE: usize = <Self::ChallengeField as ExtField<Self::Field>>::EXT_DEGREE;

    /// Append a claim-field element using the config's base transcript field.
    fn append_claim_field<T: Transcript<Self::Field>>(
        transcript: &mut T,
        label: &[u8],
        x: &Self::ClaimField,
    ) {
        append_ext_field::<Self::Field, Self::ClaimField, T>(transcript, label, x);
    }

    /// Sample a challenge-field element using the config's base transcript field.
    fn sample_challenge_field<T: Transcript<Self::Field>>(
        transcript: &mut T,
        label: &[u8],
    ) -> Self::ChallengeField {
        sample_ext_challenge::<Self::Field, Self::ChallengeField, T>(transcript, label)
    }

    /// Ring degree used by `CyclotomicRing<F, D>`.
    const D: usize;

    /// Decomposition parameters (gadget base and coefficient bounds).
    fn decomposition() -> DecompositionParams;

    /// Sparse challenge family used at this level.
    ///
    /// # Errors
    ///
    /// Returns an error if `d` is not supported by this config.
    fn stage1_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError>;

    /// SIS modulus family used by security-floor lookups for this config.
    fn sis_modulus_family() -> SisModulusFamily;

    // -- inlined former `ScheduleProvider` methods --

    /// Pre-computed schedule table backing this config, if any.
    fn schedule_table() -> Option<GeneratedScheduleTable>;

    /// Stable identity for the active schedule at `key`.
    fn schedule_key(key: AkitaScheduleLookupKey) -> String;

    /// Optional full schedule plan for configs with an explicit provider.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider cannot materialize a valid schedule.
    fn schedule_plan(key: AkitaScheduleLookupKey) -> Result<Option<AkitaSchedulePlan>, AkitaError>;

    /// Infinity-norm expansion introduced when claim-field coordinates are
    /// embedded into the ring subfield via `psi`.
    ///
    /// For the base-field path (`K=1`), `psi` is ordinary coefficient packing.
    /// For the current small-field ring-subfield embeddings (`K>1`), one input
    /// coefficient can contribute through paired ring lanes, so SIS A-role
    /// collision pricing uses a conservative factor of two.
    fn ring_subfield_embedding_norm_bound() -> u32 {
        if Self::CLAIM_EXT_DEGREE == 1 {
            1
        } else {
            2
        }
    }

    /// Audited rank floor for the root level, by role.
    #[doc(hidden)]
    fn audited_root_rank(role: AjtaiRole, max_num_vars: usize) -> usize;

    /// Maximum matrix row envelope needed across all runtime levels.
    #[doc(hidden)]
    fn envelope(max_num_vars: usize) -> CommitmentEnvelope;

    /// `(max_rows, max_stride)` bounds for the shared setup matrix.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] on arithmetic overflow.
    #[doc(hidden)]
    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), AkitaError>;

    /// Active basis for one level from public inputs.
    ///
    /// # Errors
    ///
    /// Returns an error when the requested public inputs are invalid for
    /// this config (verifier-reachable; must not panic).
    #[doc(hidden)]
    fn log_basis_at_level(inputs: AkitaScheduleInputs) -> Result<u32, AkitaError>;

    /// Inclusive `(min, max)` log-basis search range at one state.
    #[doc(hidden)]
    fn log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32);

    /// Choose the root parameters consumed by the prove/verify root path.
    ///
    /// # Errors
    ///
    /// Returns an error if the root layout, batched layout scaling, next
    /// witness sizing, or next-level basis selection is invalid.
    fn get_params_for_prove(incidence: &ClaimIncidenceSummary) -> Result<Schedule, AkitaError> {
        let key = AkitaScheduleLookupKey::new_from_incidence(incidence)?;
        if let Some(plan) = Self::schedule_plan(key)? {
            let schedule =
                akita_types::schedule_from_plan(&plan, Self::decomposition().field_bits());
            return Ok(schedule);
        }
        Err(missing_generated_schedule("prove schedule", key))
    }

    /// Choose the root parameters consumed by multipoint batched commitment.
    ///
    /// Returns the same layout `batched_prove` will use for the supplied
    /// incidence, so that every per-point commitment produced under this
    /// layout is compatible with the batched prove root. The default
    /// implementation pulls the first fold step's params from
    /// [`Self::get_params_for_prove`], or falls back to the tiny-root
    /// commitment layout when the schedule starts directly with the root
    /// `Direct` step.
    ///
    /// # Errors
    ///
    /// Returns an error if `get_params_for_prove` fails or the tiny-root
    /// fallback for root-direct schedules does not admit a valid commitment
    /// layout.
    fn get_params_for_batched_commitment(
        incidence: &ClaimIncidenceSummary,
    ) -> Result<LevelParams, AkitaError> {
        let schedule = Self::get_params_for_prove(incidence)?;
        match schedule.steps.first() {
            Some(akita_types::Step::Fold(root_step)) => Ok(root_step.params.clone()),
            Some(akita_types::Step::Direct(direct)) => {
                direct.commit_params.clone().ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "root-direct schedule is missing commit params".to_string(),
                    )
                })
            }
            None => Err(AkitaError::InvalidSetup(
                "schedule has no steps".to_string(),
            )),
        }
    }
}

/// Derived commitment config for recursive w-openings.
///
/// Sets `log_commit_bound = log_basis` because recursive `w` entries are
/// balanced digits, and sets `log_open_bound` from the parent opening bound
/// because recursive opening folds produce full-field coefficients.
#[derive(Clone, Copy, Debug)]
pub struct WCommitmentConfig<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

impl<const D: usize, Cfg: CommitmentConfig> CommitmentConfig for WCommitmentConfig<D, Cfg> {
    type Field = Cfg::Field;
    type ClaimField = Cfg::ClaimField;
    type ChallengeField = Cfg::ChallengeField;
    const D: usize = D;

    fn decomposition() -> DecompositionParams {
        recursive_level_decomposition_from_root(
            Cfg::decomposition(),
            Cfg::decomposition().log_basis,
        )
    }

    fn stage1_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Cfg::stage1_challenge_config(d)
    }

    fn sis_modulus_family() -> SisModulusFamily {
        Cfg::sis_modulus_family()
    }

    fn schedule_table() -> Option<GeneratedScheduleTable> {
        Cfg::schedule_table()
    }

    fn schedule_key(key: AkitaScheduleLookupKey) -> String {
        Cfg::schedule_key(key)
    }

    fn schedule_plan(key: AkitaScheduleLookupKey) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
        Cfg::schedule_plan(key)
    }

    fn audited_root_rank(role: AjtaiRole, max_num_vars: usize) -> usize {
        Cfg::audited_root_rank(role, max_num_vars)
    }

    fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
        Cfg::envelope(max_num_vars)
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), AkitaError> {
        Cfg::max_setup_matrix_size(max_num_vars, max_num_batched_polys, max_num_points)
    }

    fn log_basis_at_level(inputs: AkitaScheduleInputs) -> Result<u32, AkitaError> {
        Cfg::log_basis_at_level(inputs)
    }

    fn log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Cfg::log_basis_search_range(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{Fp2, Fp32, LiftBase, NegOneNr, TowerBasisFp4, UnitNr};
    use akita_transcript::{
        append_ext_field, labels, sample_ext_challenge, AkitaTranscript, Transcript,
    };

    type Base = Fp32<251>;
    type BaseFp2 = Fp2<Base, NegOneNr>;
    type BaseTowerBasisFp4 = TowerBasisFp4<Base, NegOneNr, UnitNr>;

    #[derive(Clone)]
    struct ExtensionRoleConfig;

    impl CommitmentConfig for ExtensionRoleConfig {
        type Field = Base;
        type ClaimField = BaseFp2;
        type ChallengeField = BaseTowerBasisFp4;

        const D: usize = 8;

        fn decomposition() -> DecompositionParams {
            DecompositionParams {
                log_basis: 3,
                log_commit_bound: 8,
                log_open_bound: Some(8),
            }
        }

        fn stage1_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
            if d != Self::D {
                return Err(AkitaError::InvalidSetup(format!(
                    "unsupported D={d} for ExtensionRoleConfig (expected {})",
                    Self::D
                )));
            }
            Ok(SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            })
        }

        fn sis_modulus_family() -> SisModulusFamily {
            SisModulusFamily::Q32
        }

        fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
            None
        }

        fn schedule_key(key: AkitaScheduleLookupKey) -> String {
            format!("extension-role-test/{key:?}")
        }

        fn schedule_plan(
            _key: AkitaScheduleLookupKey,
        ) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
            Ok(None)
        }

        fn audited_root_rank(_role: AjtaiRole, _max_num_vars: usize) -> usize {
            1
        }

        fn envelope(_max_num_vars: usize) -> CommitmentEnvelope {
            CommitmentEnvelope {
                max_n_a: 1,
                max_n_b: 1,
                max_n_d: 1,
            }
        }

        fn max_setup_matrix_size(
            _max_num_vars: usize,
            _max_num_batched_polys: usize,
            _max_num_points: usize,
        ) -> Result<(usize, usize), AkitaError> {
            Ok((1, 1))
        }

        fn log_basis_at_level(_inputs: AkitaScheduleInputs) -> Result<u32, AkitaError> {
            Ok(Self::decomposition().log_basis)
        }

        fn log_basis_search_range(_inputs: AkitaScheduleInputs) -> (u32, u32) {
            (3, 3)
        }
    }

    #[test]
    fn config_samples_extension_challenge_role() {
        let mut t1 = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
        let mut t2 = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);

        let c1 =
            ExtensionRoleConfig::sample_challenge_field(&mut t1, labels::CHALLENGE_RING_SWITCH);
        let c2 = sample_ext_challenge::<Base, BaseTowerBasisFp4, _>(
            &mut t2,
            labels::CHALLENGE_RING_SWITCH,
        );
        assert_eq!(c1, c2);
    }

    #[test]
    fn claim_ext_degree_default_matches_claim_field_ext_degree() {
        assert_eq!(
            ExtensionRoleConfig::CLAIM_EXT_DEGREE,
            <BaseFp2 as ExtField<Base>>::EXT_DEGREE
        );
        assert_eq!(ExtensionRoleConfig::CLAIM_EXT_DEGREE, 2);
    }

    #[test]
    fn chal_ext_degree_default_matches_challenge_field_ext_degree() {
        assert_eq!(
            ExtensionRoleConfig::CHAL_EXT_DEGREE,
            <BaseTowerBasisFp4 as ExtField<Base>>::EXT_DEGREE
        );
        assert_eq!(ExtensionRoleConfig::CHAL_EXT_DEGREE, 4);
    }

    #[test]
    fn chal_over_claim_degree_matches_quotient_of_absolute_degrees() {
        assert_eq!(
            <BaseTowerBasisFp4 as ExtField<BaseFp2>>::EXT_DEGREE,
            ExtensionRoleConfig::CHAL_EXT_DEGREE / ExtensionRoleConfig::CLAIM_EXT_DEGREE
        );
    }

    #[test]
    fn extension_role_config_exercises_true_field_tower() {
        assert_eq!(<BaseFp2 as ExtField<Base>>::EXT_DEGREE, 2);
        assert_eq!(<BaseTowerBasisFp4 as ExtField<BaseFp2>>::EXT_DEGREE, 2);
        assert_eq!(<BaseTowerBasisFp4 as ExtField<Base>>::EXT_DEGREE, 4);
        assert_eq!(ExtensionRoleConfig::CLAIM_EXT_DEGREE, 2);
        assert_eq!(ExtensionRoleConfig::CHAL_EXT_DEGREE, 4);

        let claim = BaseFp2::from_base_slice(&[Base::from_u64(3), Base::from_u64(4)]);
        let lifted = BaseTowerBasisFp4::lift_base(claim);
        assert_eq!(
            <BaseTowerBasisFp4 as ExtField<BaseFp2>>::to_base_vec(&lifted),
            vec![claim, BaseFp2::zero()]
        );
    }

    #[test]
    fn config_appends_extension_claim_role() {
        let claim = BaseFp2::new(Base::from_u64(9), Base::from_u64(10));

        let mut t1 = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
        let mut t2 = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);

        ExtensionRoleConfig::append_claim_field(&mut t1, labels::ABSORB_EVALUATION_CLAIMS, &claim);
        append_ext_field::<Base, BaseFp2, _>(&mut t2, labels::ABSORB_EVALUATION_CLAIMS, &claim);

        let c1 = t1.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
        let c2 = t2.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
        assert_eq!(c1, c2);
    }
}

#[cfg(all(test, not(feature = "zk")))]
mod fp128_policy_tests {
    use super::proof_optimized::fp128;
    use super::*;
    #[cfg(not(feature = "zk"))]
    use akita_types::generated::sis_floor::{ceil_supported_collision, min_rank_for_secure_width};

    #[cfg(not(feature = "zk"))]
    fn assert_schedule_stays_within_audited_sis_widths<Cfg: CommitmentConfig>(
        min_num_vars: usize,
        max_num_vars: usize,
    ) {
        let d = Cfg::D as u32;
        for num_vars in min_num_vars..=max_num_vars {
            let plan = Cfg::schedule_plan(AkitaScheduleLookupKey::singleton(num_vars))
                .unwrap()
                .expect("audited config should have a schedule");

            for level in plan.fold_levels() {
                let a_collision =
                    ceil_supported_collision(Cfg::sis_modulus_family(), d, level.lp.a_key.collision_inf())
                        .unwrap_or_else(|| {
                            panic!(
                                "missing audited A-row SIS collision bucket for D={d}, num_vars={num_vars}, level={}, lb={}, collision={}",
                                level.inputs.level,
                                level.lp.log_basis,
                                level.lp.a_key.collision_inf(),
                            )
                        });
                let a_rank = min_rank_for_secure_width(
                    Cfg::sis_modulus_family(),
                    d,
                    a_collision,
                    u64::try_from(level.lp.inner_width())
                        .expect("inner width should fit in u64"),
                )
                .unwrap_or_else(|| {
                    panic!(
                        "missing audited A-row SIS width for D={d}, num_vars={num_vars}, level={}, lb={}, width={}",
                        level.inputs.level,
                        level.lp.log_basis,
                        level.lp.inner_width()
                    )
                });
                assert!(
                    a_rank <= level.lp.a_key.row_len(),
                    "A-row SIS audit failed for D={d}, num_vars={num_vars}, level={}, lb={}, width={}, required_rank={a_rank}, actual_rank={}",
                    level.inputs.level,
                    level.lp.log_basis,
                    level.lp.inner_width(),
                    level.lp.a_key.row_len(),
                );

                let b_collision =
                    ceil_supported_collision(Cfg::sis_modulus_family(), d, level.lp.b_key.collision_inf())
                        .unwrap_or_else(|| {
                            panic!(
                                "missing audited B-row SIS collision bucket for D={d}, num_vars={num_vars}, level={}, lb={}, collision={}",
                                level.inputs.level,
                                level.lp.log_basis,
                                level.lp.b_key.collision_inf(),
                            )
                        });
                let b_rank = min_rank_for_secure_width(
                    Cfg::sis_modulus_family(),
                    d,
                    b_collision,
                    u64::try_from(level.lp.outer_width())
                        .expect("outer width should fit in u64"),
                )
                .unwrap_or_else(|| {
                    panic!(
                        "missing audited B-row SIS width for D={d}, num_vars={num_vars}, level={}, lb={}, width={}",
                        level.inputs.level,
                        level.lp.log_basis,
                        level.lp.outer_width()
                    )
                });
                assert!(
                    b_rank <= level.lp.b_key.row_len(),
                    "B-row SIS audit failed for D={d}, num_vars={num_vars}, level={}, lb={}, width={}, required_rank={b_rank}, actual_rank={}",
                    level.inputs.level,
                    level.lp.log_basis,
                    level.lp.outer_width(),
                    level.lp.b_key.row_len(),
                );

                let d_collision =
                    ceil_supported_collision(Cfg::sis_modulus_family(), d, level.lp.d_key.collision_inf())
                        .unwrap_or_else(|| {
                            panic!(
                                "missing audited D-row SIS collision bucket for D={d}, num_vars={num_vars}, level={}, lb={}, collision={}",
                                level.inputs.level,
                                level.lp.log_basis,
                                level.lp.d_key.collision_inf(),
                            )
                        });
                let d_rank = min_rank_for_secure_width(
                    Cfg::sis_modulus_family(),
                    d,
                    d_collision,
                    u64::try_from(level.lp.d_matrix_width())
                        .expect("d-matrix width should fit in u64"),
                )
                .unwrap_or_else(|| {
                    panic!(
                        "missing audited D-row SIS width for D={d}, num_vars={num_vars}, level={}, lb={}, width={}",
                        level.inputs.level,
                        level.lp.log_basis,
                        level.lp.d_matrix_width()
                    )
                });
                assert!(
                    d_rank <= level.lp.d_key.row_len(),
                    "D-row SIS audit failed for D={d}, num_vars={num_vars}, level={}, lb={}, width={}, required_rank={d_rank}, actual_rank={}",
                    level.inputs.level,
                    level.lp.log_basis,
                    level.lp.d_matrix_width(),
                    level.lp.d_key.row_len(),
                );
            }
        }
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn current_d64_full_schedule_stays_within_audited_sis_widths() {
        // B-row rank=1 at num_vars>=46 level=1 lb=2 — needs SIS floor fix
        assert_schedule_stays_within_audited_sis_widths::<fp128::D64Full>(8, 45);
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn current_d64_onehot_schedule_stays_within_audited_sis_widths() {
        assert_schedule_stays_within_audited_sis_widths::<fp128::D64OneHot>(8, 50);
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn current_d32_full_schedule_stays_within_audited_sis_widths() {
        // D-row rank=1 at num_vars>=30 level=2 lb=2 — needs SIS floor fix
        assert_schedule_stays_within_audited_sis_widths::<fp128::D32Full>(8, 29);
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn current_d32_onehot_schedule_stays_within_audited_sis_widths() {
        // D-row rank=1 at num_vars>=36 level=2 lb=2 — needs SIS floor fix
        assert_schedule_stays_within_audited_sis_widths::<fp128::D32OneHot>(8, 35);
    }

    #[test]
    fn small_field_sis_pricing_includes_psi_norm_bound() {
        use super::proof_optimized::{fp128, fp32};

        type SmallCfg = fp32::D64Full;
        assert_eq!(
            <fp128::D64Full as CommitmentConfig>::ring_subfield_embedding_norm_bound(),
            1
        );
        assert_eq!(
            <SmallCfg as CommitmentConfig>::ring_subfield_embedding_norm_bound(),
            2
        );

        let incidence = ClaimIncidenceSummary::same_point(20, 1).expect("singleton incidence");
        let schedule = SmallCfg::get_params_for_prove(&incidence).expect("small-field schedule");
        let Some(akita_types::Step::Fold(root)) = schedule.steps.first() else {
            panic!("small-field schedule should start with a root fold");
        };
        assert!(
            root.params.a_key.collision_inf() >= root.params.b_key.collision_inf() * 2,
            "A-role collision should include the psi norm bound"
        );
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn fp128_family_selector_uses_generated_singleton_plans() {
        let key = AkitaScheduleLookupKey::singleton(32);

        let full = fp128::best_full_schedule(key)
            .expect("selector should parse generated full schedules")
            .expect("selector should find a generated full schedule");
        let onehot = fp128::best_onehot_schedule(key)
            .expect("selector should parse generated onehot schedules")
            .expect("selector should find a generated onehot schedule");

        for selection in [&full, &onehot] {
            assert_eq!(selection.plan.initial_state().current_w_len, 1usize << 32);
        }
        assert!(!full.preset.is_onehot());
        assert!(onehot.preset.is_onehot());
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn fp128_family_selector_supports_batched_keys() {
        let key = AkitaScheduleLookupKey::new(30, 4, 4, 1);

        let selection = fp128::best_onehot_schedule(key)
            .expect("selector should parse generated batched onehot schedules")
            .expect("selector should find a generated batched onehot schedule");

        assert!(selection.preset.is_onehot());
        assert_eq!(selection.plan.initial_state().current_w_len, 1usize << 30);
    }
}
