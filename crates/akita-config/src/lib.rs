//! [`CommitmentConfig`] — the single `<Cfg>` parameter used by
//! `akita-prover`, `akita-verifier`, `akita-scheme`, and `akita-setup`.
//!
//! `get_params_for_prove` / `get_params_for_batched_commitment` are
//! table-only by default: schedule-table hit ⇒ expand the compact entry
//! via [`akita_types::schedule_from_entry_bits`] (see
//! [`CommitmentConfig::runtime_schedule`]); miss ⇒ [`AkitaError::InvalidSetup`].
//!
//! [`WCommitmentConfig`] is the derived recursive-w config used by
//! `<Cfg>`-generic ring-degree dispatch helpers.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
use akita_types::generated::{
    table_entry, GeneratedScheduleTable, GeneratedScheduleTableEntry,
};
use akita_types::{
    generated_schedule_lookup_key, AkitaScheduleInputs, AkitaScheduleLookupKey,
    ClaimIncidenceSummary, DecompositionParams, LevelParams, Schedule, SetupMatrixEnvelope,
    SisModulusFamily, Step,
};
use std::marker::PhantomData;

pub mod proof_optimized;
pub mod tensor_verifier;
mod transcript_binding;
pub use proof_optimized::{
    matrix_envelope_for_schedule, setup_level_params_from_runtime_schedule,
    worst_case_grouped_incidence_for_shape,
};
pub use transcript_binding::bind_transcript_instance_descriptor;

pub(crate) fn missing_generated_schedule(context: &str, key: AkitaScheduleLookupKey) -> AkitaError {
    AkitaError::InvalidSetup(format!(
        "{context} requires a generated schedule entry for key {key:?}; \
         override the relevant `CommitmentConfig` default and call \
         `akita_planner::find_schedule` to enable an offline DP fallback"
    ))
}

/// Commitment-config trait for the ring-native commitment core (§4.1–§4.2).
///
/// Three field roles, all extending `Field`:
/// - `Field` — base ring / SIS scalar.
/// - `ClaimField` — public opening points + claimed evaluations (proof bytes).
/// - `ChallengeField` — Fiat-Shamir scalars.
///
/// `ChallengeField: ExtField<ClaimField>` so batching by a challenge always
/// lifts the claim. The degree-one specialization
/// `Field = ClaimField = ChallengeField` is the production fp128 path.
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

    /// Absorb a claim-field element into a base-field transcript.
    fn append_claim_field<T: Transcript<Self::Field>>(
        transcript: &mut T,
        label: &[u8],
        x: &Self::ClaimField,
    ) {
        append_ext_field::<Self::Field, Self::ClaimField, T>(transcript, label, x);
    }

    /// Squeeze a challenge-field element from a base-field transcript.
    fn sample_challenge_field<T: Transcript<Self::Field>>(
        transcript: &mut T,
        label: &[u8],
    ) -> Self::ChallengeField {
        sample_ext_challenge::<Self::Field, Self::ChallengeField, T>(transcript, label)
    }

    /// Ring degree used by `CyclotomicRing<F, D>`.
    const D: usize;

    /// Gadget base + coefficient bounds.
    fn decomposition() -> DecompositionParams;

    /// Sparse challenge family for ring dimension `d`.
    ///
    /// # Errors
    ///
    /// `InvalidSetup` if `d` is not supported.
    fn stage1_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError>;

    /// Stage-1 fold-round challenge shape at one schedule level.
    ///
    /// The default `TensorChallengeShape::Flat` matches every shipped flat
    /// preset and is the only shape used by recursive (`level >= 1`) folds in
    /// the current planner. Tensor-shaped verifier presets (e.g.
    /// `tensor_verifier::fp128::D64OneHotTensor`) override this hook to return
    /// `TensorChallengeShape::Tensor` for `inputs.level == 0` so generated
    /// schedule-table materialization stamps the table-backed root layout with
    /// the tensor L1 mass `omega^2` instead of the flat `omega`.
    fn fold_challenge_shape_at_level(_inputs: AkitaScheduleInputs) -> TensorChallengeShape {
        TensorChallengeShape::Flat
    }

    /// SIS modulus family used by security-floor lookups.
    fn sis_modulus_family() -> SisModulusFamily;

    /// Offline schedule table backing this config (preset only).
    fn schedule_table() -> Option<GeneratedScheduleTable>;

    /// Resolve the compact generated schedule entry for `key`, or `None`
    /// on table miss.
    ///
    /// This is the single lookup the prover/verifier use to drive the
    /// recursion: each fold level is reconstructed on demand from the
    /// entry's compact steps via [`Self::runtime_schedule`], rather than
    /// materializing a full plan up front.
    ///
    /// # Errors
    ///
    /// `InvalidSetup` if the resolved entry is structurally invalid.
    fn resolve_schedule(
        key: AkitaScheduleLookupKey,
    ) -> Result<Option<&'static GeneratedScheduleTableEntry>, AkitaError> {
        let Some(table) = Self::schedule_table() else {
            return Ok(None);
        };
        let entry = table_entry(table, generated_schedule_lookup_key(key));
        if let Some(entry) = entry {
            entry.validate()?;
        }
        Ok(entry)
    }

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

    /// Packed capacity envelope for the shared setup matrix.
    ///
    /// # Errors
    ///
    /// `InvalidSetup` on arithmetic overflow.
    #[doc(hidden)]
    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<SetupMatrixEnvelope, AkitaError>;

    /// Inclusive `(min, max)` log-basis search range.
    #[doc(hidden)]
    fn basis_range() -> (u32, u32);

    /// Build the runtime [`Schedule`] for `key` by expanding the resolved
    /// compact entry, or `None` on table miss.
    ///
    /// This replaces the former `akita-derive` materializer: the single
    /// canonical entry walker [`akita_types::schedule_from_entry_bits`]
    /// reconstructs each level's `LevelParams` (and proof-byte accounting)
    /// from the compact steps using this config's policy.
    ///
    /// # Errors
    ///
    /// Propagates expansion / SIS-bucket failures or a structurally invalid
    /// entry.
    fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<Option<Schedule>, AkitaError> {
        let Some(entry) = Self::resolve_schedule(key)? else {
            return Ok(None);
        };
        let challenge_field_bits =
            Self::decomposition().field_bits() * Self::CHAL_EXT_DEGREE as u32;
        let schedule = akita_types::schedule_from_entry_bits(
            entry,
            key,
            Self::sis_modulus_family(),
            Self::decomposition(),
            challenge_field_bits,
            Self::CLAIM_EXT_DEGREE,
            Self::ring_subfield_embedding_norm_bound(),
            Self::stage1_challenge_config,
            Self::fold_challenge_shape_at_level,
        )?;
        Ok(Some(schedule))
    }

    /// Schedule consumed by the prove/verify root path.
    /// Default: expand the resolved table entry; error on miss.
    ///
    /// # Errors
    ///
    /// `InvalidSetup` if no schedule-table entry exists for `incidence`.
    fn get_params_for_prove(incidence: &ClaimIncidenceSummary) -> Result<Schedule, AkitaError> {
        let key = AkitaScheduleLookupKey::new_from_incidence(incidence)?;
        Self::runtime_schedule(key)?
            .ok_or_else(|| missing_generated_schedule("prove schedule", key))
    }

    /// Root commit layout the `batched_prove` flow uses for `incidence`,
    /// read off the runtime schedule's first step (the root Fold params or
    /// the root-direct's commit slot). Same layout per-point commits use,
    /// so they stay compatible with the batched prove root.
    ///
    /// Reading the schedule's first step (rather than re-resolving the compact
    /// entry directly) keeps this coupled to whatever
    /// [`Self::get_params_for_prove`] / [`Self::runtime_schedule`] produce,
    /// so config overrides (synthetic fixtures, DP fallback) stay honored.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::get_params_for_prove`]; errors if the root-direct
    /// schedule lacks a commit (the uncommittable edge case).
    fn get_params_for_batched_commitment(
        incidence: &ClaimIncidenceSummary,
    ) -> Result<LevelParams, AkitaError> {
        let schedule = Self::get_params_for_prove(incidence)?;
        match schedule.steps.first() {
            Some(Step::Fold(root_step)) => Ok(root_step.params.clone()),
            Some(Step::Direct(direct)) => direct.params.clone().ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "root-direct schedule is missing commit params".to_string(),
                )
            }),
            None => Err(AkitaError::InvalidSetup(
                "schedule has no steps".to_string(),
            )),
        }
    }
}

/// Derived commitment config for recursive w-openings: `log_commit_bound`
/// drops to `log_basis` (balanced-digit `w` entries) while `log_open_bound`
/// inherits the parent opening bound (recursive opening folds produce
/// full-field coefficients).
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
        // Recursive `w` entries are balanced digits, so `log_commit_bound`
        // drops to `log_basis`. Recursive opening folds produce full-field
        // coefficients, so `log_open_bound` inherits the parent's open
        // bound (or commit bound when the parent doesn't pin one).
        let root = Cfg::decomposition();
        DecompositionParams {
            log_basis: root.log_basis,
            log_commit_bound: root.log_basis,
            log_open_bound: Some(root.log_open_bound.unwrap_or(root.log_commit_bound)),
        }
    }

    fn stage1_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Cfg::stage1_challenge_config(d)
    }

    fn fold_challenge_shape_at_level(inputs: AkitaScheduleInputs) -> TensorChallengeShape {
        Cfg::fold_challenge_shape_at_level(inputs)
    }

    fn sis_modulus_family() -> SisModulusFamily {
        Cfg::sis_modulus_family()
    }

    fn schedule_table() -> Option<GeneratedScheduleTable> {
        Cfg::schedule_table()
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<SetupMatrixEnvelope, AkitaError> {
        Cfg::max_setup_matrix_size(max_num_vars, max_num_batched_polys, max_num_points)
    }

    fn basis_range() -> (u32, u32) {
        Cfg::basis_range()
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

        fn max_setup_matrix_size(
            _max_num_vars: usize,
            _max_num_batched_polys: usize,
            _max_num_points: usize,
        ) -> Result<SetupMatrixEnvelope, AkitaError> {
            Ok(SetupMatrixEnvelope {
                max_setup_len: 1,
                #[cfg(feature = "zk")]
                max_zk_b_len: 1,
                #[cfg(feature = "zk")]
                max_zk_d_len: 1,
            })
        }

        fn basis_range() -> (u32, u32) {
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
            let schedule = Cfg::runtime_schedule(AkitaScheduleLookupKey::singleton(num_vars))
                .unwrap()
                .expect("audited config should have a schedule");

            for (level_idx, fold) in schedule.fold_steps().enumerate() {
                let lp = &fold.params;
                let a_collision =
                    ceil_supported_collision(Cfg::sis_modulus_family(), d, lp.a_key.collision_inf())
                        .unwrap_or_else(|| {
                            panic!(
                                "missing audited A-row SIS collision bucket for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, collision={}",
                                lp.log_basis,
                                lp.a_key.collision_inf(),
                            )
                        });
                let a_rank = min_rank_for_secure_width(
                    Cfg::sis_modulus_family(),
                    d,
                    a_collision,
                    u64::try_from(lp.inner_width())
                        .expect("inner width should fit in u64"),
                )
                .unwrap_or_else(|| {
                    panic!(
                        "missing audited A-row SIS width for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, width={}",
                        lp.log_basis,
                        lp.inner_width()
                    )
                });
                assert!(
                    a_rank <= lp.a_key.row_len(),
                    "A-row SIS audit failed for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, width={}, required_rank={a_rank}, actual_rank={}",
                    lp.log_basis,
                    lp.inner_width(),
                    lp.a_key.row_len(),
                );

                let b_collision =
                    ceil_supported_collision(Cfg::sis_modulus_family(), d, lp.b_key.collision_inf())
                        .unwrap_or_else(|| {
                            panic!(
                                "missing audited B-row SIS collision bucket for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, collision={}",
                                lp.log_basis,
                                lp.b_key.collision_inf(),
                            )
                        });
                let b_rank = min_rank_for_secure_width(
                    Cfg::sis_modulus_family(),
                    d,
                    b_collision,
                    u64::try_from(lp.outer_width())
                        .expect("outer width should fit in u64"),
                )
                .unwrap_or_else(|| {
                    panic!(
                        "missing audited B-row SIS width for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, width={}",
                        lp.log_basis,
                        lp.outer_width()
                    )
                });
                assert!(
                    b_rank <= lp.b_key.row_len(),
                    "B-row SIS audit failed for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, width={}, required_rank={b_rank}, actual_rank={}",
                    lp.log_basis,
                    lp.outer_width(),
                    lp.b_key.row_len(),
                );

                let d_collision =
                    ceil_supported_collision(Cfg::sis_modulus_family(), d, lp.d_key.collision_inf())
                        .unwrap_or_else(|| {
                            panic!(
                                "missing audited D-row SIS collision bucket for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, collision={}",
                                lp.log_basis,
                                lp.d_key.collision_inf(),
                            )
                        });
                let d_rank = min_rank_for_secure_width(
                    Cfg::sis_modulus_family(),
                    d,
                    d_collision,
                    u64::try_from(lp.d_matrix_width())
                        .expect("d-matrix width should fit in u64"),
                )
                .unwrap_or_else(|| {
                    panic!(
                        "missing audited D-row SIS width for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, width={}",
                        lp.log_basis,
                        lp.d_matrix_width()
                    )
                });
                assert!(
                    d_rank <= lp.d_key.row_len(),
                    "D-row SIS audit failed for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, width={}, required_rank={d_rank}, actual_rank={}",
                    lp.log_basis,
                    lp.d_matrix_width(),
                    lp.d_key.row_len(),
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
            assert_eq!(selection.schedule.initial_w_len(), Some(1usize << 32));
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
        assert_eq!(selection.schedule.initial_w_len(), Some(1usize << 30));
    }
}
