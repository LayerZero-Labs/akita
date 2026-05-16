//! Commitment-config trait and concrete protocol configs.
//!
//! The trait [`CommitmentConfig`] is intentionally slim: presets must
//! implement every runtime hook explicitly (no thin delegating defaults).
//! Substantive helpers that encode protocol logic — `commitment_layout`,
//! `get_params_for_commitment`, `get_params_for_prove` — keep default
//! bodies because they are not policy choices and would otherwise be
//! duplicated verbatim across every config.

use akita_challenges::SparseChallengeConfig;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
#[cfg(feature = "planner")]
use akita_types::schedule_root_fold_params;
use akita_types::{
    recursive_level_decomposition_from_root, AjtaiRole, CommitmentEnvelope, DecompositionParams,
    LevelParams, SisModulusFamily,
};
use akita_types::{
    AkitaScheduleInputs, AkitaScheduleLookupKey, AkitaSchedulePlan, ClaimIncidenceSummary,
    Schedule, ScheduleProvider,
};
use std::marker::PhantomData;

pub mod proof_optimized;
pub(crate) mod schedule_policy;
pub(crate) mod sis_policy;

pub use schedule_policy::{akita_batched_root_layout, current_level_layout_with_log_basis};
use schedule_policy::{akita_root_commitment_layout, fallback_batched_root_split};

#[cfg(not(feature = "planner"))]
pub(crate) fn missing_generated_schedule(context: &str, key: AkitaScheduleLookupKey) -> AkitaError {
    AkitaError::InvalidSetup(format!(
        "{context} requires a generated schedule for key {key:?}; enable the akita-config `planner` feature to allow offline planner fallback"
    ))
}

/// Extra config bound needed only when planner-backed fallbacks are enabled.
#[cfg(feature = "planner")]
pub trait PlannerFallbackConfig: akita_planner::PlannerConfig {}

#[cfg(feature = "planner")]
impl<T: akita_planner::PlannerConfig> PlannerFallbackConfig for T {}

/// Empty marker when runtime configs are restricted to generated schedules.
#[cfg(not(feature = "planner"))]
pub trait PlannerFallbackConfig {}

#[cfg(not(feature = "planner"))]
impl<T> PlannerFallbackConfig for T {}

/// Commitment-config trait for the ring-native commitment core (§4.1–§4.2).
///
/// Concrete presets must implement every runtime hook below: the trait
/// intentionally provides no default bodies for the delegating hooks so
/// that each preset is fully explicit about which planner-backed helper
/// it uses. The substantive helpers (`commitment_layout`,
/// `get_params_for_commitment`, `get_params_for_prove`) keep defaults
/// because they encode protocol logic rather than per-config policy.
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
pub trait CommitmentConfig:
    ScheduleProvider + PlannerFallbackConfig + Clone + Send + Sync + 'static
{
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
    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig;

    /// SIS modulus family used by security-floor lookups for this config.
    fn sis_modulus_family() -> SisModulusFamily;

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

    /// Active level params for one level under an explicit basis.
    #[doc(hidden)]
    fn level_params_with_log_basis(inputs: AkitaScheduleInputs, log_basis: u32) -> LevelParams;

    /// Active root params for a concrete root layout.
    ///
    /// # Errors
    ///
    /// Returns an error if the config cannot derive a sound root parameter
    /// set for the supplied root layout.
    #[doc(hidden)]
    fn root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, AkitaError>;

    /// Root fold layout for an explicit basis.
    ///
    /// # Errors
    ///
    /// Returns an error if the root variable split underflows, overflows, or
    /// does not admit a sound root parameterization.
    #[doc(hidden)]
    fn root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, AkitaError>;

    /// Active basis for one level from public inputs.
    #[doc(hidden)]
    fn log_basis_at_level(inputs: AkitaScheduleInputs) -> u32;

    /// Inclusive `(min, max)` log-basis search range at one state.
    #[doc(hidden)]
    fn log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32);

    /// Choose the runtime commitment layout for `max_num_vars` (singleton
    /// case: one polynomial per opening point).
    ///
    /// # Errors
    ///
    /// Returns an error if `max_num_vars` does not admit a valid layout.
    fn commitment_layout(max_num_vars: usize) -> Result<LevelParams, AkitaError> {
        let key = AkitaScheduleLookupKey::singleton(max_num_vars);
        if let Some(plan) = Self::schedule_plan(key)? {
            if let Some(root_fold) = plan.fold_levels().next() {
                return Ok(root_fold.lp.clone());
            }
        }
        #[cfg(feature = "planner")]
        {
            let schedule = akita_planner::find_optimal_schedule::<Self>(key)?;
            if let Some(root_params) = schedule_root_fold_params(&schedule) {
                return Ok(root_params.clone());
            }
        }
        // Tiny-root fallback: roots that don't admit any fold step.
        akita_root_commitment_layout::<Self>(max_num_vars)
    }

    /// Choose the root parameters consumed by the commitment path.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch summary, schedule lookup, or derived
    /// layout is invalid for the requested commitment shape.
    fn get_params_for_commitment(
        num_vars: usize,
        num_polys_per_group: usize,
        max_num_points: usize,
    ) -> Result<LevelParams, AkitaError> {
        if num_polys_per_group == 0 || max_num_points == 0 {
            return Err(AkitaError::InvalidSetup(
                "commitment shape counts must be nonzero".to_string(),
            ));
        }
        let num_claims = num_polys_per_group
            .checked_mul(max_num_points)
            .ok_or_else(|| AkitaError::InvalidSetup("commitment claim count overflow".into()))?;
        if num_claims == 1 {
            return Self::commitment_layout(num_vars);
        }

        let lookup_key = AkitaScheduleLookupKey::new_with_groups(
            num_vars,
            1,
            num_polys_per_group,
            num_claims,
            max_num_points,
        );
        if let Some(plan) = Self::schedule_plan(lookup_key)? {
            if let Some(root_fold) = plan.fold_levels().next() {
                return Ok(root_fold.lp.clone());
            }
            return fallback_batched_root_split::<Self>(num_vars, num_claims);
        }

        #[cfg(feature = "planner")]
        {
            let schedule = akita_planner::find_optimal_schedule::<Self>(lookup_key)?;
            if let Some(root_params) = schedule_root_fold_params(&schedule) {
                Ok(root_params.clone())
            } else {
                fallback_batched_root_split::<Self>(num_vars, num_claims)
            }
        }

        #[cfg(not(feature = "planner"))]
        {
            let split = akita_batched_root_layout::<Self>(num_vars, num_claims)?;
            akita_types::scale_batched_root_layout(
                &split,
                num_claims,
                Self::stage1_challenge_config(Self::D).l1_norm(),
                Self::decomposition().field_bits(),
            )
        }
    }

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

        #[cfg(feature = "planner")]
        {
            akita_planner::find_optimal_schedule::<Self>(key)
        }

        #[cfg(not(feature = "planner"))]
        {
            Err(AkitaError::InvalidSetup(
                "prove schedule requires a generated schedule entry".to_string(),
            ))
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

impl<const D: usize, Cfg: CommitmentConfig> ScheduleProvider for WCommitmentConfig<D, Cfg> {
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        Cfg::schedule_table()
    }

    fn schedule_key(key: AkitaScheduleLookupKey) -> String {
        Cfg::schedule_key(key)
    }

    fn schedule_plan(key: AkitaScheduleLookupKey) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
        Cfg::schedule_plan(key)
    }
}

#[cfg(feature = "planner")]
impl<const D: usize, Cfg: CommitmentConfig> akita_planner::PlannerConfig
    for WCommitmentConfig<D, Cfg>
{
    type PlannerField = Cfg::Field;

    const PLANNER_D: usize = D;

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

    fn planner_sis_modulus_family() -> SisModulusFamily {
        <Self as CommitmentConfig>::sis_modulus_family()
    }

    fn planner_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        <Self as CommitmentConfig>::stage1_challenge_config(d)
    }

    fn planner_schedule_plan(
        key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
        <Self as ScheduleProvider>::schedule_plan(key)
    }

    fn planner_root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        <Self as CommitmentConfig>::root_level_layout_with_log_basis(inputs, log_basis)
    }

    fn planner_current_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        current_level_layout_with_log_basis::<Self>(inputs, log_basis)
    }

    fn planner_root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, AkitaError> {
        <Self as CommitmentConfig>::root_level_params_for_layout_with_log_basis(inputs, lp)
    }

    fn planner_log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        <Self as CommitmentConfig>::log_basis_search_range(inputs)
    }
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

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        Cfg::stage1_challenge_config(d)
    }

    fn sis_modulus_family() -> SisModulusFamily {
        Cfg::sis_modulus_family()
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

    fn level_params_with_log_basis(inputs: AkitaScheduleInputs, log_basis: u32) -> LevelParams {
        let params = Cfg::level_params_with_log_basis(inputs, log_basis);
        debug_assert_eq!(params.ring_dimension, D);
        params
    }

    fn root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, AkitaError> {
        Cfg::root_level_params_for_layout_with_log_basis(inputs, lp)
    }

    fn root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        Cfg::root_level_layout_with_log_basis(inputs, log_basis)
    }

    fn log_basis_at_level(inputs: AkitaScheduleInputs) -> u32 {
        Cfg::log_basis_at_level(inputs)
    }

    fn log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Cfg::log_basis_search_range(inputs)
    }

    fn commitment_layout(_max_num_vars: usize) -> Result<LevelParams, AkitaError> {
        Err(AkitaError::InvalidSetup(
            "recursive w layout requires active level params".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{Fp2, Fp32, LiftBase, NegOneNr, TowerBasisFp4, UnitNr};
    use akita_transcript::{
        append_ext_field, labels, sample_ext_challenge, Blake2bTranscript, Transcript,
    };

    type Base = Fp32<251>;
    type BaseFp2 = Fp2<Base, NegOneNr>;
    type BaseTowerBasisFp4 = TowerBasisFp4<Base, NegOneNr, UnitNr>;

    #[derive(Clone)]
    struct ExtensionRoleConfig;

    impl ScheduleProvider for ExtensionRoleConfig {
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
    }

    #[cfg(feature = "planner")]
    impl akita_planner::PlannerConfig for ExtensionRoleConfig {
        type PlannerField = Base;

        const PLANNER_D: usize = 8;

        fn planner_field_bits() -> u32 {
            8
        }

        fn planner_challenge_field_bits() -> u32 {
            8 * (<Self as CommitmentConfig>::CHAL_EXT_DEGREE as u32)
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

        fn planner_sis_modulus_family() -> SisModulusFamily {
            Self::sis_modulus_family()
        }

        fn planner_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
            Self::stage1_challenge_config(d)
        }

        fn planner_schedule_plan(
            key: AkitaScheduleLookupKey,
        ) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
            Self::schedule_plan(key)
        }

        fn planner_root_level_layout_with_log_basis(
            inputs: AkitaScheduleInputs,
            log_basis: u32,
        ) -> Result<LevelParams, AkitaError> {
            Self::root_level_layout_with_log_basis(inputs, log_basis)
        }

        fn planner_current_level_layout_with_log_basis(
            inputs: AkitaScheduleInputs,
            log_basis: u32,
        ) -> Result<LevelParams, AkitaError> {
            Ok(Self::level_params_with_log_basis(inputs, log_basis))
        }

        fn planner_root_level_params_for_layout_with_log_basis(
            inputs: AkitaScheduleInputs,
            lp: &LevelParams,
        ) -> Result<LevelParams, AkitaError> {
            Self::root_level_params_for_layout_with_log_basis(inputs, lp)
        }

        fn planner_log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
            Self::log_basis_search_range(inputs)
        }
    }

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

        fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
            assert_eq!(d, Self::D);
            SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            }
        }

        fn sis_modulus_family() -> SisModulusFamily {
            SisModulusFamily::Q32
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

        fn level_params_with_log_basis(
            _inputs: AkitaScheduleInputs,
            log_basis: u32,
        ) -> LevelParams {
            LevelParams::params_only(
                Self::sis_modulus_family(),
                Self::D,
                log_basis,
                1,
                1,
                1,
                Self::stage1_challenge_config(Self::D),
            )
        }

        fn root_level_params_for_layout_with_log_basis(
            inputs: AkitaScheduleInputs,
            lp: &LevelParams,
        ) -> Result<LevelParams, AkitaError> {
            Ok(Self::level_params_with_log_basis(inputs, lp.log_basis).with_layout(lp))
        }

        fn root_level_layout_with_log_basis(
            inputs: AkitaScheduleInputs,
            log_basis: u32,
        ) -> Result<LevelParams, AkitaError> {
            Ok(Self::level_params_with_log_basis(inputs, log_basis))
        }

        fn log_basis_at_level(_inputs: AkitaScheduleInputs) -> u32 {
            Self::decomposition().log_basis
        }

        fn log_basis_search_range(_inputs: AkitaScheduleInputs) -> (u32, u32) {
            (3, 3)
        }
    }

    #[test]
    fn config_samples_extension_challenge_role() {
        let mut t1 = Blake2bTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
        let mut t2 = Blake2bTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);

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

    #[cfg(feature = "planner")]
    #[test]
    fn planner_extension_width_matches_claim_ext_degree() {
        assert_eq!(
            <ExtensionRoleConfig as akita_planner::PlannerConfig>::planner_extension_opening_width(
            ),
            ExtensionRoleConfig::CLAIM_EXT_DEGREE
        );
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

        let mut t1 = Blake2bTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
        let mut t2 = Blake2bTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);

        ExtensionRoleConfig::append_claim_field(&mut t1, labels::ABSORB_EVALUATION_CLAIMS, &claim);
        append_ext_field::<Base, BaseFp2, _>(&mut t2, labels::ABSORB_EVALUATION_CLAIMS, &claim);

        let c1 = t1.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
        let c2 = t2.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
        assert_eq!(c1, c2);
    }
}

#[cfg(all(test, any(not(feature = "zk"), feature = "planner")))]
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
    fn current_d128_full_schedule_stays_within_audited_sis_widths() {
        assert_schedule_stays_within_audited_sis_widths::<fp128::D128Full>(8, 50);
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
    #[cfg(not(feature = "zk"))]
    fn batched_commitment_direct_fallback_scales_root_layout() {
        type Cfg = fp128::D64OneHot;

        let num_vars = 10;
        let num_claims = 4;
        let singleton = Cfg::commitment_layout(num_vars).expect("singleton layout");
        let expected = akita_types::scale_batched_root_layout(
            &singleton,
            num_claims,
            Cfg::stage1_challenge_config(Cfg::D).l1_norm(),
            Cfg::decomposition().field_bits(),
        )
        .expect("scaled layout");
        let actual =
            Cfg::get_params_for_commitment(num_vars, num_claims, 1).expect("batched layout");

        assert_eq!(actual, expected);
        assert_eq!(actual.outer_width(), singleton.outer_width() * num_claims);
        assert_eq!(
            actual.d_matrix_width(),
            singleton.d_matrix_width() * num_claims
        );
        assert!(actual.num_digits_fold >= singleton.num_digits_fold);
    }

    #[cfg(feature = "planner")]
    #[test]
    fn batched_commitment_table_miss_uses_grouped_planner_root() {
        type Cfg = fp128::D32OneHot;

        let num_vars = 30;
        let num_polys_per_group = 3;
        let max_num_points = 1;
        let lookup_key = AkitaScheduleLookupKey::new_with_groups(
            num_vars,
            1,
            num_polys_per_group,
            num_polys_per_group * max_num_points,
            max_num_points,
        );
        let schedule =
            akita_planner::find_optimal_schedule::<Cfg>(lookup_key).expect("planner schedule");
        let Some(akita_types::Step::Fold(root_step)) = schedule.steps.first() else {
            panic!("batched commitment planner schedule should start with a root fold");
        };
        let actual = Cfg::get_params_for_commitment(num_vars, num_polys_per_group, max_num_points)
            .expect("batched layout");

        assert_eq!(actual, root_step.params);
        assert_eq!(actual.outer_width(), root_step.params.outer_width());
        assert_eq!(actual.d_matrix_width(), root_step.params.d_matrix_width());
    }

    #[cfg(feature = "planner")]
    #[test]
    fn batched_commitment_shape_uses_root_schedule_params() {
        type Cfg = fp128::D32OneHot;

        // Three opening points, each bundling 2 polynomials in its commitment.
        let incidence =
            ClaimIncidenceSummary::from_point_polys(30, vec![2, 2, 2]).expect("valid incidence");
        let prove_schedule = Cfg::get_params_for_prove(&incidence).expect("prove schedule");
        let Some(akita_types::Step::Fold(root)) = prove_schedule.steps.first() else {
            panic!("batched shape should start with a root fold");
        };
        assert!(root.params.outer_width() > 0);
    }

    #[cfg(feature = "planner")]
    #[test]
    fn multipoint_prove_root_layout_is_planner_routed() {
        use super::proof_optimized::fp32;

        type Cfg = fp32::D64Full;

        let num_vars = 20;
        let num_points = 4;
        // One polynomial per point across `num_points` distinct points.
        let incidence = ClaimIncidenceSummary::from_point_polys(num_vars, vec![1; num_points])
            .expect("valid multipoint incidence");

        let prove_schedule = Cfg::get_params_for_prove(&incidence).expect("prove schedule");
        let Some(akita_types::Step::Fold(root)) = prove_schedule.steps.first() else {
            panic!("multipoint shape should start with a root fold");
        };
        // The multipoint prove root must be sized by the planner — at minimum
        // wider than the singleton root and visible to all `num_points` polys.
        let singleton = Cfg::commitment_layout(num_vars).expect("singleton layout");
        assert!(root.params.outer_width() >= singleton.outer_width());
    }

    #[cfg(feature = "planner")]
    #[test]
    fn small_field_sis_pricing_includes_psi_norm_bound() {
        use super::proof_optimized::{fp128, fp32};

        type SmallCfg = fp32::D64Full;
        assert_eq!(
            <fp128::D128Full as CommitmentConfig>::ring_subfield_embedding_norm_bound(),
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
