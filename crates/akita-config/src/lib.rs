//! [`CommitmentConfig`] — the single `<Cfg>` parameter used by
//! `akita-prover`, `akita-verifier`, `akita-pcs`, and `akita-setup`.
//!
//! Production `get_params_for_prove` implementations resolve a schedule for
//! cataloged lookup key via [`CommitmentConfig::runtime_schedule`]. Runtime
//! resolution is strict: missing generated catalog rows reject instead of
//! invoking planner search.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_schedules::PlannerPolicy;
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
#[cfg(test)]
use akita_types::PolynomialGroupLayout;
use akita_types::{
    AkitaScheduleInputs, AkitaScheduleLookupKey, ChunkedWitnessCfg, CommitmentRingDims,
    CommittedGroupParams, DecompositionParams, FoldSchedule, OpeningClaimsLayout,
    SetupMatrixCapacity, SisModulusProfileId,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};

/// Define a multi-chunk companion preset that delegates every layout-affecting
/// parameter to a base `Cfg` and overrides only the multi-chunk witness config
/// and the generated schedule catalog.
///
/// The companion shares the base's field, ring dimension, decomposition,
/// challenge config, and SIS family, so its `_multi_chunk` table enumerates the
/// same `(num_vars, num_polynomials)` keys as its sibling; the schedules differ
/// only because `policy_of` picks up the chunked `ChunkedWitnessCfg`.
macro_rules! impl_multi_chunk_companion {
    ($cfg:ty, $base:ty, $profile:expr, $feat:literal, $table:ident) => {
        impl $crate::CommitmentConfig for $cfg {
            type Field = <$base as $crate::CommitmentConfig>::Field;
            type ExtField = <$base as $crate::CommitmentConfig>::ExtField;
            const D: usize = <$base as $crate::CommitmentConfig>::D;
            const RING_DIMENSION_CANDIDATES: &'static [akita_types::CommitmentRingDims] =
                <$base as $crate::CommitmentConfig>::RING_DIMENSION_CANDIDATES;
            const DEGREE: usize = <$base as $crate::CommitmentConfig>::DEGREE;

            fn decomposition() -> akita_types::DecompositionParams {
                <$base as $crate::CommitmentConfig>::decomposition()
            }
            fn ring_challenge_config(
                d: usize,
            ) -> Result<akita_challenges::SparseChallengeConfig, akita_error::AkitaError> {
                <$base as $crate::CommitmentConfig>::ring_challenge_config(d)
            }
            fn fold_challenge_shape_at_level(
                inputs: akita_types::AkitaScheduleInputs,
            ) -> akita_challenges::TensorChallengeShape {
                <$base as $crate::CommitmentConfig>::fold_challenge_shape_at_level(inputs)
            }
            fn selection_policy() -> akita_schedules::SelectionPolicyId {
                <$base as $crate::CommitmentConfig>::selection_policy()
            }
            fn sis_modulus_profile() -> akita_types::SisModulusProfileId {
                <$base as $crate::CommitmentConfig>::sis_modulus_profile()
            }
            fn ring_subfield_embedding_norm_bound() -> u32 {
                <$base as $crate::CommitmentConfig>::ring_subfield_embedding_norm_bound()
            }
            fn setup_matrix_capacity(
                max_num_vars: usize,
                max_num_batched_polys: usize,
            ) -> Result<akita_types::SetupMatrixCapacity, akita_error::AkitaError> {
                $crate::proof_optimized::proof_optimized_setup_matrix_capacity::<$cfg>(
                    max_num_vars,
                    max_num_batched_polys,
                )
            }
            fn basis_range() -> (u32, u32) {
                <$base as $crate::CommitmentConfig>::basis_range()
            }
            fn root_honest_fold_policy() -> akita_types::sis::HonestFoldPolicySpec {
                <$base as $crate::CommitmentConfig>::root_honest_fold_policy()
            }
            fn chunked_witness_cfg() -> akita_types::ChunkedWitnessCfg {
                $profile.cfg()
            }
            fn schedule_catalog() -> Option<akita_schedules::GeneratedScheduleTable> {
                #[cfg(feature = $feat)]
                {
                    Some(akita_schedules::$table())
                }
                #[cfg(not(feature = $feat))]
                {
                    None
                }
            }

            fn get_params_for_prove(
                layout: &akita_types::OpeningClaimsLayout,
            ) -> Result<akita_types::FoldSchedule, akita_error::AkitaError> {
                Self::runtime_schedule($crate::proof_optimized::proof_optimized_schedule_key(
                    layout,
                )?)
            }
        }
    };
}

pub mod precommitted_commitment;
pub mod proof_optimized;
pub mod recursive_commitment;
pub mod schedule_selection;
pub mod setup_prefix_slots;
pub mod tensor_verifier;
#[cfg(feature = "test-support")]
pub mod test_support;
mod transcript_binding;
use akita_error::AkitaError;
pub use akita_schedules::ResolvedScheduleRow;
pub use precommitted_commitment::{
    committed_group_params, committed_group_profile, PrecommittedCommitmentConfig,
};
pub use proof_optimized::{
    ensure_prover_schedule_fits_setup, ensure_verifier_schedule_fits_setup,
    setup_level_params_from_schedule,
};
pub use recursive_commitment::RecursiveCommitmentConfig;
pub use schedule_selection::effective_batched_schedule;
pub use setup_prefix_slots::setup_prefix_slot_ids_for_capacity;
pub use transcript_binding::bind_transcript_instance_descriptor;

/// Derive the runtime schedule policy from a preset.
///
/// Every validation input is *derived* from the `Cfg` impl, so the `Cfg` impl
/// stays the one source of truth for each preset's `(D, decomposition,
/// sis_modulus_profile, ...)`.
pub fn policy_of<Cfg: CommitmentConfig>() -> PlannerPolicy {
    let recursive_setup_planning = Cfg::recursive_setup_planning();
    PlannerPolicy {
        cost_model: akita_schedules::PlannerCostModelId::ExactPayloadAndSetupEnvelope,
        selection_policy: Cfg::selection_policy(),
        setup_field_budget: None,
        min_offloaded_witness_contraction: 3,
        uniform_ring_dimension: Cfg::D,
        setup_prefix_inner_ring_dimension: Cfg::setup_prefix_inner_ring_dimension(),
        ring_dimension_candidates: Cfg::RING_DIMENSION_CANDIDATES,
        decomposition: Cfg::decomposition(),
        sis_modulus_profile: Cfg::sis_modulus_profile(),
        sis_security_policy: akita_types::DEFAULT_SIS_SECURITY_POLICY,
        sis_table_digest: akita_types::sis::SisTableDigest::CURRENT,
        ring_subfield_norm_bound: Cfg::ring_subfield_embedding_norm_bound(),
        claim_ext_degree: Cfg::DEGREE,
        chal_ext_degree: Cfg::DEGREE,
        basis_range: Cfg::basis_range(),
        witness_chunk: Cfg::chunked_witness_cfg(),
        recursive_setup_planning,
    }
}

/// Root group's source-specific policy for offline schedule generation.
pub fn honest_fold_policy_of<Cfg: CommitmentConfig>() -> akita_types::sis::HonestFoldPolicySpec {
    Cfg::root_honest_fold_policy()
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
    type Field: Field + CanonicalEncoding + AkitaSerialize + AkitaDeserialize<Context = ()> + Valid;

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
    const DEGREE: usize = <Self::ExtField as ExtField<Self::Field>>::DEGREE;

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

    /// Ring degree used by `CyclotomicRing<F, D>`.
    const D: usize;

    /// Canonically ordered A/B/D tuples admitted by offline schedule search.
    ///
    /// Uniform presets use their setup-generation dimension for every role.
    /// Adaptive presets override this with their full audited search domain.
    const RING_DIMENSION_CANDIDATES: &'static [CommitmentRingDims] =
        &[CommitmentRingDims::uniform(Self::D)];

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

    /// Stage-1 fold-round challenge policy at one schedule level.
    ///
    /// `Flat` requests independent fold coefficients. `Tensor { .. }` enables
    /// tensor pricing; the planner independently enumerates the power-of-two
    /// low-factor width and stamps the resolved shape into the schedule. The
    /// value returned in `fold_low_len` is therefore a policy marker, not a
    /// fixed layout width. Recursive levels remain flat unless a preset opts in.
    fn fold_challenge_shape_at_level(_inputs: AkitaScheduleInputs) -> TensorChallengeShape {
        TensorChallengeShape::Flat
    }

    /// Exact SIS modulus profile used by security-floor lookups.
    fn sis_modulus_profile() -> SisModulusProfileId;

    /// Prove that the concrete base field has exactly the modulus named by
    /// the SIS profile. Runtime callers use this before table lookup so a
    /// synthetic or miswired field cannot silently inherit a nearby profile.
    fn validate_sis_modulus_profile() -> Result<(), AkitaError> {
        let modulus = (-Self::Field::from_u64(1))
            .to_u128_checked()
            .expect("canonical prime-field value fits in u128")
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

    /// Infinity-norm expansion introduced when claim-field coordinates are
    /// embedded into the ring subfield via `psi`.
    ///
    /// For the base-field path (`K=1`), `psi` is ordinary coefficient packing.
    /// For the current small-field ring-subfield embeddings (`K>1`), one input
    /// coefficient can contribute through paired ring lanes, so SIS A-role
    /// collision pricing uses a conservative factor of two.
    fn ring_subfield_embedding_norm_bound() -> u32 {
        if Self::DEGREE == 1 {
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
    fn setup_matrix_capacity(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixCapacity, AkitaError>;

    /// Planner-owned A-matrix ring dimension for setup-prefix commitments.
    ///
    /// This controls how the explicitly zero-padded setup-prefix witness is
    /// chunked for commitment. It is not public-matrix identity or a setup
    /// materialization dimension.
    fn setup_prefix_inner_ring_dimension() -> usize {
        Self::D
    }

    /// Inclusive `(min, max)` log-basis search range.
    #[doc(hidden)]
    fn basis_range() -> (u32, u32);

    /// Group-owned honest sizing rule used only during offline planning.
    fn root_honest_fold_policy() -> akita_types::sis::HonestFoldPolicySpec;

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
    /// setup offloading override this and use a separate generated catalog.
    fn recursive_setup_planning() -> bool {
        false
    }

    /// Catalog-bound schedule selection objective.
    ///
    /// Uniform/direct presets minimize proof payload. Recursive setup presets
    /// minimize the first remaining direct setup footprint before payload.
    /// Mixed-dimension catalogs may opt into the physical setup-field objective
    /// explicitly; the policy is part of catalog identity and never inferred
    /// from a ring dimension.
    fn selection_policy() -> akita_schedules::SelectionPolicyId {
        if Self::recursive_setup_planning() {
            akita_schedules::SelectionPolicyId::MinFirstDirectSetupThenPayload
        } else {
            akita_schedules::SelectionPolicyId::MinEstimatedProofPayload
        }
    }

    /// Optional generated schedule catalog for this preset.
    ///
    /// Presets with generated tables override this when the matching
    /// `schedules-*` feature is enabled. The default is `None`, so runtime
    /// schedule resolution rejects catalog-backed requests.
    fn schedule_catalog() -> Option<akita_schedules::GeneratedScheduleTable> {
        None
    }

    /// Whether multi-group `commit_final_group` may run under this config adapter.
    ///
    /// Precommit adapters return `false`; multi-group final commits
    /// require the regular preset config.
    fn supports_multi_group_final_commit() -> bool {
        true
    }

    /// Build the runtime [`FoldSchedule`] for `key`.
    ///
    /// Scalar openings use `AkitaScheduleLookupKey::single(group_key)` with an
    /// empty `precommitteds` vector. Grouped roots supply frozen precommit
    /// layouts in `precommitteds`.
    ///
    /// Delegates to [`akita_schedules::resolve_group_batch_schedule`] with this
    /// preset's optional [`Self::schedule_catalog`]: validates catalog identity
    /// and expands the compact entry. A missing catalog row is unsupported.
    ///
    /// # Errors
    ///
    /// Propagates expansion / SIS-bucket failures or unsupported catalog
    /// requests. Never panics — this is verifier-reachable.
    fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<FoldSchedule, AkitaError> {
        Self::validate_sis_modulus_profile()?;
        akita_schedules::resolve_group_batch_schedule(
            &key,
            &policy_of::<Self>(),
            Self::ring_challenge_config,
            Self::fold_challenge_shape_at_level,
            Self::schedule_catalog(),
        )
    }

    /// Select the generated row accepted for exact committed profiles.
    ///
    /// This is an honest-prover operation. Verification must instead resolve
    /// the explicit public selection through [`Self::resolve_schedule_selection`].
    fn select_schedule_for_profiles(
        profiles: &akita_types::CommittedGroupBatchProfile,
    ) -> Result<akita_schedules::ResolvedScheduleRow, AkitaError> {
        Self::validate_sis_modulus_profile()?;
        profiles.validate(Self::decomposition().field_bits())?;
        akita_schedules::select_generated_schedule_row_for_profiles(
            &AkitaScheduleLookupKey {
                final_group: profiles.final_group.group,
                precommitteds: profiles.precommitteds.clone(),
            },
            profiles,
            &policy_of::<Self>(),
            Self::ring_challenge_config,
            Self::fold_challenge_shape_at_level,
            Self::schedule_catalog(),
        )
    }

    /// Resolve one explicit public selection in this config's generated catalog.
    ///
    /// This is the verifier boundary: it performs identity/digest lookup only
    /// and never reconstructs a runtime key or invokes planner search.
    fn resolve_schedule_selection(
        selection: akita_types::OpeningScheduleSelection,
    ) -> Result<akita_schedules::ResolvedScheduleRow, AkitaError> {
        Self::validate_sis_modulus_profile()?;
        akita_schedules::resolve_generated_schedule_selection(
            selection,
            &policy_of::<Self>(),
            Self::ring_challenge_config,
            Self::fold_challenge_shape_at_level,
            Self::schedule_catalog(),
        )
    }

    /// FoldSchedule consumed by the prove/verify root path.
    ///
    /// # Errors
    ///
    /// Propagates schedule-key construction, catalog expansion, or DP-search
    /// failures for `layout`.
    fn get_params_for_prove(layout: &OpeningClaimsLayout) -> Result<FoldSchedule, AkitaError>;

    /// Root commit layout the `batched_prove` flow uses for `layout`,
    /// read off the runtime schedule's root fold. Same layout per-point commits use,
    /// so they stay compatible with the batched prove root.
    ///
    /// Reading the schedule's first step (rather than re-resolving the compact
    /// entry directly) keeps this coupled to whatever
    /// [`Self::get_params_for_prove`] / [`Self::runtime_schedule`] produce,
    /// so config overrides and synthetic fixtures stay honored.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::get_params_for_prove`] and rejects malformed schedules.
    fn get_params_for_batched_commitment(
        layout: &OpeningClaimsLayout,
    ) -> Result<CommittedGroupParams, AkitaError> {
        let schedule = Self::get_params_for_prove(layout)?;
        Ok(schedule.root.params.final_group.commitment.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_transcript::{
        append_ext_field, labels, sample_ext_challenge, AkitaTranscript, Transcript,
    };
    use jolt_field::Ring;
    use jolt_field::{Fp32, FpExt4};

    type Base = Fp32<251>;
    type BaseExt = FpExt4<Base>;

    #[derive(Clone)]
    struct SingleExtensionConfig;

    impl CommitmentConfig for SingleExtensionConfig {
        type Field = Base;
        type ExtField = BaseExt;

        const D: usize = 8;

        fn decomposition() -> DecompositionParams {
            DecompositionParams {
                log_basis: 3,
                log_commit_bound: 8,
                log_open_bound: Some(8),
            }
        }

        fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
            if d != Self::D {
                return Err(AkitaError::InvalidSetup(format!(
                    "unsupported D={d} for SingleExtensionConfig (expected {})",
                    Self::D
                )));
            }
            Ok(SparseChallengeConfig::pm1_only(1))
        }

        fn sis_modulus_profile() -> SisModulusProfileId {
            SisModulusProfileId::Q32Offset99
        }

        fn setup_matrix_capacity(
            _max_num_vars: usize,
            _max_num_batched_polys: usize,
        ) -> Result<SetupMatrixCapacity, AkitaError> {
            Ok(SetupMatrixCapacity::minimum())
        }

        fn basis_range() -> (u32, u32) {
            (3, 3)
        }

        fn root_honest_fold_policy() -> akita_types::sis::HonestFoldPolicySpec {
            akita_types::sis::HonestFoldPolicySpec::BalancedSignedDigit(
                akita_types::sis::BalancedSignedDigitFoldPolicy::preserving_existing_behavior(
                    32,
                    akita_types::sis::FoldWitnessNorms::bounded(8, Self::D),
                ),
            )
        }

        fn get_params_for_prove(layout: &OpeningClaimsLayout) -> Result<FoldSchedule, AkitaError> {
            layout.check()?;
            let key = AkitaScheduleLookupKey::single(layout.root_final_group_layout()?);
            Self::runtime_schedule(key)
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
            SingleExtensionConfig::DEGREE,
            <BaseExt as ExtField<Base>>::DEGREE
        );
        assert_eq!(SingleExtensionConfig::DEGREE, 4);
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
}

#[cfg(test)]
mod sis_schedule_width_audit {
    use super::*;
    use akita_types::sis::min_secure_rank;

    pub(super) fn assert_schedule_stays_within_audited_sis_widths(
        schedule: &FoldSchedule,
        num_vars: usize,
    ) {
        for (level_idx, lp) in std::iter::once(&schedule.root.params.final_group.commitment)
            .chain(
                schedule
                    .recursive_folds
                    .iter()
                    .map(|step| &step.params.witness),
            )
            .enumerate()
        {
            let d = u32::try_from(lp.d_a()).expect("ring dimension fits in u32");

            let a_rank = min_secure_rank(
                lp.inner_commit_matrix.sis_table_key(),
                u64::try_from(lp.inner_width()).expect("inner width should fit in u64"),
            )
            .unwrap_or_else(|| {
                panic!(
                    "missing audited A-row SIS width for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, width={}",
                    lp.log_basis_inner,
                    lp.inner_width()
                )
            });
            assert!(
                a_rank <= lp.inner_commit_matrix.output_rank(),
                "A-row SIS audit failed for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, width={}, required_rank={a_rank}, actual_rank={}",
                lp.log_basis_inner,
                lp.inner_width(),
                lp.inner_commit_matrix.output_rank(),
            );

            let b_rank = min_secure_rank(
                lp.outer_commit_matrix.sis_table_key(),
                u64::try_from(lp.outer_width()).expect("outer width should fit in u64"),
            )
            .unwrap_or_else(|| {
                panic!(
                    "missing audited B-row SIS width for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, width={}",
                    lp.log_basis_outer,
                    lp.outer_width()
                )
            });
            assert!(
                b_rank <= lp.outer_commit_matrix.output_rank(),
                "B-row SIS audit failed for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, width={}, required_rank={b_rank}, actual_rank={}",
                lp.log_basis_outer,
                lp.outer_width(),
                lp.outer_commit_matrix.output_rank(),
            );

            let d_rank = min_secure_rank(
                lp.open_commit_matrix.sis_table_key(),
                u64::try_from(lp.d_matrix_width()).expect("d-matrix width should fit in u64"),
            )
            .unwrap_or_else(|| {
                panic!(
                    "missing audited D-row SIS width for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, width={}",
                    lp.log_basis_open,
                    lp.d_matrix_width()
                )
            });
            assert!(
                d_rank <= lp.open_commit_matrix.output_rank(),
                "D-row SIS audit failed for D={d}, num_vars={num_vars}, level={level_idx}, lb={}, width={}, required_rank={d_rank}, actual_rank={}",
                lp.log_basis_open,
                lp.d_matrix_width(),
                lp.open_commit_matrix.output_rank(),
            );
        }
    }
}

#[cfg(test)]
mod fp128_policy_tests {
    use super::proof_optimized::fp128;
    use super::sis_schedule_width_audit::assert_schedule_stays_within_audited_sis_widths;
    use super::*;

    fn assert_cfg_schedule_stays_within_audited_sis_widths<Cfg: CommitmentConfig>(
        num_vars_values: &[usize],
    ) {
        for &num_vars in num_vars_values {
            let schedule = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
                PolynomialGroupLayout::singleton(num_vars),
            ))
            .unwrap();
            assert_schedule_stays_within_audited_sis_widths(&schedule, num_vars);
        }
    }

    /// Spot-check keys aligned with `specs/sis-euclidean-estimator.md` plus table max.
    const CI_SIS_WIDTH_NUM_VARS: &[usize] = &[14, 16, 28, 30, 44, 50];

    #[test]
    fn current_d64_dense_schedule_stays_within_audited_sis_widths() {
        assert_cfg_schedule_stays_within_audited_sis_widths::<fp128::D64Dense>(
            CI_SIS_WIDTH_NUM_VARS,
        );
    }

    #[test]
    fn current_d64_onehot_schedule_stays_within_audited_sis_widths() {
        assert_cfg_schedule_stays_within_audited_sis_widths::<fp128::D64OneHot>(
            CI_SIS_WIDTH_NUM_VARS,
        );
    }

    #[test]
    fn small_field_sis_pricing_includes_psi_norm_bound() {
        use super::proof_optimized::{fp128, fp32};

        type SmallCfg = fp32::D128OneHot;
        assert_eq!(
            <fp128::D64Dense as CommitmentConfig>::ring_subfield_embedding_norm_bound(),
            1
        );
        assert_eq!(
            <SmallCfg as CommitmentConfig>::ring_subfield_embedding_norm_bound(),
            2
        );

        let opening_batch = OpeningClaimsLayout::new(28, 1).expect("singleton opening batch");
        let schedule =
            SmallCfg::get_params_for_prove(&opening_batch).expect("small-field schedule");
        let root_params = &schedule.root.params.final_group.commitment;
        assert!(
            root_params.inner_commit_matrix.coeff_linf_bound()
                >= root_params.outer_commit_matrix.coeff_linf_bound() * 2,
            "A-role L-infinity bound should include the psi norm bound"
        );
    }

    #[test]
    fn fp128_family_selector_uses_generated_singleton_plans() {
        let key = PolynomialGroupLayout::singleton(32);

        let dense = fp128::best_dense_schedule(key)
            .expect("selector should resolve dense schedules")
            .expect("selector should find a generated dense schedule");
        let onehot = fp128::best_onehot_schedule(key)
            .expect("selector should resolve onehot schedules")
            .expect("selector should find a generated onehot schedule");

        for selection in [&dense, &onehot] {
            assert_eq!(selection.schedule.initial_witness_len(), 1usize << 32);
        }
        assert!(!dense.preset.is_onehot());
        assert!(onehot.preset.is_onehot());
    }

    #[test]
    fn fp128_family_selector_supports_batched_keys() {
        let key = PolynomialGroupLayout::new(30, 4);

        let selection = fp128::best_onehot_schedule(key)
            .expect("selector should resolve batched onehot schedules")
            .expect("selector should find a generated batched onehot schedule");

        assert!(selection.preset.is_onehot());
        assert_eq!(selection.schedule.initial_witness_len(), 1usize << 30);
    }
}

#[cfg(test)]
mod precommit_tests {
    use super::proof_optimized::fp128;
    use super::*;

    #[test]
    fn exact_precommit_params_freeze_standalone_metadata() {
        let group = PolynomialGroupLayout::new(16, 1);
        group.validate().expect("group layout");
        let singleton =
            OpeningClaimsLayout::new(group.num_vars(), group.num_polynomials()).expect("singleton");
        let params =
            <PrecommittedCommitmentConfig<fp128::D64OneHot> as CommitmentConfig>::get_params_for_batched_commitment(
                &singleton,
            )
            .expect("precommitted group params");
        let precommitted = akita_types::CommittedGroupProfile::from_params(group, &params);
        let root_basis = fp128::D64OneHot::basis_range().0;
        assert_eq!(precommitted.log_basis_inner, root_basis);
        assert_eq!(precommitted.log_basis_outer, root_basis);
        assert_eq!(precommitted.num_positions_per_block, 256);
        assert_eq!(precommitted.num_live_blocks, 4);
        assert_ne!(precommitted.inner_commit_matrix.output_rank(), 0);
        assert_ne!(precommitted.outer_commit_matrix.output_rank(), 0);
    }

    #[test]
    fn precommit_config_rejects_prove_schedule() {
        let layout = OpeningClaimsLayout::new(2, 1).expect("opening layout");
        let err =
            <PrecommittedCommitmentConfig<fp128::D64OneHot> as CommitmentConfig>::get_params_for_prove(
                &layout,
            )
            .expect_err("precommit config must not prove");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }
}
