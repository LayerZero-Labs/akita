//! [`CommitmentConfig`] — the single `<Cfg>` parameter used by
//! `akita-prover`, `akita-verifier`, `akita-pcs`, and `akita-setup`.
//!
//! Production `get_params_for_prove` implementations resolve a schedule for
//! **any** lookup key via [`CommitmentConfig::runtime_schedule`]: a
//! schedule-table hit expands the compact entry through the planner's canonical
//! walker [`akita_planner::schedule_from_entry`]; a table miss regenerates the
//! schedule with the offline DP search [`akita_planner::find_group_batch_schedule`],
//! driven by the `Cfg`-derived [`policy_of`] bridge.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, MulBaseUnreduced,
};
use akita_planner::PlannerPolicy;
use akita_serialization::Valid;
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
#[cfg(test)]
use akita_types::PolynomialGroupLayout;
use akita_types::{
    AkitaScheduleInputs, AkitaScheduleLookupKey, ChunkedWitnessCfg, CommittedGroupParams,
    DecompositionParams, FoldSchedule, OpeningClaimsLayout, SetupMatrixEnvelope,
    SisModulusProfileId,
};

/// Define a multi-chunk companion preset that delegates every layout-affecting
/// parameter to a base `Cfg` and overrides only the multi-chunk witness config
/// and the shipped schedule catalog.
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
            const EXT_DEGREE: usize = <$base as $crate::CommitmentConfig>::EXT_DEGREE;

            fn decomposition() -> akita_types::DecompositionParams {
                <$base as $crate::CommitmentConfig>::decomposition()
            }
            fn ring_challenge_config(
                d: usize,
            ) -> Result<akita_challenges::SparseChallengeConfig, akita_field::AkitaError> {
                <$base as $crate::CommitmentConfig>::ring_challenge_config(d)
            }
            fn fold_challenge_shape_at_level(
                inputs: akita_types::AkitaScheduleInputs,
            ) -> akita_challenges::TensorChallengeShape {
                <$base as $crate::CommitmentConfig>::fold_challenge_shape_at_level(inputs)
            }
            fn sis_modulus_profile() -> akita_types::SisModulusProfileId {
                <$base as $crate::CommitmentConfig>::sis_modulus_profile()
            }
            fn ring_subfield_embedding_norm_bound() -> u32 {
                <$base as $crate::CommitmentConfig>::ring_subfield_embedding_norm_bound()
            }
            fn max_setup_matrix_size(
                max_num_vars: usize,
                max_num_batched_polys: usize,
            ) -> Result<akita_types::SetupMatrixEnvelope, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_max_setup_matrix_size::<$cfg>(
                    max_num_vars,
                    max_num_batched_polys,
                )
            }
            fn basis_range() -> (u32, u32) {
                <$base as $crate::CommitmentConfig>::basis_range()
            }
            fn onehot_chunk_size() -> usize {
                <$base as $crate::CommitmentConfig>::onehot_chunk_size()
            }
            fn root_log_basis() -> Option<u32> {
                <$base as $crate::CommitmentConfig>::root_log_basis()
            }
            fn chunked_witness_cfg() -> akita_types::ChunkedWitnessCfg {
                $profile.cfg()
            }
            fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
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
            ) -> Result<akita_types::FoldSchedule, akita_field::AkitaError> {
                Self::runtime_schedule(
                    $crate::proof_optimized::proof_optimized_schedule_key::<Self>(layout)?,
                )
            }
        }
    };
}

pub mod conservative_commitment;
pub mod generated_families;
mod matrix_envelope;
pub mod proof_optimized;
pub mod recursive_commitment;
pub mod schedule_selection;
pub mod setup_prefix_slots;
pub mod tensor_verifier;
#[cfg(feature = "test-support")]
pub mod test_support;
mod transcript_binding;
pub use conservative_commitment::ConservativeCommitmentConfig;
pub use proof_optimized::{ensure_schedule_fits_setup, setup_level_params_from_schedule};
pub use recursive_commitment::RecursiveCommitmentConfig;
pub use schedule_selection::effective_batched_schedule;
pub use setup_prefix_slots::setup_prefix_slot_ids_for_capacity;
pub use transcript_binding::bind_transcript_instance_descriptor;

/// Derive the `Cfg`-free [`PlannerPolicy`] the planner DP consumes from a
/// preset.
///
/// This is the single bridge between a [`CommitmentConfig`] preset and
/// [`akita_planner::find_group_batch_schedule`]: every brute-force input is *derived*
/// from the `Cfg` impl, so the `Cfg` impl stays the one source of truth for
/// each preset's `(D, decomposition, sis_modulus_profile, …)`. Never hand-write a
/// `PlannerPolicy` literal per preset.
/// Build the canonical schedule key for a root opening batch under `Cfg`.
///
/// Scalar layouts yield an empty `precommitteds` vector. Multi-group layouts
/// freeze each earlier group through the conservative commit adapter.
pub fn opening_schedule_key<Cfg: CommitmentConfig>(
    layout: &OpeningClaimsLayout,
) -> Result<AkitaScheduleLookupKey, AkitaError> {
    proof_optimized::proof_optimized_schedule_key::<Cfg>(layout)
}

pub fn policy_of<Cfg: CommitmentConfig>() -> PlannerPolicy {
    PlannerPolicy {
        ring_dimension: Cfg::D,
        decomposition: Cfg::decomposition(),
        sis_modulus_profile: Cfg::sis_modulus_profile(),
        sis_security_policy: akita_types::DEFAULT_SIS_SECURITY_POLICY,
        sis_table_digest: akita_types::sis::SisTableDigest::CURRENT,
        ring_subfield_norm_bound: Cfg::ring_subfield_embedding_norm_bound(),
        claim_ext_degree: Cfg::EXT_DEGREE,
        chal_ext_degree: Cfg::EXT_DEGREE,
        basis_range: Cfg::basis_range(),
        root_log_basis: root_log_basis_override().unwrap_or_else(Cfg::root_log_basis),
        onehot_chunk_size: Cfg::onehot_chunk_size(),
        witness_chunk: Cfg::chunked_witness_cfg(),
        recursive_setup_planning: Cfg::recursive_setup_planning(),
    }
}

/// Benchmark/experiment hook: parse `AKITA_ROOT_LOG_BASIS` into a
/// [`PlannerPolicy::root_log_basis`] override.
///
/// The outer `Option` distinguishes "variable unset" (returns `None`, so
/// `policy_of` falls back to the preset's `Cfg::root_log_basis()`) from an
/// explicit override:
///
/// - unset → `None` (use the preset default)
/// - `unpinned` / `none` / empty → `Some(None)` (force unpinned root)
/// - `2` → `Some(Some(2))` (pin the root fold to `log_basis = 2`)
///
/// This lets the profile harness and planner sweeps exercise every root pin
/// from a single build (the pin is otherwise a compile-time preset constant).
fn root_log_basis_override() -> Option<Option<u32>> {
    let raw = std::env::var("AKITA_ROOT_LOG_BASIS").ok()?;
    let raw = raw.trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("none") || raw.eq_ignore_ascii_case("unpinned") {
        return Some(None);
    }
    Some(raw.parse::<u32>().ok())
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
    type Field: CanonicalField + FieldCore;

    /// Field used by public openings and all proof scalars.
    type ExtField: ExtField<Self::Field> + MulBaseUnreduced<Self::Field> + Valid;

    /// Extension degree `K = [ExtField : Field]`.
    ///
    /// This is the `K` consumed by [`field_reduction::psi_embed`] and
    /// [`field_reduction::embed_subfield`] in `akita-types`, and the `K` that
    /// validates `SubfieldParams<D, K>`. Default body delegates to
    /// `<ExtField as ExtField<Field>>::EXT_DEGREE`; presets should not
    /// override unless they have a reason to disagree with that.
    ///
    /// [`field_reduction::psi_embed`]: akita_types::field_reduction::psi_embed
    /// [`field_reduction::embed_subfield`]: akita_types::field_reduction::embed_subfield
    const EXT_DEGREE: usize = <Self::ExtField as ExtField<Self::Field>>::EXT_DEGREE;

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
            .to_canonical_u128()
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
        if Self::EXT_DEGREE == 1 {
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
    ) -> Result<SetupMatrixEnvelope, AkitaError>;

    /// Inclusive `(min, max)` log-basis search range.
    #[doc(hidden)]
    fn basis_range() -> (u32, u32);

    /// Pin for the root-fold (level 0) `log_basis`.
    ///
    /// `Some(lb)` forces the offline DP to use `log_basis = lb` at the root fold
    /// (clamped into [`Self::basis_range`]); `None` leaves the root to the
    /// ordinary search. Fold levels `≥ 1` are never pinned. The default `None` is
    /// unpinned and reproduces the historical schedules byte-for-byte. Copied
    /// into [`PlannerPolicy::root_log_basis`] by `policy_of::<Cfg>()`.
    fn root_log_basis() -> Option<u32> {
        None
    }

    /// One-hot chunk size `K` of the committed witnesses under this config.
    ///
    /// Bounds the committed one-hot witness L1 mass per ring element as
    /// `nonzeros = ceil(D / K)`, which feeds the Hachi Lemma 7 weak-binding
    /// collision norm and the folded-witness digit count. The value must be a
    /// true worst case (the smallest `K`, i.e. the largest `nonzeros`, any
    /// instance under this config may commit). It is only consulted for a root
    /// level whose `log_commit_bound == 1` (one-hot commitment); dense configs
    /// always use `nonzeros = D` regardless of this hook.
    ///
    /// The default `1` is the fully generic one-hot case: it safely covers every
    /// valid chunking accepted by `OneHotPoly`, including `K < D` multi-chunk
    /// roots. A config that publicly guarantees a larger minimum chunk size may
    /// override this hook to recover tighter one-hot schedules.
    fn onehot_chunk_size() -> usize {
        1
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
    /// setup offloading override this and use a separate generated catalog.
    fn recursive_setup_planning() -> bool {
        false
    }

    /// Optional shipped schedule catalog for this preset.
    ///
    /// Presets with generated tables override this when the matching
    /// `schedules-*` feature is enabled. The default is `None` (DP-only).
    fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
        None
    }

    /// Whether multi-group `commit_final_group` may run under this config adapter.
    ///
    /// Conservative precommit adapters return `false`; multi-group final commits
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
    /// Delegates to [`akita_planner::resolve_group_batch_schedule`] with this
    /// preset's optional [`Self::schedule_catalog`]: validates catalog identity
    /// on a hit, expands the compact entry, and regenerates from scratch with
    /// the offline DP on a miss.
    ///
    /// # Errors
    ///
    /// Propagates expansion / SIS-bucket failures or DP-search failures
    /// (invalid key dimensions, witness overflow). Never panics — this is
    /// verifier-reachable.
    fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<FoldSchedule, AkitaError> {
        Self::validate_sis_modulus_profile()?;
        akita_planner::resolve_group_batch_schedule(
            &key,
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
    /// so config overrides (synthetic fixtures, DP fallback) stay honored.
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
    use akita_field::{Fp32, FpExt4};
    use akita_transcript::{
        append_ext_field, labels, sample_ext_challenge, AkitaTranscript, Transcript,
    };

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

        fn max_setup_matrix_size(
            _max_num_vars: usize,
            _max_num_batched_polys: usize,
        ) -> Result<SetupMatrixEnvelope, AkitaError> {
            Ok(SetupMatrixEnvelope { max_setup_len: 1 })
        }

        fn basis_range() -> (u32, u32) {
            (3, 3)
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
            SingleExtensionConfig::EXT_DEGREE,
            <BaseExt as ExtField<Base>>::EXT_DEGREE
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
    const CI_SIS_WIDTH_NUM_VARS: &[usize] = &[13, 16, 28, 30, 44, 50];

    /// Dense `D64Full` pins the root to `log_basis = 2`, which cannot fold
    /// `nv = 50` into a valid schedule, so its supported range (and table) is
    /// capped at `nv = 49` — spot-check its own table max instead of `50`.
    const CI_SIS_WIDTH_NUM_VARS_D64_FULL: &[usize] = &[13, 16, 28, 30, 44, 49];

    #[test]
    fn current_d64_full_schedule_stays_within_audited_sis_widths() {
        assert_cfg_schedule_stays_within_audited_sis_widths::<fp128::D64Full>(
            CI_SIS_WIDTH_NUM_VARS_D64_FULL,
        );
    }

    #[test]
    fn current_d64_onehot_schedule_stays_within_audited_sis_widths() {
        assert_cfg_schedule_stays_within_audited_sis_widths::<fp128::D64OneHot>(
            CI_SIS_WIDTH_NUM_VARS,
        );
    }

    #[test]
    #[ignore = "full nv sweep is slow; run manually before SIS table or schedule changes"]
    fn current_d64_full_schedule_stays_within_audited_sis_widths_full_range() {
        // `D64Full` root=2 supports up to `nv = 49` (see `CI_SIS_WIDTH_NUM_VARS_D64_FULL`).
        let num_vars: Vec<usize> = (13..=49).collect();
        assert_cfg_schedule_stays_within_audited_sis_widths::<fp128::D64Full>(&num_vars);
    }

    #[test]
    #[ignore = "full nv sweep is slow; run manually before SIS table or schedule changes"]
    fn current_d64_onehot_schedule_stays_within_audited_sis_widths_full_range() {
        let num_vars: Vec<usize> = (12..=50).collect();
        assert_cfg_schedule_stays_within_audited_sis_widths::<fp128::D64OneHot>(&num_vars);
    }

    #[test]
    fn small_field_sis_pricing_includes_psi_norm_bound() {
        use super::proof_optimized::{fp128, fp32};

        type SmallCfg = fp32::D128OneHot;
        assert_eq!(
            <fp128::D64Full as CommitmentConfig>::ring_subfield_embedding_norm_bound(),
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

        let full = fp128::best_full_schedule(key)
            .expect("selector should resolve full schedules")
            .expect("selector should find a generated full schedule");
        let onehot = fp128::best_onehot_schedule(key)
            .expect("selector should resolve onehot schedules")
            .expect("selector should find a generated onehot schedule");

        for selection in [&full, &onehot] {
            assert_eq!(selection.schedule.initial_witness_len(), 1usize << 32);
        }
        assert!(!full.preset.is_onehot());
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
mod conservative_precommit_tests {
    use super::proof_optimized::fp128;
    use super::*;

    #[test]
    fn conservative_precommit_params_freeze_standalone_metadata() {
        let precommitted = conservative_commitment::conservative_precommitted_group_params::<
            fp128::D64OneHot,
        >(PolynomialGroupLayout::new(16, 1))
        .expect("precommitted group params");
        assert_eq!(precommitted.group, PolynomialGroupLayout::new(16, 1));
        assert_ne!(precommitted.log_basis_outer, 0);
        assert_ne!(precommitted.n_a, 0);
        assert_ne!(precommitted.n_b, 0);
    }

    #[test]
    fn conservative_config_rejects_prove_schedule() {
        let layout = OpeningClaimsLayout::new(2, 1).expect("opening layout");
        let err =
            <ConservativeCommitmentConfig<fp128::D64OneHot> as CommitmentConfig>::get_params_for_prove(
                &layout,
            )
            .expect_err("conservative config must not prove");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }
}
