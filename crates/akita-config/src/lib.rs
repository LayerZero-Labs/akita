//! [`CommitmentConfig`] — the single `<Cfg>` parameter used by
//! `akita-prover`, `akita-verifier`, `akita-pcs`, and `akita-setup`.
//!
//! `get_params_for_prove` / `get_params_for_batched_commitment` resolve a
//! schedule for **any** lookup key via [`CommitmentConfig::runtime_schedule`]:
//! a schedule-table hit expands the compact entry through the planner's
//! canonical walker [`akita_planner::schedule_from_entry`]; a table miss
//! regenerates the schedule with the offline DP search
//! [`akita_planner::find_schedule`], driven by the `Cfg`-derived
//! [`policy_of`] bridge. Fallback is the default for every preset.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, MulBaseUnreduced};
use akita_planner::PlannerPolicy;
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
use akita_types::sis::{min_secure_rank, rounded_up_collision_norm_t};
use akita_types::{
    AjtaiKeyParams, AkitaScheduleInputs, AkitaScheduleLookupKey, ChunkedWitnessCfg,
    CommitmentGroupScheduleKey, DecompositionParams, LevelParams, OpeningBatchShape, Schedule,
    SetupMatrixEnvelope, SisModulusFamily, Step,
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
            const TIERED_COMMITMENT: bool = <$base as $crate::CommitmentConfig>::TIERED_COMMITMENT;

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
            fn sis_modulus_family() -> akita_types::SisModulusFamily {
                <$base as $crate::CommitmentConfig>::sis_modulus_family()
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
        }
    };
}

pub mod conservative_commitment;
pub mod generated_families;
pub mod proof_optimized;
pub mod tensor_verifier;
#[cfg(feature = "test-support")]
pub mod test_support;
mod transcript_binding;
pub use conservative_commitment::ConservativeCommitmentConfig;
pub use proof_optimized::{
    matrix_envelope_for_schedule, setup_level_params_from_runtime_schedule,
    worst_case_grouped_opening_batch_for_shape,
};
pub use transcript_binding::bind_transcript_instance_descriptor;

/// Derive the `Cfg`-free [`PlannerPolicy`] the planner DP consumes from a
/// preset.
///
/// This is the single bridge between a [`CommitmentConfig`] preset and
/// [`akita_planner::find_schedule`]: every brute-force input is *derived*
/// from the `Cfg` impl, so the `Cfg` impl stays the one source of truth for
/// each preset's `(D, decomposition, sis_family, …)`. Never hand-write a
/// `PlannerPolicy` literal per preset.
pub fn policy_of<Cfg: CommitmentConfig>() -> PlannerPolicy {
    PlannerPolicy {
        ring_dimension: Cfg::D,
        decomposition: Cfg::decomposition(),
        sis_family: Cfg::sis_modulus_family(),
        ring_subfield_norm_bound: Cfg::ring_subfield_embedding_norm_bound(),
        claim_ext_degree: Cfg::EXT_DEGREE,
        chal_ext_degree: Cfg::EXT_DEGREE,
        basis_range: Cfg::basis_range(),
        onehot_chunk_size: Cfg::onehot_chunk_size(),
        tiered: Cfg::TIERED_COMMITMENT,
        witness_chunk: Cfg::chunked_witness_cfg(),
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
    type Field: CanonicalField + FieldCore;

    /// Field used by public openings and all proof scalars.
    type ExtField: ExtField<Self::Field> + MulBaseUnreduced<Self::Field>;

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

    /// Enable the second commitment tier (matrix `F`).
    ///
    /// When `true`, the planner is allowed
    /// to reuse a smaller first-tier matrix `B` across `f` witness slices and
    /// commit the partial images with a second-tier matrix `F`
    /// (`u_final = F · decompose(u_1 ‖ … ‖ u_f)`), shrinking the shared
    /// preprocessing matrix and the verifier setup-contribution scan. See
    /// `specs/tiered-commitment.md`. Threaded into the planner via
    /// [`PlannerPolicy::tiered`] (see [`policy_of`]).
    const TIERED_COMMITMENT: bool = false;

    /// Gadget base + coefficient bounds.
    fn decomposition() -> DecompositionParams;

    /// Short ring challenge family for ring dimension `d`.
    ///
    /// This is the short ring element `c(X)` that folds the committed witness
    /// (the weak-binding challenge). It is sampled before the stage-1 sumcheck,
    /// so it is not itself a sumcheck-stage challenge. "Short" means bounded
    /// norm, not sparse: the `d == 32` policy is a low-norm ball that may be
    /// dense, while larger degrees use sparse fixed-weight families.
    ///
    /// # Errors
    ///
    /// `InvalidSetup` if `d` is not supported.
    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError>;

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

    /// Optional shipped schedule catalog for this preset.
    ///
    /// Presets with generated tables override this when the matching
    /// `schedules-*` feature is enabled. The default is `None` (DP-only).
    fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
        None
    }

    /// Build the runtime [`Schedule`] for `key`.
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
    fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<Schedule, AkitaError> {
        akita_planner::resolve_group_batch_schedule(
            &key,
            &policy_of::<Self>(),
            Self::ring_challenge_config,
            Self::fold_challenge_shape_at_level,
            Self::schedule_catalog(),
        )
    }

    /// Root commit layout for a grouped final root plan.
    ///
    /// Reads the first schedule step from [`Self::runtime_schedule`], so config
    /// overrides and DP fallback stay honored.
    fn get_params_for_grouped_batched_commitment(
        key: &AkitaScheduleLookupKey,
    ) -> Result<LevelParams, AkitaError> {
        let schedule = Self::runtime_schedule(key.clone())?;
        match schedule.steps.first() {
            Some(Step::Fold(root_step)) => Ok(root_step.params.clone()),
            Some(Step::Direct(direct)) => direct.params.clone().ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "grouped root-direct schedule is missing commit params".to_string(),
                )
            }),
            None => Err(AkitaError::InvalidSetup(
                "grouped schedule has no steps".to_string(),
            )),
        }
    }

    /// Schedule used to derive the standalone conservative layout for `commit_group`.
    ///
    /// The group layout is planned at the minimum configured root basis, then
    /// its B rank is widened separately by [`Self::get_params_for_group_commit`].
    ///
    /// # Errors
    ///
    /// Returns `InvalidSetup` for dense or tiered configs, malformed group keys,
    /// or unsupported SIS buckets.
    fn group_commit_schedule(key: &CommitmentGroupScheduleKey) -> Result<Schedule, AkitaError> {
        if Self::TIERED_COMMITMENT {
            return Err(AkitaError::InvalidSetup(
                "tiered standalone commitment groups are not supported; see specs/multi-group-batching.md"
                    .to_string(),
            ));
        }
        if Self::decomposition().log_commit_bound != 1 {
            return Err(AkitaError::InvalidSetup(
                "standalone commitment groups require a one-hot config".to_string(),
            ));
        }
        if key.num_polynomials == 0 {
            return Err(AkitaError::InvalidSetup(
                "standalone commitment group key must contain at least one polynomial".to_string(),
            ));
        }
        key.validate()?;

        let (min_basis, _) = Self::basis_range();
        let mut policy = policy_of::<Self>();
        policy.basis_range = (min_basis, min_basis);
        policy.decomposition.log_basis = min_basis;
        akita_planner::find_schedule(
            *key,
            &policy,
            Self::ring_challenge_config,
            Self::fold_challenge_shape_at_level,
        )
    }

    /// Whether `commit_group`'s own min-basis schedule starts with a root fold.
    ///
    /// This is deliberately derived from [`Self::group_commit_schedule`], not the
    /// normal prove schedule, because standalone precommit may be valid for a
    /// conservative layout even when the runtime prove schedule has a different
    /// shape or is unavailable.
    fn group_commit_schedule_starts_with_fold(
        key: &CommitmentGroupScheduleKey,
    ) -> Result<bool, AkitaError> {
        Ok(matches!(
            Self::group_commit_schedule(key)?.steps.first(),
            Some(Step::Fold(_))
        ))
    }

    /// Standalone conservative layout used by `commit_group`.
    ///
    /// The group layout is planned at the minimum configured root basis, then
    /// its B rank is widened for the maximum configured root basis.
    ///
    /// # Errors
    ///
    /// Returns `InvalidSetup` for dense or tiered configs, malformed group keys,
    /// unsupported SIS buckets, or schedules without root commit params.
    fn get_params_for_group_commit(
        key: &CommitmentGroupScheduleKey,
    ) -> Result<LevelParams, AkitaError> {
        let (min_basis, max_basis) = Self::basis_range();
        let mut params =
            Self::group_commit_schedule(key).and_then(|schedule| match schedule.steps.first() {
                Some(Step::Fold(root_step)) => Ok(root_step.params.clone()),
                Some(Step::Direct(direct)) => direct.params.clone().ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "root-direct group schedule is missing commit params".to_string(),
                    )
                }),
                None => Err(AkitaError::InvalidSetup(
                    "group commit schedule has no steps".to_string(),
                )),
            })?;
        if params.log_basis != min_basis {
            return Err(AkitaError::InvalidSetup(
                "group commit planner did not use the minimum configured log_basis".to_string(),
            ));
        }

        let conservative_norm =
            rounded_up_collision_norm_t(Self::sis_modulus_family(), Self::D, max_basis)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "no conservative B-role norm for standalone commitment group".to_string(),
                    )
                })?;
        let conservative_n_b = min_secure_rank(
            Self::sis_modulus_family(),
            Self::D as u32,
            conservative_norm,
            params.b_key.col_len() as u64,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "no conservative B-role rank for standalone commitment group".to_string(),
            )
        })?;
        params.b_key = AjtaiKeyParams::try_new(
            Self::sis_modulus_family(),
            conservative_n_b,
            params.b_key.col_len(),
            conservative_norm,
            Self::D,
        )?;
        Ok(params)
    }

    /// Schedule consumed by the prove/verify root path.
    /// Default: expand the resolved table entry; error on miss.
    ///
    /// # Errors
    ///
    /// `InvalidSetup` if no schedule-table entry exists for `opening_batch`.
    fn get_params_for_prove(opening_batch: &OpeningBatchShape) -> Result<Schedule, AkitaError> {
        let key = CommitmentGroupScheduleKey::new_from_opening_batch(opening_batch)?;
        Self::runtime_schedule(AkitaScheduleLookupKey::single(key))
    }

    /// Root commit layout the `batched_prove` flow uses for `opening_batch`,
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
        opening_batch: &OpeningBatchShape,
    ) -> Result<LevelParams, AkitaError> {
        let schedule = Self::get_params_for_prove(opening_batch)?;
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
            Ok(SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            })
        }

        fn sis_modulus_family() -> SisModulusFamily {
            SisModulusFamily::Q32
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
        schedule: &Schedule,
        num_vars: usize,
    ) {
        for (level_idx, fold) in schedule.fold_steps().enumerate() {
            let lp = &fold.params;
            let d = u32::try_from(lp.ring_dimension).expect("ring dimension fits in u32");
            let family = lp.a_key.sis_family();

            let a_collision = lp.a_key.collision_l2_sq();
            let a_rank = min_secure_rank(
                family,
                d,
                a_collision,
                u64::try_from(lp.inner_width()).expect("inner width should fit in u64"),
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

            let b_collision = lp.b_key.collision_l2_sq();
            let b_rank = min_secure_rank(
                family,
                d,
                b_collision,
                u64::try_from(lp.outer_width()).expect("outer width should fit in u64"),
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

            let d_collision = lp.d_key.collision_l2_sq();
            let d_rank = min_secure_rank(
                family,
                d,
                d_collision,
                u64::try_from(lp.d_matrix_width()).expect("d-matrix width should fit in u64"),
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
                CommitmentGroupScheduleKey::singleton(num_vars),
            ))
            .unwrap();
            assert_schedule_stays_within_audited_sis_widths(&schedule, num_vars);
        }
    }

    /// Spot-check keys aligned with `specs/sis-euclidean-estimator.md` plus table max.
    const CI_SIS_WIDTH_NUM_VARS: &[usize] = &[8, 16, 28, 30, 44, 50];

    #[test]
    fn current_d64_full_schedule_stays_within_audited_sis_widths() {
        assert_cfg_schedule_stays_within_audited_sis_widths::<fp128::D64Full>(
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
    #[ignore = "full nv sweep is slow; run manually before SIS table or schedule changes"]
    fn current_d64_full_schedule_stays_within_audited_sis_widths_full_range() {
        let num_vars: Vec<usize> = (8..=50).collect();
        assert_cfg_schedule_stays_within_audited_sis_widths::<fp128::D64Full>(&num_vars);
    }

    #[test]
    #[ignore = "full nv sweep is slow; run manually before SIS table or schedule changes"]
    fn current_d64_onehot_schedule_stays_within_audited_sis_widths_full_range() {
        let num_vars: Vec<usize> = (8..=50).collect();
        assert_cfg_schedule_stays_within_audited_sis_widths::<fp128::D64OneHot>(&num_vars);
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

        let opening_batch = OpeningBatchShape::new(20, 1).expect("singleton opening batch");
        let schedule =
            SmallCfg::get_params_for_prove(&opening_batch).expect("small-field schedule");
        let Some(akita_types::Step::Fold(root)) = schedule.steps.first() else {
            panic!("small-field schedule should start with a root fold");
        };
        assert!(
            root.params.a_key.collision_l2_sq() >= root.params.b_key.collision_l2_sq() * 2,
            "A-role collision should include the psi norm bound"
        );
    }

    #[test]
    fn fp128_family_selector_uses_generated_singleton_plans() {
        let key = CommitmentGroupScheduleKey::singleton(32);

        let full = fp128::best_full_schedule(key)
            .expect("selector should resolve full schedules")
            .expect("selector should find a generated full schedule");
        let onehot = fp128::best_onehot_schedule(key)
            .expect("selector should resolve onehot schedules")
            .expect("selector should find a generated onehot schedule");

        for selection in [&full, &onehot] {
            assert_eq!(selection.schedule.initial_w_len(), Some(1usize << 32));
        }
        assert!(!full.preset.is_onehot());
        assert!(onehot.preset.is_onehot());
    }

    #[test]
    fn fp128_family_selector_supports_batched_keys() {
        let key = CommitmentGroupScheduleKey::new(30, 4);

        let selection = fp128::best_onehot_schedule(key)
            .expect("selector should resolve batched onehot schedules")
            .expect("selector should find a generated batched onehot schedule");

        assert!(selection.preset.is_onehot());
        assert_eq!(selection.schedule.initial_w_len(), Some(1usize << 30));
    }
}
