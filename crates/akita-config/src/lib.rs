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
use akita_types::{
    AkitaScheduleInputs, AkitaScheduleLookupKey, DecompositionParams, LevelParams, OpeningBatch,
    Schedule, SetupMatrixEnvelope, SisModulusFamily, Step,
};

pub mod generated_families;
pub mod proof_optimized;
pub mod tensor_verifier;
#[cfg(feature = "test-support")]
pub mod test_support;
mod transcript_binding;
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

    /// Absorb a claim-field element into a base-field transcript.
    fn append_claim_field<T: Transcript<Self::Field>>(
        transcript: &mut T,
        label: &[u8],
        x: &Self::ExtField,
    ) {
        append_ext_field::<Self::Field, Self::ExtField, T>(transcript, label, x);
    }

    /// Squeeze a challenge-field element from a base-field transcript.
    fn sample_challenge_field<T: Transcript<Self::Field>>(
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

    /// Build the runtime [`Schedule`] for `key`.
    ///
    /// Delegates entirely to the planner's cache-then-generate entry point
    /// [`akita_planner::get_schedule`]: the planner owns the shipped tables,
    /// so it selects the matching table from the `Cfg`-derived
    /// [`policy_of::<Self>()`][policy_of] (and the level-0 fold shape),
    /// expands the compact entry on a hit, and regenerates from scratch with
    /// the offline DP on a miss. The result is deterministic in
    /// `(policy, key)` plus this config's `stage1` / `fold_shape` hooks, so
    /// prover and verifier resolve identical schedules and the Fiat-Shamir
    /// `PlanSection` digest stays consistent. Any lookup key is supported
    /// with no reliance on a pre-shipped table.
    ///
    /// # Errors
    ///
    /// Propagates expansion / SIS-bucket failures or DP-search failures
    /// (invalid key dimensions, witness overflow). Never panics — this is
    /// verifier-reachable.
    fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<Schedule, AkitaError> {
        akita_planner::get_schedule(
            key,
            &policy_of::<Self>(),
            Self::ring_challenge_config,
            Self::fold_challenge_shape_at_level,
        )
    }

    /// Schedule consumed by the prove/verify root path.
    /// Default: expand the resolved table entry; error on miss.
    ///
    /// # Errors
    ///
    /// `InvalidSetup` if no schedule-table entry exists for `opening_batch`.
    fn get_params_for_prove(opening_batch: &OpeningBatch) -> Result<Schedule, AkitaError> {
        let key = AkitaScheduleLookupKey::new_from_opening_batch(opening_batch)?;
        Self::runtime_schedule(key)
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
        opening_batch: &OpeningBatch,
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
    use akita_field::{Fp32, NegOneNr, TowerBasisFpExt4, UnitNr};
    use akita_transcript::{
        append_ext_field, labels, sample_ext_challenge, AkitaTranscript, Transcript,
    };

    type Base = Fp32<251>;
    type BaseExt = TowerBasisFpExt4<Base, NegOneNr, UnitNr>;

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
    fn config_samples_extension_challenge() {
        let mut t1 = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
        let mut t2 = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);

        let c1 =
            SingleExtensionConfig::sample_challenge_field(&mut t1, labels::CHALLENGE_RING_SWITCH);
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

        SingleExtensionConfig::append_claim_field(
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

#[cfg(all(test, feature = "zk"))]
mod zk_generated_family_sis_audit {
    use super::sis_schedule_width_audit::assert_schedule_stays_within_audited_sis_widths;
    use super::*;

    const GENERATED_FAMILY_NV_SAMPLES: &[usize] = &[8, 16, 28, 30];

    fn audit_generated_family_sparse(
        family: &generated_families::GeneratedFamily,
        nv_samples: &[usize],
    ) {
        for key in generated_families::family_keys(family).expect("family keys") {
            if !nv_samples.contains(&key.num_vars) {
                continue;
            }
            let schedule = (family.table_backed)(key).expect("runtime schedule");
            assert_schedule_stays_within_audited_sis_widths(&schedule, key.num_vars);
        }
    }

    #[test]
    fn generated_families_stay_within_audited_sis_widths() {
        for family in generated_families::ALL_GENERATED_FAMILIES {
            audit_generated_family_sparse(family, GENERATED_FAMILY_NV_SAMPLES);
        }
    }
}

#[cfg(all(test, not(feature = "zk")))]
mod fp128_policy_tests {
    use super::proof_optimized::fp128;
    use super::sis_schedule_width_audit::assert_schedule_stays_within_audited_sis_widths;
    use super::*;

    fn assert_cfg_schedule_stays_within_audited_sis_widths<Cfg: CommitmentConfig>(
        min_num_vars: usize,
        max_num_vars: usize,
    ) {
        for num_vars in min_num_vars..=max_num_vars {
            let schedule =
                Cfg::runtime_schedule(AkitaScheduleLookupKey::singleton(num_vars)).unwrap();
            assert_schedule_stays_within_audited_sis_widths(&schedule, num_vars);
        }
    }

    #[test]
    fn current_d64_full_schedule_stays_within_audited_sis_widths() {
        assert_cfg_schedule_stays_within_audited_sis_widths::<fp128::D64Full>(8, 50);
    }

    #[test]
    fn current_d64_onehot_schedule_stays_within_audited_sis_widths() {
        assert_cfg_schedule_stays_within_audited_sis_widths::<fp128::D64OneHot>(8, 50);
    }

    #[test]
    fn current_d32_full_schedule_stays_within_audited_sis_widths() {
        assert_cfg_schedule_stays_within_audited_sis_widths::<fp128::D32Full>(8, 50);
    }

    #[test]
    fn current_d32_onehot_schedule_stays_within_audited_sis_widths() {
        assert_cfg_schedule_stays_within_audited_sis_widths::<fp128::D32OneHot>(8, 50);
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

        let opening_batch = OpeningBatch::same_point(20, 1).expect("singleton opening batch");
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
    #[cfg(not(feature = "zk"))]
    fn fp128_family_selector_uses_generated_singleton_plans() {
        let key = AkitaScheduleLookupKey::singleton(32);

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
    #[cfg(not(feature = "zk"))]
    fn fp128_family_selector_supports_batched_keys() {
        let key = AkitaScheduleLookupKey::new(30, 4, 4, 1);

        let selection = fp128::best_onehot_schedule(key)
            .expect("selector should resolve batched onehot schedules")
            .expect("selector should find a generated batched onehot schedule");

        assert!(selection.preset.is_onehot());
        assert_eq!(selection.schedule.initial_w_len(), Some(1usize << 30));
    }
}
