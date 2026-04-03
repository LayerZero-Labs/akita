//! Field-profile definitions for planner-backed commitment presets.

use super::config::{CommitmentConfig, CommitmentPreset, DecompositionParams};
use super::schedule::{
    generated_schedule_plan_from_table, hachi_root_schedule_artifact,
    planned_log_basis_at_level_from_schedule, planned_recursive_suffix_bytes_from_schedule,
    planned_schedule, planned_schedule_key_from_schedule, HachiRootBatchSummary,
    HachiScheduleInputs, HachiScheduleLookupKey, HachiSchedulePlan,
};
use super::schedule_tables::{
    fp128_adaptive_bounded_table, fp128_adaptive_onehot_d64_table, GeneratedScheduleTableEntry,
};
use crate::algebra::Prime128Offset275;
use crate::algebra::SparseChallengeConfig;
use crate::error::HachiError;
use crate::{CanonicalField, FieldCore};

/// Dynamic proof-family selector used by prime-profile root planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DynamicScheduleFamily {
    /// Full-field coefficient family.
    Full,
    /// Onehot/small-coefficient family.
    OneHot,
}

/// Selected root schedule for one dynamic commitment/proof context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DynamicRootScheduleSelection {
    /// Chosen root ring degree.
    pub root_d: usize,
    /// Estimated total proof bytes for the chosen root schedule.
    pub total_proof_bytes: usize,
}

fn generated_or_planned_schedule<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    table: &'static [GeneratedScheduleTableEntry],
    min_log_basis: u32,
    max_log_basis: u32,
) -> Result<HachiSchedulePlan, HachiError> {
    if let Some(plan) = generated_schedule_plan_from_table::<Cfg>(max_num_vars, table)? {
        return Ok(plan);
    }
    planned_schedule::<Cfg>(max_num_vars, min_log_basis, max_log_basis)
}

fn select_root_schedule_for_cfg<Cfg: CommitmentConfig, const D: usize>(
    key: HachiScheduleLookupKey,
) -> Result<DynamicRootScheduleSelection, HachiError> {
    let artifact = hachi_root_schedule_artifact::<Cfg, D>(key)?;
    Ok(DynamicRootScheduleSelection {
        root_d: D,
        total_proof_bytes: artifact.total_proof_bytes,
    })
}

fn select_smallest_root_schedule<Cfg32, Cfg64, Cfg128>(
    key: HachiScheduleLookupKey,
) -> Result<DynamicRootScheduleSelection, HachiError>
where
    Cfg32: CommitmentConfig,
    Cfg64: CommitmentConfig,
    Cfg128: CommitmentConfig,
{
    let mut best: Option<DynamicRootScheduleSelection> = None;
    for candidate in [
        select_root_schedule_for_cfg::<Cfg32, 32>(key),
        select_root_schedule_for_cfg::<Cfg64, 64>(key),
        select_root_schedule_for_cfg::<Cfg128, 128>(key),
    ] {
        let Ok(candidate) = candidate else {
            continue;
        };
        if best.as_ref().is_none_or(|best_sel| {
            candidate.total_proof_bytes < best_sel.total_proof_bytes
                || (candidate.total_proof_bytes == best_sel.total_proof_bytes
                    && candidate.root_d < best_sel.root_d)
        }) {
            best = Some(candidate);
        }
    }

    best.ok_or_else(|| {
        HachiError::InvalidInput(format!(
            "dynamic root selection found no supported root D for num_vars={}, layout_num_claims={}, batch_claims={}",
            key.num_vars, key.layout_num_claims, key.batch.num_claims
        ))
    })
}

/// Planner/security profile for one commitment base field.
///
/// A profile owns the field-specific challenge families, generated schedule
/// tables, audited root-rank escalations, and dynamic root-ring selection
/// policy used by the public preset surface.
pub trait CommitmentFieldProfile: Clone + Copy + Default + Send + Sync + 'static {
    /// Base field for this commitment profile.
    type Field: CanonicalField + FieldCore + Send + Sync + 'static;

    /// Typed config to use when the dynamic full family chooses `D=32`.
    type FullCfg32: CommitmentConfig<Field = Self::Field>;
    /// Typed config to use when the dynamic full family chooses `D=64`.
    type FullCfg64: CommitmentConfig<Field = Self::Field>;
    /// Typed config to use when the dynamic full family chooses `D=128`.
    type FullCfg128: CommitmentConfig<Field = Self::Field>;

    /// Typed config to use when the dynamic onehot family chooses `D=32`.
    type OneHotCfg32: CommitmentConfig<Field = Self::Field>;
    /// Typed config to use when the dynamic onehot family chooses `D=64`.
    type OneHotCfg64: CommitmentConfig<Field = Self::Field>;
    /// Typed config to use when the dynamic onehot family chooses `D=128`.
    type OneHotCfg128: CommitmentConfig<Field = Self::Field>;

    /// Build decomposition parameters for this field.
    fn decomposition(log_commit_bound: u32, log_basis: u32) -> DecompositionParams;

    /// Sparse stage-1 challenge family for a given root/recursive ring degree.
    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig;

    /// Inclusive search range for adaptive basis planning.
    fn adaptive_log_basis_search_range() -> (u32, u32) {
        (2, 6)
    }

    /// Generated table for one adaptive bounded family, if this profile ships one.
    fn generated_adaptive_bounded_table<
        const D: usize,
        const LOG_COMMIT_BOUND: u32,
        const N_A: usize,
        const N_B: usize,
        const N_D: usize,
    >() -> Option<&'static [GeneratedScheduleTableEntry]> {
        let _ = (D, LOG_COMMIT_BOUND, N_A, N_B, N_D);
        None
    }

    /// Generated table for the coarse `D=64` onehot family, if shipped.
    fn generated_onehot_d64_table() -> Option<&'static [GeneratedScheduleTableEntry]> {
        None
    }

    /// Minimum audited root rank for outer `B/D` rows at this level.
    fn audited_root_outer_rank(d: usize, level: usize, max_num_vars: usize) -> usize {
        let _ = (d, level, max_num_vars);
        1
    }

    /// Minimum audited root rank for inner `A` rows at this level.
    fn audited_root_a_rank<const LOG_COMMIT_BOUND: u32>(
        d: usize,
        level: usize,
        max_num_vars: usize,
    ) -> usize {
        let _ = (d, level, max_num_vars, LOG_COMMIT_BOUND);
        1
    }

    /// Root-rank policy for the coarse `D=64` onehot family.
    fn onehot_d64_root_rank(level: usize, max_num_vars: usize) -> usize {
        let _ = (level, max_num_vars);
        1
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ProfileScheduleSource {
    exact_plan: HachiSchedulePlan,
    min_log_basis: u32,
    max_log_basis: u32,
}

impl ProfileScheduleSource {
    fn new(exact_plan: HachiSchedulePlan, min_log_basis: u32, max_log_basis: u32) -> Self {
        Self {
            exact_plan,
            min_log_basis,
            max_log_basis,
        }
    }

    pub(crate) fn log_basis_at_level<Cfg: CommitmentConfig>(
        &self,
        inputs: HachiScheduleInputs,
    ) -> Result<u32, HachiError> {
        planned_log_basis_at_level_from_schedule::<Cfg>(
            &self.exact_plan,
            inputs,
            self.min_log_basis,
            self.max_log_basis,
        )
    }

    pub(crate) fn schedule_key(&self) -> String {
        planned_schedule_key_from_schedule(&self.exact_plan)
    }

    pub(crate) fn schedule_plan(&self) -> HachiSchedulePlan {
        self.exact_plan.clone()
    }

    pub(crate) fn recursive_suffix_bytes<Cfg: CommitmentConfig>(
        &self,
        max_num_vars: usize,
        level: usize,
        current_w_len: usize,
    ) -> Result<usize, HachiError> {
        planned_recursive_suffix_bytes_from_schedule::<Cfg>(
            &self.exact_plan,
            max_num_vars,
            level,
            current_w_len,
            self.min_log_basis,
            self.max_log_basis,
        )
    }
}

/// Internal schedule authority for profile-backed planner families.
pub(crate) trait CommitmentFieldProfileSchedule: CommitmentFieldProfile {
    /// Exact schedule source for one adaptive bounded family.
    ///
    /// # Errors
    ///
    /// Returns an error if the bounded family cannot derive a valid exact
    /// schedule source at `max_num_vars`.
    fn adaptive_bounded_schedule_source<
        Cfg: CommitmentConfig,
        const D: usize,
        const LOG_COMMIT_BOUND: u32,
        const N_A: usize,
        const N_B: usize,
        const N_D: usize,
    >(
        max_num_vars: usize,
    ) -> Result<ProfileScheduleSource, HachiError> {
        let (min_log_basis, max_log_basis) = Self::adaptive_log_basis_search_range();
        let exact_plan = if let Some(table) =
            Self::generated_adaptive_bounded_table::<D, LOG_COMMIT_BOUND, N_A, N_B, N_D>()
        {
            generated_or_planned_schedule::<Cfg>(max_num_vars, table, min_log_basis, max_log_basis)?
        } else {
            planned_schedule::<Cfg>(max_num_vars, min_log_basis, max_log_basis)?
        };
        Ok(ProfileScheduleSource::new(
            exact_plan,
            min_log_basis,
            max_log_basis,
        ))
    }

    /// Exact schedule source for the coarse `D=64` onehot family.
    ///
    /// # Errors
    ///
    /// Returns an error if the onehot `D=64` family cannot derive a valid
    /// exact schedule source at `max_num_vars`.
    fn onehot_d64_schedule_source<Cfg: CommitmentConfig>(
        max_num_vars: usize,
    ) -> Result<ProfileScheduleSource, HachiError> {
        let (min_log_basis, max_log_basis) = Self::adaptive_log_basis_search_range();
        let exact_plan = if let Some(table) = Self::generated_onehot_d64_table() {
            generated_or_planned_schedule::<Cfg>(max_num_vars, table, min_log_basis, max_log_basis)?
        } else {
            planned_schedule::<Cfg>(max_num_vars, min_log_basis, max_log_basis)?
        };
        Ok(ProfileScheduleSource::new(
            exact_plan,
            min_log_basis,
            max_log_basis,
        ))
    }
}

/// Internal dynamic-root selection hooks layered on top of a public profile.
pub(crate) trait CommitmentFieldProfileDynamic:
    CommitmentFieldProfile + CommitmentFieldProfileSchedule
{
    /// Optional profile-level override for dynamic root selection.
    fn preferred_dynamic_root_d(
        family: DynamicScheduleFamily,
        key: HachiScheduleLookupKey,
    ) -> Option<usize> {
        let _ = (family, key);
        None
    }

    /// Select the dynamic root schedule for one public proof family.
    ///
    /// # Errors
    ///
    /// Returns an error if the profile cannot choose or materialize a
    /// supported root schedule for the provided family and public batch key.
    fn select_dynamic_root_schedule(
        family: DynamicScheduleFamily,
        key: HachiScheduleLookupKey,
    ) -> Result<DynamicRootScheduleSelection, HachiError> {
        if let Some(root_d) = Self::preferred_dynamic_root_d(family, key) {
            return match (family, root_d) {
                (DynamicScheduleFamily::Full, 32) => {
                    select_root_schedule_for_cfg::<Self::FullCfg32, 32>(key)
                }
                (DynamicScheduleFamily::Full, 64) => {
                    select_root_schedule_for_cfg::<Self::FullCfg64, 64>(key)
                }
                (DynamicScheduleFamily::Full, 128) => {
                    select_root_schedule_for_cfg::<Self::FullCfg128, 128>(key)
                }
                (DynamicScheduleFamily::OneHot, 32) => {
                    select_root_schedule_for_cfg::<Self::OneHotCfg32, 32>(key)
                }
                (DynamicScheduleFamily::OneHot, 64) => {
                    select_root_schedule_for_cfg::<Self::OneHotCfg64, 64>(key)
                }
                (DynamicScheduleFamily::OneHot, 128) => {
                    select_root_schedule_for_cfg::<Self::OneHotCfg128, 128>(key)
                }
                _ => Err(HachiError::InvalidSetup(format!(
                    "unsupported dynamic root D={root_d} for {family:?} family"
                ))),
            };
        }

        match family {
            DynamicScheduleFamily::Full => {
                select_smallest_root_schedule::<Self::FullCfg32, Self::FullCfg64, Self::FullCfg128>(
                    key,
                )
            }
            DynamicScheduleFamily::OneHot => select_smallest_root_schedule::<
                Self::OneHotCfg32,
                Self::OneHotCfg64,
                Self::OneHotCfg128,
            >(key),
        }
    }

    /// Select only the dynamic root ring degree for one public proof family.
    ///
    /// # Errors
    ///
    /// Returns an error if the profile cannot choose or materialize a
    /// supported root schedule for the provided family and public batch key.
    fn select_dynamic_root_ring_dim(
        family: DynamicScheduleFamily,
        key: HachiScheduleLookupKey,
    ) -> Result<usize, HachiError> {
        Ok(Self::select_dynamic_root_schedule(family, key)?.root_d)
    }
}

fn uniform_pm1_stage1_challenge(weight: usize) -> SparseChallengeConfig {
    SparseChallengeConfig::Uniform {
        weight,
        nonzero_coeffs: vec![-1, 1],
    }
}

fn uniform_range_stage1_challenge(weight: usize, max_abs_coeff: i16) -> SparseChallengeConfig {
    SparseChallengeConfig::Uniform {
        weight,
        nonzero_coeffs: (-max_abs_coeff..=max_abs_coeff)
            .filter(|&c| c != 0)
            .collect(),
    }
}

fn d32_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
    assert_eq!(d, 32, "d32_stage1_challenge_config requires d=32, got {d}");
    uniform_range_stage1_challenge(32, 8)
}

fn d64_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
    assert_eq!(d, 64, "d64_stage1_challenge_config requires d=64, got {d}");
    SparseChallengeConfig::SplitRing {
        half_weight: 21,
        max_mag2_per_half: 6,
    }
}

fn d128_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
    assert_eq!(
        d, 128,
        "d128_stage1_challenge_config requires d=128, got {d}"
    );
    uniform_pm1_stage1_challenge(31)
}

const FP128_D128_AUDITED_ROOT_RANK2_FROM_NV: usize = 54;
const FP128_D128_AUDITED_ROOT_A_RANK2_FROM_NV: usize = 59;

fn fp128_exact_fit_singleton_prefers_d32(key: HachiScheduleLookupKey) -> bool {
    key.max_num_vars == key.num_vars
        && key.layout_num_claims == 1
        && key.batch == HachiRootBatchSummary::singleton()
        && (6..=63).contains(&key.num_vars)
}

/// Planner/security profile for the blessed fp128 prime `2^128 - 275`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fp128PrimeProfile;

impl CommitmentFieldProfile for Fp128PrimeProfile {
    type Field = Prime128Offset275;
    type FullCfg32 =
        CommitmentPreset<Self::Field, super::config::AdaptiveBoundedPolicy<Self, 32, 128, 2, 2, 2>>;
    type FullCfg64 =
        CommitmentPreset<Self::Field, super::config::AdaptiveBoundedPolicy<Self, 64, 128, 1, 1, 1>>;
    type FullCfg128 = CommitmentPreset<
        Self::Field,
        super::config::AdaptiveBoundedPolicy<Self, 128, 128, 1, 1, 1>,
    >;
    type OneHotCfg32 =
        CommitmentPreset<Self::Field, super::config::AdaptiveBoundedPolicy<Self, 32, 1, 2, 2, 2>>;
    type OneHotCfg64 = CommitmentPreset<Self::Field, super::config::AdaptiveOneHotD64Policy<Self>>;
    type OneHotCfg128 =
        CommitmentPreset<Self::Field, super::config::AdaptiveBoundedPolicy<Self, 128, 1, 1, 1, 1>>;

    fn decomposition(log_commit_bound: u32, log_basis: u32) -> DecompositionParams {
        DecompositionParams {
            log_basis,
            log_commit_bound,
            log_open_bound: if log_commit_bound < 128 {
                Some(128)
            } else {
                None
            },
        }
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        match d {
            32 => d32_stage1_challenge_config(d),
            64 => d64_stage1_challenge_config(d),
            128 => d128_stage1_challenge_config(d),
            _ => panic!("unsupported fp128 ring dim {d}"),
        }
    }

    fn generated_adaptive_bounded_table<
        const D: usize,
        const LOG_COMMIT_BOUND: u32,
        const N_A: usize,
        const N_B: usize,
        const N_D: usize,
    >() -> Option<&'static [GeneratedScheduleTableEntry]> {
        fp128_adaptive_bounded_table::<D, LOG_COMMIT_BOUND, N_A, N_B, N_D>()
    }

    fn generated_onehot_d64_table() -> Option<&'static [GeneratedScheduleTableEntry]> {
        Some(fp128_adaptive_onehot_d64_table())
    }

    fn audited_root_outer_rank(d: usize, level: usize, max_num_vars: usize) -> usize {
        if d == 128 && level == 0 && max_num_vars >= FP128_D128_AUDITED_ROOT_RANK2_FROM_NV {
            2
        } else {
            1
        }
    }

    fn audited_root_a_rank<const LOG_COMMIT_BOUND: u32>(
        d: usize,
        level: usize,
        max_num_vars: usize,
    ) -> usize {
        if d == 128
            && LOG_COMMIT_BOUND != 1
            && level == 0
            && max_num_vars >= FP128_D128_AUDITED_ROOT_A_RANK2_FROM_NV
        {
            2
        } else {
            1
        }
    }

    fn onehot_d64_root_rank(level: usize, max_num_vars: usize) -> usize {
        usize::from(max_num_vars >= 38 && level == 0) + 1
    }
}

impl CommitmentFieldProfileSchedule for Fp128PrimeProfile {}

impl CommitmentFieldProfileDynamic for Fp128PrimeProfile {
    fn preferred_dynamic_root_d(
        _family: DynamicScheduleFamily,
        key: HachiScheduleLookupKey,
    ) -> Option<usize> {
        fp128_exact_fit_singleton_prefers_d32(key).then_some(32)
    }
}

#[cfg(test)]
mod schedule_source_tests {
    use super::*;
    use crate::protocol::commitment::presets::fp128;

    #[test]
    fn bounded_schedule_source_matches_cfg_hooks() {
        type Cfg = fp128::D128Full;

        let max_num_vars = 30usize;
        let inputs = HachiScheduleInputs {
            max_num_vars,
            level: 4,
            current_w_len: 245_888,
        };

        let source = <Fp128PrimeProfile as CommitmentFieldProfileSchedule>::adaptive_bounded_schedule_source::<
            Cfg,
            128,
            128,
            1,
            1,
            1,
        >(max_num_vars)
        .unwrap();

        assert_eq!(source.schedule_key(), Cfg::schedule_key(max_num_vars));
        assert_eq!(
            source.schedule_plan(),
            Cfg::schedule_plan(max_num_vars).unwrap().unwrap()
        );
        assert_eq!(
            source.log_basis_at_level::<Cfg>(inputs).unwrap(),
            Cfg::log_basis_at_level(inputs)
        );
        assert_eq!(
            source
                .recursive_suffix_bytes::<Cfg>(max_num_vars, inputs.level, inputs.current_w_len)
                .unwrap(),
            Cfg::recursive_suffix_bytes(max_num_vars, inputs.level, inputs.current_w_len)
                .unwrap()
                .unwrap()
        );
    }

    #[test]
    fn onehot_d64_schedule_source_matches_cfg_hooks() {
        type Cfg = fp128::D64OneHot;

        let max_num_vars = 30usize;
        let inputs = HachiScheduleInputs {
            max_num_vars,
            level: 4,
            current_w_len: 245_888,
        };

        let source =
            <Fp128PrimeProfile as CommitmentFieldProfileSchedule>::onehot_d64_schedule_source::<
                Cfg,
            >(max_num_vars)
            .unwrap();

        assert_eq!(source.schedule_key(), Cfg::schedule_key(max_num_vars));
        assert_eq!(
            source.schedule_plan(),
            Cfg::schedule_plan(max_num_vars).unwrap().unwrap()
        );
        assert_eq!(
            source.log_basis_at_level::<Cfg>(inputs).unwrap(),
            Cfg::log_basis_at_level(inputs)
        );
        assert_eq!(
            source
                .recursive_suffix_bytes::<Cfg>(max_num_vars, inputs.level, inputs.current_w_len)
                .unwrap(),
            Cfg::recursive_suffix_bytes(max_num_vars, inputs.level, inputs.current_w_len)
                .unwrap()
                .unwrap()
        );
    }
}
