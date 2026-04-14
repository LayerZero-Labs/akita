//! Field-profile definitions for planner-backed commitment presets.

use super::config::{CommitmentConfig, DecompositionParams};
use super::generated::{
    fp128_d128_full_table, fp128_d128_onehot_table, fp128_d32_full_table, fp128_d32_onehot_table,
    fp128_d64_full_table, fp128_d64_onehot_table, table_entry_envelope_for_max_num_vars,
    GeneratedScheduleTable,
};
use super::schedule::{
    generated_schedule_plan_from_table, planned_log_basis_at_level_from_schedule,
    planned_recursive_suffix_bytes_from_schedule, planned_schedule_key_from_schedule,
    HachiScheduleInputs, HachiScheduleLookupKey, HachiSchedulePlan,
};
use crate::algebra::Prime128Offset2355;
use crate::algebra::SparseChallengeConfig;
use crate::error::HachiError;
use crate::{CanonicalField, FieldCore};

fn generated_schedule<Cfg: CommitmentConfig, const D: usize>(
    key: HachiScheduleLookupKey,
    table: GeneratedScheduleTable,
) -> Result<HachiSchedulePlan, HachiError> {
    generated_schedule_plan_from_table::<Cfg, D>(key, table)?.ok_or_else(|| {
        HachiError::InvalidSetup(format!(
            "missing generated schedule for {} at key={key:?}",
            std::any::type_name::<Cfg>()
        ))
    })
}

/// Planner/security profile for one commitment base field.
///
/// A profile owns the field-specific challenge families, generated schedule
/// tables, and audited root-rank escalations used by the public preset surface.
pub trait CommitmentFieldProfile: Clone + Copy + Default + Send + Sync + 'static {
    /// Base field for this commitment profile.
    type Field: CanonicalField + FieldCore + Send + Sync + 'static;

    /// Build decomposition parameters for this field.
    fn decomposition(log_commit_bound: u32, log_basis: u32) -> DecompositionParams;

    /// Sparse stage-1 challenge family for a given root/recursive ring degree.
    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig;

    /// Inclusive search range for adaptive basis planning.
    fn adaptive_log_basis_search_range() -> (u32, u32) {
        (2, 6)
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
    key: HachiScheduleLookupKey,
    exact_plan: HachiSchedulePlan,
    min_log_basis: u32,
    max_log_basis: u32,
}

impl ProfileScheduleSource {
    fn new(
        key: HachiScheduleLookupKey,
        exact_plan: HachiSchedulePlan,
        min_log_basis: u32,
        max_log_basis: u32,
    ) -> Self {
        Self {
            key,
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
        planned_schedule_key_from_schedule(self.key, &self.exact_plan)
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
    /// Generated table for one shipped planner-backed family, if available.
    fn generated_schedule_table<const D: usize, const LOG_COMMIT_BOUND: u32>(
    ) -> Option<GeneratedScheduleTable> {
        let _ = (D, LOG_COMMIT_BOUND);
        None
    }

    /// Maximum `(n_a, n_b, n_d)` required by the generated entry at
    /// `max_num_vars`, if this profile ships one.
    fn generated_schedule_envelope<const D: usize, const LOG_COMMIT_BOUND: u32>(
        max_num_vars: usize,
    ) -> Option<(usize, usize, usize)> {
        Self::generated_schedule_table::<D, LOG_COMMIT_BOUND>()
            .and_then(|table| table_entry_envelope_for_max_num_vars(table, max_num_vars))
    }

    /// Exact generated schedule source for one shipped generated family.
    ///
    /// # Errors
    ///
    /// Returns an error if the family does not ship a generated schedule table
    /// or cannot derive a valid exact schedule source at
    /// `max_num_vars`.
    fn generated_schedule_source<
        Cfg: CommitmentConfig,
        const D: usize,
        const LOG_COMMIT_BOUND: u32,
    >(
        key: HachiScheduleLookupKey,
    ) -> Result<ProfileScheduleSource, HachiError> {
        let (min_log_basis, max_log_basis) = Self::adaptive_log_basis_search_range();
        let table = Self::generated_schedule_table::<D, LOG_COMMIT_BOUND>().ok_or_else(|| {
            HachiError::InvalidSetup(format!(
                "missing generated schedule table for {}",
                std::any::type_name::<Cfg>()
            ))
        })?;
        let exact_plan = generated_schedule::<Cfg, D>(key, table)?;
        Ok(ProfileScheduleSource::new(
            key,
            exact_plan,
            min_log_basis,
            max_log_basis,
        ))
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

/// Planner/security profile for the blessed fp128 prime `2^128 - 275`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fp128PrimeProfile;

impl CommitmentFieldProfile for Fp128PrimeProfile {
    type Field = Prime128Offset2355;

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

impl CommitmentFieldProfileSchedule for Fp128PrimeProfile {
    fn generated_schedule_table<const D: usize, const LOG_COMMIT_BOUND: u32>(
    ) -> Option<GeneratedScheduleTable> {
        match (D, LOG_COMMIT_BOUND) {
            (32, 128) => Some(fp128_d32_full_table()),
            (32, 1) => Some(fp128_d32_onehot_table()),
            (64, 128) => Some(fp128_d64_full_table()),
            (64, 1) => Some(fp128_d64_onehot_table()),
            (128, 128) => Some(fp128_d128_full_table()),
            (128, 1) => Some(fp128_d128_onehot_table()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod schedule_source_tests {
    use super::*;
    use crate::protocol::commitment::presets::fp128;

    #[test]
    fn d128_full_schedule_source_matches_cfg_hooks() {
        type Cfg = fp128::D128Full;

        let max_num_vars = 30usize;
        let key = HachiScheduleLookupKey::singleton(max_num_vars, max_num_vars, 1);
        let inputs = HachiScheduleInputs {
            max_num_vars,
            level: 4,
            current_w_len: 245_888,
        };

        let source =
            <Fp128PrimeProfile as CommitmentFieldProfileSchedule>::generated_schedule_source::<
                Cfg,
                128,
                128,
            >(key)
            .unwrap();

        assert_eq!(source.schedule_key(), Cfg::schedule_key(key));
        assert_eq!(
            source.schedule_plan(),
            Cfg::schedule_plan(key).unwrap().unwrap()
        );
        assert_eq!(
            source.log_basis_at_level::<Cfg>(inputs).unwrap(),
            Cfg::log_basis_at_level(inputs)
        );
        assert_eq!(
            source
                .recursive_suffix_bytes::<Cfg>(max_num_vars, inputs.level, inputs.current_w_len)
                .unwrap(),
            Cfg::recursive_suffix_bytes(key, inputs.level, inputs.current_w_len)
                .unwrap()
                .unwrap()
        );
    }

    #[test]
    fn onehot_d64_schedule_source_matches_cfg_hooks() {
        type Cfg = fp128::D64OneHot;

        let max_num_vars = 30usize;
        let key = HachiScheduleLookupKey::singleton(max_num_vars, max_num_vars, 1);
        let inputs = HachiScheduleInputs {
            max_num_vars,
            level: 4,
            current_w_len: 245_888,
        };

        let source =
            <Fp128PrimeProfile as CommitmentFieldProfileSchedule>::generated_schedule_source::<
                Cfg,
                64,
                1,
            >(key)
            .unwrap();

        assert_eq!(source.schedule_key(), Cfg::schedule_key(key));
        assert_eq!(
            source.schedule_plan(),
            Cfg::schedule_plan(key).unwrap().unwrap()
        );
        assert_eq!(
            source.log_basis_at_level::<Cfg>(inputs).unwrap(),
            Cfg::log_basis_at_level(inputs)
        );
        assert_eq!(
            source
                .recursive_suffix_bytes::<Cfg>(max_num_vars, inputs.level, inputs.current_w_len)
                .unwrap(),
            Cfg::recursive_suffix_bytes(key, inputs.level, inputs.current_w_len)
                .unwrap()
                .unwrap()
        );
    }
}
