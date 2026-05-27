//! Shared metadata describing every `Cfg` family that ships with a
//! generated schedule table in `akita-types::generated`.
//!
//! Both `gen_schedule_tables` (the offline table emitter) and the
//! cross-crate drift-guard test consume [`ALL_GENERATED_FAMILIES`] so
//! the two cannot drift apart: a missing `Cfg` here is missing in both
//! the emitted artifact and the regression guard.
//!
//! Each entry carries:
//!
//! - the on-disk module/const names of the generated table,
//! - the inclusive `num_vars` range to enumerate,
//! - the keys cross-product to emit (singleton and 4-batched at every
//!   `num_vars`),
//! - a `find_schedule::<Cfg>(_, false)` regen hook,
//! - a `Cfg::schedule_table()` accessor so consumers can validate the
//!   shipped artifact against the regen hook without ever needing to
//!   know about `Cfg` directly.

use akita_config::proof_optimized::{fp128, fp16, fp32, fp64};
use akita_config::tensor_verifier;
use akita_config::CommitmentConfig;
use akita_field::AkitaError;
use akita_types::generated::GeneratedScheduleTable;
use akita_types::{AkitaScheduleLookupKey, ClaimIncidenceSummary, Schedule};

use crate::find_schedule;

/// One generated schedule-table family.
///
/// Function-pointer fields (instead of generic `Fn` closures) keep the
/// list `const`-constructible and `'static`.
#[derive(Clone, Copy)]
pub struct GeneratedFamily {
    /// On-disk module file name (without `.rs`) and the basename used
    /// to derive the static `&[GeneratedScheduleTableEntry]` const name.
    /// The `_zk` suffix is appended by the binary at emit time.
    pub module_name: &'static str,
    /// On-disk const name for the table entries array. The binary
    /// rewrites `_SCHEDULES` -> `_ZK_SCHEDULES` when the `zk` feature
    /// is enabled.
    pub const_name: &'static str,
    /// Inclusive lower bound of the `num_vars` range enumerated for
    /// this family.
    pub min_num_vars: usize,
    /// Inclusive upper bound of the `num_vars` range enumerated for
    /// this family.
    pub max_num_vars: usize,
    /// `find_schedule::<Cfg>(key, false)` — DP regeneration that ignores
    /// any prior shipped table for this Cfg.
    pub regen: fn(AkitaScheduleLookupKey) -> Result<Schedule, AkitaError>,
    /// `Cfg::schedule_table()` for the family. Returns the table the
    /// linked binary currently ships for the active feature set
    /// (non-zk vs zk), or `None` when the Cfg has no shipped table.
    pub schedule_table: fn() -> Option<GeneratedScheduleTable>,
}

/// Build the ordered key cross-product emitted for `family`.
///
/// The order matches what `gen_schedule_tables.rs` writes to disk: all
/// singleton (`num_polys = 1`) keys first, then all 4-batched
/// (`num_polys = 4`) keys, each block ordered by `num_vars` ascending.
/// Drift-guard tests assert positional equality against the shipped
/// table, so this ordering doubles as the canonical row order.
///
/// # Errors
///
/// Returns an error if the synthetic incidence summary fails to build
/// or the lookup-key derivation fails (both indicate a malformed
/// `(min_num_vars, max_num_vars)` range).
pub fn family_keys(family: &GeneratedFamily) -> Result<Vec<AkitaScheduleLookupKey>, AkitaError> {
    let mut keys = Vec::with_capacity(2 * (family.max_num_vars - family.min_num_vars + 1));
    for num_polys in [1, 4] {
        for nv in family.min_num_vars..=family.max_num_vars {
            let incidence = ClaimIncidenceSummary::same_point(nv, num_polys)?;
            keys.push(AkitaScheduleLookupKey::new_from_incidence(&incidence)?);
        }
    }
    Ok(keys)
}

fn regen<Cfg: CommitmentConfig>(key: AkitaScheduleLookupKey) -> Result<Schedule, AkitaError> {
    find_schedule::<Cfg>(key, false)
}

/// Every `Cfg` that ships with a generated schedule table.
///
/// Adding a new preset with a generated table requires adding a row
/// here; both the table emitter and the drift-guard test pick it up
/// automatically.
pub const ALL_GENERATED_FAMILIES: &[GeneratedFamily] = &[
    GeneratedFamily {
        module_name: "fp128_d32_full",
        const_name: "FP128_D32_FULL_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        regen: regen::<fp128::D32Full>,
        schedule_table: fp128::D32Full::schedule_table,
    },
    GeneratedFamily {
        module_name: "fp128_d32_onehot",
        const_name: "FP128_D32_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        regen: regen::<fp128::D32OneHot>,
        schedule_table: fp128::D32OneHot::schedule_table,
    },
    GeneratedFamily {
        module_name: "fp128_d64_full",
        const_name: "FP128_D64_FULL_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        regen: regen::<fp128::D64Full>,
        schedule_table: fp128::D64Full::schedule_table,
    },
    GeneratedFamily {
        module_name: "fp128_d64_onehot",
        const_name: "FP128_D64_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        regen: regen::<fp128::D64OneHot>,
        schedule_table: fp128::D64OneHot::schedule_table,
    },
    GeneratedFamily {
        module_name: "fp128_d64_onehot_tensor",
        const_name: "FP128_D64_ONEHOT_TENSOR_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        regen: regen::<tensor_verifier::fp128::D64OneHotTensor>,
        schedule_table: tensor_verifier::fp128::D64OneHotTensor::schedule_table,
    },
    GeneratedFamily {
        module_name: "fp32_d32",
        const_name: "FP32_D32_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        regen: regen::<fp32::D32Full>,
        schedule_table: fp32::D32Full::schedule_table,
    },
    GeneratedFamily {
        module_name: "fp32_d32_onehot",
        const_name: "FP32_D32_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        regen: regen::<fp32::D32OneHot>,
        schedule_table: fp32::D32OneHot::schedule_table,
    },
    GeneratedFamily {
        module_name: "fp32_d64",
        const_name: "FP32_D64_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        regen: regen::<fp32::D64Full>,
        schedule_table: fp32::D64Full::schedule_table,
    },
    GeneratedFamily {
        module_name: "fp32_d64_onehot",
        const_name: "FP32_D64_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        regen: regen::<fp32::D64OneHot>,
        schedule_table: fp32::D64OneHot::schedule_table,
    },
    GeneratedFamily {
        module_name: "fp16_d32_full",
        const_name: "FP16_D32_FULL_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        regen: regen::<fp16::D32Full>,
        schedule_table: fp16::D32Full::schedule_table,
    },
    GeneratedFamily {
        module_name: "fp16_d32_onehot",
        const_name: "FP16_D32_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        regen: regen::<fp16::D32OneHot>,
        schedule_table: fp16::D32OneHot::schedule_table,
    },
    GeneratedFamily {
        module_name: "fp16_d64_full",
        const_name: "FP16_D64_FULL_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        regen: regen::<fp16::D64Full>,
        schedule_table: fp16::D64Full::schedule_table,
    },
    GeneratedFamily {
        module_name: "fp16_d64_onehot",
        const_name: "FP16_D64_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        regen: regen::<fp16::D64OneHot>,
        schedule_table: fp16::D64OneHot::schedule_table,
    },
    GeneratedFamily {
        module_name: "fp64_d32",
        const_name: "FP64_D32_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        regen: regen::<fp64::D32Full>,
        schedule_table: fp64::D32Full::schedule_table,
    },
    GeneratedFamily {
        module_name: "fp64_d32_onehot",
        const_name: "FP64_D32_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        regen: regen::<fp64::D32OneHot>,
        schedule_table: fp64::D32OneHot::schedule_table,
    },
    GeneratedFamily {
        module_name: "fp64_d64",
        const_name: "FP64_D64_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        regen: regen::<fp64::D64Full>,
        schedule_table: fp64::D64Full::schedule_table,
    },
    GeneratedFamily {
        module_name: "fp64_d64_onehot",
        const_name: "FP64_D64_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        regen: regen::<fp64::D64OneHot>,
        schedule_table: fp64::D64OneHot::schedule_table,
    },
];
