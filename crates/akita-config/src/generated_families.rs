//! Shared metadata describing every `Cfg` family that ships with a
//! generated schedule table in `akita-types::generated`.
//!
//! Both the `gen_schedule_tables` binary (the offline table emitter) and
//! the drift-guard test consume [`ALL_GENERATED_FAMILIES`] so the two
//! cannot drift apart: a missing `Cfg` here is missing in both the emitted
//! artifact and the regression guard.
//!
//! This list is the one place a preset `Cfg` type is bound to its regen
//! hook and shipped table, so it lives in `akita-config` (the only crate
//! that can name the presets). The `Cfg`-free DP itself lives in
//! `akita-planner` and is reached through the `regen` glue below, which
//! derives a [`akita_planner::PlannerPolicy`] from each preset via
//! [`crate::policy_of`].

use akita_field::AkitaError;
use akita_planner::find_schedule;
use akita_types::{AkitaScheduleLookupKey, ClaimIncidenceSummary, Schedule};

use crate::proof_optimized::{fp128, fp32, fp64};
use crate::{policy_of, tensor_verifier, CommitmentConfig};

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
    /// Pure DP regeneration that ignores any shipped table
    /// (`find_schedule(key, &policy_of::<Cfg>(), …)`).
    pub regen: fn(AkitaScheduleLookupKey) -> Result<Schedule, AkitaError>,
    /// `Cfg::runtime_schedule(key)` — the table fast path when an entry
    /// exists, falling through to the DP otherwise. Used by diagnostic
    /// comparisons against the shipped table.
    pub table_backed: fn(AkitaScheduleLookupKey) -> Result<Schedule, AkitaError>,
}

/// Build the ordered key cross-product emitted for `family`.
///
/// The order matches what `gen_schedule_tables` writes to disk: all
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

/// Pure DP regeneration for `Cfg` — never consults the shipped table.
fn regen<Cfg: CommitmentConfig>(key: AkitaScheduleLookupKey) -> Result<Schedule, AkitaError> {
    find_schedule(
        key,
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )
}

/// Table-backed resolution for `Cfg` — table hit when present, otherwise
/// the DP fallback baked into `runtime_schedule`.
fn table_backed<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
) -> Result<Schedule, AkitaError> {
    Cfg::runtime_schedule(key)
}

/// Every `Cfg` that ships with a generated schedule table.
///
/// Adding a new preset with a generated table requires adding a row
/// here; both the table emitter and the drift-guard test pick it up
/// automatically.
pub const ALL_GENERATED_FAMILIES: &[GeneratedFamily] = &[
    GeneratedFamily {
        module_name: "fp128_d128_full",
        const_name: "FP128_D128_FULL_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        regen: regen::<fp128::D128Full>,
        table_backed: table_backed::<fp128::D128Full>,
    },
    GeneratedFamily {
        module_name: "fp128_d128_onehot",
        const_name: "FP128_D128_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        regen: regen::<fp128::D128OneHot>,
        table_backed: table_backed::<fp128::D128OneHot>,
    },
    GeneratedFamily {
        module_name: "fp128_d64_onehot",
        const_name: "FP128_D64_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        regen: regen::<fp128::D64OneHot>,
        table_backed: table_backed::<fp128::D64OneHot>,
    },
    GeneratedFamily {
        module_name: "fp128_d64_onehot_tensor",
        const_name: "FP128_D64_ONEHOT_TENSOR_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        regen: regen::<tensor_verifier::fp128::D64OneHotTensor>,
        table_backed: table_backed::<tensor_verifier::fp128::D64OneHotTensor>,
    },
    // Tiered companion of `fp128_d64_onehot`
    #[cfg(not(feature = "zk"))]
    GeneratedFamily {
        module_name: "fp128_d64_onehot_tiered",
        const_name: "FP128_D64_ONEHOT_TIERED_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        regen: regen::<fp128::D64OneHotTiered>,
        table_backed: table_backed::<fp128::D64OneHotTiered>,
    },
    GeneratedFamily {
        module_name: "fp64_d128",
        const_name: "FP64_D128_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        regen: regen::<fp64::D128Full>,
        table_backed: table_backed::<fp64::D128Full>,
    },
    GeneratedFamily {
        module_name: "fp64_d128_onehot",
        const_name: "FP64_D128_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        regen: regen::<fp64::D128OneHot>,
        table_backed: table_backed::<fp64::D128OneHot>,
    },
    GeneratedFamily {
        module_name: "fp64_d256_onehot",
        const_name: "FP64_D256_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        regen: regen::<fp64::D256OneHot>,
        table_backed: table_backed::<fp64::D256OneHot>,
    },
    GeneratedFamily {
        module_name: "fp32_d128_onehot",
        const_name: "FP32_D128_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        regen: regen::<fp32::D128OneHot>,
        table_backed: table_backed::<fp32::D128OneHot>,
    },
    GeneratedFamily {
        module_name: "fp32_d256_onehot",
        const_name: "FP32_D256_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        regen: regen::<fp32::D256OneHot>,
        table_backed: table_backed::<fp32::D256OneHot>,
    },
];
