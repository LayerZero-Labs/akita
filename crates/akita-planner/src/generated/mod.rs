#![allow(missing_docs)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedFoldStep {
    pub ring_d: u32,
    pub log_basis: u32,
    pub m_vars: u32,
    pub r_vars: u32,
    pub n_a: u32,
    /// First-tier `B` rank **before** any tiering split. Under a tiered policy,
    /// expansion re-derives the shrunk `B'` and the second-tier `F` by replaying
    /// `apply_tiering`, so the table stores the un-tiered rank here.
    pub n_b: u32,
    pub n_d: u32,
}

/// Terminal direct-send step in a generated schedule.
///
/// `commit` is `Some` only for a **root-direct** entry (a schedule whose
/// single step is this `Direct`): it carries the brute-forced root commit
/// layout — the same 7-field shape as a fold step — so the runtime can
/// expand it into the committed `LevelParams` via
/// [`GeneratedFoldStep::expand_to_level_params`] without re-running the
/// offline SIS derivation.
///
/// Terminal-direct steps that follow one or more folds ship the cleartext
/// witness without committing, so they carry `commit: None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedDirectStep {
    pub commit: Option<GeneratedFoldStep>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratedStep {
    Fold(GeneratedFoldStep),
    Direct(GeneratedDirectStep),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedScheduleKey {
    pub num_vars: usize,
    pub num_commitment_groups: usize,
    pub num_t_vectors: usize,
    pub num_w_vectors: usize,
    pub num_z_vectors: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedScheduleTableEntry {
    pub key: GeneratedScheduleKey,
    pub steps: &'static [GeneratedStep],
}

#[derive(Debug, Clone, Copy)]
pub struct GeneratedScheduleTable {
    pub sis_family: SisModulusFamily,
    pub entries: &'static [GeneratedScheduleTableEntry],
}

pub mod expand;
#[cfg(not(feature = "zk"))]
pub mod fp128_d128_full;
#[cfg(feature = "zk")]
pub mod fp128_d128_full_zk;
#[cfg(not(feature = "zk"))]
pub mod fp128_d128_onehot;
#[cfg(feature = "zk")]
pub mod fp128_d128_onehot_zk;
#[cfg(not(feature = "zk"))]
pub mod fp128_d64_onehot;
#[cfg(not(feature = "zk"))]
pub mod fp128_d64_onehot_tensor;
#[cfg(not(feature = "zk"))]
pub mod fp128_d64_onehot_tiered;
#[cfg(feature = "zk")]
pub mod fp128_d64_onehot_tensor_zk;
#[cfg(feature = "zk")]
pub mod fp128_d64_onehot_zk;
#[cfg(not(feature = "zk"))]
pub mod fp32_d128_onehot;
#[cfg(feature = "zk")]
pub mod fp32_d128_onehot_zk;
#[cfg(not(feature = "zk"))]
pub mod fp32_d256_onehot;
#[cfg(feature = "zk")]
pub mod fp32_d256_onehot_zk;
#[cfg(not(feature = "zk"))]
pub mod fp64_d128;
#[cfg(not(feature = "zk"))]
pub mod fp64_d128_onehot;
#[cfg(feature = "zk")]
pub mod fp64_d128_onehot_zk;
#[cfg(feature = "zk")]
pub mod fp64_d128_zk;
#[cfg(not(feature = "zk"))]
pub mod fp64_d256_onehot;
#[cfg(feature = "zk")]
pub mod fp64_d256_onehot_zk;
pub use akita_types::SisModulusFamily;

pub fn table_entry(
    table: GeneratedScheduleTable,
    key: GeneratedScheduleKey,
) -> Option<&'static GeneratedScheduleTableEntry> {
    table.entries.iter().find(|entry| entry.key == key)
}

pub fn fp128_d128_full_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q128,
            entries: fp128_d128_full_zk::FP128_D128_FULL_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q128,
        entries: fp128_d128_full::FP128_D128_FULL_SCHEDULES,
    }
}

pub fn fp128_d128_onehot_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q128,
            entries: fp128_d128_onehot_zk::FP128_D128_ONEHOT_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q128,
        entries: fp128_d128_onehot::FP128_D128_ONEHOT_SCHEDULES,
    }
}

pub fn fp128_d64_onehot_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q128,
            entries: fp128_d64_onehot_zk::FP128_D64_ONEHOT_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q128,
        entries: fp128_d64_onehot::FP128_D64_ONEHOT_SCHEDULES,
    }
}

/// Tiered-commitment companion of [`fp128_d64_onehot_table`]: entries whose
/// first-tier `B` footprint exceeds inner `A` carry the un-tiered `n_b`, and
/// expansion replays `apply_tiering` to recover the `B'`/`F` split. Tiering is
/// a non-ZK optimization, so this family has no `_zk` variant.
#[cfg(not(feature = "zk"))]
pub fn fp128_d64_onehot_tiered_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q128,
        entries: fp128_d64_onehot_tiered::FP128_D64_ONEHOT_TIERED_SCHEDULES,
    }
}

pub fn fp128_d64_onehot_tensor_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q128,
            entries: fp128_d64_onehot_tensor_zk::FP128_D64_ONEHOT_TENSOR_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q128,
        entries: fp128_d64_onehot_tensor::FP128_D64_ONEHOT_TENSOR_SCHEDULES,
    }
}

macro_rules! small_field_table_fn {
    ($fn_name:ident, $family:expr, $non_zk_mod:ident, $zk_mod:ident, $non_zk_const:ident, $zk_const:ident) => {
        pub fn $fn_name() -> GeneratedScheduleTable {
            #[cfg(feature = "zk")]
            {
                GeneratedScheduleTable {
                    sis_family: $family,
                    entries: $zk_mod::$zk_const,
                }
            }
            #[cfg(not(feature = "zk"))]
            GeneratedScheduleTable {
                sis_family: $family,
                entries: $non_zk_mod::$non_zk_const,
            }
        }
    };
}

small_field_table_fn!(
    fp32_d128_onehot_table,
    SisModulusFamily::Q32,
    fp32_d128_onehot,
    fp32_d128_onehot_zk,
    FP32_D128_ONEHOT_SCHEDULES,
    FP32_D128_ONEHOT_ZK_SCHEDULES
);
small_field_table_fn!(
    fp32_d256_onehot_table,
    SisModulusFamily::Q32,
    fp32_d256_onehot,
    fp32_d256_onehot_zk,
    FP32_D256_ONEHOT_SCHEDULES,
    FP32_D256_ONEHOT_ZK_SCHEDULES
);
small_field_table_fn!(
    fp64_d128_table,
    SisModulusFamily::Q64,
    fp64_d128,
    fp64_d128_zk,
    FP64_D128_SCHEDULES,
    FP64_D128_ZK_SCHEDULES
);
small_field_table_fn!(
    fp64_d128_onehot_table,
    SisModulusFamily::Q64,
    fp64_d128_onehot,
    fp64_d128_onehot_zk,
    FP64_D128_ONEHOT_SCHEDULES,
    FP64_D128_ONEHOT_ZK_SCHEDULES
);
small_field_table_fn!(
    fp64_d256_onehot_table,
    SisModulusFamily::Q64,
    fp64_d256_onehot,
    fp64_d256_onehot_zk,
    FP64_D256_ONEHOT_SCHEDULES,
    FP64_D256_ONEHOT_ZK_SCHEDULES
);
