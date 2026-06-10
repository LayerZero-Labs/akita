#![allow(missing_docs)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedFoldStep {
    pub ring_d: u32,
    pub log_basis: u32,
    pub m_vars: u32,
    pub r_vars: u32,
    pub n_a: u32,
    /// Stored first-tier `B` rank. This is the actual committed rank: the shrunk
    /// `B'` rank when the step is tiered (`tier_split.is_some()`), and the full
    /// `B` rank otherwise.
    pub n_b: u32,
    pub n_d: u32,
    /// Tiered split factor `f`. `None` for single-tier steps; `Some(f)` when the
    /// step reuses a smaller `B'` across `f` column-slices (paired with `n_f`).
    pub tier_split: Option<u32>,
    /// Second-tier `F` rank. `None` for single-tier steps; `Some` iff
    /// `tier_split` is `Some`. Expansion sizes `F` from `tier_split`, `n_b`, and
    /// the level's `num_digits_open`.
    pub n_f: Option<u32>,
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
// @generated schedule module wiring begin
#[cfg(not(feature = "zk"))]
pub mod fp128_d128_full;
#[cfg(feature = "zk")]
pub mod fp128_d128_full_zk;
#[cfg(not(feature = "zk"))]
pub mod fp128_d128_onehot;
#[cfg(feature = "zk")]
pub mod fp128_d128_onehot_zk;
#[cfg(not(feature = "zk"))]
pub mod fp128_d32_full;
#[cfg(feature = "zk")]
pub mod fp128_d32_full_zk;
#[cfg(not(feature = "zk"))]
pub mod fp128_d32_onehot;
#[cfg(feature = "zk")]
pub mod fp128_d32_onehot_zk;
#[cfg(not(feature = "zk"))]
pub mod fp128_d64_onehot;
#[cfg(not(feature = "zk"))]
pub mod fp128_d64_onehot_tensor;
#[cfg(feature = "zk")]
pub mod fp128_d64_onehot_tensor_zk;
#[cfg(not(feature = "zk"))]
pub mod fp128_d64_onehot_tiered;
#[cfg(feature = "zk")]
pub mod fp128_d64_onehot_tiered_zk;
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

pub fn fp128_d32_full_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q128,
            entries: fp128_d32_full_zk::FP128_D32_FULL_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q128,
        entries: fp128_d32_full::FP128_D32_FULL_SCHEDULES,
    }
}

pub fn fp128_d32_onehot_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q128,
            entries: fp128_d32_onehot_zk::FP128_D32_ONEHOT_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q128,
        entries: fp128_d32_onehot::FP128_D32_ONEHOT_SCHEDULES,
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

pub fn fp128_d64_onehot_tiered_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q128,
            entries: fp128_d64_onehot_tiered_zk::FP128_D64_ONEHOT_TIERED_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q128,
        entries: fp128_d64_onehot_tiered::FP128_D64_ONEHOT_TIERED_SCHEDULES,
    }
}

pub fn fp32_d128_onehot_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q32,
            entries: fp32_d128_onehot_zk::FP32_D128_ONEHOT_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q32,
        entries: fp32_d128_onehot::FP32_D128_ONEHOT_SCHEDULES,
    }
}

pub fn fp32_d256_onehot_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q32,
            entries: fp32_d256_onehot_zk::FP32_D256_ONEHOT_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q32,
        entries: fp32_d256_onehot::FP32_D256_ONEHOT_SCHEDULES,
    }
}

pub fn fp64_d128_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q64,
            entries: fp64_d128_zk::FP64_D128_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q64,
        entries: fp64_d128::FP64_D128_SCHEDULES,
    }
}

pub fn fp64_d128_onehot_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q64,
            entries: fp64_d128_onehot_zk::FP64_D128_ONEHOT_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q64,
        entries: fp64_d128_onehot::FP64_D128_ONEHOT_SCHEDULES,
    }
}

pub fn fp64_d256_onehot_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q64,
            entries: fp64_d256_onehot_zk::FP64_D256_ONEHOT_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q64,
        entries: fp64_d256_onehot::FP64_D256_ONEHOT_SCHEDULES,
    }
}
// @generated schedule module wiring end
pub use akita_types::SisModulusFamily;

pub fn table_entry(
    table: GeneratedScheduleTable,
    key: GeneratedScheduleKey,
) -> Option<&'static GeneratedScheduleTableEntry> {
    table.entries.iter().find(|entry| entry.key == key)
}
