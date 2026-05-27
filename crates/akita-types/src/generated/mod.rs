#![allow(missing_docs)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedFoldStep {
    pub ring_d: u32,
    pub log_basis: u32,
    pub m_vars: u32,
    pub r_vars: u32,
    pub n_a: u32,
    pub n_b: u32,
    pub n_d: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedDirectStep;

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

#[cfg(not(feature = "zk"))]
pub mod fp128_d32_full;
#[cfg(feature = "zk")]
pub mod fp128_d32_full_zk;
#[cfg(not(feature = "zk"))]
pub mod fp128_d32_onehot;
#[cfg(not(feature = "zk"))]
pub mod fp128_d32_onehot_fast_verify;
#[cfg(feature = "zk")]
pub mod fp128_d32_onehot_fast_verify_zk;
#[cfg(feature = "zk")]
pub mod fp128_d32_onehot_zk;
#[cfg(not(feature = "zk"))]
pub mod fp128_d64_full;
#[cfg(feature = "zk")]
pub mod fp128_d64_full_zk;
#[cfg(not(feature = "zk"))]
pub mod fp128_d64_onehot;
#[cfg(not(feature = "zk"))]
pub mod fp128_d64_onehot_tensor;
#[cfg(feature = "zk")]
pub mod fp128_d64_onehot_tensor_zk;
#[cfg(feature = "zk")]
pub mod fp128_d64_onehot_zk;
#[cfg(not(feature = "zk"))]
pub mod fp16_d32_full;
#[cfg(feature = "zk")]
pub mod fp16_d32_full_zk;
#[cfg(not(feature = "zk"))]
pub mod fp16_d32_onehot;
#[cfg(feature = "zk")]
pub mod fp16_d32_onehot_zk;
#[cfg(not(feature = "zk"))]
pub mod fp16_d64_full;
#[cfg(feature = "zk")]
pub mod fp16_d64_full_zk;
#[cfg(not(feature = "zk"))]
pub mod fp16_d64_onehot;
#[cfg(feature = "zk")]
pub mod fp16_d64_onehot_zk;
#[cfg(not(feature = "zk"))]
pub mod fp32_d32;
#[cfg(not(feature = "zk"))]
pub mod fp32_d32_onehot;
#[cfg(feature = "zk")]
pub mod fp32_d32_onehot_zk;
#[cfg(feature = "zk")]
pub mod fp32_d32_zk;
#[cfg(not(feature = "zk"))]
pub mod fp32_d64;
#[cfg(not(feature = "zk"))]
pub mod fp32_d64_onehot;
#[cfg(feature = "zk")]
pub mod fp32_d64_onehot_zk;
#[cfg(feature = "zk")]
pub mod fp32_d64_zk;
#[cfg(not(feature = "zk"))]
pub mod fp64_d32;
#[cfg(not(feature = "zk"))]
pub mod fp64_d32_onehot;
#[cfg(feature = "zk")]
pub mod fp64_d32_onehot_zk;
#[cfg(feature = "zk")]
pub mod fp64_d32_zk;
#[cfg(not(feature = "zk"))]
pub mod fp64_d64;
#[cfg(not(feature = "zk"))]
pub mod fp64_d64_onehot;
#[cfg(feature = "zk")]
pub mod fp64_d64_onehot_zk;
#[cfg(feature = "zk")]
pub mod fp64_d64_zk;
pub mod sis_floor;

use sis_floor::SisModulusFamily;

pub fn table_entry(
    table: GeneratedScheduleTable,
    key: GeneratedScheduleKey,
) -> Option<&'static GeneratedScheduleTableEntry> {
    table.entries.iter().find(|entry| entry.key == key)
}

pub fn table_entry_envelope_up_to_num_vars(
    table: GeneratedScheduleTable,
    upper_num_vars: usize,
) -> Option<(usize, usize, usize)> {
    let mut max_n_a = 0usize;
    let mut max_n_b = 0usize;
    let mut max_n_d = 0usize;
    let mut saw_entry = false;
    for entry in table
        .entries
        .iter()
        .filter(|entry| entry.key.num_vars <= upper_num_vars)
    {
        for step in entry.steps {
            if let GeneratedStep::Fold(fold) = step {
                saw_entry = true;
                max_n_a = max_n_a.max(fold.n_a as usize);
                max_n_b = max_n_b.max(fold.n_b as usize);
                max_n_d = max_n_d.max(fold.n_d as usize);
            }
        }
    }
    saw_entry.then_some((max_n_a, max_n_b, max_n_d))
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

/// Generated schedule table for the fast-verify onehot `D=32` preset
/// (`fp128::D32OneHotFastVerify`). Produced by
/// `cargo run -p akita-config --features planner --bin gen_schedule_tables`.
pub fn fp128_d32_onehot_fast_verify_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q128,
            entries: fp128_d32_onehot_fast_verify_zk::FP128_D32_ONEHOT_FAST_VERIFY_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q128,
        entries: fp128_d32_onehot_fast_verify::FP128_D32_ONEHOT_FAST_VERIFY_SCHEDULES,
    }
}

pub fn fp128_d64_full_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q128,
            entries: fp128_d64_full_zk::FP128_D64_FULL_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q128,
        entries: fp128_d64_full::FP128_D64_FULL_SCHEDULES,
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
    fp32_d32_table,
    SisModulusFamily::Q32,
    fp32_d32,
    fp32_d32_zk,
    FP32_D32_SCHEDULES,
    FP32_D32_ZK_SCHEDULES
);
small_field_table_fn!(
    fp32_d32_onehot_table,
    SisModulusFamily::Q32,
    fp32_d32_onehot,
    fp32_d32_onehot_zk,
    FP32_D32_ONEHOT_SCHEDULES,
    FP32_D32_ONEHOT_ZK_SCHEDULES
);
small_field_table_fn!(
    fp32_d64_table,
    SisModulusFamily::Q32,
    fp32_d64,
    fp32_d64_zk,
    FP32_D64_SCHEDULES,
    FP32_D64_ZK_SCHEDULES
);
small_field_table_fn!(
    fp32_d64_onehot_table,
    SisModulusFamily::Q32,
    fp32_d64_onehot,
    fp32_d64_onehot_zk,
    FP32_D64_ONEHOT_SCHEDULES,
    FP32_D64_ONEHOT_ZK_SCHEDULES
);
small_field_table_fn!(
    fp16_d32_full_table,
    SisModulusFamily::Q16,
    fp16_d32_full,
    fp16_d32_full_zk,
    FP16_D32_FULL_SCHEDULES,
    FP16_D32_FULL_ZK_SCHEDULES
);
small_field_table_fn!(
    fp16_d32_onehot_table,
    SisModulusFamily::Q16,
    fp16_d32_onehot,
    fp16_d32_onehot_zk,
    FP16_D32_ONEHOT_SCHEDULES,
    FP16_D32_ONEHOT_ZK_SCHEDULES
);
small_field_table_fn!(
    fp16_d64_full_table,
    SisModulusFamily::Q16,
    fp16_d64_full,
    fp16_d64_full_zk,
    FP16_D64_FULL_SCHEDULES,
    FP16_D64_FULL_ZK_SCHEDULES
);
small_field_table_fn!(
    fp16_d64_onehot_table,
    SisModulusFamily::Q16,
    fp16_d64_onehot,
    fp16_d64_onehot_zk,
    FP16_D64_ONEHOT_SCHEDULES,
    FP16_D64_ONEHOT_ZK_SCHEDULES
);
small_field_table_fn!(
    fp64_d32_table,
    SisModulusFamily::Q64,
    fp64_d32,
    fp64_d32_zk,
    FP64_D32_SCHEDULES,
    FP64_D32_ZK_SCHEDULES
);
small_field_table_fn!(
    fp64_d32_onehot_table,
    SisModulusFamily::Q64,
    fp64_d32_onehot,
    fp64_d32_onehot_zk,
    FP64_D32_ONEHOT_SCHEDULES,
    FP64_D32_ONEHOT_ZK_SCHEDULES
);
small_field_table_fn!(
    fp64_d64_table,
    SisModulusFamily::Q64,
    fp64_d64,
    fp64_d64_zk,
    FP64_D64_SCHEDULES,
    FP64_D64_ZK_SCHEDULES
);
small_field_table_fn!(
    fp64_d64_onehot_table,
    SisModulusFamily::Q64,
    fp64_d64_onehot,
    fp64_d64_onehot_zk,
    FP64_D64_ONEHOT_SCHEDULES,
    FP64_D64_ONEHOT_ZK_SCHEDULES
);
