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
    pub max_num_vars: usize,
    pub num_vars: usize,
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
    pub entries: &'static [GeneratedScheduleTableEntry],
}

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
pub mod fp128_d64_full;
#[cfg(feature = "zk")]
pub mod fp128_d64_full_zk;
#[cfg(not(feature = "zk"))]
pub mod fp128_d64_onehot;
#[cfg(feature = "zk")]
pub mod fp128_d64_onehot_zk;
pub mod sis_floor;

pub fn table_entry(
    table: GeneratedScheduleTable,
    key: GeneratedScheduleKey,
) -> Option<&'static GeneratedScheduleTableEntry> {
    table.entries.iter().find(|entry| entry.key == key)
}

pub fn table_entry_envelope_for_max_num_vars(
    table: GeneratedScheduleTable,
    max_num_vars: usize,
) -> Option<(usize, usize, usize)> {
    let mut max_n_a = 0usize;
    let mut max_n_b = 0usize;
    let mut max_n_d = 0usize;
    let mut saw_entry = false;
    for entry in table
        .entries
        .iter()
        .filter(|entry| entry.key.max_num_vars == max_num_vars)
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
            entries: fp128_d32_full_zk::FP128_D32_FULL_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        entries: fp128_d32_full::FP128_D32_FULL_SCHEDULES,
    }
}

pub fn fp128_d32_onehot_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            entries: fp128_d32_onehot_zk::FP128_D32_ONEHOT_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        entries: fp128_d32_onehot::FP128_D32_ONEHOT_SCHEDULES,
    }
}

pub fn fp128_d128_full_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            entries: fp128_d128_full_zk::FP128_D128_FULL_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        entries: fp128_d128_full::FP128_D128_FULL_SCHEDULES,
    }
}

pub fn fp128_d128_onehot_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            entries: fp128_d128_onehot_zk::FP128_D128_ONEHOT_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        entries: fp128_d128_onehot::FP128_D128_ONEHOT_SCHEDULES,
    }
}

pub fn fp128_d64_full_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            entries: fp128_d64_full_zk::FP128_D64_FULL_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        entries: fp128_d64_full::FP128_D64_FULL_SCHEDULES,
    }
}

pub fn fp128_d64_onehot_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            entries: fp128_d64_onehot_zk::FP128_D64_ONEHOT_ZK_SCHEDULES,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        entries: fp128_d64_onehot::FP128_D64_ONEHOT_SCHEDULES,
    }
}
