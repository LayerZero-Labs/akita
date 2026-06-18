#![allow(missing_docs)]

pub use akita_planner::generated::{
    GeneratedDirectStep, GeneratedFoldStep, GeneratedScheduleCatalogIdentity, GeneratedScheduleKey,
    GeneratedScheduleTable, GeneratedScheduleTableEntry, GeneratedStep, SisModulusFamily,
};

// @generated schedule module wiring begin
#[cfg(not(feature = "zk"))]
#[cfg(all(feature = "fp128-d128-full"))]
pub mod fp128_d128_full;
#[cfg(feature = "zk")]
#[cfg(all(feature = "fp128-d128-full"))]
pub mod fp128_d128_full_zk;
#[cfg(not(feature = "zk"))]
#[cfg(all(feature = "fp128-d128-onehot"))]
pub mod fp128_d128_onehot;
#[cfg(feature = "zk")]
#[cfg(all(feature = "fp128-d128-onehot"))]
pub mod fp128_d128_onehot_zk;
#[cfg(not(feature = "zk"))]
#[cfg(all(feature = "fp128-d64-full"))]
pub mod fp128_d64_full;
#[cfg(feature = "zk")]
#[cfg(all(feature = "fp128-d64-full"))]
pub mod fp128_d64_full_zk;
#[cfg(not(feature = "zk"))]
#[cfg(all(feature = "fp128-d64-onehot"))]
pub mod fp128_d64_onehot;
#[cfg(not(feature = "zk"))]
#[cfg(all(feature = "fp128-d64-onehot-tensor"))]
pub mod fp128_d64_onehot_tensor;
#[cfg(feature = "zk")]
#[cfg(all(feature = "fp128-d64-onehot-tensor"))]
pub mod fp128_d64_onehot_tensor_zk;
#[cfg(not(feature = "zk"))]
#[cfg(all(feature = "fp128-d64-onehot-tiered", not(feature = "zk")))]
pub mod fp128_d64_onehot_tiered;
#[cfg(feature = "zk")]
#[cfg(all(feature = "fp128-d64-onehot"))]
pub mod fp128_d64_onehot_zk;
#[cfg(not(feature = "zk"))]
#[cfg(all(feature = "fp32-d128-onehot"))]
pub mod fp32_d128_onehot;
#[cfg(feature = "zk")]
#[cfg(all(feature = "fp32-d128-onehot"))]
pub mod fp32_d128_onehot_zk;
#[cfg(not(feature = "zk"))]
#[cfg(all(feature = "fp32-d256-onehot"))]
pub mod fp32_d256_onehot;
#[cfg(feature = "zk")]
#[cfg(all(feature = "fp32-d256-onehot"))]
pub mod fp32_d256_onehot_zk;
#[cfg(not(feature = "zk"))]
#[cfg(all(feature = "fp64-d128"))]
pub mod fp64_d128;
#[cfg(not(feature = "zk"))]
#[cfg(all(feature = "fp64-d128-onehot"))]
pub mod fp64_d128_onehot;
#[cfg(feature = "zk")]
#[cfg(all(feature = "fp64-d128-onehot"))]
pub mod fp64_d128_onehot_zk;
#[cfg(feature = "zk")]
#[cfg(all(feature = "fp64-d128"))]
pub mod fp64_d128_zk;
#[cfg(not(feature = "zk"))]
#[cfg(all(feature = "fp64-d256-onehot"))]
pub mod fp64_d256_onehot;
#[cfg(feature = "zk")]
#[cfg(all(feature = "fp64-d256-onehot"))]
pub mod fp64_d256_onehot_zk;

#[cfg(all(feature = "fp128-d128-full"))]
pub fn fp128_d128_full_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q128,
            entries: fp128_d128_full_zk::FP128_D128_FULL_ZK_SCHEDULES,
            identity: None,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q128,
        entries: fp128_d128_full::FP128_D128_FULL_SCHEDULES,
        identity: None,
    }
}

#[cfg(all(feature = "fp128-d128-onehot"))]
pub fn fp128_d128_onehot_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q128,
            entries: fp128_d128_onehot_zk::FP128_D128_ONEHOT_ZK_SCHEDULES,
            identity: None,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q128,
        entries: fp128_d128_onehot::FP128_D128_ONEHOT_SCHEDULES,
        identity: None,
    }
}

#[cfg(all(feature = "fp128-d64-full"))]
pub fn fp128_d64_full_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q128,
            entries: fp128_d64_full_zk::FP128_D64_FULL_ZK_SCHEDULES,
            identity: None,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q128,
        entries: fp128_d64_full::FP128_D64_FULL_SCHEDULES,
        identity: None,
    }
}

#[cfg(all(feature = "fp128-d64-onehot"))]
pub fn fp128_d64_onehot_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q128,
            entries: fp128_d64_onehot_zk::FP128_D64_ONEHOT_ZK_SCHEDULES,
            identity: None,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q128,
        entries: fp128_d64_onehot::FP128_D64_ONEHOT_SCHEDULES,
        identity: None,
    }
}

#[cfg(all(feature = "fp128-d64-onehot-tensor"))]
pub fn fp128_d64_onehot_tensor_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q128,
            entries: fp128_d64_onehot_tensor_zk::FP128_D64_ONEHOT_TENSOR_ZK_SCHEDULES,
            identity: None,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q128,
        entries: fp128_d64_onehot_tensor::FP128_D64_ONEHOT_TENSOR_SCHEDULES,
        identity: None,
    }
}

#[cfg(all(feature = "fp32-d128-onehot"))]
pub fn fp32_d128_onehot_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q32,
            entries: fp32_d128_onehot_zk::FP32_D128_ONEHOT_ZK_SCHEDULES,
            identity: None,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q32,
        entries: fp32_d128_onehot::FP32_D128_ONEHOT_SCHEDULES,
        identity: None,
    }
}

#[cfg(all(feature = "fp32-d256-onehot"))]
pub fn fp32_d256_onehot_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q32,
            entries: fp32_d256_onehot_zk::FP32_D256_ONEHOT_ZK_SCHEDULES,
            identity: None,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q32,
        entries: fp32_d256_onehot::FP32_D256_ONEHOT_SCHEDULES,
        identity: None,
    }
}

#[cfg(all(feature = "fp64-d128"))]
pub fn fp64_d128_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q64,
            entries: fp64_d128_zk::FP64_D128_ZK_SCHEDULES,
            identity: None,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q64,
        entries: fp64_d128::FP64_D128_SCHEDULES,
        identity: None,
    }
}

#[cfg(all(feature = "fp64-d128-onehot"))]
pub fn fp64_d128_onehot_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q64,
            entries: fp64_d128_onehot_zk::FP64_D128_ONEHOT_ZK_SCHEDULES,
            identity: None,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q64,
        entries: fp64_d128_onehot::FP64_D128_ONEHOT_SCHEDULES,
        identity: None,
    }
}

#[cfg(all(feature = "fp64-d256-onehot"))]
pub fn fp64_d256_onehot_table() -> GeneratedScheduleTable {
    #[cfg(feature = "zk")]
    {
        GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q64,
            entries: fp64_d256_onehot_zk::FP64_D256_ONEHOT_ZK_SCHEDULES,
            identity: None,
        }
    }
    #[cfg(not(feature = "zk"))]
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q64,
        entries: fp64_d256_onehot::FP64_D256_ONEHOT_SCHEDULES,
        identity: None,
    }
}

/// Tiered-commitment companion of [`fp128_d64_onehot_table`]: tiered entries
/// store the committed `B'`/`F` layout directly (`tier_split` + `n_f` set, with
/// `n_b` the shrunk `B'` rank), so expansion rebuilds `B'`/`F` from the stored
/// fields. Tiering is a non-ZK optimization, so this family has no `_zk` variant.
#[cfg(all(feature = "fp128-d64-onehot-tiered", not(feature = "zk")))]
pub fn fp128_d64_onehot_tiered_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        sis_family: SisModulusFamily::Q128,
        entries: fp128_d64_onehot_tiered::FP128_D64_ONEHOT_TIERED_SCHEDULES,
        identity: None,
    }
}
// @generated schedule module wiring end
