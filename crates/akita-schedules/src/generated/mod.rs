#![allow(missing_docs)]

pub use akita_planner::generated::{
    GeneratedCommitmentGroupLayout, GeneratedCommitmentGroupScheduleKey, GeneratedDirectStep,
    GeneratedFoldStep, GeneratedGroupBatchScheduleTableEntry, GeneratedScheduleCatalogIdentity,
    GeneratedScheduleLookupKey, GeneratedScheduleTable, GeneratedScheduleTableEntry, GeneratedStep,
    SisModulusFamily,
};
pub use akita_planner::{ChunkedWitnessCfg, DecompositionParams, TensorChallengeShape};

// @generated schedule module wiring begin
#[cfg(feature = "fp128-d128-full")]
pub mod fp128_d128_full;
#[cfg(feature = "fp128-d128-onehot")]
pub mod fp128_d128_onehot;
#[cfg(feature = "fp128-d128-onehot")]
pub mod fp128_d128_onehot_group_batch;
#[cfg(feature = "fp128-d64-full")]
pub mod fp128_d64_full;
#[cfg(feature = "fp128-d64-full-multi-chunk")]
pub mod fp128_d64_full_multi_chunk;
#[cfg(feature = "fp128-d64-onehot")]
pub mod fp128_d64_onehot;
#[cfg(feature = "fp128-d64-onehot")]
pub mod fp128_d64_onehot_group_batch;
#[cfg(feature = "fp128-d64-onehot-multi-chunk")]
pub mod fp128_d64_onehot_multi_chunk;
#[cfg(feature = "fp128-d64-onehot-multi-chunk-w2r2")]
pub mod fp128_d64_onehot_multi_chunk_w2r2;
#[cfg(feature = "fp128-d64-onehot-multi-chunk-w4r2")]
pub mod fp128_d64_onehot_multi_chunk_w4r2;
#[cfg(feature = "fp128-d64-onehot-tensor")]
pub mod fp128_d64_onehot_tensor;
#[cfg(feature = "fp128-d64-onehot-tensor")]
pub mod fp128_d64_onehot_tensor_group_batch;
#[cfg(feature = "fp128-d64-onehot-tiered")]
pub mod fp128_d64_onehot_tiered;
#[cfg(feature = "fp32-d128-onehot")]
pub mod fp32_d128_onehot;
#[cfg(feature = "fp32-d256-onehot")]
pub mod fp32_d256_onehot;
#[cfg(feature = "fp64-d128")]
pub mod fp64_d128;
#[cfg(feature = "fp64-d128-onehot")]
pub mod fp64_d128_onehot;
#[cfg(feature = "fp64-d256-onehot")]
pub mod fp64_d256_onehot;

#[cfg(feature = "fp128-d128-full")]
pub fn fp128_d128_full_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d128_full::FP128_D128_FULL_SCHEDULES,
        group_batch_entries: &[],
        identity: fp128_d128_full::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d128-onehot")]
pub fn fp128_d128_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d128_onehot::FP128_D128_ONEHOT_SCHEDULES,
        group_batch_entries: fp128_d128_onehot_group_batch::FP128_D128_ONEHOT_GROUP_BATCH_SCHEDULES,
        identity: fp128_d128_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-full")]
pub fn fp128_d64_full_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_full::FP128_D64_FULL_SCHEDULES,
        group_batch_entries: &[],
        identity: fp128_d64_full::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-full-multi-chunk")]
pub fn fp128_d64_full_multi_chunk_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_full_multi_chunk::FP128_D64_FULL_MULTI_CHUNK_SCHEDULES,
        group_batch_entries: &[],
        identity: fp128_d64_full_multi_chunk::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot")]
pub fn fp128_d64_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot::FP128_D64_ONEHOT_SCHEDULES,
        group_batch_entries: fp128_d64_onehot_group_batch::FP128_D64_ONEHOT_GROUP_BATCH_SCHEDULES,
        identity: fp128_d64_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot-multi-chunk")]
pub fn fp128_d64_onehot_multi_chunk_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_multi_chunk::FP128_D64_ONEHOT_MULTI_CHUNK_SCHEDULES,
        group_batch_entries: &[],
        identity: fp128_d64_onehot_multi_chunk::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot-multi-chunk-w2r2")]
pub fn fp128_d64_onehot_multi_chunk_w2r2_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_multi_chunk_w2r2::FP128_D64_ONEHOT_MULTI_CHUNK_W2R2_SCHEDULES,
        group_batch_entries: &[],
        identity: fp128_d64_onehot_multi_chunk_w2r2::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot-multi-chunk-w4r2")]
pub fn fp128_d64_onehot_multi_chunk_w4r2_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_multi_chunk_w4r2::FP128_D64_ONEHOT_MULTI_CHUNK_W4R2_SCHEDULES,
        group_batch_entries: &[],
        identity: fp128_d64_onehot_multi_chunk_w4r2::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot-tensor")]
pub fn fp128_d64_onehot_tensor_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_tensor::FP128_D64_ONEHOT_TENSOR_SCHEDULES,
        group_batch_entries:
            fp128_d64_onehot_tensor_group_batch::FP128_D64_ONEHOT_TENSOR_GROUP_BATCH_SCHEDULES,
        identity: fp128_d64_onehot_tensor::CATALOG_IDENTITY,
    }
}

/// Tiered-commitment companion of [`fp128_d64_onehot_table`]: tiered entries
/// store the committed `B'`/`F` layout directly (`tier_split` + `n_f` set, with
/// `n_b` the shrunk `B'` rank), so expansion rebuilds `B'`/`F` from the stored
/// fields.
#[cfg(feature = "fp128-d64-onehot-tiered")]
pub fn fp128_d64_onehot_tiered_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_tiered::FP128_D64_ONEHOT_TIERED_SCHEDULES,
        group_batch_entries: &[],
        identity: fp128_d64_onehot_tiered::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp32-d128-onehot")]
pub fn fp32_d128_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp32_d128_onehot::FP32_D128_ONEHOT_SCHEDULES,
        group_batch_entries: &[],
        identity: fp32_d128_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp32-d256-onehot")]
pub fn fp32_d256_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp32_d256_onehot::FP32_D256_ONEHOT_SCHEDULES,
        group_batch_entries: &[],
        identity: fp32_d256_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp64-d128")]
pub fn fp64_d128_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp64_d128::FP64_D128_SCHEDULES,
        group_batch_entries: &[],
        identity: fp64_d128::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp64-d128-onehot")]
pub fn fp64_d128_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp64_d128_onehot::FP64_D128_ONEHOT_SCHEDULES,
        group_batch_entries: &[],
        identity: fp64_d128_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp64-d256-onehot")]
pub fn fp64_d256_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp64_d256_onehot::FP64_D256_ONEHOT_SCHEDULES,
        group_batch_entries: &[],
        identity: fp64_d256_onehot::CATALOG_IDENTITY,
    }
}
// @generated schedule module wiring end
