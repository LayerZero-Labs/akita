#![allow(missing_docs)]

pub use akita_planner::generated::{
    GeneratedDirectStep, GeneratedFoldStep, GeneratedFoldStepWithSetupMetadata,
    GeneratedScheduleCatalogIdentity, GeneratedScheduleTable, GeneratedScheduleTableEntry,
    GeneratedSetupPrefixGroup, GeneratedStep, PolynomialGroupLayout, PrecommittedGroupParams,
    SetupContributionMode, SisModulusProfileId, SisTableDigest,
};
pub use akita_planner::{
    ChunkedWitnessCfg, DecompositionParams, SisSecurityPolicyId, TensorChallengeShape,
};

// @generated schedule module wiring begin
#[cfg(feature = "fp128-d128-full")]
pub mod fp128_d128_full;
#[cfg(feature = "fp128-d128-onehot")]
pub mod fp128_d128_onehot;
#[cfg(feature = "fp128-d64-full")]
pub mod fp128_d64_full;
#[cfg(feature = "fp128-d64-full-multi-chunk")]
pub mod fp128_d64_full_multi_chunk;
#[cfg(feature = "fp128-d64-onehot")]
pub mod fp128_d64_onehot;
#[cfg(feature = "fp128-d64-onehot-multi-chunk")]
pub mod fp128_d64_onehot_multi_chunk;
#[cfg(feature = "fp128-d64-onehot-multi-chunk-w2r2")]
pub mod fp128_d64_onehot_multi_chunk_w2r2;
#[cfg(feature = "fp128-d64-onehot-multi-chunk-w4r2")]
pub mod fp128_d64_onehot_multi_chunk_w4r2;
#[cfg(feature = "fp128-d64-onehot-recursive")]
pub mod fp128_d64_onehot_recursive;
#[cfg(feature = "fp128-d64-onehot-tensor")]
pub mod fp128_d64_onehot_tensor;
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
        identity: fp128_d128_full::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d128-onehot")]
pub fn fp128_d128_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d128_onehot::FP128_D128_ONEHOT_SCHEDULES,
        identity: fp128_d128_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-full")]
pub fn fp128_d64_full_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_full::FP128_D64_FULL_SCHEDULES,
        identity: fp128_d64_full::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-full-multi-chunk")]
pub fn fp128_d64_full_multi_chunk_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_full_multi_chunk::FP128_D64_FULL_MULTI_CHUNK_SCHEDULES,
        identity: fp128_d64_full_multi_chunk::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot")]
pub fn fp128_d64_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot::FP128_D64_ONEHOT_SCHEDULES,
        identity: fp128_d64_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot-multi-chunk")]
pub fn fp128_d64_onehot_multi_chunk_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_multi_chunk::FP128_D64_ONEHOT_MULTI_CHUNK_SCHEDULES,
        identity: fp128_d64_onehot_multi_chunk::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot-multi-chunk-w2r2")]
pub fn fp128_d64_onehot_multi_chunk_w2r2_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_multi_chunk_w2r2::FP128_D64_ONEHOT_MULTI_CHUNK_W2R2_SCHEDULES,
        identity: fp128_d64_onehot_multi_chunk_w2r2::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot-multi-chunk-w4r2")]
pub fn fp128_d64_onehot_multi_chunk_w4r2_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_multi_chunk_w4r2::FP128_D64_ONEHOT_MULTI_CHUNK_W4R2_SCHEDULES,
        identity: fp128_d64_onehot_multi_chunk_w4r2::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot-recursive")]
pub fn fp128_d64_onehot_recursive_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_recursive::FP128_D64_ONEHOT_RECURSIVE_SCHEDULES,
        identity: fp128_d64_onehot_recursive::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot-tensor")]
pub fn fp128_d64_onehot_tensor_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_tensor::FP128_D64_ONEHOT_TENSOR_SCHEDULES,
        identity: fp128_d64_onehot_tensor::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp32-d128-onehot")]
pub fn fp32_d128_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp32_d128_onehot::FP32_D128_ONEHOT_SCHEDULES,
        identity: fp32_d128_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp32-d256-onehot")]
pub fn fp32_d256_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp32_d256_onehot::FP32_D256_ONEHOT_SCHEDULES,
        identity: fp32_d256_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp64-d128")]
pub fn fp64_d128_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp64_d128::FP64_D128_SCHEDULES,
        identity: fp64_d128::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp64-d128-onehot")]
pub fn fp64_d128_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp64_d128_onehot::FP64_D128_ONEHOT_SCHEDULES,
        identity: fp64_d128_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp64-d256-onehot")]
pub fn fp64_d256_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp64_d256_onehot::FP64_D256_ONEHOT_SCHEDULES,
        identity: fp64_d256_onehot::CATALOG_IDENTITY,
    }
}
// @generated schedule module wiring end
