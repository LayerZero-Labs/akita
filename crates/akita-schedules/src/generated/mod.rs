//! Generated schedule table accessors.
//!
//! Slice 4 moves static table data here; until then these delegate to
//! `akita-planner` so the feature graph can be wired without changing behavior.

use akita_planner::GeneratedScheduleTable;

#[cfg(feature = "fp128-d128-full")]
pub fn fp128_d128_full_table() -> GeneratedScheduleTable {
    akita_planner::generated::fp128_d128_full_table()
}

#[cfg(feature = "fp128-d128-onehot")]
pub fn fp128_d128_onehot_table() -> GeneratedScheduleTable {
    akita_planner::generated::fp128_d128_onehot_table()
}

#[cfg(feature = "fp128-d64-full")]
pub fn fp128_d64_full_table() -> GeneratedScheduleTable {
    akita_planner::generated::fp128_d64_full_table()
}

#[cfg(feature = "fp128-d64-onehot")]
pub fn fp128_d64_onehot_table() -> GeneratedScheduleTable {
    akita_planner::generated::fp128_d64_onehot_table()
}

#[cfg(feature = "fp128-d64-onehot-tensor")]
pub fn fp128_d64_onehot_tensor_table() -> GeneratedScheduleTable {
    akita_planner::generated::fp128_d64_onehot_tensor_table()
}

#[cfg(all(feature = "fp128-d64-onehot-tiered", not(feature = "zk")))]
pub fn fp128_d64_onehot_tiered_table() -> GeneratedScheduleTable {
    akita_planner::generated::fp128_d64_onehot_tiered_table()
}

#[cfg(feature = "fp32-d128-onehot")]
pub fn fp32_d128_onehot_table() -> GeneratedScheduleTable {
    akita_planner::generated::fp32_d128_onehot_table()
}

#[cfg(feature = "fp32-d256-onehot")]
pub fn fp32_d256_onehot_table() -> GeneratedScheduleTable {
    akita_planner::generated::fp32_d256_onehot_table()
}

#[cfg(feature = "fp64-d128")]
pub fn fp64_d128_table() -> GeneratedScheduleTable {
    akita_planner::generated::fp64_d128_table()
}

#[cfg(feature = "fp64-d128-onehot")]
pub fn fp64_d128_onehot_table() -> GeneratedScheduleTable {
    akita_planner::generated::fp64_d128_onehot_table()
}

#[cfg(feature = "fp64-d256-onehot")]
pub fn fp64_d256_onehot_table() -> GeneratedScheduleTable {
    akita_planner::generated::fp64_d256_onehot_table()
}
