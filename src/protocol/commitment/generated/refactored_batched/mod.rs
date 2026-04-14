pub(crate) mod fp128_d128_full;
pub(crate) mod fp128_d128_onehot;
pub(crate) mod fp128_d32_full;
pub(crate) mod fp128_d32_onehot;
pub(crate) mod fp128_d64_full;
pub(crate) mod fp128_d64_onehot;

use super::GeneratedScheduleTable;

pub(crate) fn fp128_d128_full_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d128_full::FP128_D128_FULL_SCHEDULES,
    }
}

pub(crate) fn fp128_d128_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d128_onehot::FP128_D128_ONEHOT_SCHEDULES,
    }
}

pub(crate) fn fp128_d32_full_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d32_full::FP128_D32_FULL_SCHEDULES,
    }
}

pub(crate) fn fp128_d32_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d32_onehot::FP128_D32_ONEHOT_SCHEDULES,
    }
}

pub(crate) fn fp128_d64_full_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_full::FP128_D64_FULL_SCHEDULES,
    }
}

pub(crate) fn fp128_d64_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot::FP128_D64_ONEHOT_SCHEDULES,
    }
}
