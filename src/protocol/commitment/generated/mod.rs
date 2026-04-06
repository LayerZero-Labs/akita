#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GeneratedDirectWitnessShape {
    PackedDigits {
        num_elems: usize,
        bits_per_elem: u32,
    },
    FieldElements {
        num_elems: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GeneratedFoldStep {
    pub current_w_len: usize,
    pub d: u32,
    pub log_basis: u32,
    pub challenge_l1_mass: usize,
    pub m_vars: u32,
    pub r_vars: u32,
    pub n_a: u32,
    pub n_b: u32,
    pub n_d: u32,
    pub delta_open: usize,
    pub delta_fold: usize,
    pub delta_commit: usize,
    pub w_ring: usize,
    pub next_w_len: usize,
    pub level_bytes: usize,
    pub label: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GeneratedDirectStep {
    pub current_w_len: usize,
    pub witness_shape: GeneratedDirectWitnessShape,
    pub entry_d: Option<u32>,
    pub entry_nb: Option<u32>,
    pub direct_bytes: usize,
    pub total_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GeneratedStep {
    Fold(GeneratedFoldStep),
    Direct(GeneratedDirectStep),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GeneratedScheduleTableEntry {
    pub max_num_vars: usize,
    pub total_bytes: usize,
    pub steps: &'static [GeneratedStep],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GeneratedScheduleTable {
    pub entries: &'static [GeneratedScheduleTableEntry],
}

pub(crate) mod fp128_d128_full;
pub(crate) mod fp128_d128_onehot;
pub(crate) mod fp128_d32_full;
pub(crate) mod fp128_d32_onehot;
pub(crate) mod fp128_d64_full;
pub(crate) mod fp128_d64_onehot;
pub(crate) mod sis_floor;

pub(crate) fn table_entry(
    table: GeneratedScheduleTable,
    max_num_vars: usize,
) -> Option<&'static GeneratedScheduleTableEntry> {
    table
        .entries
        .iter()
        .find(|entry| entry.max_num_vars == max_num_vars)
}

pub(crate) fn table_entry_envelope(
    table: GeneratedScheduleTable,
    max_num_vars: usize,
) -> Option<(usize, usize, usize)> {
    let entry = table_entry(table, max_num_vars)?;
    let mut max_n_a = 0usize;
    let mut max_n_b = 0usize;
    let mut max_n_d = 0usize;
    let mut saw_fold = false;
    for step in entry.steps {
        let GeneratedStep::Fold(fold) = step else {
            continue;
        };
        saw_fold = true;
        max_n_a = max_n_a.max(fold.n_a as usize);
        max_n_b = max_n_b.max(fold.n_b as usize);
        max_n_d = max_n_d.max(fold.n_d as usize);
    }
    saw_fold.then_some((max_n_a, max_n_b, max_n_d))
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
