#![allow(missing_docs)]

use crate::DirectWitnessShape;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratedStage1ChallengeShape {
    Flat,
    Tensor,
}

impl From<GeneratedStage1ChallengeShape> for akita_challenges::Stage1ChallengeShape {
    fn from(shape: GeneratedStage1ChallengeShape) -> Self {
        match shape {
            GeneratedStage1ChallengeShape::Flat => Self::Flat,
            GeneratedStage1ChallengeShape::Tensor => Self::Tensor,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratedDirectWitnessShape {
    PackedDigits {
        num_elems: usize,
        bits_per_elem: u32,
    },
    #[allow(dead_code)]
    FieldElements { num_elems: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedFoldStep {
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
pub struct GeneratedDirectStep {
    pub current_w_len: usize,
    pub witness_shape: GeneratedDirectWitnessShape,
    pub entry_d: Option<u32>,
    pub entry_nb: Option<u32>,
    pub direct_bytes: usize,
    pub total_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratedStep {
    Fold(GeneratedFoldStep),
    Direct(GeneratedDirectStep),
}

pub fn generated_direct_witness_shape(shape: GeneratedDirectWitnessShape) -> DirectWitnessShape {
    match shape {
        GeneratedDirectWitnessShape::PackedDigits {
            num_elems,
            bits_per_elem,
        } => DirectWitnessShape::PackedDigits((num_elems, bits_per_elem)),
        GeneratedDirectWitnessShape::FieldElements { num_elems } => {
            DirectWitnessShape::FieldElements(num_elems)
        }
    }
}

pub fn generated_direct_log_basis(
    shape: GeneratedDirectWitnessShape,
    field_element_log_basis: u32,
) -> u32 {
    match shape {
        GeneratedDirectWitnessShape::PackedDigits { bits_per_elem, .. } => bits_per_elem,
        GeneratedDirectWitnessShape::FieldElements { .. } => field_element_log_basis,
    }
}

pub fn generated_step_current_w_len(step: &GeneratedStep) -> usize {
    match step {
        GeneratedStep::Fold(level) => level.current_w_len,
        GeneratedStep::Direct(direct) => direct.current_w_len,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedScheduleKey {
    pub max_num_vars: usize,
    pub num_vars: usize,
    pub layout_num_claims: usize,
    pub batch_num_claims: usize,
    pub batch_num_commitment_groups: usize,
    pub batch_num_points: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedScheduleTableEntry {
    pub key: GeneratedScheduleKey,
    pub total_bytes: usize,
    pub steps: &'static [GeneratedStep],
}

#[derive(Debug, Clone, Copy)]
pub struct GeneratedScheduleTable {
    pub entries: &'static [GeneratedScheduleTableEntry],
    pub stage1_challenge_shape: GeneratedStage1ChallengeShape,
}

pub mod fp128_d128_full;
pub mod fp128_d128_onehot;
pub mod fp128_d32_full;
pub mod fp128_d32_onehot;
pub mod fp128_d64_full;
pub mod fp128_d64_onehot;
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
            match step {
                GeneratedStep::Fold(fold) => {
                    saw_entry = true;
                    max_n_a = max_n_a.max(fold.n_a as usize);
                    max_n_b = max_n_b.max(fold.n_b as usize);
                    max_n_d = max_n_d.max(fold.n_d as usize);
                }
                GeneratedStep::Direct(direct) => {
                    if let Some(entry_nb) = direct.entry_nb {
                        saw_entry = true;
                        max_n_b = max_n_b.max(entry_nb as usize);
                    }
                }
            }
        }
    }
    saw_entry.then_some((max_n_a, max_n_b, max_n_d))
}

pub fn fp128_d32_full_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d32_full::FP128_D32_FULL_SCHEDULES,
        stage1_challenge_shape: GeneratedStage1ChallengeShape::Tensor,
    }
}

pub fn fp128_d32_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d32_onehot::FP128_D32_ONEHOT_SCHEDULES,
        stage1_challenge_shape: GeneratedStage1ChallengeShape::Tensor,
    }
}

pub fn fp128_d128_full_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d128_full::FP128_D128_FULL_SCHEDULES,
        stage1_challenge_shape: GeneratedStage1ChallengeShape::Tensor,
    }
}

pub fn fp128_d128_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d128_onehot::FP128_D128_ONEHOT_SCHEDULES,
        stage1_challenge_shape: GeneratedStage1ChallengeShape::Tensor,
    }
}

pub fn fp128_d64_full_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_full::FP128_D64_FULL_SCHEDULES,
        stage1_challenge_shape: GeneratedStage1ChallengeShape::Tensor,
    }
}

pub fn fp128_d64_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot::FP128_D64_ONEHOT_SCHEDULES,
        stage1_challenge_shape: GeneratedStage1ChallengeShape::Tensor,
    }
}
