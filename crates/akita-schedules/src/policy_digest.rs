//! Stable policy identity shared by trusted artifacts and offline generation.

use crate::{PlannerPolicy, RingDimensionScheduleMode};

/// Fixed-width digest of every planner-policy field that affects an admitted row.
pub fn policy_digest(policy: &PlannerPolicy) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut h = Fnv64::new();
    h.write_u64(sis_modulus_profile_tag(policy.sis_modulus_profile));
    h.write_u64(u64::from(policy.sis_security_policy.tag()));
    h.write_bytes(&policy.sis_table_digest.0);
    h.write_bytes(&policy.sis_l2_table_digest.0);
    h.write_u64(u64::from(policy.selective_l2_response_model.tag()));
    write_ring_dimension_schedule_mode(&mut h, policy.ring_dimension_schedule_mode);
    write_decomposition(&mut h, policy.decomposition);
    h.write_u64(policy.claim_ext_degree as u64);
    h.write_u64(policy.chal_ext_degree as u64);
    h.write_u64(u64::from(policy.inner_basis_range.0));
    h.write_u64(u64::from(policy.inner_basis_range.1));
    h.write_u64(u64::from(policy.opening_basis_range.0));
    h.write_u64(u64::from(policy.opening_basis_range.1));
    h.write_u64(policy.witness_chunk.num_chunks as u64);
    h.write_u64(policy.witness_chunk.num_activated_levels as u64);
    h.write_u64(u64::from(policy.recursive_setup_planning));
    h.write_u64(u64::from(policy.cost_model.tag()));
    h.write_u64(u64::from(policy.selection_policy.tag()));
    h.write_u64(u64::from(policy.recursive_split_search_policy.tag()));
    h.write_u64(u64::from(policy.recursive_setup_search_policy.tag()));
    write_optional_usize(&mut h, policy.setup_field_budget);
    h.write_u64(policy.min_offloaded_witness_contraction as u64);
    let digest = h.finish();
    out[..8].copy_from_slice(&digest.to_le_bytes());
    out
}

fn sis_modulus_profile_tag(family: akita_types::SisModulusProfileId) -> u64 {
    match family {
        akita_types::SisModulusProfileId::Q32Offset99 => 0,
        akita_types::SisModulusProfileId::Q64Offset59 => 1,
        akita_types::SisModulusProfileId::Q128OffsetA7F7 => 2,
    }
}

fn write_ring_dimension_schedule_mode(h: &mut Fnv64, mode: RingDimensionScheduleMode) {
    match mode {
        RingDimensionScheduleMode::UniformDimension { ring_dimension } => {
            h.write_u64(0);
            h.write_u64(ring_dimension as u64);
        }
        RingDimensionScheduleMode::AdaptiveDimension {
            num_search_levels,
            suffix_dimensions,
            potential_a_dimensions,
            potential_b_dimensions,
            potential_d_dimensions,
        } => {
            h.write_u64(1);
            h.write_u64(num_search_levels as u64);
            h.write_u64(suffix_dimensions.len() as u64);
            for &dimension in suffix_dimensions {
                h.write_u64(dimension as u64);
            }
            for dimensions in [
                potential_a_dimensions,
                potential_b_dimensions,
                potential_d_dimensions,
            ] {
                h.write_u64(dimensions.len() as u64);
                for &dimension in dimensions {
                    h.write_u64(dimension as u64);
                }
            }
        }
    }
}

fn write_decomposition(h: &mut Fnv64, decomposition: akita_types::DecompositionParams) {
    h.write_u64(u64::from(decomposition.log_basis));
    h.write_u64(u64::from(decomposition.log_commit_bound));
    match decomposition.log_open_bound {
        Some(value) => {
            h.write_u64(1);
            h.write_u64(u64::from(value));
        }
        None => h.write_u64(0),
    }
}

fn write_optional_usize(h: &mut Fnv64, value: Option<usize>) {
    match value {
        Some(value) => {
            h.write_u64(1);
            h.write_u64(value as u64);
        }
        None => h.write_u64(0),
    }
}

struct Fnv64 {
    state: u64,
}

impl Fnv64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    const fn new() -> Self {
        Self {
            state: Self::OFFSET,
        }
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state ^= u64::from(*byte);
            self.state = self.state.wrapping_mul(Self::PRIME);
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.write_bytes(&value.to_le_bytes());
    }

    const fn finish(self) -> u64 {
        self.state
    }
}
