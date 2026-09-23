//! Stable policy identity shared by trusted artifacts and offline generation.

use akita_types::digest_descriptor_bytes;

use crate::{PlannerPolicy, RingDimensionScheduleMode};

/// Fixed-width digest of every planner-policy field that affects an admitted row.
pub fn policy_digest(policy: &PlannerPolicy) -> [u8; 32] {
    let PlannerPolicy {
        cost_model,
        selective_l2_response_model,
        selection_policy,
        recursive_split_search_policy,
        recursive_setup_search_policy,
        setup_field_budget,
        min_offloaded_witness_contraction,
        ring_dimension_schedule_mode,
        decomposition,
        sis_modulus_profile,
        sis_security_policy,
        sis_table_digest,
        sis_l2_table_digest,
        claim_ext_degree,
        chal_ext_degree,
        inner_basis_range,
        opening_basis_range,
        witness_chunk,
        recursive_setup_planning,
    } = *policy;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"AKITA-PLANNER-POLICY-V1");
    write_u64(&mut bytes, sis_modulus_profile_tag(sis_modulus_profile));
    write_u64(&mut bytes, u64::from(sis_security_policy.tag()));
    bytes.extend_from_slice(&sis_table_digest.0);
    bytes.extend_from_slice(&sis_l2_table_digest.0);
    write_u64(&mut bytes, u64::from(selective_l2_response_model.tag()));
    write_ring_dimension_schedule_mode(&mut bytes, ring_dimension_schedule_mode);
    write_decomposition(&mut bytes, decomposition);
    write_u64(&mut bytes, claim_ext_degree as u64);
    write_u64(&mut bytes, chal_ext_degree as u64);
    write_u64(&mut bytes, u64::from(inner_basis_range.0));
    write_u64(&mut bytes, u64::from(inner_basis_range.1));
    write_u64(&mut bytes, u64::from(opening_basis_range.0));
    write_u64(&mut bytes, u64::from(opening_basis_range.1));
    write_u64(&mut bytes, witness_chunk.num_chunks as u64);
    write_u64(&mut bytes, witness_chunk.num_activated_levels as u64);
    write_u64(&mut bytes, u64::from(recursive_setup_planning));
    write_u64(&mut bytes, u64::from(cost_model.tag()));
    write_u64(&mut bytes, u64::from(selection_policy.tag()));
    write_u64(&mut bytes, u64::from(recursive_split_search_policy.tag()));
    write_u64(&mut bytes, u64::from(recursive_setup_search_policy.tag()));
    write_optional_usize(&mut bytes, setup_field_budget);
    write_u64(&mut bytes, min_offloaded_witness_contraction as u64);
    digest_descriptor_bytes(&bytes)
}

fn sis_modulus_profile_tag(family: akita_types::SisModulusProfileId) -> u64 {
    match family {
        akita_types::SisModulusProfileId::Q32Offset99 => 0,
        akita_types::SisModulusProfileId::Q64Offset59 => 1,
        akita_types::SisModulusProfileId::Q128OffsetA7F7 => 2,
    }
}

fn write_ring_dimension_schedule_mode(bytes: &mut Vec<u8>, mode: RingDimensionScheduleMode) {
    match mode {
        RingDimensionScheduleMode::UniformDimension { ring_dimension } => {
            write_u64(bytes, 0);
            write_u64(bytes, ring_dimension as u64);
        }
        RingDimensionScheduleMode::AdaptiveDimension {
            num_search_levels,
            suffix_dimensions,
            potential_a_dimensions,
            potential_b_dimensions,
            potential_d_dimensions,
        } => {
            write_u64(bytes, 1);
            write_u64(bytes, num_search_levels as u64);
            write_u64(bytes, suffix_dimensions.len() as u64);
            for &dimension in suffix_dimensions {
                write_u64(bytes, dimension as u64);
            }
            for dimensions in [
                potential_a_dimensions,
                potential_b_dimensions,
                potential_d_dimensions,
            ] {
                write_u64(bytes, dimensions.len() as u64);
                for &dimension in dimensions {
                    write_u64(bytes, dimension as u64);
                }
            }
        }
    }
}

fn write_decomposition(bytes: &mut Vec<u8>, decomposition: akita_types::DecompositionParams) {
    write_u64(bytes, u64::from(decomposition.log_basis));
    write_u64(bytes, u64::from(decomposition.log_commit_bound));
    match decomposition.log_open_bound {
        Some(value) => {
            write_u64(bytes, 1);
            write_u64(bytes, u64::from(value));
        }
        None => write_u64(bytes, 0),
    }
}

fn write_optional_usize(bytes: &mut Vec<u8>, value: Option<usize>) {
    match value {
        Some(value) => {
            write_u64(bytes, 1);
            write_u64(bytes, value as u64);
        }
        None => write_u64(bytes, 0),
    }
}

fn write_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use crate::SelectionPolicyId;

    #[test]
    fn current_selection_objectives_do_not_reuse_retired_identity_tags() {
        assert_eq!(SelectionPolicyId::MinEstimatedProofPayloadV2.tag(), 4);
        assert_eq!(SelectionPolicyId::MinFirstDirectSetupThenPayloadV2.tag(), 5);
        assert_eq!(
            SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3.tag(),
            6
        );
        assert!(![1, 2, 3].contains(&SelectionPolicyId::MinEstimatedProofPayloadV2.tag()));
        assert!(![1, 2, 3].contains(&SelectionPolicyId::MinFirstDirectSetupThenPayloadV2.tag()));
        assert!(![1, 2, 3, 4, 5].contains(
            &SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3.tag()
        ));
    }
}
