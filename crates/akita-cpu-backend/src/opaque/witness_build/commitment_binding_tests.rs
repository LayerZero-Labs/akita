use super::CpuCommitmentMaterialHandle;
use akita_types::RingVec;
use jolt_field::{One, Prime128OffsetA7F7 as F, Zero};

#[test]
fn retained_material_rejects_substituted_public_rows() {
    let plan = crate::opaque::CommitInnerPlan {
        ring_dimension: 4,
        num_live_blocks: 1,
        n_a: 1,
        num_positions_per_block: 1,
        num_digits_inner: 1,
        log_basis_inner: 1,
    };
    let rows = RingVec::from_coeffs_with_ring_dim(vec![F::one(); 4], 4).unwrap();
    let inner =
        crate::commitment::InnerRelationStateMaterial::new(&plan, 1, vec![rows.clone()]).unwrap();
    let mut material = CpuCommitmentMaterialHandle {
        binding: crate::opaque::OperationBinding::legacy_unscoped(),
        inner,
        compression: None,
        source_count: 1,
        commitment_id: None,
        public_commitment: None,
    };
    assert!(material.validate_public_commitment(&rows).is_err());
    material.bind_commitment(1, Some(rows.clone()));
    material.validate_public_commitment(&rows).unwrap();
    material
        .validate_public_commitment(&RingVec::from_coeffs(rows.coeffs().to_vec()))
        .unwrap();
    assert!(material
        .validate_public_commitment(&RingVec::from_coeffs(vec![F::zero(); 4]))
        .is_err());
    assert!(material
        .validate_public_commitment(
            &RingVec::from_coeffs_with_ring_dim(rows.coeffs().to_vec(), 2).unwrap(),
        )
        .is_err());
}
