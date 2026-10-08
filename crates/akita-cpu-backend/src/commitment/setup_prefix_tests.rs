use super::*;
use akita_params::test_fixtures::{retarget_group_role_dims_wide, sample_level_params};
use akita_params::*;
use akita_types::*;
use jolt_field::Zero;
#[test]
fn prover_registry_duplicate_insert_does_not_replace_existing_slot() {
    use jolt_field::Prime32Offset99 as F;

    let natural_len = 64;
    let mut level_params = sample_level_params();
    retarget_group_role_dims_wide(&mut level_params, 64, 64, 1024);
    let commitment_params =
        setup_prefix_precommitted_params(&level_params, natural_len).expect("prefix params");
    let id = scheduled_setup_prefix(natural_len, commitment_params)
        .slot_id()
        .expect("setup prefix group");
    let slot = || {
        let inner_rows =
            RingVec::from_coeffs_with_ring_dim(vec![F::zero(); 64], 64).expect("inner rows");
        let matrix = &id.commitment_profile.outer.matrix;
        let plan = akita_params::CompressionChainPlan::for_complete_source(
            matrix.sis_modulus_profile(),
            matrix.output_rank() * matrix.ring_dimension(),
        )
        .expect("compression plan");
        let stages = plan
            .maps()
            .iter()
            .map(|map| {
                akita_params::PackedNegativeBinary::from_bytes(
                    *map,
                    vec![0; map.packed_digit_bytes()],
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("packed stages");
        let witness = akita_params::CompressionChainWitness::new(plan.clone(), stages)
            .expect("compression witness");
        let quotients = plan
            .maps()
            .iter()
            .map(|map| {
                RingVec::from_coeffs_with_ring_dim(
                    vec![F::zero(); map.output_coefficients()],
                    map.ring_dimension(),
                )
                .expect("quotient")
            })
            .collect::<Vec<_>>();
        let hint = PortableCommitmentHandle::<F>::singleton_with_outer_compression(
            inner_rows, &witness, &quotients,
        )
        .expect("hint");
        SetupPrefixSlot {
            id: id.clone(),
            commitment: SetupPrefixPublicCommitment {
                rows: vec![RingVec::from_coeffs(vec![
                    F::zero();
                    plan.terminal_coefficients()
                ])],
            },
            hint,
        }
    };

    let mut registry = SetupPrefixProverRegistry::<F>::new([0; 32].into());
    registry.insert(slot()).expect("first insert");
    registry
        .insert(slot())
        .expect_err("duplicate insert must fail");

    assert_eq!(registry.len(), 1);

    let mut missing_stages = slot();
    missing_stages.hint = PortableCommitmentHandle::singleton(
        RingVec::from_coeffs_with_ring_dim(vec![F::zero(); 64], 64).expect("inner rows"),
    )
    .expect("uncompressed hint");
    let mut missing_registry = SetupPrefixProverRegistry::<F>::new([0; 32].into());
    missing_registry
        .insert(missing_stages)
        .expect_err("setup-prefix hints must retain both compression stages");
}
