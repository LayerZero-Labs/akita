use super::*;
use akita_challenges::SparseChallengeConfig;
use akita_types::*;
use jolt_field::Zero;
fn sample_level_params() -> CommittedGroupParams {
    CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        64,
        3,
        3,
        3,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(4, 3, 2, 2, 2)
    .expect("sample level params")
}

fn retarget_group_role_dims(
    params: &mut CommittedGroupParams,
    inner_ring_dimension: usize,
    outer_ring_dimension: usize,
) {
    params.own_group_mut().opening.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(inner_ring_dimension)
            .expect("production challenge");
    let a_bound = *akita_types::sis::inner_coeff_linf_bounds(
        params.inner().matrix.sis_modulus_profile(),
        u32::try_from(inner_ring_dimension).expect("test ring dimension"),
    )
    .first()
    .expect("retargeted exact A bounds");
    let inner = params.inner().matrix;
    params.own_group_mut().profile.inner.matrix =
        akita_types::InnerCommitMatrixParams::try_new_with_min_rank(
            akita_types::SisTableKey {
                policy: inner.security_policy(),
                table_digest: inner
                    .sis_table_key()
                    .expect("L infinity test matrix")
                    .table_digest,
                modulus_profile: inner.sis_modulus_profile(),
                role: akita_types::sis::SisMatrixRole::Inner,
                ring_dimension: u32::try_from(inner_ring_dimension).expect("test ring dimension"),
                coeff_linf_bound: a_bound,
            },
            inner.input_width(),
        )
        .expect("audited retargeted A matrix");
    let inner_output_rank = params.inner().matrix.output_rank();
    let projected_outer_width = inner_output_rank
        .checked_mul(params.outer().digits.num_digits)
        .and_then(|width| width.checked_mul(params.blocks().live_blocks))
        .and_then(|width| width.checked_mul(inner_ring_dimension / outer_ring_dimension))
        .expect("retargeted B width");
    let outer = params.outer().matrix;
    params.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::try_new_with_min_rank(
        akita_types::SisTableKey {
            policy: outer.security_policy(),
            table_digest: outer.sis_table_key().table_digest,
            modulus_profile: outer.sis_modulus_profile(),
            role: akita_types::sis::SisMatrixRole::Outer,
            ring_dimension: u32::try_from(outer_ring_dimension).expect("test ring dimension"),
            coeff_linf_bound: 3,
        },
        projected_outer_width,
    )
    .expect("audited retargeted B matrix");
}

fn retarget_group_role_dims_wide(
    params: &mut CommittedGroupParams,
    inner_ring_dimension: usize,
    outer_ring_dimension: usize,
    min_input_width: usize,
) {
    retarget_group_role_dims(params, inner_ring_dimension, outer_ring_dimension);
    let inner = params.inner().matrix;
    params.own_group_mut().profile.inner.matrix =
        akita_types::InnerCommitMatrixParams::try_new_with_min_rank(
            inner.sis_table_key().expect("L infinity test matrix"),
            inner.input_width().max(min_input_width),
        )
        .expect("wide audited A matrix");
    let inner_output_rank = params.inner().matrix.output_rank();
    let projected_outer_width = inner_output_rank
        .checked_mul(params.outer().digits.num_digits)
        .and_then(|width| width.checked_mul(params.blocks().live_blocks))
        .and_then(|width| width.checked_mul(inner_ring_dimension / outer_ring_dimension))
        .expect("wide retargeted B width")
        .max(min_input_width);
    let outer = params.outer().matrix;
    params.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::try_new_with_min_rank(
        akita_types::SisTableKey {
            policy: outer.security_policy(),
            table_digest: outer.sis_table_key().table_digest,
            modulus_profile: outer.sis_modulus_profile(),
            role: akita_types::sis::SisMatrixRole::Outer,
            ring_dimension: u32::try_from(outer_ring_dimension).expect("test ring dimension"),
            coeff_linf_bound: 3,
        },
        projected_outer_width,
    )
    .expect("wide audited B matrix");
}

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
        let plan = akita_types::CompressionChainPlan::for_complete_source(
            matrix.sis_modulus_profile(),
            matrix.output_rank() * matrix.ring_dimension(),
        )
        .expect("compression plan");
        let stages = plan
            .maps()
            .iter()
            .map(|map| {
                akita_types::PackedNegativeBinary::from_bytes(
                    *map,
                    vec![0; map.packed_digit_bytes()],
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("packed stages");
        let witness = akita_types::CompressionChainWitness::new(plan.clone(), stages)
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
