use super::*;
use crate::test_fixtures::{
    prefix_eligible_level_params, retarget_group_role_dims, retarget_group_role_dims_wide,
    sample_level_params,
};
use crate::{
    CommittedGroupParams, GroupCommitPhaseParams, GroupOpenPhaseParams, OpeningClaimsLayout,
    OpeningMethod, OuterCommitMatrixParams, PolynomialGroupLayout, SisModulusProfileId,
};
use akita_challenges::SparseChallengeConfig;
use std::collections::{BTreeSet, HashSet};

fn precommitted_group(
    params: &CommittedGroupParams,
    group: PolynomialGroupLayout,
) -> GroupOpenPhaseParams {
    GroupOpenPhaseParams {
        setup_natural_len: None,
        profile: GroupCommitPhaseParams::from_params_unchecked_for_test(group, params),
        opening: crate::GroupOpeningPlan::evaluation_trace(
            params.fold_challenge_config(),
            params.open().digits.log_basis,
            params.open().digits.num_digits,
            params.num_digits_fold(),
        ),
    }
}

#[test]
fn active_setup_field_len_matches_packed_role_maximum() {
    let lp = sample_level_params();
    let opening_batch = OpeningClaimsLayout::new(5, 3).expect("opening batch");
    let w_a = lp.blocks().positions_per_block * lp.inner().digits.num_digits;
    let w_b = lp.outer().matrix.input_width();
    let w_d = opening_batch.num_total_polynomials()
        * lp.blocks().live_blocks
        * lp.open().digits.num_digits;
    let expected_ring_slots = lp
        .inner()
        .matrix
        .output_rank()
        .checked_mul(w_a)
        .unwrap()
        .max(lp.outer().matrix.output_rank().checked_mul(w_b).unwrap())
        .max(lp.open().matrix.output_rank().checked_mul(w_d).unwrap());
    let geometry =
        active_setup_projection_geometry(&lp, &opening_batch).expect("projection geometry");
    assert_eq!(geometry.required(), expected_ring_slots);
    let dims = lp.role_dims();
    let base_d = dims.d_a().min(dims.d_b()).min(dims.d_d());
    assert_eq!(
        active_setup_field_len(&lp, &opening_batch).expect("field len"),
        expected_ring_slots * base_d
    );
}

#[test]
fn active_setup_field_len_prices_one_physical_sliced_b_matrix() {
    let mut lp = CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        64,
        3,
        3,
        3,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(4, 32, 2, 2, 2)
    .expect("sliced level params");
    let opening_batch = OpeningClaimsLayout::new(7, 3).expect("opening batch");
    lp.own_group_mut().profile.outer_slice_count = crate::CommitmentSliceCount::FOUR;
    let slice_geometry = crate::CommitmentSliceGeometry::try_new(
        lp.outer_slice_count(),
        lp.blocks().live_blocks,
        opening_batch.num_total_polynomials(),
        lp.inner().matrix.output_rank(),
        lp.outer().digits.num_digits,
        lp.inner().matrix.ring_dimension(),
        lp.outer().matrix.ring_dimension(),
    )
    .expect("slice geometry");
    let outer = lp.outer().matrix;
    lp.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        slice_geometry.physical_input_width(),
        outer.coeff_linf_bound(),
        outer.ring_dimension(),
    );

    let geometry =
        active_setup_projection_geometry(&lp, &opening_batch).expect("projection geometry");
    let base_d = geometry.base_ring_dim();
    let expected_b_projection = lp.outer().matrix.output_rank()
        * slice_geometry.physical_input_width()
        * (lp.outer().matrix.ring_dimension() / base_d);
    assert!(geometry.required() >= expected_b_projection);

    let logical_unsliced_projection = lp
        .outer_slice_count()
        .logical_output_rows(lp.outer().matrix.output_rank())
        .expect("logical B rows")
        * opening_batch.num_total_polynomials()
        * lp.inner().matrix.output_rank()
        * lp.blocks().live_blocks
        * lp.outer().digits.num_digits
        * (lp.outer().matrix.ring_dimension() / base_d);
    assert!(geometry.required() < logical_unsliced_projection);
}

#[test]
fn active_setup_field_len_projects_each_group_at_its_native_dimensions() {
    let mut final_params = sample_level_params();
    retarget_group_role_dims(&mut final_params, 128, 64);

    let mut precommitted_params = sample_level_params();
    retarget_group_role_dims(&mut precommitted_params, 256, 128);
    let precommitted_layout = PolynomialGroupLayout::new(5, 1);
    final_params
        .set_precommitted_groups(vec![precommitted_group(
            &precommitted_params,
            precommitted_layout,
        )])
        .unwrap();
    let opening_batch = OpeningClaimsLayout::from_root_groups(
        &[precommitted_layout],
        PolynomialGroupLayout::new(5, 3),
    )
    .expect("heterogeneous opening batch");

    let base_ring_dimension = 64usize;
    let mut expected_a_projection = 0usize;
    let mut expected_b_projection = 0usize;
    for group_index in 0..opening_batch.num_groups() {
        let group_params = final_params
            .group_params(&opening_batch, group_index)
            .expect("group params");
        let dims = final_params
            .group_role_dims(&opening_batch, group_index)
            .expect("group role dimensions");
        let a_cols = group_params.num_positions_per_block() * group_params.num_digits_inner();
        let b_cols = group_params.b_col_len();
        expected_a_projection = expected_a_projection
            .max(group_params.a_rows_len() * a_cols * (dims.d_a() / base_ring_dimension));
        expected_b_projection = expected_b_projection
            .max(group_params.b_rows_len() * b_cols * (dims.d_b() / base_ring_dimension));
    }
    let expected_d_projection = final_params.open().matrix.output_rank()
        * final_params.open().matrix.input_width()
        * (final_params.role_dims().d_d() / base_ring_dimension);
    let expected_ring_slots = expected_a_projection
        .max(expected_b_projection)
        .max(expected_d_projection);
    let geometry = active_setup_projection_geometry(&final_params, &opening_batch)
        .expect("heterogeneous projection geometry");

    assert_eq!(geometry.base_ring_dim(), base_ring_dimension);
    assert_eq!(geometry.required(), expected_ring_slots);
    assert_eq!(
        active_setup_field_len(&final_params, &opening_batch).expect("active setup field length"),
        expected_ring_slots * base_ring_dimension
    );
}

#[test]
fn setup_prefix_slot_identity_binds_outer_slice_count() {
    let mut params = prefix_eligible_level_params();
    retarget_group_role_dims_wide(&mut params, 64, 64, 1024);
    let unsliced = setup_prefix_precommitted_params(&params, 1024).expect("unsliced prefix");

    let mut sliced_params = params;
    sliced_params.own_group_mut().profile.outer_slice_count = crate::CommitmentSliceCount::TWO;
    let sliced = setup_prefix_precommitted_params(&sliced_params, 1024).expect("sliced prefix");

    let unsliced_id = scheduled_setup_prefix(777, unsliced);
    let sliced_id = scheduled_setup_prefix(777, sliced);
    assert_ne!(unsliced_id.slot_id(), sliced_id.slot_id());
    let mut unsliced_bytes = Vec::new();
    unsliced_id.append_descriptor_bytes(&mut unsliced_bytes);
    let mut sliced_bytes = Vec::new();
    sliced_id.append_descriptor_bytes(&mut sliced_bytes);
    assert_ne!(unsliced_bytes, sliced_bytes);
}

#[test]
fn setup_prefix_slot_identity_excludes_consuming_opening_plan() {
    let mut params = prefix_eligible_level_params();
    retarget_group_role_dims_wide(&mut params, 64, 64, 1024);
    let evaluation_trace = setup_prefix_precommitted_params(&params, 1024).expect("prefix params");
    let mut subring_packing = evaluation_trace;
    subring_packing.opening.opening_method = OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 64,
    };

    let evaluation_trace = scheduled_setup_prefix(777, evaluation_trace);
    let subring_packing = scheduled_setup_prefix(777, subring_packing);
    let evaluation_trace_id = evaluation_trace.slot_id().expect("setup prefix group");
    let subring_packing_id = subring_packing.slot_id().expect("setup prefix group");

    assert_eq!(evaluation_trace_id, subring_packing_id);
    assert_eq!(
        BTreeSet::from([evaluation_trace_id.clone(), subring_packing_id.clone()]).len(),
        1
    );
    assert_eq!(
        HashSet::from([evaluation_trace_id.clone(), subring_packing_id.clone()]).len(),
        1
    );
    let mut evaluation_trace_wire = Vec::new();
    evaluation_trace_id
        .serialize_with_mode(&mut evaluation_trace_wire, Compress::Yes)
        .expect("serialize evaluation-trace slot id");
    assert_eq!(&evaluation_trace_wire[..4], b"SPF4");
    assert_eq!(
        SetupPrefixSlotId::deserialize_with_mode(
            evaluation_trace_wire.as_slice(),
            Compress::Yes,
            Validate::Yes,
            &(),
        )
        .expect("deserialize SPF4 setup-prefix slot"),
        evaluation_trace_id,
    );
    let mut previous_wire = evaluation_trace_wire.clone();
    previous_wire[..4].copy_from_slice(b"SPF3");
    assert!(SetupPrefixSlotId::deserialize_with_mode(
        previous_wire.as_slice(),
        Compress::Yes,
        Validate::Yes,
        &(),
    )
    .is_err());
    let mut subring_packing_wire = Vec::new();
    subring_packing_id
        .serialize_with_mode(&mut subring_packing_wire, Compress::Yes)
        .expect("serialize subring-packing slot id");
    assert_eq!(evaluation_trace_wire, subring_packing_wire);

    let mut evaluation_trace_schedule = Vec::new();
    evaluation_trace.append_descriptor_bytes(&mut evaluation_trace_schedule);
    let mut subring_packing_schedule = Vec::new();
    subring_packing.append_descriptor_bytes(&mut subring_packing_schedule);
    assert_ne!(evaluation_trace_schedule, subring_packing_schedule);
    assert!(subring_packing.validate().is_ok());
}
