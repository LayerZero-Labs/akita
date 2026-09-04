use super::*;

#[test]
fn fold_schedule_estimate_separates_direct_and_stage3_payloads() {
    let estimate = FoldScheduleEstimate {
        nonce_stream_bytes: 0,
        estimated_root_direct_payload_bytes: 100,
        estimated_root_stage3_payload_bytes: 11,
        estimated_recursive_direct_payload_bytes: vec![200, 300],
        estimated_recursive_stage3_payload_bytes: vec![22, 0],
        estimated_terminal_direct_payload_bytes: 400,
        estimated_terminal_response_payload_bytes: 350,
        estimated_num_setup_field_elements: 512,
        first_direct_setup_field_len: Some(1_024),
        selected_offload_edges: 2,
    };

    assert_eq!(
        estimate.estimated_direct_proof_payload_bytes().unwrap(),
        1_000
    );
    assert_eq!(estimate.estimated_stage3_payload_bytes().unwrap(), 33);
    assert_eq!(estimate.estimated_proof_payload_bytes().unwrap(), 1_033);
}
use crate::golomb_rice::golomb_rice_encode_vec;
use crate::GrindingPlan;
use crate::{
    canonical_proof_shape, extension_opening_reduction_level_bytes, level_proof_bytes,
    sumcheck_rounds, terminal_response_bytes, AkitaStage1Proof, AkitaStage1StageProof,
    AkitaStage2Proof, Commitment, CommitmentPayloadMode, CommittedGroup,
    CommittedGroupBatchProfile, DigitRangePlan, ExtensionOpeningReductionProof, FoldLevelProof,
    NextWitnessBinding, OpeningClaimsLayout, PolynomialGroupLayout, RingRelationMode, RingVec,
    SisModulusProfileId, TailSegmentGroupLayout, TailSegmentLayout, TerminalLevelProof,
    TerminalResponse, TerminalResponseShape, EXTENSION_OPENING_REDUCTION_DEGREE,
};
use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_serialization::{AkitaSerialize, Compress};
use akita_sumcheck::EqFactoredUniPoly;
use akita_sumcheck::{CompressedUniPoly, EqFactoredSumcheckProof, SumcheckProof};
use jolt_field::{CanonicalEncoding, Field, Prime128OffsetA7F7, Zero};

#[path = "schedule_tests/descriptor.rs"]
mod descriptor;
#[path = "schedule_tests/execution_admission.rs"]
mod execution_admission;
#[path = "schedule_tests/group_topology.rs"]
mod group_topology;
#[path = "schedule_tests/proof_shapes.rs"]
mod proof_shapes;
#[path = "schedule_tests/relation_mode.rs"]
mod relation_mode;
#[path = "schedule_tests/sis_occurrences.rs"]
mod sis_occurrences;
type F = Prime128OffsetA7F7;
const TEST_TERMINAL_A_BOUND: u128 = 104_244;
fn committed_params(ring_dimension: usize) -> CommittedGroupParams {
    committed_params_with_geometry(ring_dimension, 4, 4)
}

fn committed_params_with_geometry(
    ring_dimension: usize,
    num_positions_per_block: usize,
    num_live_ring_elements_per_claim: usize,
) -> CommittedGroupParams {
    let mut params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        ring_dimension,
        3,
        2,
        2,
        2,
        SparseChallengeConfig::production_for_ring_dim(ring_dimension)
            .expect("production test challenge"),
    )
    .with_decomp(
        num_positions_per_block,
        num_live_ring_elements_per_claim,
        2,
        2,
        2,
    )
    .expect("schedule validation params");
    let a_bound = execution_admission::exact_test_a_bound(&params);
    let inner = params.inner().matrix;
    params.own_group_mut().profile.inner.matrix = crate::InnerCommitMatrixParams::try_new(
        inner.security_policy(),
        inner
            .sis_table_key()
            .expect("L infinity test matrix")
            .table_digest,
        inner.sis_modulus_profile(),
        inner.output_rank(),
        inner.input_width(),
        a_bound,
        inner.ring_dimension(),
    )
    .expect("audited schedule A matrix");
    let outer = params.outer().matrix;
    params.own_group_mut().profile.outer.matrix = crate::OuterCommitMatrixParams::try_new(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width(),
        3,
        outer.ring_dimension(),
    )
    .expect("audited schedule B matrix");
    let source_len = num_live_ring_elements_per_claim * ring_dimension;
    assert!(source_len.is_power_of_two());
    params.own_group_mut().profile.group =
        PolynomialGroupLayout::new(source_len.trailing_zeros() as usize, 1);
    params
}

fn provision_setup_prefix_capacity(params: &mut CommittedGroupParams, n_prefix: usize) {
    let d_setup = params.inner().matrix.ring_dimension();
    let d_outer = params.outer().matrix.ring_dimension();
    let ring_slots = n_prefix / d_setup;
    let setup_num_digits = crate::sis::compute_num_digits_field_width(
        params.inner().matrix.sis_modulus_profile().field_bits(),
        params.inner().digits.log_basis,
    );
    let required_inner_width = ring_slots
        .checked_mul(setup_num_digits)
        .expect("setup-prefix A width");
    let required_positions = required_inner_width
        .div_ceil(params.inner().digits.num_digits)
        .next_power_of_two();
    params.own_group_mut().profile.blocks.positions_per_block =
        params.blocks().positions_per_block.max(required_positions);
    params.own_group_mut().profile.blocks.live_blocks = params
        .blocks()
        .live_ring_elements_per_claim
        .div_ceil(params.blocks().positions_per_block);
    let inner_width = params
        .blocks()
        .positions_per_block
        .checked_mul(params.inner().digits.num_digits)
        .expect("recursive witness A width");
    let inner_key = params
        .inner()
        .matrix
        .sis_table_key()
        .expect("L infinity setup-prefix matrix");
    params.own_group_mut().profile.inner.matrix =
        crate::InnerCommitMatrixParams::try_new_with_min_rank(inner_key, inner_width)
            .expect("full-field setup-prefix A matrix");

    let outer_width = crate::CommitmentSliceGeometry::try_new(
        params.outer_slice_count(),
        params.blocks().live_blocks,
        1,
        params.inner().matrix.output_rank(),
        params.outer().digits.num_digits,
        d_setup,
        d_outer,
    )
    .expect("setup-prefix slice geometry")
    .physical_input_width();
    let outer_key = params.outer().matrix.sis_table_key();
    params.own_group_mut().profile.outer.matrix =
        crate::OuterCommitMatrixParams::try_new_with_min_rank(outer_key, outer_width)
            .expect("setup-prefix B matrix");
}

fn retarget_outer_dimension(
    params: &mut CommittedGroupParams,
    ring_dimension: usize,
) -> Result<(), AkitaError> {
    let outer = &params.outer().matrix;
    let column_scale = outer.ring_dimension() / ring_dimension;
    params.own_group_mut().profile.outer.matrix =
        crate::sis::OuterCommitMatrixParams::new_unchecked(
            outer.security_policy(),
            outer.sis_table_key().table_digest,
            outer.sis_modulus_profile(),
            outer.output_rank(),
            outer.input_width() * column_scale,
            outer.coeff_linf_bound(),
            ring_dimension,
        );
    Ok(())
}

fn retarget_open_dimension(
    params: &mut CommittedGroupParams,
    ring_dimension: usize,
) -> Result<(), AkitaError> {
    let open = &params.open().matrix;
    let column_scale = open.ring_dimension() / ring_dimension;
    params.open_matrix = crate::sis::OpenCommitMatrixParams::new_unchecked(
        open.security_policy(),
        open.sis_table_key().table_digest,
        open.sis_modulus_profile(),
        open.output_rank(),
        open.input_width() * column_scale,
        open.coeff_linf_bound(),
        ring_dimension,
    );
    Ok(())
}

fn preceding_group_params(
    params: &CommittedGroupParams,
    group: PolynomialGroupLayout,
) -> crate::GroupOpenPhaseParams {
    crate::GroupOpenPhaseParams {
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

fn recursive_schedule(
    predecessor_ring_dimension: usize,
    successor_ring_dimension: usize,
    offload: bool,
) -> FoldSchedule {
    let predecessor = committed_params(predecessor_ring_dimension);
    let mut successor = committed_params(successor_ring_dimension);
    if offload {
        successor.own_group_mut().opening.fold_challenge_config =
            SparseChallengeConfig::production_for_ring_dim(successor_ring_dimension)
                .expect("production setup-prefix challenge");
    }
    let incoming_setup_prefix = offload.then(|| {
        let natural_len = successor_ring_dimension;
        provision_setup_prefix_capacity(&mut successor, natural_len);
        let commitment_params = crate::setup_prefix_precommitted_params(&successor, natural_len)
            .expect("setup-prefix commitment params");
        crate::scheduled_setup_prefix(natural_len, commitment_params)
    });
    successor.set_setup_prefix(incoming_setup_prefix).unwrap();
    let terminal =
        TerminalFoldParams::from_expanded_group(committed_params(successor_ring_dimension));
    let terminal_response_len = 3 * successor_ring_dimension;
    let root_handoff_len = predecessor_ring_dimension.max(successor_ring_dimension);

    FoldSchedule {
        root: FoldParams {
            params: predecessor.clone(),
            input_witness_len: predecessor_ring_dimension,
            output_witness_len: root_handoff_len,
        },
        recursive_folds: vec![FoldParams {
            params: {
                let mut params = successor.clone();
                params.set_setup_prefix(incoming_setup_prefix).unwrap();
                params
            },
            input_witness_len: root_handoff_len,
            output_witness_len: successor_ring_dimension,
        }],
        terminal: TerminalFoldParams {
            fold_challenge_config: SparseChallengeConfig::pm1_only(3),
            response_shape: TerminalResponseShape {
                layout: TailSegmentLayout {
                    ring_dimension: successor_ring_dimension,
                    groups: vec![TailSegmentGroupLayout {
                        z_coords: successor_ring_dimension,
                        e_field_elems: successor_ring_dimension,
                        t_field_elems: successor_ring_dimension,
                        z_linf_cap: Some(1),
                        z_payload_bytes: 1,
                        z_rice_low_bits: 0,
                    }],
                    logical_num_elems: terminal_response_len,
                },
            },
            input_witness_len: successor_ring_dimension,
            ..terminal
        },
    }
}

fn append_recursive_fold(schedule: &mut FoldSchedule) {
    let mut step = schedule
        .recursive_folds
        .last()
        .expect("recursive fixture has one fold")
        .clone();
    step.params.set_setup_prefix(None).unwrap();
    step.params.set_setup_prefix(None).unwrap();
    step.input_witness_len = schedule
        .recursive_folds
        .last()
        .expect("recursive fixture has one fold")
        .output_witness_len;
    schedule.terminal.input_witness_len = step.output_witness_len;
    schedule.recursive_folds.push(step);
}

#[test]
fn schedule_rejects_raw_root_payload() {
    let mut schedule = recursive_schedule(64, 64, false);
    schedule.root.params.payload_mode = CommitmentPayloadMode::Raw;

    assert!(matches!(
        schedule.validate_structure(),
        Err(AkitaError::InvalidSetup(message)) if message.contains("root fold payload")
    ));
}

#[test]
fn schedule_rejects_raw_first_recursive_payload() {
    let mut schedule = recursive_schedule(64, 64, false);
    schedule.recursive_folds[0].params.payload_mode = CommitmentPayloadMode::Raw;

    let validation = schedule.validate_structure();
    assert!(
        matches!(
            validation,
            Err(AkitaError::InvalidSetup(ref message)) if message.contains("cutover policy")
        ),
        "unexpected validation result: {validation:?}"
    );
}

#[test]
fn schedule_rejects_compression_after_raw_suffix_starts() {
    let mut schedule = recursive_schedule(64, 64, false);
    append_recursive_fold(&mut schedule);
    append_recursive_fold(&mut schedule);
    schedule.recursive_folds[1].params.payload_mode = CommitmentPayloadMode::Raw;

    let validation = schedule.validate_structure();
    assert!(
        matches!(
            validation,
            Err(AkitaError::InvalidSetup(ref message)) if message.contains("cutover policy")
        ),
        "unexpected validation result: {validation:?}"
    );
}

#[test]
fn schedule_accepts_extended_compressed_prefix() {
    let mut schedule = recursive_schedule(64, 64, false);
    append_recursive_fold(&mut schedule);

    schedule.validate_structure().unwrap();
}

#[test]
fn schedule_rejects_setup_prefix_inside_raw_suffix() {
    let mut schedule = recursive_schedule(64, 64, false);
    append_recursive_fold(&mut schedule);
    let raw = &mut schedule.recursive_folds[1];
    raw.params.payload_mode = CommitmentPayloadMode::Raw;
    let natural_len = 64;
    provision_setup_prefix_capacity(&mut raw.params, natural_len);
    let commitment_params = crate::setup_prefix_precommitted_params(&raw.params, natural_len)
        .expect("setup-prefix commitment params");
    let prefix = crate::scheduled_setup_prefix(natural_len, commitment_params);
    raw.params.set_setup_prefix(Some(prefix)).unwrap();
    raw.params.set_setup_prefix(Some(prefix)).unwrap();

    let validation = schedule.validate_structure();
    assert!(
        matches!(
            validation,
            Err(AkitaError::InvalidSetup(ref message)) if message.contains("cutover policy")
        ),
        "unexpected validation result: {validation:?}"
    );
}

#[test]
fn schedule_rejects_setup_prefix_that_resumes_compression() {
    let mut schedule = recursive_schedule(64, 64, false);
    append_recursive_fold(&mut schedule);
    append_recursive_fold(&mut schedule);
    schedule.recursive_folds[1].params.payload_mode = CommitmentPayloadMode::Raw;
    let resumed = &mut schedule.recursive_folds[2];
    let natural_len = 64;
    provision_setup_prefix_capacity(&mut resumed.params, natural_len);
    let commitment_params = crate::setup_prefix_precommitted_params(&resumed.params, natural_len)
        .expect("setup-prefix commitment params");
    let prefix = crate::scheduled_setup_prefix(natural_len, commitment_params);
    resumed.params.set_setup_prefix(Some(prefix)).unwrap();
    resumed.params.set_setup_prefix(Some(prefix)).unwrap();

    let validation = schedule.validate_structure();
    assert!(
        matches!(
            validation,
            Err(AkitaError::InvalidSetup(ref message)) if message.contains("resume compression")
        ),
        "unexpected validation result: {validation:?}"
    );
}

/// The setup-prefix edge used to be stored twice — on the consuming fold and on
/// its witness params — with `validate_structure` rejecting disagreement. This
/// test used to construct that disagreement.
///
/// It cannot any more: one field holds the prefix, so "the two authorities
/// disagree" is not a representable state and needs no runtime check. What is
/// still worth pinning is that clearing the prefix changes the schedule rather
/// than being silently ignored, which is what proves the surviving field is the
/// one the validator and the descriptor both read.
#[test]
fn clearing_the_only_setup_prefix_field_changes_the_schedule() {
    let with_prefix = recursive_schedule(64, 64, true);
    assert!(
        with_prefix.recursive_folds[0]
            .params
            .setup_prefix()
            .is_some(),
        "fixture must consume a setup prefix"
    );
    let mut without_prefix = with_prefix.clone();
    without_prefix.recursive_folds[0]
        .params
        .set_setup_prefix(None)
        .unwrap();
    assert_ne!(
        with_prefix.canonical_descriptor_bytes(),
        without_prefix.canonical_descriptor_bytes(),
        "the surviving prefix field must be the one the descriptor reads"
    );
    assert_ne!(
        with_prefix.recursive_folds[0].predecessor_setup_contribution_mode(),
        without_prefix.recursive_folds[0].predecessor_setup_contribution_mode(),
        "presence of the prefix is the sole authority for the contribution mode"
    );
}

#[test]
fn schedule_accepts_prefix_dimension_independent_of_producer_projection() {
    let schedule = recursive_schedule(128, 64, true);

    schedule
        .validate_structure()
        .expect("prefix commitment A dimension is independent of producer projection");
}

#[test]
fn schedule_accepts_stage2_points_within_successor_capacity() {
    recursive_schedule(128, 64, false)
        .validate_structure()
        .expect("successor cubes may be wider than their incoming Stage 2 points");
}

#[test]
fn schedule_rejects_root_stage2_point_wider_than_successor() {
    let mut schedule = recursive_schedule(64, 64, false);
    let narrow_successor = committed_params_with_geometry(64, 1, 1);
    schedule.root.output_witness_len = 128;
    schedule.recursive_folds[0].input_witness_len = 128;
    schedule.recursive_folds[0].params.open_matrix = narrow_successor.open().matrix;
    schedule.recursive_folds[0].params = narrow_successor;

    let err = schedule
        .validate_structure()
        .expect_err("the successor cube cannot hold the root Stage 2 point");
    assert!(
        matches!(err, AkitaError::InvalidSetup(message) if message.contains("root fold Stage 2 point"))
    );
}

#[test]
fn schedule_rejects_recursive_stage2_point_wider_than_terminal() {
    let mut schedule = recursive_schedule(64, 64, false);
    let narrow_terminal = committed_params_with_geometry(64, 1, 1);
    schedule.recursive_folds[0].output_witness_len = 128;
    // Replace the terminal first: after the three-type merge `input_witness_len`
    // lives on the same value, so assigning it before the replacement would be
    // silently discarded.
    schedule.terminal = TerminalFoldParams::from_expanded_group(narrow_terminal);
    schedule.terminal.input_witness_len = 128;

    let err = schedule
        .validate_structure()
        .expect_err("the terminal cube cannot hold the recursive Stage 2 point");
    assert!(
        matches!(err, AkitaError::InvalidSetup(message) if message.contains("recursive fold 0 Stage 2 point"))
    );
}

#[test]
fn schedule_accepts_offload_at_uniform_successor_dimension() {
    recursive_schedule(64, 64, true)
        .validate_structure()
        .expect("offload supports uniform predecessor/successor geometry");
}

#[test]
fn schedule_accepts_mixed_producer_projecting_to_prefix_dimension() {
    let mut schedule = recursive_schedule(128, 64, true);
    let producer = &mut schedule.root.params;
    retarget_outer_dimension(producer, 64).expect("retarget producer B role");
    retarget_open_dimension(producer, 64).expect("retarget producer D role");

    schedule
        .validate_structure()
        .expect("mixed A128/B64/D64 producer projects its setup prefix at D64");
}

#[test]
fn schedule_accepts_prefix_commitment_roles_independent_of_consumer_roles() {
    let mut schedule = recursive_schedule(64, 128, true);
    retarget_outer_dimension(&mut schedule.recursive_folds[0].params, 64)
        .expect("retarget consumer B role");

    schedule
        .validate_structure()
        .expect("prefix commitment roles are independent of consumer witness roles");
}

#[test]
fn schedule_accepts_exact_multi_group_prefix_from_mixed_producer() {
    let mut schedule = recursive_schedule(128, 64, false);
    let producer = &mut schedule.root.params;
    retarget_outer_dimension(producer, 64).expect("retarget producer B role");
    retarget_open_dimension(producer, 64).expect("retarget producer D role");

    let final_group = PolynomialGroupLayout::new(9, 1);
    let singleton_layout =
        OpeningClaimsLayout::from_groups(vec![final_group]).expect("singleton layout");
    let singleton_natural_len = crate::active_setup_field_len(producer, &singleton_layout)
        .expect("singleton setup geometry");

    let precommitted_group = PolynomialGroupLayout::new(9, 1);
    let mut group_params = producer.clone();
    group_params.own_group_mut().opening.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(group_params.d_a())
            .expect("precommitted test group uses a production ring dimension");
    let a_bound = execution_admission::exact_test_a_bound(&group_params);
    let inner = &group_params.inner().matrix;
    group_params.own_group_mut().profile.inner.matrix =
        crate::sis::InnerCommitMatrixParams::new_unchecked(
            inner.security_policy(),
            inner
                .sis_table_key()
                .expect("L infinity test matrix")
                .table_digest,
            inner.sis_modulus_profile(),
            inner.output_rank(),
            inner.input_width(),
            a_bound,
            inner.ring_dimension(),
        );
    let outer = &group_params.outer().matrix;
    group_params.own_group_mut().profile.outer.matrix =
        crate::sis::OuterCommitMatrixParams::new_unchecked(
            outer.security_policy(),
            outer.sis_table_key().table_digest,
            outer.sis_modulus_profile(),
            outer.output_rank(),
            outer.input_width(),
            3,
            outer.ring_dimension(),
        );
    let precommitted = preceding_group_params(&group_params, precommitted_group);
    let one_precommitted_d_width = precommitted
        .d_segment_width(1, producer.role_dims().d_d())
        .expect("precommitted D width");
    let preceding_group_count = 8;
    producer
        .set_precommitted_groups(vec![precommitted; preceding_group_count])
        .unwrap();
    let precommitted_d_width = one_precommitted_d_width * preceding_group_count;

    let open = &producer.open().matrix;
    producer.open_matrix = crate::sis::OpenCommitMatrixParams::new_unchecked(
        open.security_policy(),
        open.sis_table_key().table_digest,
        open.sis_modulus_profile(),
        open.output_rank(),
        open.input_width() + precommitted_d_width,
        open.coeff_linf_bound(),
        open.ring_dimension(),
    );

    let mut groups = vec![precommitted_group; preceding_group_count];
    groups.push(final_group);
    let opening_layout = OpeningClaimsLayout::from_groups(groups).expect("multi-group layout");
    let natural_len = crate::active_setup_field_len(producer, &opening_layout)
        .expect("multi-group mixed setup geometry");
    assert!(
        natural_len > singleton_natural_len,
        "the exact prefix must include the larger multi-group setup footprint"
    );

    let n_prefix = crate::padded_setup_prefix_len(natural_len);
    let prefix_ring_slots = n_prefix / 64;
    let mut consumer = committed_params_with_geometry(64, prefix_ring_slots, 64);
    consumer.own_group_mut().opening.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(64)
            .expect("production setup-prefix challenge");
    provision_setup_prefix_capacity(&mut consumer, n_prefix);
    let commitment_params = crate::setup_prefix_precommitted_params(&consumer, n_prefix)
        .expect("consumer-compatible prefix commitment");
    let prefix = crate::scheduled_setup_prefix(natural_len, commitment_params);
    schedule.recursive_folds[0].params = consumer.clone();
    schedule.recursive_folds[0].params.open_matrix = consumer.open().matrix;
    schedule.recursive_folds[0]
        .params
        .set_setup_prefix(Some(prefix))
        .unwrap();
    schedule.recursive_folds[0]
        .params
        .set_setup_prefix(Some(prefix))
        .unwrap();

    schedule
        .validate_structure()
        .expect("mixed multi-group producer offloads its exact D64 setup projection");
}

#[test]
fn terminal_projection_preserves_the_fixed_inner_matrix() {
    let sparse = SparseChallengeConfig::pm1_only(3);
    let mut committed = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        64,
        3,
        4,
        3,
        2,
        sparse,
    )
    .with_decomp(4, 32, 2, 2, 2)
    .expect("committed params");
    let inner = committed.inner().matrix;
    committed.own_group_mut().profile.inner.matrix =
        crate::sis::InnerCommitMatrixParams::new_unchecked(
            inner.security_policy(),
            inner
                .sis_table_key()
                .expect("L infinity test matrix")
                .table_digest,
            inner.sis_modulus_profile(),
            inner.output_rank(),
            inner.input_width(),
            TEST_TERMINAL_A_BOUND,
            inner.ring_dimension(),
        );
    let expected_inner = committed.inner().matrix;

    let (terminal, response_cap) =
        TerminalFoldParams::try_from_expanded_group(committed).expect("terminal projection");
    assert_eq!(terminal.inner.matrix, expected_inner);
    assert_eq!(
        response_cap,
        terminal.certified_response_linf_cap().unwrap()
    );
    assert!(response_cap > 0);
}

fn terminal_response_fixture(
    lp: &CommittedGroupParams,
    num_claims: usize,
) -> (TerminalResponse<F>, TerminalResponseShape) {
    let field_bits = F::MODULUS_BITS;
    let shape = TerminalResponseShape::from_groups(
        lp,
        field_bits,
        [(
            lp.final_group_scalar().expect("scalar final group"),
            num_claims,
            num_claims,
            1,
            127,
        )],
    )
    .expect("terminal response shape");
    let layout = shape.layout.clone();
    let group = layout.groups[0];
    let rice_low_bits = group.z_rice_low_bits;
    let zigzag_w =
        crate::golomb_rice::golomb_rice_zigzag_width(group.z_linf_cap.unwrap_or(i16::MAX as u128));
    let z_payload = golomb_rice_encode_vec(&vec![0i64; group.z_coords], rice_low_bits, zigzag_w)
        .expect("encode zero z segment");
    let witness = TerminalResponse {
        layout: layout.clone(),
        z_payloads: vec![z_payload],
        e_fields: RingVec::from_coeffs(vec![F::zero(); group.e_field_elems]),
        t_fields: RingVec::from_coeffs(vec![F::zero(); group.t_field_elems]),
    };
    (witness, shape)
}

fn dummy_sumcheck<F: Field>(rounds: usize, degree: usize) -> SumcheckProof<F> {
    SumcheckProof {
        round_polys: (0..rounds)
            .map(|_| CompressedUniPoly {
                coeffs_except_linear_term: vec![F::zero(); degree],
            })
            .collect(),
    }
}

fn dummy_eq_factored_sumcheck<F: Field>(
    rounds: usize,
    degree: usize,
) -> EqFactoredSumcheckProof<F> {
    EqFactoredSumcheckProof {
        round_polys: (0..rounds)
            .map(|_| EqFactoredUniPoly {
                coeffs_except_constant_term: vec![F::zero(); degree],
            })
            .collect(),
    }
}

fn dummy_stage1_proof<F: Field>(rounds: usize, b: usize) -> AkitaStage1Proof<F> {
    AkitaStage1Proof {
        stages: DigitRangePlan::new(b)
            .expect("test range basis")
            .stage_shapes(rounds)
            .into_iter()
            .map(|shape| AkitaStage1StageProof {
                sumcheck_proof: dummy_eq_factored_sumcheck(rounds, shape.sumcheck_proof.1),
                child_claims: vec![F::zero(); shape.child_claims],
            })
            .collect(),
        range_image_evaluation: F::zero(),
        norm_proof: None,
    }
}

fn exact_level_proof_bytes<F: Field + CanonicalEncoding + AkitaSerialize>(
    lp: &CommittedGroupParams,
    next_lp: &CommittedGroupParams,
    output_witness_len: usize,
) -> Result<usize, AkitaError> {
    let current_source_coeffs = lp
        .open()
        .matrix
        .output_rank()
        .checked_mul(lp.role_dims().d_d())
        .ok_or_else(|| AkitaError::InvalidSetup("recursive proof sizing overflow".to_string()))?;
    let current_coeffs = crate::CompressionChainPlan::for_complete_source(
        lp.open().matrix.sis_modulus_profile(),
        current_source_coeffs,
    )?
    .terminal_coefficients();
    let next_commit_source_coeffs = next_lp
        .outer()
        .matrix
        .output_rank()
        .checked_mul(next_lp.role_dims().d_b())
        .ok_or_else(|| AkitaError::InvalidSetup("recursive proof sizing overflow".to_string()))?;
    let next_commit_coeffs = crate::CompressionChainPlan::for_complete_source(
        next_lp.outer().matrix.sis_modulus_profile(),
        next_commit_source_coeffs,
    )?
    .terminal_coefficients();
    let rounds = sumcheck_rounds(lp.d_a(), output_witness_len);
    let b = 1usize << lp.open().digits.log_basis;

    let proof = FoldLevelProof {
        extension_opening_reduction: None,
        opening_payload: RingVec::from_coeffs(vec![F::zero(); current_coeffs]),
        stage1: dummy_stage1_proof(rounds, b),
        stage2: AkitaStage2Proof {
            sumcheck_proof: dummy_sumcheck(rounds, 3),
            next_witness_binding: NextWitnessBinding::OuterPayload(RingVec::from_coeffs(vec![
                F::zero();
                next_commit_coeffs
            ])),
            next_w_eval: F::zero(),
        },
        stage3_sumcheck_proof: None,
    };
    Ok(proof.serialized_size(Compress::No))
}

#[test]
fn planned_level_bytes_match_non_offloaded_payload_at_all_bases() {
    const D: usize = 64;
    let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
    let next_lp = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        D,
        2,
        2,
        3,
        2,
        fold_challenge_config,
    );
    let output_witness_len = D * 8;

    for log_basis in 2..=6 {
        let lp = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            D,
            log_basis,
            2,
            2,
            2,
            fold_challenge_config,
        )
        .with_decomp(1, 1, 1, 1, 1)
        .unwrap();
        let opening_layout =
            OpeningClaimsLayout::new(sumcheck_rounds(D, output_witness_len), 1).unwrap();
        assert_eq!(
                level_proof_bytes(
                    128,
                    128,
                    &lp,
                    lp.relation_address_geometry(
                        &opening_layout,
                        1,
                        next_lp.d_a(),
                        output_witness_len,
                    )
                    .unwrap(),
                    Some(&next_lp),
                )
                .unwrap(),
                exact_level_proof_bytes::<F>(&lp, &next_lp, output_witness_len).unwrap(),
                "planned level bytes should match the serialized non-offloaded body at log_basis={log_basis}"
            );
    }
}

#[test]
fn planned_terminal_level_bytes_match_terminal_payload_at_all_bases() {
    const D: usize = 64;
    let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
    let num_claims = 3;

    for log_basis in 2..=6 {
        let mut lp = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            D,
            log_basis,
            2,
            2,
            2,
            fold_challenge_config,
        )
        .with_decomp(1, 1, 1, 1, 1)
        .unwrap();
        lp.own_group_mut().opening.num_digits_fold = 2;
        let inner = lp.inner().matrix;
        lp.own_group_mut().profile.inner.matrix =
            crate::sis::InnerCommitMatrixParams::new_unchecked(
                inner.security_policy(),
                inner
                    .sis_table_key()
                    .expect("L infinity test matrix")
                    .table_digest,
                inner.sis_modulus_profile(),
                inner.output_rank(),
                inner.input_width(),
                TEST_TERMINAL_A_BOUND,
                inner.ring_dimension(),
            );

        let (terminal_response, witness_shape) = terminal_response_fixture(&lp, num_claims);
        let terminal_response_bytes_runtime = terminal_response.serialized_size(Compress::No);
        let terminal_proof = TerminalLevelProof::<F, F>::new_with_extension_opening_reduction(
            None,
            terminal_response,
        );

        // The planner accounts for the final witness separately
        // (`terminal_response_bytes` on the terminal plan). Subtract
        // it from the serialized terminal level. The proof-level packed nonce
        // stream is accounted separately.
        let serialized_without_witness =
            terminal_proof.serialized_size(Compress::No) - terminal_response_bytes_runtime;

        assert_eq!(
            0, serialized_without_witness,
            "planned terminal-level bytes should match the serialized terminal body \
                 (less terminal_response) at log_basis={log_basis}"
        );

        let scheduled_bytes = terminal_response_bytes(128, &witness_shape);
        assert!(
            scheduled_bytes >= terminal_response_bytes_runtime,
            "scheduled direct witness budget must cover serialized terminal response \
                 at log_basis={log_basis}"
        );
    }
}

#[test]
fn planned_batched_root_bytes_match_non_offloaded_payload_at_all_bases() {
    const D: usize = 64;
    let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
    let next_lp = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        D,
        2,
        2,
        3,
        2,
        fold_challenge_config,
    );
    let output_witness_len = D * 8;

    for log_basis in 2..=6 {
        let lp = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            D,
            log_basis,
            2,
            2,
            2,
            fold_challenge_config,
        )
        .with_decomp(1, 1, 1, 1, 1)
        .unwrap();
        let rounds = sumcheck_rounds(D, output_witness_len);
        let opening_layout = OpeningClaimsLayout::new(rounds, 1).unwrap();
        let b = 1usize << log_basis;
        let level_proof = FoldLevelProof {
            extension_opening_reduction: None,
            opening_payload: RingVec::from_coeffs(vec![
                F::zero();
                crate::CompressionChainPlan::for_complete_source(
                    lp.open().matrix.sis_modulus_profile(),
                    lp.open().matrix.output_rank() * lp.role_dims().d_d(),
                )
                .unwrap()
                .terminal_coefficients()
            ]),
            stage1: dummy_stage1_proof(rounds, b),
            stage2: AkitaStage2Proof {
                sumcheck_proof: dummy_sumcheck(rounds, 3),
                next_witness_binding: NextWitnessBinding::OuterPayload(RingVec::from_coeffs(vec![
                    F::zero();
                    crate::CompressionChainPlan::for_complete_source(
                        next_lp.outer().matrix.sis_modulus_profile(),
                        next_lp.outer().matrix.output_rank() * next_lp.role_dims().d_b(),
                    )
                    .unwrap()
                    .terminal_coefficients()
                ])),
                next_w_eval: F::zero(),
            },
            stage3_sumcheck_proof: None,
        };
        assert_eq!(
                level_proof_bytes(
                    128,
                    128,
                    &lp,
                    lp.relation_address_geometry(
                        &opening_layout,
                        1,
                        next_lp.d_a(),
                        output_witness_len,
                    )
                    .unwrap(),
                    Some(&next_lp),
                )
                .unwrap(),
                level_proof.serialized_size(Compress::No),
                "planned batched root bytes should match the serialized non-offloaded body at log_basis={log_basis}"
            );
    }
}

#[test]
fn planned_extension_reduction_bytes_match_headerless_payload() {
    let extension_width = 4usize;
    let num_claims = 3usize;
    let opening_vars = 12usize;
    let partials = extension_width.saturating_mul(num_claims);
    let reduction = ExtensionOpeningReductionProof {
        partials: vec![F::zero(); partials],
        sumcheck: dummy_sumcheck(
            opening_vars - extension_width.trailing_zeros() as usize,
            EXTENSION_OPENING_REDUCTION_DEGREE,
        ),
        final_claims: vec![F::zero(); num_claims],
    };
    let sumcheck_bytes = reduction.sumcheck.serialized_size(Compress::No);

    assert_eq!(
        extension_opening_reduction_level_bytes(
            128,
            extension_width,
            PolynomialGroupLayout::new(opening_vars, num_claims),
        )
        .unwrap(),
        reduction
            .partials
            .iter()
            .map(|partial| partial.serialized_size(Compress::No))
            .sum::<usize>()
            + sumcheck_bytes
            + reduction
                .final_claims
                .iter()
                .map(|claim| claim.serialized_size(Compress::No))
                .sum::<usize>(),
        "planned EOR bytes should match the headerless serialized payload"
    );
}

#[test]
fn scalar_schedule_key_accepts_single_group_layout() {
    let layout = OpeningClaimsLayout::new(4, 2).expect("scalar layout");
    let key =
        AkitaScheduleLookupKey::single(layout.root_final_group_layout().expect("final group"));
    assert_eq!(key.final_group, PolynomialGroupLayout::new(4, 2));
    assert!(key.precommitteds.is_empty());
    assert_eq!(key.num_commitment_groups(), 1);
}

#[test]
fn validate_rejects_zero_dimensions() {
    assert!(
        AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(0, 1))
            .validate(128)
            .is_err()
    );
    assert!(
        AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(20, 0))
            .validate(128)
            .is_err()
    );
    assert!(
        AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(20, 4))
            .validate(128)
            .is_ok()
    );
}

fn precommitted_descriptor(num_vars: usize) -> GroupCommitPhaseParams {
    let num_live_blocks = 1usize << (num_vars - 10);
    let inner_commit_matrix = crate::InnerCommitMatrixParams::try_new_with_min_rank(
        crate::SisTableKey {
            policy: crate::sis::DEFAULT_SIS_SECURITY_POLICY,
            table_digest: crate::SisTableDigest::CURRENT,
            modulus_profile: crate::SisModulusProfileId::Q128OffsetA7F7,
            role: crate::sis::SisMatrixRole::Inner,
            ring_dimension: 64,
            coeff_linf_bound: TEST_TERMINAL_A_BOUND,
        },
        16,
    )
    .expect("audited precommitted A matrix");
    let outer_width = inner_commit_matrix.output_rank() * num_live_blocks;
    GroupCommitPhaseParams {
        version: GroupCommitPhaseParams::VERSION,
        group: PolynomialGroupLayout::new(num_vars, 1),
        blocks: crate::BlockGeometry::new(1usize << (num_vars - 6), 16, num_live_blocks),
        outer_slice_count: crate::CommitmentSliceCount::ONE,
        inner: crate::RoleParams::new(crate::GadgetDigits::new(1, 1), inner_commit_matrix),
        outer: crate::RoleParams::new(
            crate::GadgetDigits::new(2, 1),
            crate::OuterCommitMatrixParams::try_new_with_min_rank(
                crate::SisTableKey {
                    policy: crate::sis::DEFAULT_SIS_SECURITY_POLICY,
                    table_digest: crate::SisTableDigest::CURRENT,
                    modulus_profile: crate::SisModulusProfileId::Q128OffsetA7F7,
                    role: crate::sis::SisMatrixRole::Outer,
                    ring_dimension: 64,
                    coeff_linf_bound: 3,
                },
                outer_width,
            )
            .expect("audited precommitted B matrix"),
        ),
    }
}

fn committed_group_for_extractor(num_vars: usize) -> CommittedGroup<F> {
    CommittedGroup::new(
        precommitted_descriptor(num_vars),
        Commitment::new(RingVec::from_coeffs(vec![F::zero(); 64])),
    )
}

#[test]
fn ordered_group_profile_extractor_rejects_empty_input() {
    let groups: [&CommittedGroup<F>; 0] = [];
    let error = CommittedGroupBatchProfile::from_ordered_groups(groups)
        .expect_err("empty group sequence must reject");
    assert!(matches!(error, AkitaError::InvalidInput(_)));
}

#[test]
fn ordered_group_profile_extractor_handles_a_group_without_precommitted_groups() {
    let final_group = committed_group_for_extractor(12);
    let batch = CommittedGroupBatchProfile::from_ordered_groups([&final_group])
        .expect("profile without precommitted groups");
    assert!(batch.precommitteds.is_empty());
    assert_eq!(batch.final_group, *final_group.profile());
}

#[test]
fn ordered_group_profile_extractor_preserves_prefix_order() {
    let first = committed_group_for_extractor(12);
    let second = committed_group_for_extractor(13);
    let final_group = committed_group_for_extractor(14);
    let batch = CommittedGroupBatchProfile::from_ordered_groups([&first, &second, &final_group])
        .expect("ordered grouped profile");
    assert_eq!(
        batch.precommitteds,
        vec![*first.profile(), *second.profile()]
    );
    assert_eq!(batch.final_group, *final_group.profile());
}

#[test]
fn precommitted_group_profiles_reject_an_empty_prefix() {
    let empty: [&CommittedGroup<F>; 0] = [];
    assert!(matches!(
        PrecommittedGroupProfiles::from_ordered_groups(empty)
            .expect_err("empty prefix must reject"),
        AkitaError::InvalidInput(_)
    ));
    assert!(matches!(
        PrecommittedGroupProfiles::from_profiles(Vec::new()).expect_err("empty prefix must reject"),
        AkitaError::InvalidInput(_)
    ));
}

#[test]
fn precommitted_group_profiles_preserve_caller_order() {
    let first = committed_group_for_extractor(12);
    let second = committed_group_for_extractor(13);
    let prefix =
        PrecommittedGroupProfiles::from_ordered_groups([&first, &second]).expect("nonempty prefix");
    assert_eq!(
        prefix.as_slice(),
        &[*first.profile(), *second.profile()][..]
    );
}

#[test]
fn group_batch_key_separates_final_arity_from_max_opening_arity() {
    let multi_group_key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(14, 3),
        precommitteds: vec![precommitted_descriptor(20)],
    };

    multi_group_key
        .validate(128)
        .expect("commit order must not impose an arity ordering");
    assert_eq!(multi_group_key.final_group.num_vars(), 14);
    assert_eq!(multi_group_key.max_num_vars(), 20);
    assert!(!multi_group_key.fits_setup_capacity(19, 4).unwrap());
    assert!(multi_group_key.fits_setup_capacity(20, 4).unwrap());

    let opening_layout = multi_group_key.opening_layout().expect("opening layout");
    assert_eq!(opening_layout.max_num_vars(), 20);
    assert_eq!(
        opening_layout.groups(),
        &[
            PolynomialGroupLayout::new(20, 1),
            PolynomialGroupLayout::new(14, 3),
        ],
        "opening layout must preserve precommitted-then-final transcript order"
    );
}

#[test]
fn group_batch_key_allows_independent_precommitted_num_vars() {
    let multi_group_key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(20, 3),
        precommitteds: vec![precommitted_descriptor(12)],
    };

    multi_group_key
        .validate(128)
        .expect("precommitted group arity is not derived from the final group");
}

#[test]
fn group_batch_key_allows_precommitted_num_vars_equal_to_main() {
    let multi_group_key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(20, 3),
        precommitteds: vec![precommitted_descriptor(20)],
    };

    multi_group_key
        .validate(128)
        .expect("precommitted groups may use the final group's full arity");
}

#[test]
fn group_batch_key_allows_mixed_polynomial_counts() {
    let multi_group_key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(20, 3),
        precommitteds: vec![{
            let mut descriptor = precommitted_descriptor(10);
            descriptor.group = PolynomialGroupLayout::new(10, 2);
            descriptor.outer.matrix = crate::OuterCommitMatrixParams::try_new_with_min_rank(
                crate::SisTableKey {
                    policy: crate::sis::DEFAULT_SIS_SECURITY_POLICY,
                    table_digest: crate::SisTableDigest::CURRENT,
                    modulus_profile: crate::SisModulusProfileId::Q128OffsetA7F7,
                    role: crate::sis::SisMatrixRole::Outer,
                    ring_dimension: 64,
                    coeff_linf_bound: 3,
                },
                descriptor.inner.matrix.output_rank()
                    * descriptor.blocks.live_blocks
                    * descriptor.group.num_polynomials(),
            )
            .expect("audited multi-polynomial B matrix");
            descriptor
        }],
    };

    multi_group_key
        .validate(128)
        .expect("a precommitted group may contain multiple polynomials");
    assert_eq!(multi_group_key.num_commitment_groups(), 2);
    assert_eq!(multi_group_key.num_polynomials().unwrap(), 5);
    assert!(!multi_group_key.fits_setup_capacity(20, 4).unwrap());
    assert!(multi_group_key.fits_setup_capacity(20, 5).unwrap());
}

#[test]
fn group_batch_key_identity_binds_ordered_profiles() {
    let first = precommitted_descriptor(12);
    let second = precommitted_descriptor(14);
    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(16, 1),
        precommitteds: vec![first, second],
    };

    let mut reordered = key.clone();
    reordered.precommitteds.swap(0, 1);
    assert_ne!(
        key.canonical_descriptor_bytes(),
        reordered.canonical_descriptor_bytes(),
        "group ordering is transcript and catalog identity"
    );
}

#[test]
fn validate_frozen_precommit_rejects_geometry_mismatch() {
    let mut layout = precommitted_descriptor(20);
    layout.blocks.live_ring_elements_per_claim = 1;
    layout.blocks.live_blocks = 1;
    let err = layout
        .validate_frozen_precommit(128)
        .expect_err("geometry must match num_vars");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn checked_committed_profile_construction_rejects_invalid_params() {
    let schedule = recursive_schedule(64, 64, false);
    let mut params = schedule.root.params;
    params
        .own_group_mut()
        .profile
        .blocks
        .live_ring_elements_per_claim = 1;
    params.own_group_mut().profile.blocks.live_blocks = 1;

    assert!(matches!(
        GroupCommitPhaseParams::try_from_params(PolynomialGroupLayout::singleton(8), &params),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn validate_frozen_precommit_rejects_unsupported_inner_decomposition() {
    let mut unsupported_basis = precommitted_descriptor(20);
    unsupported_basis.inner.digits.log_basis = crate::MAX_I16_LOG_BASIS + 1;
    assert!(matches!(
        unsupported_basis.validate_frozen_precommit(128),
        Err(AkitaError::InvalidSetup(_))
    ));

    let mut excessive_depth = precommitted_descriptor(20);
    excessive_depth.inner.digits.num_digits =
        crate::sis::compute_num_digits_field_width(128, excessive_depth.inner.digits.log_basis) + 1;
    assert!(matches!(
        excessive_depth.validate_frozen_precommit(128),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn schedule_row_identity_binds_profiles_and_expanded_schedule() {
    let schedule = recursive_schedule(64, 64, false);
    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::from_params_unchecked_for_test(
            PolynomialGroupLayout::singleton(8),
            &schedule.root.params,
        ),
        precommitteds: Vec::new(),
    };
    let digest = crate::schedule_row_digest(&profiles, &schedule).expect("row digest");
    assert_eq!(
        digest,
        crate::schedule_row_digest(&profiles, &schedule).expect("stable row digest")
    );

    let mut changed_profiles = profiles.clone();
    changed_profiles.final_group.group = PolynomialGroupLayout::new(8, 2);
    assert_ne!(
        digest,
        crate::schedule_row_digest(&changed_profiles, &schedule).expect("changed-profile digest")
    );

    let mut changed_schedule = schedule.clone();
    changed_schedule.terminal.input_witness_len += 1;
    assert_ne!(
        digest,
        crate::schedule_row_digest(&profiles, &changed_schedule).expect("changed-row digest")
    );

    for field in ["cap", "rice", "budget"] {
        let mut changed_schedule = schedule.clone();
        let group = &mut changed_schedule.terminal.response_shape.layout.groups[0];
        match field {
            "cap" => {
                group.z_linf_cap = Some(group.z_linf_cap.unwrap_or_default() + 1);
            }
            "rice" => group.z_rice_low_bits += 1,
            "budget" => group.z_payload_bytes += 1,
            _ => unreachable!(),
        }
        assert_ne!(
            digest,
            crate::schedule_row_digest(&profiles, &changed_schedule)
                .expect("terminal-shape mutation digest"),
            "terminal response-shape field {field} must change the row digest"
        );
    }
}

#[test]
fn schedule_row_identity_binds_setup_prefix_opening_method() {
    let schedule = recursive_schedule(64, 64, true);
    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::from_params_unchecked_for_test(
            PolynomialGroupLayout::singleton(8),
            &schedule.root.params,
        ),
        precommitteds: Vec::new(),
    };
    let digest = crate::schedule_row_digest(&profiles, &schedule).expect("row digest");

    let mut changed = schedule;
    let mut changed_prefix = changed.recursive_folds[0]
        .params
        .setup_prefix()
        .copied()
        .expect("setup prefix");
    changed_prefix.opening.opening_method = crate::OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 64,
    };
    changed.recursive_folds[0]
        .params
        .set_setup_prefix(Some(changed_prefix))
        .unwrap();
    assert_ne!(
        digest,
        crate::schedule_row_digest(&profiles, &changed).expect("changed opening-method digest")
    );
    let validation = changed.validate_structure();
    assert!(
        validation.is_ok(),
        "unexpected validation result: {validation:?}"
    );
    assert!(changed.validate_nonterminal_opening_execution(1).is_err());

    let mut changed_dimension = changed.clone();
    let mut widened_prefix = changed_dimension.recursive_folds[0]
        .params
        .setup_prefix()
        .copied()
        .expect("setup prefix");
    widened_prefix.opening.opening_method = crate::OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 128,
    };
    changed_dimension.recursive_folds[0]
        .params
        .set_setup_prefix(Some(widened_prefix))
        .unwrap();
    assert_ne!(
        crate::schedule_row_digest(&profiles, &changed).expect("subring-dimension-64 digest"),
        crate::schedule_row_digest(&profiles, &changed_dimension)
            .expect("subring-dimension-128 digest")
    );
}

#[test]
fn schedule_row_identity_binds_main_opening_method() {
    let schedule = recursive_schedule(64, 64, false);
    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::from_params_unchecked_for_test(
            PolynomialGroupLayout::singleton(8),
            &schedule.root.params,
        ),
        precommitteds: Vec::new(),
    };
    let digest = crate::schedule_row_digest(&profiles, &schedule).expect("row digest");
    let mut changed = schedule;
    changed.root.params.own_group_mut().opening.opening_method =
        crate::OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        };
    changed
        .root
        .params
        .own_group_mut()
        .opening
        .fold_challenge_config =
        akita_challenges::SparseChallengeConfig::production_for_ring_dim(64).unwrap();
    assert_ne!(
        digest,
        crate::schedule_row_digest(&profiles, &changed)
            .expect("changed main opening-method digest")
    );
    assert!(changed.validate_structure().is_ok());
    assert!(changed.validate_nonterminal_opening_execution(1).is_err());
}
