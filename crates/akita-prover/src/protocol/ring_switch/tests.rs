use super::coeffs::balanced_decompose_centered_i32_i8_into;
use super::coeffs::{
    quotient_decomposition_calls, reset_quotient_decomposition_calls, ring_switch_build_w,
};
use super::evals::build_w_evals_compact;
use crate::backend::packed_digits::PackedSignedDigits;
use crate::compute::compression::{execute_compression_chains, CompressionExecutionInput};
use crate::compute::{ComputeBackendSetup, CpuBackend, OperationCtx};
use crate::protocol::ring_relation::{
    materialize_compression_witness, CompressionSourceId, CompressionSourceWitness,
};
use crate::protocol::ring_relation::{
    multi_group_quotient_calls, reset_multi_group_quotient_calls,
};
use crate::protocol::ring_relation_witness::{
    FoldChunkCoefficients, RelationDQuotientWitness, RingRelationGroupWitness, RingRelationWitness,
};
use crate::{AkitaProverSetup, DecomposeFoldWitness};
use akita_algebra::{poly::multilinear_eval, CyclotomicRing, EqPolynomial};
use akita_challenges::{Challenges, SparseChallenge, SparseChallengeConfig};
use akita_types::{
    active_setup_field_len, relation_rhs_coeff_len, shared_setup_fold_gadget, AkitaCommitmentHint,
    CommitmentPayloadMode, CommittedGroupParams, CompressionWitnessSpan, DigitBlocks,
    DigitRangePlan, OpeningClaimsLayout, PreparedCoefficientFunctional, PreparedRelationAddress,
    RelationAddressGeometry, RelationRangeImagePlan, RingMultiplierOpeningPoint, RingOpeningPoint,
    RingRelationGroupOpening, RingRelationInstance, RingRelationMode, RingVec,
    SetupContributionGroupInputs, SetupContributionPlan, SetupMatrixCapacity, SisModulusProfileId,
};
use jolt_field::{Prime128OffsetA7F7, Prime64Offset59, Ring, Zero};
use std::array::from_fn;

type ReducedF = Prime64Offset59;
const REDUCED_D: usize = 64;

fn reduced_params(payload_mode: CommitmentPayloadMode) -> CommittedGroupParams {
    let mut params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q64Offset59,
        REDUCED_D,
        2,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(1),
    )
    .with_decomp(4, 8, 1, 2, 2)
    .expect("reduced ring-switch params");
    params.own_group_mut().opening.num_digits_fold = 3;
    params.payload_mode = payload_mode;
    params.ring_relation_mode = RingRelationMode::ReducedEvaluation;
    params
}

fn reduced_instance(
    params: &CommittedGroupParams,
) -> (
    RingRelationInstance<ReducedF>,
    akita_types::RelationRhsLayout,
) {
    let opening_batch = OpeningClaimsLayout::new(8, 1).expect("opening batch");
    let group_params = params
        .group_params(&opening_batch, 0)
        .expect("group params");
    let blocks = group_params.num_live_blocks();
    let challenges = Challenges::from_sparse(
        vec![
            SparseChallenge {
                positions: vec![0].into(),
                coeffs: vec![1].into(),
            };
            blocks
        ],
        blocks,
        1,
    )
    .expect("fold challenges");
    let multiplier = RingMultiplierOpeningPoint::from_base(&RingOpeningPoint {
        position_weights: (0..group_params.num_positions_per_block())
            .map(|position| ReducedF::from_u64(position as u64 + 2))
            .collect(),
        live_block_weights: vec![ReducedF::zero(); blocks],
    });
    let geometry = akita_types::RelationWitnessGeometry::for_level(params, &opening_batch, 1)
        .expect("relation geometry");
    let rhs_layout = geometry.rhs_layout().clone();
    let rhs_len = relation_rhs_coeff_len(&rhs_layout).expect("relation RHS length");
    let gamma = ReducedF::from_u64(7);
    let mut gamma_ring = vec![ReducedF::zero(); REDUCED_D];
    gamma_ring[0] = gamma;
    let instance = RingRelationInstance::new(
        vec![RingRelationGroupOpening::evaluation_trace(
            challenges, multiplier,
        )],
        1,
        opening_batch,
        vec![gamma],
        RingVec::from_coeffs_with_ring_dim(gamma_ring, REDUCED_D).expect("gamma ring"),
        RingVec::from_coeffs(vec![ReducedF::zero(); rhs_len]),
        RingVec::from_coeffs_with_ring_dim(vec![ReducedF::from_i64(-1); REDUCED_D], REDUCED_D)
            .expect("D rows"),
        params.role_dims(),
    )
    .expect("reduced relation instance");
    (instance, rhs_layout)
}

fn reduced_group_witness(
    params: &CommittedGroupParams,
    hint: AkitaCommitmentHint<ReducedF>,
) -> RingRelationGroupWitness<ReducedF> {
    let opening_batch = OpeningClaimsLayout::new(8, 1).expect("opening batch");
    let group_params = params
        .group_params(&opening_batch, 0)
        .expect("group params");
    let z_rows = group_params
        .num_positions_per_block()
        .checked_mul(group_params.num_digits_inner())
        .expect("Z row count");
    let blocks = group_params.num_live_blocks();
    let e_planes = blocks
        .checked_mul(group_params.num_digits_open())
        .expect("E plane count");
    RingRelationGroupWitness::from_parts(
        DecomposeFoldWitness::from_coefficient_parts::<REDUCED_D>(
            vec![[ReducedF::zero(); REDUCED_D]; z_rows],
            vec![[0; REDUCED_D]; z_rows],
        ),
        FoldChunkCoefficients::single(),
        DigitBlocks::new(
            vec![-1; e_planes * REDUCED_D],
            vec![group_params.num_digits_open(); blocks],
            REDUCED_D,
        )
        .expect("E digits"),
        RingVec::from_coeffs_with_ring_dim(vec![ReducedF::zero(); blocks * REDUCED_D], REDUCED_D)
            .expect("folded opening"),
        hint,
        params.role_dims(),
    )
}

fn inner_rows(params: &CommittedGroupParams) -> RingVec<ReducedF> {
    let opening_batch = OpeningClaimsLayout::new(8, 1).expect("opening batch");
    let group_params = params
        .group_params(&opening_batch, 0)
        .expect("group params");
    let rows = group_params
        .num_live_blocks()
        .checked_mul(group_params.a_rows_len())
        .expect("inner row count");
    RingVec::from_coeffs_with_ring_dim(vec![ReducedF::from_i64(-1); rows * REDUCED_D], REDUCED_D)
        .expect("inner rows")
}

fn expected_packed_digits(
    span: &CompressionWitnessSpan,
    packed: &akita_types::PackedNegativeBinary,
) -> Vec<i8> {
    assert_eq!(span.map(), packed.map());
    (0..span.range().len())
        .map(|linear| {
            if linear < packed.map().real_digit_count()
                && packed.bytes()[linear / 8] >> (linear % 8) & 1 == 1
            {
                -1
            } else {
                0
            }
        })
        .collect()
}

struct ReducedCompilerFixture {
    witness_layout: akita_types::WitnessLayout,
    live_len: usize,
    physical_field_len: usize,
    opening_source_len: usize,
    geometry: RelationAddressGeometry,
    relation_plan: RelationRangeImagePlan,
    tau1: Vec<ReducedF>,
    alpha: ReducedF,
    point: Vec<ReducedF>,
}

fn reduced_compiler_fixture(
    params: &CommittedGroupParams,
    instance: &RingRelationInstance<ReducedF>,
) -> ReducedCompilerFixture {
    let opening_batch = instance.opening_batch();
    let witness_layout = instance
        .segment_layout(params, None)
        .expect("reduced witness layout");
    let live_len = witness_layout.live_coeff_len();
    let physical_field_len = live_len.next_power_of_two();
    let geometry = RelationAddressGeometry::for_relation(
        &akita_types::RelationWitnessGeometry::for_level(params, opening_batch, 1)
            .expect("relation geometry"),
        REDUCED_D,
        live_len,
    )
    .expect("relation address geometry");
    let relation_geometry =
        akita_types::RelationWitnessGeometry::for_level(params, opening_batch, 1)
            .expect("relation geometry");
    let relation_plan = RelationRangeImagePlan::new(
        relation_geometry,
        geometry,
        DigitRangePlan::new(1usize << params.open().digits.log_basis).expect("digit range plan"),
        witness_layout.clone(),
        opening_batch,
    )
    .expect("relation plan");
    let tau1 = (0..params
        .relation_row_index_num_vars(opening_batch)
        .expect("row variables"))
        .map(|index| ReducedF::from_u64(17 + index as u64))
        .collect();
    let point = (0..geometry.relation_point_variable_count())
        .map(|index| ReducedF::from_u64(41 + index as u64))
        .collect();
    ReducedCompilerFixture {
        witness_layout,
        live_len,
        physical_field_len,
        opening_source_len: physical_field_len / REDUCED_D,
        geometry,
        relation_plan,
        tau1,
        alpha: ReducedF::from_u64(29),
        point,
    }
}

fn structured_reduced_evaluation(
    params: &CommittedGroupParams,
    instance: &RingRelationInstance<ReducedF>,
    setup: &akita_types::AkitaExpandedSetup<ReducedF>,
    fixture: &ReducedCompilerFixture,
) -> ReducedF {
    let opening_batch = instance.opening_batch();
    let row_count = params
        .relation_matrix_row_count(opening_batch.num_groups())
        .expect("relation rows");
    let eq_tau1 = EqPolynomial::evals_prefix(&fixture.tau1, row_count)
        .expect("row equality")
        .into();
    let group_params = params.group_params(opening_batch, 0).expect("group params");
    let setup_groups = vec![SetupContributionGroupInputs {
        group_id: 0,
        num_claims: opening_batch
            .group_layout(0)
            .expect("group layout")
            .num_polynomials(),
        depth_fold: group_params.num_digits_fold(),
        a_row_start: params.a_row_range(opening_batch, 0).expect("A rows").start,
        b_row_start: params
            .commitment_row_range(opening_batch, 0)
            .expect("B rows")
            .start,
    }];
    let fold_gadget = shared_setup_fold_gadget(params, opening_batch, &setup_groups)
        .expect("evaluation-trace fold gadget");
    let coefficient_bits = fixture.geometry.relation_coefficient_variable_count();
    let mut setup_plan = SetupContributionPlan::prepare::<ReducedF>(
        params,
        opening_batch,
        1,
        eq_tau1,
        &fixture.witness_layout,
        &setup_groups,
        PreparedRelationAddress::new(&fixture.point[coefficient_bits..]).expect("relation address"),
        Some(&fold_gadget),
        fixture.geometry,
    )
    .expect("setup contribution plan");
    setup_plan
        .materialize_direct_scan(
            PreparedCoefficientFunctional::reduced_evaluation(
                fixture.alpha,
                &fixture.point[..coefficient_bits],
                fixture.geometry,
            )
            .expect("reduced coefficient functional"),
        )
        .expect("reduced direct scan");
    let structured = setup_plan
        .evaluate_reduced_structured_group::<ReducedF>(
            0,
            instance
                .group_ambient_a_challenges(0)
                .expect("ambient challenges"),
            &instance
                .group_ring_multiplier_point(0)
                .expect("ring multiplier")
                .prepare_functional_multiplier(),
        )
        .expect("structured reduced evaluation");
    let compression = if params.payload_mode.is_compressed() {
        akita_types::build_reduced_compression_relation_weights::<ReducedF, ReducedF>(
            fixture.alpha,
            params,
            opening_batch,
            1,
            &fixture.tau1,
            &fixture.witness_layout,
            REDUCED_D,
            fixture.physical_field_len,
        )
        .expect("reduced compression relation weights")
        .evaluate_at_point(setup, &fixture.point)
        .expect("reduced compression evaluation")
    } else {
        ReducedF::zero()
    };
    structured
        + setup_plan
            .evaluate_direct::<ReducedF>(setup)
            .expect("direct reduced setup evaluation")
        + compression
}

fn assert_reduced_compiler_matches_structured_verifier(
    params: &CommittedGroupParams,
    instance: &RingRelationInstance<ReducedF>,
    setup: &akita_types::AkitaExpandedSetup<ReducedF>,
) {
    let fixture = reduced_compiler_fixture(params, instance);
    let dense = super::relation_weights::build_reduced_dense_relation_weights(
        setup,
        instance,
        fixture.alpha,
        params,
        &fixture.tau1,
        fixture.opening_source_len,
        REDUCED_D,
        &fixture.relation_plan,
    )
    .expect("reduced dense relation weights");
    let dense_evaluation =
        multilinear_eval(dense.evaluations(), &fixture.point).expect("dense MLE");
    assert_eq!(
        dense_evaluation,
        structured_reduced_evaluation(params, instance, setup, &fixture),
    );
    assert!(dense.evaluations()[fixture.live_len..]
        .iter()
        .all(ReducedF::is_zero));
}

#[test]
fn centered_i32_decompose_matches_ring_decompose() {
    type F = Prime128OffsetA7F7;
    const D: usize = 128;

    let centered = from_fn(|i| ((37 * i as i32 + 11) % 95) - 47);
    let ring =
        CyclotomicRing::<F, D>::from_coefficients(from_fn(|i| F::from_i64(centered[i] as i64)));

    for (num_digits, log_basis) in [
        (7usize, 3u32),
        (10usize, 2u32),
        (5usize, 5u32),
        (4usize, 6u32),
    ] {
        let mut got = vec![[0i8; D]; num_digits];
        balanced_decompose_centered_i32_i8_into(&centered, &mut got, log_basis);

        let mut expected = vec![[0i8; D]; num_digits];
        ring.balanced_decompose_pow2_i8_into(&mut expected, log_basis);
        assert_eq!(
            got, expected,
            "centered i32 decomposition mismatch for num_digits={num_digits} log_basis={log_basis}"
        );
    }
}

#[test]
fn compact_witness_keeps_exact_live_prefix() {
    let witness = (0..(5 * 8)).map(|value| value as i8).collect::<Vec<_>>();
    let (compact, col_bits, ring_bits) = build_w_evals_compact(
        PackedSignedDigits::from_i8_digits_auto(witness.clone()),
        8,
        1,
        7,
    )
    .expect("valid compact witness");

    assert_eq!(compact.decode(), witness);
    assert_eq!(col_bits, 3);
    assert_eq!(ring_bits, 3);
}

#[test]
fn packed_compact_witness_keeps_exact_live_prefix() {
    let witness = (0..(5 * 8)).map(|value| value as i8).collect::<Vec<_>>();
    let (compact, col_bits, ring_bits) =
        build_w_evals_compact(PackedSignedDigits::from_i8_digits_auto(witness), 8, 2, 7)
            .expect("valid packed witness");

    assert_eq!(compact.len(), 5 * 4);
    assert_eq!(col_bits, 3);
    assert_eq!(ring_bits, 2);
}
fn with_reduced_setup<R>(
    setup_coefficients: usize,
    test: impl FnOnce(&OperationCtx<'_, ReducedF, CpuBackend>, &AkitaProverSetup<ReducedF>) -> R,
) -> R {
    let setup = AkitaProverSetup::<ReducedF>::generate_with_capacity(
        8,
        1,
        SetupMatrixCapacity {
            num_field_elements: setup_coefficients,
        },
    )
    .expect("prover setup");
    let backend = CpuBackend::DEFAULT;
    let prepared = backend
        .prepare_expanded(setup.expanded.clone())
        .expect("prepared setup");
    let ctx =
        OperationCtx::new(&backend, &prepared, setup.expanded.as_ref()).expect("operation context");
    test(&ctx, &setup)
}

fn assert_reduced_compression_report(
    report: &crate::compute::compression::CompressionExecutionReport,
) {
    assert_eq!(report.quotient_lift_batches, 0);
    assert_eq!(report.quotient_rows, 0);
    assert_eq!(report.reduced_evaluation_batches, report.maps);
    assert!(report
        .batches
        .iter()
        .all(|batch| batch.relation_mode == RingRelationMode::ReducedEvaluation));
}

fn build_reduced_without_quotients(
    instance: &RingRelationInstance<ReducedF>,
    witness: RingRelationWitness<ReducedF>,
    ctx: &OperationCtx<'_, ReducedF, CpuBackend>,
    params: &CommittedGroupParams,
) -> crate::RecursiveWitnessFlat {
    reset_multi_group_quotient_calls();
    reset_quotient_decomposition_calls();
    let flat = ring_switch_build_w(instance, witness, ctx, params)
        .expect("valid reduced-evaluation witness");
    assert_eq!(multi_group_quotient_calls(), 0);
    assert_eq!(quotient_decomposition_calls(), 0);
    flat
}

fn retained_compression_digits(
    layout: &akita_types::WitnessLayout,
    compression: &crate::protocol::ring_relation::CompressionWitnessMaterialization<ReducedF>,
) -> Vec<(CompressionWitnessSpan, Vec<i8>)> {
    layout
        .compression_layers()
        .iter()
        .flat_map(|layer| {
            let outer = compression
                .source(CompressionSourceId::Outer { group_index: 0 })
                .expect("outer source");
            let opening = compression
                .source(CompressionSourceId::Opening)
                .expect("opening source");
            let f_span = layer
                .f_spans()
                .first()
                .expect("F compression span")
                .1
                .clone();
            let h_span = layer.h_span().clone();
            [
                (
                    f_span.clone(),
                    expected_packed_digits(&f_span, &outer.witness.stages()[layer.map_index()]),
                ),
                (
                    h_span.clone(),
                    expected_packed_digits(&h_span, &opening.witness.stages()[layer.map_index()]),
                ),
            ]
        })
        .collect()
}

#[test]
fn compressed_reduced_ring_switch_keeps_quotient_paths_cold() {
    let params = reduced_params(CommitmentPayloadMode::Compressed);
    let (instance, rhs_layout) = reduced_instance(&params);
    let outer_plan = rhs_layout
        .group_compression_plan(0)
        .expect("outer compression plan")
        .1
        .clone();
    let opening_plan = rhs_layout
        .opening_compression_plan()
        .expect("opening compression plan")
        .clone();
    let compression_setup_coefficients = outer_plan
        .maps()
        .iter()
        .chain(opening_plan.maps())
        .map(|map| map.input_width() * map.ring_dimension())
        .max()
        .expect("compression maps");
    let setup_coefficients = compression_setup_coefficients.max(
        active_setup_field_len(&params, instance.opening_batch())
            .expect("active setup field length"),
    );
    with_reduced_setup(setup_coefficients, |ctx, setup| {
        let (mut outer_outputs, outer_report) = execute_compression_chains(
            ctx,
            vec![CompressionExecutionInput {
                id: (),
                plan: outer_plan.clone(),
                coefficients: vec![ReducedF::from_i64(-1); outer_plan.source_coefficients()],
                relation_mode: RingRelationMode::ReducedEvaluation,
            }],
        )
        .expect("reduced outer compression");
        assert_reduced_compression_report(&outer_report);
        let outer_output = outer_outputs.pop().expect("outer output");
        let hint = AkitaCommitmentHint::singleton_with_reduced_outer_compression(
            inner_rows(&params),
            &outer_output.witness,
        )
        .expect("compressed hint");
        let outer_source = CompressionSourceWitness::from_outer_hint(
            0,
            &outer_plan,
            &hint,
            outer_output.terminal.coefficients().to_vec(),
            RingRelationMode::ReducedEvaluation,
        )
        .expect("retained outer source");
        let (compression, opening_report) = materialize_compression_witness(
            ctx,
            &rhs_layout,
            vec![outer_source],
            instance.v(),
            RingRelationMode::ReducedEvaluation,
        )
        .expect("reduced relation compression");
        assert_reduced_compression_report(&opening_report);

        let layout = instance
            .segment_layout(&params, None)
            .expect("compressed witness layout");
        let expected_digits = retained_compression_digits(&layout, &compression);
        let witness = RingRelationWitness::from_groups(
            vec![reduced_group_witness(&params, hint)],
            RelationDQuotientWitness::ReducedEvaluation,
            Some(compression),
        );
        let flat = build_reduced_without_quotients(&instance, witness, ctx, &params);
        assert!(layout.r_rows().is_empty());
        assert_eq!(layout.quotient_depth(), None);
        assert_eq!(flat.live_coeff_len(), layout.live_coeff_len());
        let digits = flat.to_i8_digits();
        for (span, expected) in expected_digits {
            assert_eq!(&digits[span.range()], expected);
        }
        assert_reduced_compiler_matches_structured_verifier(
            &params,
            &instance,
            setup.expanded.as_ref(),
        );
    });
}

#[test]
fn raw_reduced_ring_switch_keeps_quotient_paths_cold() {
    let params = reduced_params(CommitmentPayloadMode::Raw);
    let (instance, _) = reduced_instance(&params);
    let setup_coefficients = active_setup_field_len(&params, instance.opening_batch())
        .expect("active setup field length");
    with_reduced_setup(setup_coefficients, |ctx, setup| {
        let hint = AkitaCommitmentHint::singleton(inner_rows(&params)).expect("raw hint");
        let witness = RingRelationWitness::from_groups(
            vec![reduced_group_witness(&params, hint)],
            RelationDQuotientWitness::ReducedEvaluation,
            None,
        );
        let flat = build_reduced_without_quotients(&instance, witness, ctx, &params);
        let layout = instance
            .segment_layout(&params, None)
            .expect("raw witness layout");
        assert!(layout.r_rows().is_empty());
        assert_eq!(layout.quotient_depth(), None);
        assert!(layout.compression_layers().is_empty());
        assert_eq!(flat.live_coeff_len(), layout.live_coeff_len());
        assert_reduced_compiler_matches_structured_verifier(
            &params,
            &instance,
            setup.expanded.as_ref(),
        );
    });
}

#[test]
fn reduced_ring_switch_rejects_quotient_witness_without_running_quotients() {
    let params = reduced_params(CommitmentPayloadMode::Raw);
    let (instance, _) = reduced_instance(&params);
    let setup_coefficients = active_setup_field_len(&params, instance.opening_batch())
        .expect("active setup field length");
    with_reduced_setup(setup_coefficients, |ctx, _| {
        let hint = AkitaCommitmentHint::singleton(inner_rows(&params)).expect("mismatched hint");
        let witness = RingRelationWitness::from_groups(
            vec![reduced_group_witness(&params, hint)],
            RelationDQuotientWitness::QuotientLift(RingVec::from_coeffs(Vec::new())),
            None,
        );
        reset_multi_group_quotient_calls();
        reset_quotient_decomposition_calls();
        assert!(ring_switch_build_w(&instance, witness, ctx, &params).is_err());
        assert_eq!(multi_group_quotient_calls(), 0);
        assert_eq!(quotient_decomposition_calls(), 0);
    });
}
