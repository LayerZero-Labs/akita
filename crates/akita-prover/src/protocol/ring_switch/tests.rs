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
use akita_algebra::CyclotomicRing;
use akita_challenges::{Challenges, SparseChallenge, SparseChallengeConfig};
use akita_types::{
    relation_rhs_coeff_len, AkitaCommitmentHint, CommitmentPayloadMode, CommittedGroupParams,
    CompressionWitnessSpan, DigitBlocks, OpeningClaimsLayout, RingMultiplierOpeningPoint,
    RingOpeningPoint, RingRelationGroupOpening, RingRelationInstance, RingRelationMode, RingVec,
    SetupMatrixCapacity, SisModulusProfileId,
};
use jolt_field::{Prime128OffsetA7F7, Prime64Offset59, Ring};
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
        position_weights: vec![ReducedF::zero(); group_params.num_positions_per_block()],
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

#[test]
fn reduced_ring_switch_build_w_keeps_every_quotient_path_cold() {
    let compressed_params = reduced_params(CommitmentPayloadMode::Compressed);
    let (compressed_instance, rhs_layout) = reduced_instance(&compressed_params);
    let outer_plan = rhs_layout
        .group_compression_plan(0)
        .expect("outer compression plan")
        .1
        .clone();
    let opening_plan = rhs_layout
        .opening_compression_plan()
        .expect("opening compression plan")
        .clone();
    let setup_coefficients = outer_plan
        .maps()
        .iter()
        .chain(opening_plan.maps())
        .map(|map| map.input_width() * map.ring_dimension())
        .max()
        .expect("compression maps");
    let setup = AkitaProverSetup::<ReducedF>::generate_with_capacity(
        8,
        1,
        SetupMatrixCapacity {
            num_field_elements: setup_coefficients,
        },
    )
    .expect("prover setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_expanded(setup.expanded.clone())
        .expect("prepared setup");
    let ctx = OperationCtx::new(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
        .expect("operation context");

    let compressed_inner_rows = inner_rows(&compressed_params);
    let (mut outer_outputs, outer_report) = execute_compression_chains(
        &ctx,
        vec![CompressionExecutionInput {
            id: (),
            plan: outer_plan.clone(),
            coefficients: vec![ReducedF::from_i64(-1); outer_plan.source_coefficients()],
            relation_mode: RingRelationMode::ReducedEvaluation,
        }],
    )
    .expect("reduced outer compression");
    assert_eq!(outer_report.quotient_lift_batches, 0);
    assert_eq!(outer_report.quotient_rows, 0);
    assert_eq!(outer_report.reduced_evaluation_batches, outer_report.maps);
    assert!(outer_report
        .batches
        .iter()
        .all(|batch| batch.relation_mode == RingRelationMode::ReducedEvaluation));
    let outer_output = outer_outputs.pop().expect("outer output");
    let compressed_hint = AkitaCommitmentHint::singleton_with_reduced_outer_compression(
        compressed_inner_rows,
        &outer_output.witness,
    )
    .expect("compressed hint");
    let outer_source = CompressionSourceWitness::from_outer_hint(
        0,
        &outer_plan,
        &compressed_hint,
        outer_output.terminal.coefficients().to_vec(),
        RingRelationMode::ReducedEvaluation,
    )
    .expect("retained outer source");
    let (compression, opening_report) = materialize_compression_witness(
        &ctx,
        &rhs_layout,
        vec![outer_source],
        compressed_instance.v(),
        RingRelationMode::ReducedEvaluation,
    )
    .expect("reduced relation compression");
    assert_eq!(opening_report.quotient_lift_batches, 0);
    assert_eq!(opening_report.quotient_rows, 0);
    assert_eq!(
        opening_report.reduced_evaluation_batches,
        opening_report.maps
    );
    assert!(opening_report
        .batches
        .iter()
        .all(|batch| batch.relation_mode == RingRelationMode::ReducedEvaluation));

    let compressed_layout = compressed_instance
        .segment_layout(&compressed_params, None)
        .expect("compressed witness layout");
    let retained_compression_digits = compressed_layout
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
        .collect::<Vec<_>>();
    let compressed_witness = RingRelationWitness::from_groups(
        vec![reduced_group_witness(&compressed_params, compressed_hint)],
        RelationDQuotientWitness::ReducedEvaluation,
        Some(compression),
    );
    reset_multi_group_quotient_calls();
    reset_quotient_decomposition_calls();
    let compressed_w = ring_switch_build_w(
        &compressed_instance,
        compressed_witness,
        &ctx,
        &compressed_params,
    )
    .expect("compressed reduced witness");
    assert_eq!(multi_group_quotient_calls(), 0);
    assert_eq!(quotient_decomposition_calls(), 0);
    assert!(compressed_layout.r_rows().is_empty());
    assert_eq!(compressed_layout.quotient_depth(), None);
    assert_eq!(
        compressed_w.live_coeff_len(),
        compressed_layout.live_coeff_len()
    );
    let compressed_digits = compressed_w.to_i8_digits();
    for (span, expected) in retained_compression_digits {
        assert_eq!(&compressed_digits[span.range()], expected);
    }

    let raw_params = reduced_params(CommitmentPayloadMode::Raw);
    let (raw_instance, _) = reduced_instance(&raw_params);
    let raw_hint = AkitaCommitmentHint::singleton(inner_rows(&raw_params)).expect("raw hint");
    let raw_witness = RingRelationWitness::from_groups(
        vec![reduced_group_witness(&raw_params, raw_hint)],
        RelationDQuotientWitness::ReducedEvaluation,
        None,
    );
    reset_multi_group_quotient_calls();
    reset_quotient_decomposition_calls();
    let raw_w = ring_switch_build_w(&raw_instance, raw_witness, &ctx, &raw_params)
        .expect("raw reduced witness");
    let raw_layout = raw_instance
        .segment_layout(&raw_params, None)
        .expect("raw witness layout");
    assert_eq!(multi_group_quotient_calls(), 0);
    assert_eq!(quotient_decomposition_calls(), 0);
    assert!(raw_layout.r_rows().is_empty());
    assert_eq!(raw_layout.quotient_depth(), None);
    assert!(raw_layout.compression_layers().is_empty());
    assert_eq!(raw_w.live_coeff_len(), raw_layout.live_coeff_len());

    let mismatched_hint =
        AkitaCommitmentHint::singleton(inner_rows(&raw_params)).expect("mismatched hint");
    let mismatched = RingRelationWitness::from_groups(
        vec![reduced_group_witness(&raw_params, mismatched_hint)],
        RelationDQuotientWitness::QuotientLift(RingVec::from_coeffs(Vec::new())),
        None,
    );
    reset_multi_group_quotient_calls();
    reset_quotient_decomposition_calls();
    assert!(ring_switch_build_w(&raw_instance, mismatched, &ctx, &raw_params).is_err());
    assert_eq!(multi_group_quotient_calls(), 0);
    assert_eq!(quotient_decomposition_calls(), 0);
}
