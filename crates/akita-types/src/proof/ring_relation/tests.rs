#[path = "tests/dimension_tests.rs"]
mod dimension_tests;
#[path = "tests/support.rs"]
mod support;

use super::*;
use crate::layout::GroupOpenPhaseParams;
use crate::r_decomp_levels;
use crate::DigitBlocks;
use crate::{
    emit_witness_e_planes, emit_witness_t_planes, emit_witness_z_planes, relation_rhs_coeff_len,
    InnerCommitMatrixParams, OpenCommitMatrixParams, OuterCommitMatrixParams,
    PolynomialGroupLayout, RingOpeningPoint,
};
use akita_challenges::{SparseChallenge, SparseChallengeConfig};
use jolt_field::{Fp32, One, Zero};
use support::{flatten_markers, marker};

type F = Fp32<251>;
const D: usize = 32;
const MULTI_GROUP_D: usize = 64;

fn relation_layout(
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
) -> crate::RelationRhsLayout {
    crate::RelationWitnessGeometry::for_evaluation_trace_execution(lp, opening_batch)
        .expect("relation geometry")
        .rhs_layout()
        .clone()
}

fn certify_test_sis_bounds(lp: &mut CommittedGroupParams) {
    let inner_bound = *crate::sis::inner_coeff_linf_bounds(
        lp.inner().matrix.sis_modulus_profile(),
        u32::try_from(lp.d_a()).expect("test ring dimension"),
    )
    .first()
    .expect("exact test A bounds");
    lp.own_group_mut().profile.inner.matrix = InnerCommitMatrixParams::new_unchecked(
        lp.inner().matrix.security_policy(),
        lp.inner()
            .matrix
            .sis_table_key()
            .expect("test matrix is L infinity")
            .table_digest,
        lp.inner().matrix.sis_modulus_profile(),
        lp.inner().matrix.output_rank(),
        lp.inner().matrix.input_width(),
        inner_bound,
        lp.d_a(),
    );
    lp.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        lp.outer().matrix.security_policy(),
        lp.outer().matrix.sis_table_key().table_digest,
        lp.outer().matrix.sis_modulus_profile(),
        lp.outer().matrix.output_rank(),
        lp.outer().matrix.input_width(),
        3,
        lp.d_a(),
    );
}

fn fold_challenge_config() -> SparseChallengeConfig {
    SparseChallengeConfig::pm1_only(1)
}

fn opening_point(lp: &CommittedGroupParams) -> RingOpeningPoint<F> {
    RingOpeningPoint {
        position_weights: vec![F::zero(); lp.blocks().positions_per_block],
        live_block_weights: vec![F::zero(); lp.blocks().live_blocks],
    }
}

fn test_level_params(_num_fold_claims: usize) -> CommittedGroupParams {
    let mut params = CommittedGroupParams::params_only(
        crate::SisModulusProfileId::Q32Offset99,
        D,
        2,
        1,
        1,
        1,
        fold_challenge_config(),
    )
    .with_decomp(4, 8, 1, 2, 2)
    .expect("test params");
    params.own_group_mut().opening.num_digits_fold = 3;
    params
}

fn test_challenges(lp: &CommittedGroupParams, num_claims: usize) -> Challenges {
    let total = lp.blocks().live_blocks * num_claims;
    Challenges::from_sparse(
        vec![
            SparseChallenge {
                positions: vec![0].into(),
                coeffs: vec![1].into(),
            };
            total
        ],
        lp.blocks().live_blocks,
        num_claims,
    )
    .expect("challenges")
}

fn packing_challenges(lp: &CommittedGroupParams, num_claims: usize) -> Challenges {
    let config = lp.fold_challenge_config();
    let sparse = (0..lp.blocks().live_blocks * num_claims)
        .map(|_| SparseChallenge {
            positions: (0..config.weight())
                .map(|position| position as u32)
                .collect(),
            coeffs: (0..config.count_pm1)
                .map(|_| 1)
                .chain((0..config.count_pm2).map(|_| 2))
                .collect(),
        })
        .collect();
    Challenges::from_sparse(sparse, lp.blocks().live_blocks, num_claims)
        .expect("packing challenges")
}

fn evaluation_trace_openings(
    challenges: Vec<Challenges>,
    points: Vec<RingMultiplierOpeningPoint<F>>,
) -> Vec<RingRelationGroupOpening<F>> {
    challenges
        .into_iter()
        .zip(points)
        .map(|(challenges, ring_multiplier_point)| {
            RingRelationGroupOpening::evaluation_trace(challenges, ring_multiplier_point)
        })
        .collect()
}

#[test]
fn relation_instance_rejects_empty_y() {
    let lp = test_level_params(1);
    let opening_batch = OpeningClaimsLayout::new(2, 1).expect("valid opening batch");
    let opening_point = opening_point(&lp);
    let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
    let err = RingRelationInstance::<F>::new(
        evaluation_trace_openings(
            vec![test_challenges(&lp, opening_batch.num_total_polynomials())],
            vec![ring_multiplier_point],
        ),
        1,
        opening_batch,
        vec![F::one()],
        RingVec::from_ring_elems::<D>(&[CyclotomicRing::one()]),
        RingVec::from_ring_elems::<D>(&[]),
        RingVec::from_ring_elems::<D>(&[]),
        CommitmentRingDims::uniform(D),
    )
    .expect_err("empty rhs must be rejected");
    assert!(
        format!("{err:?}").contains("ring relation rhs must contain at least the consistency row"),
        "unexpected error: {err:?}"
    );
}

fn chunk_test_level_params(
    block_index_bits: usize,
    _num_fold_claims: usize,
) -> CommittedGroupParams {
    // num_live_blocks = 2^block_index_bits, num_positions_per_block = 2^position_index_bits, single-tier.
    let mut params = CommittedGroupParams::params_only(
        crate::SisModulusProfileId::Q32Offset99,
        D,
        2,
        1,
        1,
        1,
        fold_challenge_config(),
    )
    .with_decomp(4, 1usize << (2 + block_index_bits), 1, 2, 2)
    .expect("test params");
    params.own_group_mut().opening.num_digits_fold = 3;
    params
}

/// Build a minimal non-terminal relation instance whose layout-relevant
/// shape is `opening_batch.num_total_polynomials() = num_claims` and
/// `y.len() = num_rows` (the only fields
/// [`RingRelationInstance::segment_layout`] reads).
fn build_instance(
    lp: &CommittedGroupParams,
    num_claims: usize,
    _num_rows: usize,
) -> RingRelationInstance<F> {
    let opening_batch = OpeningClaimsLayout::new(8, num_claims).expect("opening batch");
    let rhs_coeff_len =
        relation_rhs_coeff_len(&relation_layout(lp, &opening_batch)).expect("relation rhs length");
    let opening_point = opening_point(lp);
    let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
    RingRelationInstance::<F>::new(
        evaluation_trace_openings(
            vec![test_challenges(lp, num_claims)],
            vec![ring_multiplier_point],
        ),
        1,
        opening_batch,
        vec![F::one(); num_claims],
        RingVec::from_ring_elems::<D>(&vec![CyclotomicRing::one(); num_claims]),
        RingVec::from_coeffs(vec![F::zero(); rhs_coeff_len]),
        RingVec::from_ring_elems::<D>(&[]),
        CommitmentRingDims::uniform(D),
    )
    .expect("instance")
}

#[test]
fn resolve_single_chunk_matches_legacy_offsets() {
    let num_claims = 3;
    let lp = chunk_test_level_params(1, num_claims);
    assert_eq!(lp.witness_chunk.num_chunks, 1);
    let _lens = ring_relation_segment_lengths::<F>(
        &lp,
        RingRelationOpeningCounts {
            num_claims,
            num_t_vectors: num_claims,
        },
    )
    .expect("lengths");

    let resolved = build_instance(&lp, num_claims, 4)
        .segment_layout(&lp, None)
        .expect("resolved layout");
    assert_eq!(resolved.num_chunks_for_group(0), 1);
    let unit = &resolved.units()[0];
    // Single-unit compact offsets: z first, then e, t, and the shared r tail.
    assert_eq!(unit.z_range().start, 0);
    assert_eq!(unit.e_range().start, unit.z_range().end);
    assert_eq!(unit.t_range().start, unit.e_range().end);
    // The shared r tail follows the unit's compact z, e, and t ranges.
    assert_eq!(resolved.tail_range().start, unit.t_range().end);
    assert_eq!(unit.global_block_start(), 0);
    assert_eq!(unit.num_live_blocks(), lp.blocks().live_blocks);
}

#[test]
fn resolve_multi_chunk_offsets_contiguous_and_cover_blocks() {
    let num_claims = 2;
    for w in [1usize, 2, 4, 8] {
        let mut lp = chunk_test_level_params(3, num_claims); // num_live_blocks = 8
        if w > 1 {
            lp.witness_chunk = crate::witness::ChunkedWitnessCfg {
                num_chunks: w,
                num_activated_levels: 1,
            };
        }
        let layout = build_instance(&lp, num_claims, 4)
            .segment_layout(&lp, None)
            .expect("layout");
        assert_eq!(layout.num_chunks_for_group(0), w);
        let blocks_per_chunk = lp.blocks().live_blocks / w;

        // Partitioned e/t lengths sum to the single-machine totals; z replicated.
        let e_sum: usize = layout.units().iter().map(|unit| unit.e_range().len()).sum();
        let t_sum: usize = layout.units().iter().map(|unit| unit.t_range().len()).sum();
        assert_eq!(
            e_sum,
            lp.open().digits.num_digits * lp.blocks().live_blocks * num_claims * D
        );
        assert_eq!(
            t_sum,
            lp.outer().digits.num_digits
                * lp.inner().matrix.output_rank()
                * lp.blocks().live_blocks
                * num_claims
                * D
        );
        for unit in layout.units() {
            assert_eq!(
                unit.z_range().len(),
                lp.blocks().positions_per_block
                    * lp.inner().digits.num_digits
                    * lp.num_digits_fold()
                    * D
            );
        }

        // Ownership units are contiguous and z-first; the shared r tail follows all units.
        let stride = layout.units()[0].t_range().end;
        for (j, unit) in layout.units().iter().enumerate() {
            let base = j * stride;
            assert_eq!(unit.z_range().start, base);
            assert_eq!(unit.e_range().start, unit.z_range().end);
            assert_eq!(unit.t_range().start, unit.e_range().end);
            assert_eq!(unit.global_block_start(), j * blocks_per_chunk);
        }
        assert_eq!(layout.tail_range().start, w * stride);
        // Block windows tile [0, num_live_blocks).
        assert_eq!(
            layout.units().last().unwrap().global_block_start() + blocks_per_chunk,
            lp.blocks().live_blocks
        );
    }
}

#[test]
fn resolve_rejects_bad_chunk_count() {
    let num_claims = 2;
    // num_chunks = 3 is not a power of two.
    let mut lp = chunk_test_level_params(3, num_claims);
    lp.witness_chunk = crate::witness::ChunkedWitnessCfg {
        num_chunks: 3,
        num_activated_levels: 1,
    };
    assert!(build_instance(&lp, num_claims, 4)
        .segment_layout(&lp, None)
        .is_err());
}

#[test]
fn resolve_preserves_empty_chunk_slots() {
    let num_claims = 2;
    let mut lp = chunk_test_level_params(2, num_claims);
    lp.witness_chunk = crate::witness::ChunkedWitnessCfg {
        num_chunks: 8,
        num_activated_levels: 1,
    };
    let layout = build_instance(&lp, num_claims, 4)
        .segment_layout(&lp, None)
        .expect("layout with empty chunk slots");
    let expected_ranges = crate::dyadic_block_ranges(4, 8).expect("chunk ranges");
    assert_eq!(layout.units().len(), 8);
    for (unit, expected_range) in layout.units().iter().zip(expected_ranges) {
        assert_eq!(unit.global_block_range(), expected_range);
        assert_eq!(unit.e_range().is_empty(), unit.num_live_blocks() == 0);
        assert_eq!(unit.t_range().is_empty(), unit.num_live_blocks() == 0);
        assert!(!unit.z_range().is_empty());
    }
}

#[test]
fn resolve_rejects_capacity_overflow() {
    let num_claims = 2;
    let lp = chunk_test_level_params(3, num_claims);
    // A witness ring capacity of 1 is far smaller than offset_r + r_len.
    assert!(
        build_instance(&lp, num_claims, 4)
            .segment_layout(&lp, Some(1))
            .is_err(),
        "tiny witness capacity must be rejected"
    );
    // A generous capacity passes.
    build_instance(&lp, num_claims, 4)
        .segment_layout(&lp, Some(1 << 20))
        .expect("ample capacity");
}

#[test]
fn relation_segment_layout_uses_same_axis_contract() {
    let lp = test_level_params(3);
    let opening_batch = OpeningClaimsLayout::new(2, 3).expect("valid batch");
    let opening_point = opening_point(&lp);
    let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
    let relation_rhs_layout = relation_layout(&lp, &opening_batch);
    let rhs_coeff_len = relation_rhs_coeff_len(&relation_rhs_layout).expect("rhs length");
    let instance = RingRelationInstance::<F>::new(
        evaluation_trace_openings(
            vec![test_challenges(&lp, opening_batch.num_total_polynomials())],
            vec![ring_multiplier_point],
        ),
        1,
        opening_batch,
        vec![F::one(); 3],
        RingVec::from_ring_elems::<D>(&[CyclotomicRing::one(); 3]),
        RingVec::from_coeffs(vec![F::zero(); rhs_coeff_len]),
        RingVec::from_ring_elems::<D>(&vec![CyclotomicRing::zero(); relation_rhs_layout.n_d]),
        CommitmentRingDims::uniform(D),
    )
    .expect("same-axis relation");

    let layout = instance.segment_layout(&lp, None).expect("layout");
    let unit = &layout.units()[0];
    assert_eq!(layout.num_chunks_for_group(0), 1);
    assert_eq!(unit.z_range().start, 0);
    assert_eq!(unit.e_range().start, unit.z_range().end);
    assert_eq!(unit.t_range().start, unit.e_range().end);
    assert_eq!(layout.tail_range().start, unit.t_range().end);
    instance
        .check_v_shape_for_level(&lp)
        .expect("v rows match layout");
}

fn multi_group_one_three_fixture() -> (CommittedGroupParams, OpeningClaimsLayout) {
    use crate::schedule::GroupCommitPhaseParams;
    let fold_challenge_config = SparseChallengeConfig::production_for_ring_dim(MULTI_GROUP_D)
        .expect("multi-group test ring dimension has a production challenge");
    let lp = CommittedGroupParams::params_only(
        crate::SisModulusProfileId::Q128OffsetA7F7,
        MULTI_GROUP_D,
        3,
        2,
        4,
        3,
        fold_challenge_config,
    )
    .with_decomp(4, 16, 2, 2, 2)
    .expect("multi-group main params");
    let mut precommit_lp = CommittedGroupParams::params_only(
        crate::SisModulusProfileId::Q128OffsetA7F7,
        MULTI_GROUP_D,
        3,
        2,
        4,
        3,
        fold_challenge_config,
    )
    .with_decomp(4, 16, 2, 2, 2)
    .expect("multi-group precommit params");
    certify_test_sis_bounds(&mut precommit_lp);
    let precommit = GroupOpenPhaseParams {
        setup_natural_len: None,
        profile: GroupCommitPhaseParams::from_params_unchecked_for_test(
            PolynomialGroupLayout::new(4, 1),
            &precommit_lp,
        ),
        opening: crate::GroupOpeningPlan::evaluation_trace(
            precommit_lp.fold_challenge_config(),
            precommit_lp.open().digits.log_basis,
            precommit_lp.open().digits.num_digits,
            precommit_lp.num_digits_fold(),
        ),
    };
    let mut multi_group_lp = lp;
    multi_group_lp
        .set_precommitted_groups(vec![precommit])
        .unwrap();
    let batch = OpeningClaimsLayout::from_root_groups(
        &[PolynomialGroupLayout::new(4, 1)],
        PolynomialGroupLayout::new(4, 1),
    )
    .expect("multi-group opening batch");
    (multi_group_lp, batch)
}

#[test]
fn multi_group_segment_layout_total_matches_next_w_len() {
    let (lp, opening_batch) = multi_group_one_three_fixture();
    let relation_rhs_coefficients =
        relation_rhs_coeff_len(&relation_layout(&lp, &opening_batch)).expect("rhs length");
    let opening_point_pre = opening_point(&lp);
    let opening_point_final = opening_point(&lp);
    let ring_multiplier_pre = RingMultiplierOpeningPoint::from_base(&opening_point_pre);
    let ring_multiplier_final = RingMultiplierOpeningPoint::from_base(&opening_point_final);
    let instance = RingRelationInstance::<F>::new(
        evaluation_trace_openings(
            vec![test_challenges(&lp, 1), test_challenges(&lp, 1)],
            vec![ring_multiplier_pre, ring_multiplier_final],
        ),
        1,
        opening_batch.clone(),
        vec![F::one(); opening_batch.num_total_polynomials()],
        RingVec::from_ring_elems::<MULTI_GROUP_D>(&vec![
            CyclotomicRing::one();
            opening_batch.num_total_polynomials()
        ]),
        RingVec::from_coeffs(vec![F::zero(); relation_rhs_coefficients]),
        RingVec::from_ring_elems::<MULTI_GROUP_D>(&vec![
            CyclotomicRing::zero();
            lp.open().matrix.output_rank()
        ]),
        CommitmentRingDims::uniform(MULTI_GROUP_D),
    )
    .expect("multi-group instance");

    let layout = instance
        .segment_layout(&lp, None)
        .expect("multi-group segment layout");
    let num_groups = opening_batch.num_groups();
    // With one chunk, authenticated group order gives one contiguous
    // `[z_g | e_g | t_g]` unit per group before the shared R tail.
    assert_eq!(layout.units().len(), num_groups);
    let quotient_depth = r_decomp_levels::<F>(lp.open().digits.log_basis);
    let quotient_coeff_len = layout
        .r_rows()
        .iter()
        .map(|row| row.geometry().physical_coefficient_width() * quotient_depth)
        .sum::<usize>();

    let mut base = 0usize;
    for (p, unit) in layout.units().iter().enumerate() {
        let z_g = unit.z_range().len();
        let e_g = unit.e_range().len();
        let t_g = unit.t_range().len();
        assert_eq!(unit.z_range().start, base);
        assert_eq!(unit.e_range().start, base + z_g);
        assert_eq!(unit.t_range().start, base + z_g + e_g);
        if p + 1 == num_groups {
            assert_eq!(layout.tail_range().start, base + z_g + e_g + t_g);
            assert!(layout.tail_range().len() >= quotient_coeff_len);
        }
        base += z_g + e_g + t_g;
    }

    let expected_witness_len = lp
        .output_witness_len::<F>(&opening_batch, 1)
        .expect("next w len");
    assert_eq!(layout.live_coeff_len(), expected_witness_len);
}

#[test]
fn multi_group_segment_layout_resolves_group_shard_product() {
    let (mut lp, opening_batch) = multi_group_one_three_fixture();
    lp.witness_chunk = crate::witness::ChunkedWitnessCfg {
        num_chunks: 2,
        num_activated_levels: 1,
    };
    let relation_rhs_coefficients =
        relation_rhs_coeff_len(&relation_layout(&lp, &opening_batch)).expect("rhs length");
    let opening_point_pre = opening_point(&lp);
    let opening_point_final = opening_point(&lp);
    let ring_multiplier_pre = RingMultiplierOpeningPoint::from_base(&opening_point_pre);
    let ring_multiplier_final = RingMultiplierOpeningPoint::from_base(&opening_point_final);
    let gamma_len = opening_batch.num_total_polynomials();
    let instance = RingRelationInstance::<F>::new(
        evaluation_trace_openings(
            vec![test_challenges(&lp, 1), test_challenges(&lp, 1)],
            vec![ring_multiplier_pre, ring_multiplier_final],
        ),
        1,
        opening_batch,
        vec![F::one(); gamma_len],
        RingVec::from_ring_elems::<MULTI_GROUP_D>(&vec![CyclotomicRing::one(); gamma_len]),
        RingVec::from_coeffs(vec![F::zero(); relation_rhs_coefficients]),
        RingVec::from_ring_elems::<MULTI_GROUP_D>(&vec![
            CyclotomicRing::zero();
            lp.open().matrix.output_rank()
        ]),
        CommitmentRingDims::uniform(MULTI_GROUP_D),
    )
    .expect("multi-group instance");
    let layout = instance
        .segment_layout(&lp, None)
        .expect("multi-group multi-chunk layout");
    assert_eq!(layout.units().len(), 4);
    assert_eq!(
        layout
            .units()
            .iter()
            .map(|unit| (unit.group_index(), unit.chunk_index()))
            .collect::<Vec<_>>(),
        vec![(1, 0), (0, 0), (1, 1), (0, 1)]
    );
    for group_index in [1, 0] {
        let mut units = layout.units_for_group(group_index).expect("group units");
        let first = units.next().expect("first group unit");
        let second = units.next().expect("second group unit");
        assert!(units.next().is_none());
        assert_eq!(first.global_block_range(), 0..2);
        assert_eq!(second.global_block_range(), 2..4);
        assert!(first.t_range().end < second.z_range().start);
    }
    assert_eq!(
        layout.units().last().expect("last unit").t_range().end,
        layout.tail_range().start
    );

    // Independent dense emitter oracle: each physical range must contain
    // the corresponding semantic source planes in digit-innermost order.
    let mut emitted = vec![0i8; layout.live_coeff_len()];
    for group_index in [1, 0] {
        let params = lp
            .group_params(instance.opening_batch(), group_index)
            .expect("group params");
        let num_claims = instance
            .opening_batch()
            .group_layout(group_index)
            .expect("group layout")
            .num_polynomials();
        let num_live_blocks = params.num_live_blocks();
        let depth_witness = params.num_digits_inner();
        let depth_commit = params.num_digits_outer();
        let depth_open = params.num_digits_open();
        let n_a = params.a_rows_len();
        let e_source = (0..num_claims * num_live_blocks * depth_open)
            .map(|index| marker::<MULTI_GROUP_D>(100 * group_index + index))
            .collect::<Vec<_>>();
        let t_source = (0..num_claims * num_live_blocks * n_a * depth_commit)
            .map(|index| marker::<MULTI_GROUP_D>(300 * group_index + index))
            .collect::<Vec<_>>();
        let e_digits = DigitBlocks::new(
            e_source.as_flattened().to_vec(),
            vec![depth_open; num_claims * num_live_blocks],
            MULTI_GROUP_D,
        )
        .expect("E digits");
        let t_digits = DigitBlocks::new(
            t_source.as_flattened().to_vec(),
            vec![n_a * depth_commit; num_claims * num_live_blocks],
            MULTI_GROUP_D,
        )
        .expect("T digits");
        let depth_fold = params.num_digits_fold();
        for unit in layout.units_for_group(group_index).expect("units") {
            emit_witness_e_planes::<MULTI_GROUP_D>(
                &mut emitted,
                unit,
                MULTI_GROUP_D,
                num_claims,
                depth_open,
                &e_digits,
                num_live_blocks,
            )
            .expect("emit E");
            emit_witness_t_planes::<MULTI_GROUP_D, MULTI_GROUP_D>(
                &mut emitted,
                unit,
                num_claims,
                n_a,
                depth_commit,
                &t_digits,
                num_live_blocks,
            )
            .expect("emit T");
            let z_source = (0..params.num_positions_per_block() * depth_witness * depth_fold)
                .map(|index| {
                    marker::<MULTI_GROUP_D>(500 * group_index + 100 * unit.chunk_index() + index)
                })
                .collect::<Vec<_>>();
            emit_witness_z_planes::<MULTI_GROUP_D>(
                &mut emitted,
                unit,
                params.num_positions_per_block(),
                depth_witness,
                depth_fold,
                &z_source,
            )
            .expect("emit Z");
            let z_range = unit.z_range();
            assert_eq!(&emitted[z_range], flatten_markers(z_source).as_slice());

            let mut expected_e = Vec::new();
            for claim in 0..num_claims {
                for block_idx in unit.global_block_range() {
                    for digit in 0..depth_open {
                        expected_e.push(
                            e_source[(claim * num_live_blocks + block_idx) * depth_open + digit],
                        );
                    }
                }
            }
            let e_range = unit.e_range();
            assert_eq!(&emitted[e_range], flatten_markers(expected_e).as_slice());

            let mut expected_t = Vec::new();
            for claim in 0..num_claims {
                for block_idx in unit.global_block_range() {
                    for a_row in 0..n_a {
                        for digit in 0..depth_commit {
                            expected_t.push(
                                t_source[((claim * num_live_blocks + block_idx) * n_a + a_row)
                                    * depth_commit
                                    + digit],
                            );
                        }
                    }
                }
            }
            let t_range = unit.t_range();
            assert_eq!(&emitted[t_range], flatten_markers(expected_t).as_slice());
        }
    }
    let quotient_depth = r_decomp_levels::<F>(lp.open().digits.log_basis);
    for (row_index, row) in layout.r_rows().iter().enumerate() {
        for digit in 0..quotient_depth {
            for coefficient in 0..row.geometry().polynomial_modulus_dimension() {
                let address = layout
                    .r_coefficient_index(row_index, digit, 0, coefficient)
                    .expect("R address");
                let value = ((row_index + digit + coefficient) % 100 + 1) as i8;
                emitted[address] = value;
                assert_eq!(emitted[address], value);
            }
        }
    }
}

#[test]
fn packing_instance_emits_all_physical_e_coordinate_planes() {
    const PACK_D_A: usize = 256;
    const PACK_D_D: usize = 64;
    let mut lp = CommittedGroupParams::params_only(
        crate::SisModulusProfileId::Q32Offset99,
        PACK_D_A,
        2,
        1,
        1,
        1,
        SparseChallengeConfig::production_for_ring_dim(PACK_D_A).expect("ambient config"),
    )
    .with_decomp(4, 8, 1, 2, 2)
    .expect("packing params");
    lp.own_group_mut().opening.num_digits_fold = 3;
    lp.own_group_mut().opening.opening_method = crate::OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 64,
    };
    lp.own_group_mut().opening.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(64).expect("packing config");
    certify_test_sis_bounds(&mut lp);
    lp.open_matrix = OpenCommitMatrixParams::new_unchecked(
        lp.open().matrix.security_policy(),
        lp.open().matrix.sis_table_key().table_digest,
        lp.open().matrix.sis_modulus_profile(),
        lp.open().matrix.output_rank(),
        8,
        lp.open().matrix.coeff_linf_bound(),
        PACK_D_D,
    );
    let opening_batch = OpeningClaimsLayout::new(8, 1).expect("opening batch");
    let packing_geometry =
        SubringCoefficientPackingGeometry::try_new(2, PACK_D_A, 64).expect("packing geometry");
    let zero_shell = Challenges::from_sparse(
        vec![
            SparseChallenge {
                positions: Vec::new().into(),
                coeffs: Vec::new().into(),
            };
            lp.blocks().live_blocks
        ],
        lp.blocks().live_blocks,
        1,
    )
    .expect("structurally valid zero shell");
    assert!(CoefficientPackingChallenges::new(packing_geometry, zero_shell).is_err());
    let config = lp.fold_challenge_config();
    let wrong_magnitude = Challenges::from_sparse(
        vec![
            SparseChallenge {
                positions: (0..config.weight())
                    .map(|position| position as u32)
                    .collect(),
                coeffs: std::iter::once(3)
                    .chain((1..config.weight()).map(|_| 1))
                    .collect(),
            };
            lp.blocks().live_blocks
        ],
        lp.blocks().live_blocks,
        1,
    )
    .expect("structurally valid wrong shell");
    assert!(CoefficientPackingChallenges::new(packing_geometry, wrong_magnitude).is_err());
    let subring_challenges = packing_challenges(&lp, 1);
    let group_opening = RingRelationGroupOpening::coefficient_packing(
        CoefficientPackingChallenges::new(packing_geometry, subring_challenges)
            .expect("packing challenges"),
    );
    assert_eq!(
        group_opening.coefficient_packing_geometry(),
        Some(packing_geometry)
    );
    assert!(group_opening.evaluation_trace_multiplier_point().is_err());

    let relation_geometry = crate::RelationWitnessGeometry::for_level(&lp, &opening_batch, 2)
        .expect("relation geometry");
    let rhs_len = relation_rhs_coeff_len(relation_geometry.rhs_layout()).expect("rhs len");
    let instance = RingRelationInstance::<F>::new(
        vec![group_opening],
        2,
        opening_batch.clone(),
        vec![F::one()],
        RingVec::from_ring_elems::<PACK_D_A>(&[CyclotomicRing::one()]),
        RingVec::from_coeffs(vec![F::zero(); rhs_len]),
        RingVec::from_ring_elems::<PACK_D_D>(&[]),
        lp.role_dims(),
    )
    .expect("packing instance");
    let layout = instance.segment_layout(&lp, None).expect("packing layout");
    let unit = layout.units().first().expect("packing witness unit");
    assert_eq!(unit.e_geometry().polynomial_modulus_dimension(), 64);
    assert_eq!(unit.e_geometry().coordinate_plane_count(), 2);
    assert_eq!(unit.e_geometry().physical_coefficient_width(), 128);

    let depth_open = lp.open().digits.num_digits;
    let blocks = lp.blocks().live_blocks;
    let role_subcolumns = packing_geometry.partial_base_field_width() / PACK_D_D;
    let source = (0..blocks * role_subcolumns * depth_open)
        .map(|index| marker::<PACK_D_D>(700 + index))
        .collect::<Vec<_>>();
    let digits = DigitBlocks::new(
        source.as_flattened().to_vec(),
        vec![role_subcolumns * depth_open; blocks],
        PACK_D_D,
    )
    .expect("packing digits");
    let mut emitted = vec![0i8; layout.live_coeff_len()];
    emit_witness_e_planes::<PACK_D_D>(
        &mut emitted,
        unit,
        packing_geometry.partial_base_field_width(),
        1,
        depth_open,
        &digits,
        blocks,
    )
    .expect("emit packing E");
    assert_eq!(&emitted[unit.e_range()], digits.digits());

    let packing_r_row = layout
        .r_rows()
        .iter()
        .position(|row| row.geometry() == unit.e_geometry())
        .expect("packing consistency quotient row");
    let plane_zero = layout
        .r_coefficient_index(packing_r_row, 0, 0, 0)
        .expect("packing Q plane zero");
    let plane_one = layout
        .r_coefficient_index(packing_r_row, 0, 1, 0)
        .expect("packing Q plane one");
    assert_eq!(plane_one - plane_zero, 64);
    assert!(layout.r_coefficient_index(packing_r_row, 0, 2, 0).is_err());

    let aliased_width_geometry =
        SubringCoefficientPackingGeometry::try_new(2, 128, 64).expect("aliased geometry");
    let wrong_opening = RingRelationGroupOpening::coefficient_packing(
        CoefficientPackingChallenges::new(aliased_width_geometry, packing_challenges(&lp, 1))
            .expect("individually valid packing challenges"),
    );
    let wrong_instance = RingRelationInstance::<F>::new(
        vec![wrong_opening],
        2,
        opening_batch.clone(),
        vec![F::one()],
        RingVec::from_ring_elems::<PACK_D_A>(&[CyclotomicRing::one()]),
        RingVec::from_coeffs(vec![F::zero(); rhs_len]),
        RingVec::from_ring_elems::<PACK_D_D>(&[]),
        lp.role_dims(),
    )
    .expect("carrier construction is schedule-independent");
    assert!(wrong_instance.segment_layout(&lp, None).is_err());

    let wrong_k_opening = RingRelationGroupOpening::coefficient_packing(
        CoefficientPackingChallenges::new(
            SubringCoefficientPackingGeometry::try_new(1, PACK_D_A, 64).expect("wrong-k geometry"),
            packing_challenges(&lp, 1),
        )
        .expect("individually valid wrong-k packing challenges"),
    );
    let wrong_k_instance = RingRelationInstance::<F>::new(
        vec![wrong_k_opening],
        2,
        opening_batch,
        vec![F::one()],
        RingVec::from_ring_elems::<PACK_D_A>(&[CyclotomicRing::one()]),
        RingVec::from_coeffs(vec![F::zero(); rhs_len]),
        RingVec::from_ring_elems::<PACK_D_D>(&[]),
        lp.role_dims(),
    )
    .expect("wrong-k carrier construction is schedule-independent");
    assert!(wrong_k_instance.segment_layout(&lp, None).is_err());
}
