use super::*;
use std::sync::Arc;

use akita_algebra::poly::multilinear_eval;
use akita_challenges::{Challenges, SparseChallenge, SparseChallengeConfig};
use jolt_field::{
    CanonicalEncoding, Ext2, ExtField, Field, FpExt4, One, Prime128OffsetA7F7, Prime32Offset99,
    Prime64Offset59, Ring, Zero,
};

use crate::{
    fold_coefficient_packing_partials, relation_claim_from_compressed_rhs_extension,
    relation_rhs_coeff_len, BasisMode, ChunkedWitnessCfg, CoefficientPackingChallenges,
    CommitmentPayloadMode, CommitmentRingDims, DigitRangePlan, OpenCommitMatrixParams,
    OuterCommitMatrixParams, PolynomialGroupLayout, RelationAddressGeometry,
    RingMultiplierOpeningPoint, RingOpeningPoint, RingRelationGroupOpening, RingVec,
    SisModulusProfileId, WitnessLayout,
};
use crate::{
    GroupCommitPhaseParams, GroupOpenPhaseParams, GroupOpeningPlan, InnerCommitMatrixParams,
};

type F = Prime64Offset59;
type E = Ext2<F>;

struct Fixture<Base: Field, Extension: Field> {
    params: CommittedGroupParams,
    opening_batch: OpeningClaimsLayout,
    relation_plan: RelationRangeImagePlan,
    relation: RingRelationInstance<Base>,
    prepared_point: PreparedSubringCoefficientPackingPoint<Extension>,
    claim_coefficients: Vec<Extension>,
    tau1: Vec<Extension>,
}

#[allow(clippy::too_many_arguments)]
fn fixture<Base, Extension>(
    profile: SisModulusProfileId,
    d_a: usize,
    d_d: usize,
    s: usize,
    live_positions: usize,
    positions_per_block: usize,
    num_vars: usize,
    num_claims: usize,
    num_chunks: usize,
) -> Fixture<Base, Extension>
where
    Base: Field + CanonicalEncoding + Ring,
    Extension: ExtField<Base> + FpExtEncoding<Base> + Ring + ExtField<Base>,
{
    let config = SparseChallengeConfig::production_for_ring_dim(s).unwrap();
    let mut params = CommittedGroupParams::params_only(profile, d_a, 2, 2, 2, 2, config)
        .with_decomp(positions_per_block, live_positions, 2, 2, 2)
        .unwrap();
    params.payload_mode = CommitmentPayloadMode::Raw;
    params.witness_chunk = ChunkedWitnessCfg {
        num_chunks,
        num_activated_levels: usize::from(num_chunks > 1),
    };
    params.own_group_mut().opening.opening_method = OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: s,
    };
    let outer = params.outer().matrix;
    params.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width(),
        outer.coeff_linf_bound(),
        64,
    );
    let opening = params.open().matrix;
    params.open_matrix = OpenCommitMatrixParams::new_unchecked(
        opening.security_policy(),
        opening.sis_table_key().table_digest,
        opening.sis_modulus_profile(),
        opening.output_rank(),
        opening.input_width(),
        opening.coeff_linf_bound(),
        d_d,
    );
    let opening_batch =
        OpeningClaimsLayout::from_groups(vec![PolynomialGroupLayout::new(num_vars, num_claims)])
            .unwrap();
    let extension_degree = <Extension as ExtField<Base>>::DEGREE;
    let relation_geometry =
        RelationWitnessGeometry::for_level(&params, &opening_batch, extension_degree).unwrap();
    let witness_layout = WitnessLayout::new(
        &params,
        &opening_batch,
        &relation_geometry,
        params.witness_chunk.num_chunks,
        crate::RelationQuotientPlan::quotient_lift(r_decomp_levels::<Base>(
            params.open().digits.log_basis,
        ))
        .unwrap(),
    )
    .unwrap();
    let relation_address_geometry = RelationAddressGeometry::for_relation(
        &relation_geometry,
        d_d,
        witness_layout.live_coeff_len(),
    )
    .unwrap();
    let relation_plan = RelationRangeImagePlan::new(
        relation_geometry.clone(),
        relation_address_geometry,
        DigitRangePlan::new(4).unwrap(),
        witness_layout,
        &opening_batch,
    )
    .unwrap();
    let geometry = SubringCoefficientPackingGeometry::try_new(extension_degree, d_a, s).unwrap();
    let point_values = (0..num_vars)
        .map(|index| Extension::from_u64((index + 2) as u64))
        .collect::<Vec<_>>();
    let prepared_point = PreparedSubringCoefficientPackingPoint::new(
        geometry,
        BasisMode::Lagrange,
        live_positions,
        positions_per_block,
        num_vars,
        &point_values,
    )
    .unwrap();
    let challenge_count = num_claims * prepared_point.num_live_blocks();
    let sparse = (0..challenge_count)
        .map(|challenge| SparseChallenge {
            positions: (0..config.weight())
                .map(|term| ((term + challenge) % s) as u32)
                .collect(),
            coeffs: (0..config.count_pm1)
                .map(|term| if term.is_multiple_of(2) { 1 } else { -1 })
                .chain((0..config.count_pm2).map(|_| 2))
                .collect(),
        })
        .collect();
    let challenges =
        Challenges::from_sparse(sparse, prepared_point.num_live_blocks(), num_claims).unwrap();
    let group_opening = RingRelationGroupOpening::coefficient_packing(
        CoefficientPackingChallenges::new(geometry, challenges).unwrap(),
    );
    let gamma = (0..num_claims)
        .map(|claim| Base::from_u64((claim + 3) as u64))
        .collect::<Vec<_>>();
    let mut row_coefficients = vec![Base::zero(); num_claims * d_a];
    for (claim, &coefficient) in gamma.iter().enumerate() {
        row_coefficients[claim * d_a] = coefficient;
    }
    let rhs = RingVec::from_coeffs(vec![
        Base::zero();
        relation_rhs_coeff_len(relation_geometry.rhs_layout())
            .unwrap()
    ]);
    let relation = RingRelationInstance::new(
        vec![group_opening],
        extension_degree,
        opening_batch.clone(),
        gamma,
        RingVec::from_coeffs_with_ring_dim(row_coefficients, d_a).unwrap(),
        rhs,
        RingVec::from_coeffs(Vec::new()),
        params.role_dims(),
    )
    .unwrap();
    let claim_coefficients = (0..num_claims)
        .map(|claim| Extension::from_u64((claim + 5) as u64))
        .collect();
    let tau1 = (0..relation_plan.relation_row_index_num_vars().unwrap())
        .map(|index| Extension::from_u64((index + 7) as u64))
        .collect();
    Fixture {
        params,
        opening_batch,
        relation_plan,
        relation,
        prepared_point,
        claim_coefficients,
        tau1,
    }
}

#[test]
fn packing_rejects_tensor_projected_commitment_source() {
    let mut fixture = fixture::<F, E>(
        SisModulusProfileId::Q64Offset59,
        256,
        64,
        64,
        4,
        4,
        10,
        1,
        1,
    );
    let extension_degree = <E as ExtField<F>>::DEGREE;
    fixture.params.source_encoding =
        crate::CommittedSourceEncoding::TensorSubfieldProjection { extension_degree };
    assert!(matches!(
        RelationWitnessGeometry::for_level(
            &fixture.params,
            &fixture.opening_batch,
            extension_degree,
        ),
        Err(AkitaError::InvalidSetup(_))
    ));
}

fn prepare<Base, Extension>(
    fixture: &Fixture<Base, Extension>,
    alpha: Extension,
) -> CoefficientPackingGroupSemantics<Extension>
where
    Base: Field + CanonicalEncoding + Ring,
    Extension: ExtField<Base> + FpExtEncoding<Base> + Ring + ExtField<Base>,
{
    prepare_coefficient_packing_group_semantics(CoefficientPackingGroupSemanticInputs {
        level_params: &fixture.params,
        opening_batch: &fixture.opening_batch,
        relation_plan: &fixture.relation_plan,
        relation: &fixture.relation,
        group_index: 0,
        prepared_point: &fixture.prepared_point,
        alpha,
        tau1: &fixture.tau1,
        claim_coefficients: &fixture.claim_coefficients,
    })
    .unwrap()
}

fn prepare_compact<Base, Extension>(
    fixture: &Fixture<Base, Extension>,
    alpha: Extension,
) -> CoefficientPackingVerifierGroupSemantics<Extension>
where
    Base: Field + CanonicalEncoding + Ring,
    Extension: ExtField<Base> + FpExtEncoding<Base> + Ring + ExtField<Base>,
{
    prepare_coefficient_packing_verifier_batch_semantics(CoefficientPackingBatchSemanticInputs {
        level_params: &fixture.params,
        opening_batch: &fixture.opening_batch,
        relation_plan: &fixture.relation_plan,
        relation: &fixture.relation,
        prepared_points: &[(0, &fixture.prepared_point)],
        alpha,
        tau1: &fixture.tau1,
        claim_coefficients: &fixture.claim_coefficients,
    })
    .unwrap()
    .groups()[0]
        .clone()
}

fn materialize_events<Extension: Field>(
    events: &CoefficientPackingRelationEvents<Extension>,
) -> Vec<Extension> {
    let mut dense = vec![Extension::zero(); events.physical_field_len()];
    for event in events.events() {
        for (offset, index) in event.physical_coefficients().enumerate() {
            dense[index] +=
                event.scalar() * events.alpha_powers()[event.alpha_exponent_start() + offset];
        }
    }
    dense
}

fn materialize_stage2_source<Extension: Field>(
    terms: &CoefficientPackingStage2Terms<Extension>,
    selected_source: CoefficientPackingStage2Source,
) -> Vec<Extension> {
    let mut dense = vec![Extension::zero(); terms.physical_field_len()];
    let source = match selected_source {
        CoefficientPackingStage2Source::DirectOpening => terms.direct_opening_source(),
        CoefficientPackingStage2Source::PackingZ => terms.packing_z_source(),
    };
    for term in terms
        .terms()
        .iter()
        .filter(|term| term.source() == selected_source)
    {
        for segment in &terms.segments()[term.segments()] {
            let physical = segment.physical_coefficients();
            let source_range = segment.source_coefficients();
            for offset in 0..physical.len() {
                dense[physical.start + offset] +=
                    term.factor() * source[source_range.start + offset];
            }
        }
    }
    dense
}

#[test]
fn compact_consumers_match_dense_event_and_stage2_oracles() {
    let fixture = fixture::<F, E>(
        SisModulusProfileId::Q64Offset59,
        256,
        128,
        64,
        6,
        4,
        11,
        2,
        2,
    );
    for alpha in [E::zero(), E::one(), E::from_u64(17)] {
        let semantics = prepare(&fixture, alpha);
        let compact = prepare_compact(&fixture, alpha);
        let padded_len = semantics
            .relation_events()
            .physical_field_len()
            .next_power_of_two();
        let point = (0..padded_len.trailing_zeros())
            .map(|index| E::from_u64(23 + index as u64))
            .collect::<Vec<_>>();
        let mut dense_events = materialize_events(semantics.relation_events());
        dense_events.resize(padded_len, E::zero());
        assert_eq!(
            semantics
                .relation_events()
                .evaluate_at_point(&point)
                .unwrap(),
            multilinear_eval(&dense_events, &point).unwrap()
        );
        let mut reordered_events = semantics.relation_events().clone();
        reordered_events.events.reverse();
        assert_eq!(
            reordered_events.evaluate_at_point(&point).unwrap(),
            multilinear_eval(&dense_events, &point).unwrap(),
            "direct-index alpha caching must not depend on event order"
        );
        let mut dense_stage2 = materialize_stage2_source(
            semantics.stage2_terms(),
            CoefficientPackingStage2Source::DirectOpening,
        );
        let z = materialize_stage2_source(
            semantics.stage2_terms(),
            CoefficientPackingStage2Source::PackingZ,
        );
        for (sum, contribution) in dense_stage2.iter_mut().zip(z) {
            *sum += contribution;
        }
        dense_stage2.resize(padded_len, E::zero());
        assert_eq!(
            semantics.stage2_terms().evaluate_at_point(&point).unwrap(),
            multilinear_eval(&dense_stage2, &point).unwrap()
        );
        assert_eq!(
            compact
                .compact_factors()
                .evaluate_relation_at_point(&point)
                .unwrap(),
            multilinear_eval(&dense_events, &point).unwrap()
        );
        assert_eq!(
            compact
                .compact_factors()
                .evaluate_stage2_at_point(&point)
                .unwrap(),
            multilinear_eval(&dense_stage2, &point).unwrap()
        );
        let mut reordered_terms = semantics.stage2_terms().clone();
        reordered_terms.terms.reverse();
        assert_eq!(
            reordered_terms.evaluate_at_point(&point).unwrap(),
            multilinear_eval(&dense_stage2, &point).unwrap(),
            "direct-index source caching must not depend on term order"
        );
        assert!(semantics
            .relation_events()
            .evaluate_at_point(&point[..point.len() - 1])
            .is_err());
        assert!(semantics
            .stage2_terms()
            .evaluate_at_point(&point[..point.len() - 1])
            .is_err());
    }
}

fn assert_compact_factors_match_dense<Base, Extension>(fixture: &Fixture<Base, Extension>)
where
    Base: Field + CanonicalEncoding + Ring,
    Extension: ExtField<Base> + FpExtEncoding<Base> + Ring + ExtField<Base>,
{
    for alpha in [Extension::zero(), Extension::one(), Extension::from_u64(19)] {
        let semantics = prepare(fixture, alpha);
        let compact = prepare_compact(fixture, alpha);
        let padded_len = semantics
            .relation_events()
            .physical_field_len()
            .next_power_of_two();
        let point = (0..padded_len.trailing_zeros())
            .map(|index| Extension::from_u64(29 + u64::from(index)))
            .collect::<Vec<_>>();
        let mut relation = materialize_events(semantics.relation_events());
        relation.resize(padded_len, Extension::zero());
        let mut stage2 = materialize_stage2_source(
            semantics.stage2_terms(),
            CoefficientPackingStage2Source::DirectOpening,
        );
        let packing_z = materialize_stage2_source(
            semantics.stage2_terms(),
            CoefficientPackingStage2Source::PackingZ,
        );
        for (sum, contribution) in stage2.iter_mut().zip(packing_z) {
            *sum += contribution;
        }
        stage2.resize(padded_len, Extension::zero());
        assert_eq!(
            compact
                .compact_factors()
                .evaluate_relation_at_point(&point)
                .unwrap(),
            multilinear_eval(&relation, &point).unwrap()
        );
        assert_eq!(
            compact
                .compact_factors()
                .evaluate_stage2_at_point(&point)
                .unwrap(),
            multilinear_eval(&stage2, &point).unwrap()
        );
        assert!(compact
            .compact_factors()
            .evaluate_relation_at_point(&point[..point.len() - 1])
            .is_err());
    }
}

#[test]
fn compact_factors_cover_overlap_and_fp32_h4_geometries() {
    let overlap = fixture::<Prime128OffsetA7F7, Prime128OffsetA7F7>(
        SisModulusProfileId::Q128OffsetA7F7,
        64,
        64,
        64,
        6,
        4,
        9,
        2,
        2,
    );
    assert_compact_factors_match_dense(&overlap);

    for d_d in [64, 128] {
        let h4 = fixture::<Prime32Offset99, FpExt4<Prime32Offset99>>(
            SisModulusProfileId::Q32Offset99,
            1024,
            d_d,
            64,
            6,
            4,
            13,
            2,
            2,
        );
        assert_eq!(
            h4.prepared_point.geometry().packing_factor(),
            4,
            "k=4,dA=1024,s=64 must exercise h=4"
        );
        assert_compact_factors_match_dense(&h4);
    }
    let recursive_role_subcolumns = fixture::<Prime128OffsetA7F7, Prime128OffsetA7F7>(
        SisModulusProfileId::Q128OffsetA7F7,
        512,
        64,
        512,
        6,
        4,
        12,
        2,
        2,
    );
    assert_compact_factors_match_dense(&recursive_role_subcolumns);
    let compact = prepare_compact(&recursive_role_subcolumns, Prime128OffsetA7F7::from_u64(19));
    let role_stride = recursive_role_subcolumns.params.open().digits.num_digits * 64;
    let direct_families = &compact.compact_factors().direct_opening_families;
    assert!(direct_families
        .iter()
        .any(|family| family.axes.iter().any(|axis| axis.len == 8
            && axis.left_stride == 64
            && axis.right_stride == role_stride)));
}

#[test]
fn compact_factors_skip_empty_distributed_witness_units() {
    let fixture = fixture::<Prime128OffsetA7F7, Prime128OffsetA7F7>(
        SisModulusProfileId::Q128OffsetA7F7,
        256,
        64,
        64,
        2,
        1,
        9,
        2,
        8,
    );
    assert!(fixture
        .relation_plan
        .witness_layout()
        .units_for_group(0)
        .unwrap()
        .any(|unit| unit.num_live_blocks() == 0));
    assert_compact_factors_match_dense(&fixture);
}

#[test]
fn compact_affine_e_relation_handles_the_production_fp128_root_stride() {
    type Extension = Prime128OffsetA7F7;

    const K: usize = 1;
    const S: usize = 64;
    const H: usize = 4;
    const D_A: usize = 256;
    const D_D: usize = 64;
    const OPENING_DIGITS: usize = 43;
    const LIVE_BLOCKS: usize = 8192;
    assert_eq!(D_A, K * S * H);
    assert_eq!(D_D, S);

    let alpha = Extension::from_u64(7);
    let coefficient_weights = scalar_powers(alpha, S);
    let digit_weights = scalar_powers(Extension::from_u64(3), OPENING_DIGITS);
    let outer_weights = (0..LIVE_BLOCKS)
        .map(|block| Extension::from_u64(11 + (block % 251) as u64))
        .collect::<Vec<_>>();
    let coefficient_bits = S.trailing_zeros() as usize;
    let outer_domain = LIVE_BLOCKS * OPENING_DIGITS;
    let outer_bits = outer_domain.next_power_of_two().trailing_zeros() as usize;
    let point = (0..coefficient_bits + outer_bits)
        .map(|bit| match bit % 5 {
            0 => Extension::zero(),
            1 => Extension::one(),
            _ => Extension::from_u64(17 + bit as u64),
        })
        .collect::<Vec<_>>();
    let family = CoefficientPackingAffineRelationFamily {
        scalar: Extension::from_u64(13),
        coefficient_weights: coefficient_weights.clone().into(),
        coefficient_len: S,
        base_offset: 0,
        outer_len: LIVE_BLOCKS,
        outer_stride: OPENING_DIGITS,
        digit_stride: 1,
        digit_weights: digit_weights.clone().into(),
        outer_weights: outer_weights.clone().into(),
    };
    let family_scalar = family.scalar;
    let compact = CoefficientPackingCompactFactors {
        basis: BasisMode::Lagrange,
        physical_field_len: 1usize << point.len(),
        direct_opening_point: Arc::from([]),
        packing_z_point: Arc::from([]),
        affine_relation_families: vec![family],
        quotient_families: Vec::new(),
        direct_opening_families: Vec::new(),
        packing_z_families: Vec::new(),
    };

    let coefficient_evaluation = coefficient_weights.iter().enumerate().fold(
        Extension::zero(),
        |sum, (coefficient, &weight)| {
            sum + weight * eq_eval_at_index(&point[..coefficient_bits], coefficient)
        },
    );
    let outer_evaluation =
        outer_weights
            .iter()
            .enumerate()
            .fold(Extension::zero(), |sum, (block, &block_weight)| {
                sum + digit_weights.iter().enumerate().fold(
                    Extension::zero(),
                    |digit_sum, (digit, &digit_weight)| {
                        digit_sum
                            + block_weight
                                * digit_weight
                                * eq_eval_at_index(
                                    &point[coefficient_bits..],
                                    block * OPENING_DIGITS + digit,
                                )
                    },
                )
            });
    assert_eq!(
        compact.evaluate_relation_at_point(&point).unwrap(),
        family_scalar * coefficient_evaluation * outer_evaluation
    );
    assert!(compact
        .evaluate_relation_at_point(&point[..coefficient_bits - 1])
        .is_err());
}

#[test]
fn compact_consumers_parallel_branch_matches_dense_oracles() {
    let fixture = fixture::<F, E>(
        SisModulusProfileId::Q64Offset59,
        256,
        128,
        64,
        6,
        4,
        11,
        2,
        2,
    );
    let semantics = prepare(&fixture, E::from_u64(17));
    let padded_len = semantics
        .relation_events()
        .physical_field_len()
        .next_power_of_two();
    let point = (0..padded_len.trailing_zeros())
        .map(|index| E::from_u64(31 + index as u64))
        .collect::<Vec<_>>();

    let mut parallel_events = semantics.relation_events().clone();
    let event_work = parallel_events
        .events()
        .iter()
        .map(|event| {
            event.physical_coefficients().len() / parallel_events.relation_coefficient_block_len()
        })
        .sum::<usize>();
    let event_repetitions = 1024usize.div_ceil(event_work);
    let original_events = parallel_events.events.clone();
    parallel_events.events = original_events
        .iter()
        .cloned()
        .cycle()
        .take(original_events.len() * event_repetitions)
        .collect();
    let parallel_event_work = parallel_events
        .events()
        .iter()
        .map(|event| {
            event.physical_coefficients().len() / parallel_events.relation_coefficient_block_len()
        })
        .sum::<usize>();
    assert!(parallel_event_work >= 1024);
    let mut dense_events = materialize_events(&parallel_events);
    dense_events.resize(padded_len, E::zero());
    assert_eq!(
        parallel_events.evaluate_at_point(&point).unwrap(),
        multilinear_eval(&dense_events, &point).unwrap()
    );

    let mut parallel_terms = semantics.stage2_terms().clone();
    let term_work = parallel_terms
        .terms()
        .iter()
        .map(|term| {
            parallel_terms.segments()[term.segments()]
                .iter()
                .map(|segment| {
                    segment.physical_coefficients().len()
                        / parallel_terms.relation_coefficient_block_len
                })
                .sum::<usize>()
        })
        .sum::<usize>();
    let term_repetitions = 1024usize.div_ceil(term_work);
    let original_terms = parallel_terms.terms.clone();
    parallel_terms.terms = original_terms
        .iter()
        .cloned()
        .cycle()
        .take(original_terms.len() * term_repetitions)
        .collect();
    let parallel_term_work = parallel_terms
        .terms()
        .iter()
        .map(|term| {
            parallel_terms.segments()[term.segments()]
                .iter()
                .map(|segment| {
                    segment.physical_coefficients().len()
                        / parallel_terms.relation_coefficient_block_len
                })
                .sum::<usize>()
        })
        .sum::<usize>();
    assert!(parallel_term_work >= 1024);
    let mut dense_stage2 = materialize_stage2_source(
        &parallel_terms,
        CoefficientPackingStage2Source::DirectOpening,
    );
    let packing_z =
        materialize_stage2_source(&parallel_terms, CoefficientPackingStage2Source::PackingZ);
    for (sum, contribution) in dense_stage2.iter_mut().zip(packing_z) {
        *sum += contribution;
    }
    dense_stage2.resize(padded_len, E::zero());
    assert_eq!(
        parallel_terms.evaluate_at_point(&point).unwrap(),
        multilinear_eval(&dense_stage2, &point).unwrap()
    );
}

#[test]
fn semantics_bind_partial_blocks_claims_planes_and_positive_q_convention() {
    let fixture = fixture::<F, E>(
        SisModulusProfileId::Q64Offset59,
        256,
        128,
        64,
        6,
        4,
        11,
        2,
        1,
    );
    let alpha = E::from_u64(13);
    let semantics = prepare(&fixture, alpha);
    assert_eq!(semantics.geometry().packing_factor(), 2);
    assert_eq!(semantics.stage2_terms().group_claim_range(), 0..2);
    assert_eq!(semantics.stage2_terms().direct_opening_source().len(), 128);
    assert_eq!(semantics.stage2_terms().packing_z_source().len(), 256);
    assert_eq!(
        semantics.stage2_terms().scalar_claim_weight(),
        relation_row_weight(
            fixture.relation_plan.scalar_opening_row_index().unwrap(),
            &fixture.tau1,
        )
        .unwrap()
    );

    let depth_open = fixture.params.open().digits.num_digits;
    let quotient_depth = fixture
        .relation_plan
        .witness_layout()
        .quotient_depth()
        .expect("quotient-lift fixture");
    let extension_degree = <E as ExtField<F>>::DEGREE;
    let expected_e_events = 2 * 2 * depth_open * extension_degree;
    let expected_q_events = quotient_depth * extension_degree;
    let events = semantics.relation_events().events();
    assert_eq!(events.len(), expected_e_events + expected_q_events);
    for event in &events[..expected_e_events] {
        assert_eq!(event.physical_coefficients().len(), 64);
        assert_eq!(event.alpha_exponent_start(), 0);
    }
    for event in &events[expected_e_events..] {
        assert_eq!(event.physical_coefficients().len(), 64);
        assert_eq!(event.alpha_exponent_start(), 0);
        assert!(!event.scalar().is_zero());
    }

    let consistency_weight = relation_row_weight(
        fixture.relation_plan.consistency_row_index(0).unwrap(),
        &fixture.tau1,
    )
    .unwrap();
    let geometry = semantics.geometry();
    let alpha_powers = scalar_powers(alpha, geometry.challenge_subring_dimension());
    let basis = [
        E::from_base_slice(&[F::one(), F::zero()]),
        E::from_base_slice(&[F::zero(), F::one()]),
    ];
    let challenges = match fixture.relation.group_opening_view(0).unwrap() {
        RingRelationGroupOpeningView::SubringCoefficientPacking {
            canonical_subring_challenges,
            ambient_a_challenges,
            ..
        } => {
            let ambient_powers = scalar_powers(alpha, geometry.a_ring_dimension());
            let ambient_base = ambient_powers[geometry.subring_embedding_stride()];
            let embedded_subring_powers =
                scalar_powers(ambient_base, geometry.challenge_subring_dimension());
            assert_eq!(
                ambient_a_challenges
                    .eval_at_pows::<F, E>(0, &ambient_powers)
                    .unwrap(),
                canonical_subring_challenges
                    .eval_at_pows::<F, E>(0, &embedded_subring_powers)
                    .unwrap()
            );
            canonical_subring_challenges
        }
        RingRelationGroupOpeningView::EvaluationTrace { .. } => panic!("method was erased"),
    };
    let opening_gadget = gadget_row_scalars::<F>(
        fixture.params.open().digits.num_digits,
        fixture.params.open().digits.log_basis,
    );
    let first_challenge = challenges.eval_at_pows::<F, E>(0, &alpha_powers).unwrap();
    assert_eq!(
        events[0].scalar(),
        consistency_weight * first_challenge * E::lift_base(opening_gadget[0]) * basis[0]
    );
    let denominator = alpha_powers.last().copied().unwrap() * alpha + E::one();
    let quotient_gadget = gadget_row_scalars::<F>(
        fixture
            .relation_plan
            .witness_layout()
            .quotient_depth()
            .expect("quotient-lift fixture"),
        fixture.params.open().digits.log_basis,
    );
    assert_eq!(
        events[expected_e_events].scalar(),
        -(consistency_weight * E::lift_base(quotient_gadget[0]) * basis[0] * denominator)
    );

    let direct_source = semantics.stage2_terms().direct_opening_source();
    for (plane, &basis_element) in basis.iter().enumerate() {
        for (coefficient, &tail_weight) in fixture.prepared_point.tail_weights().iter().enumerate()
        {
            assert_eq!(
                direct_source[plane * geometry.challenge_subring_dimension() + coefficient],
                basis_element * tail_weight
            );
        }
    }
    let packing_z_source = semantics.stage2_terms().packing_z_source();
    for (low, &packing_weight) in fixture.prepared_point.packing_weights().iter().enumerate() {
        for (coefficient, &alpha_power) in alpha_powers.iter().enumerate() {
            let physical = geometry.a_ring_coefficient_index(low, coefficient).unwrap();
            assert_eq!(packing_z_source[physical], packing_weight * alpha_power);
        }
    }

    let direct_terms = semantics
        .stage2_terms()
        .terms()
        .iter()
        .filter(|term| term.source() == CoefficientPackingStage2Source::DirectOpening)
        .count();
    let z_terms = semantics
        .stage2_terms()
        .terms()
        .iter()
        .filter(|term| term.source() == CoefficientPackingStage2Source::PackingZ)
        .count();
    assert_eq!(direct_terms, 2 * 2 * depth_open);
    assert_eq!(
        z_terms,
        fixture.params.blocks().positions_per_block
            * fixture.params.inner().digits.num_digits
            * fixture.params.num_digits_fold()
    );
    for term in semantics.stage2_terms().terms() {
        let source_len = match term.source() {
            CoefficientPackingStage2Source::DirectOpening => {
                semantics.stage2_terms().direct_opening_source().len()
            }
            CoefficientPackingStage2Source::PackingZ => {
                semantics.stage2_terms().packing_z_source().len()
            }
        };
        for segment in &semantics.stage2_terms().segments()[term.segments()] {
            assert_eq!(
                segment.physical_coefficients().len(),
                segment.source_coefficients().len()
            );
            assert!(
                segment.physical_coefficients().end
                    <= semantics.stage2_terms().physical_field_len()
            );
            assert!(segment.source_coefficients().end <= source_len);
        }
    }
}

#[test]
fn packing_and_ambient_challenge_evaluations_cannot_be_substituted_when_h_exceeds_one() {
    type F4 = Prime32Offset99;
    type E4 = FpExt4<F4>;

    let fixture = fixture::<F4, E4>(
        SisModulusProfileId::Q32Offset99,
        1024,
        128,
        64,
        6,
        4,
        13,
        2,
        1,
    );
    let alpha = E4::from_u64(41);
    let (geometry, canonical, ambient) = match fixture.relation.group_opening_view(0).unwrap() {
        RingRelationGroupOpeningView::SubringCoefficientPacking {
            geometry,
            canonical_subring_challenges,
            ambient_a_challenges,
        } => (geometry, canonical_subring_challenges, ambient_a_challenges),
        RingRelationGroupOpeningView::EvaluationTrace { .. } => panic!("method was erased"),
    };
    assert!(geometry.packing_factor() > 1);
    let packing_eval = canonical
        .eval_at_pows::<F4, E4>(
            0,
            &scalar_powers(alpha, geometry.challenge_subring_dimension()),
        )
        .unwrap();
    let ambient_eval = ambient
        .eval_at_pows::<F4, E4>(0, &scalar_powers(alpha, geometry.a_ring_dimension()))
        .unwrap();
    let canonical_at_ambient_argument = canonical
        .eval_at_pows::<F4, E4>(
            0,
            &scalar_powers(
                scalar_powers(alpha, geometry.subring_embedding_stride() + 1)
                    [geometry.subring_embedding_stride()],
                geometry.challenge_subring_dimension(),
            ),
        )
        .unwrap();
    assert_eq!(ambient_eval, canonical_at_ambient_argument);
    assert_ne!(packing_eval, ambient_eval);

    let nonzero_packed_opening = E4::from_u64(17);
    assert_ne!(
        packing_eval * nonzero_packed_opening,
        ambient_eval * nonzero_packed_opening,
        "swapping c(alpha) and c(alpha^(k h)) changes both role-specific numerators"
    );
}

#[test]
fn every_extension_plane_is_bound_by_the_packing_divisibility_identity() {
    type F4 = Prime32Offset99;
    type E4 = FpExt4<F4>;

    let fixture = fixture::<F4, E4>(
        SisModulusProfileId::Q32Offset99,
        1024,
        128,
        64,
        6,
        4,
        13,
        1,
        1,
    );
    let (geometry, challenges) = match fixture.relation.group_opening_view(0).unwrap() {
        RingRelationGroupOpeningView::SubringCoefficientPacking {
            geometry,
            canonical_subring_challenges,
            ..
        } => (geometry, canonical_subring_challenges),
        RingRelationGroupOpeningView::EvaluationTrace { .. } => panic!("method was erased"),
    };
    let challenge = &challenges.as_slice()[..1];
    let partials = (0..geometry.partial_base_field_width())
        .map(|index| F4::from_u64((index + 3) as u64))
        .collect::<Vec<_>>();
    let product = fold_coefficient_packing_partials(geometry, challenge, &partials).unwrap();
    let alpha = E4::from_u64(43);
    let alpha_powers = scalar_powers(alpha, geometry.challenge_subring_dimension());
    let challenge_eval = challenge[0].eval_at_pows::<F4, E4>(&alpha_powers).unwrap();
    let denominator = alpha_powers.last().copied().unwrap() * alpha + E4::one();
    let basis = canonical_extension_basis::<F4, E4>(geometry.extension_degree()).unwrap();
    let plane_eval = |coefficients: &[F4]| {
        coefficients
            .iter()
            .zip(&alpha_powers)
            .fold(E4::zero(), |sum, (&coefficient, &power)| {
                sum + E4::lift_base(coefficient) * power
            })
    };
    let residual = |reduced: &[F4]| {
        (0..geometry.extension_degree()).fold(E4::zero(), |sum, plane| {
            let range = plane * geometry.challenge_subring_dimension()
                ..(plane + 1) * geometry.challenge_subring_dimension();
            sum + basis[plane]
                * (challenge_eval * plane_eval(&partials[range.clone()])
                    - plane_eval(&reduced[range.clone()])
                    - denominator
                        * plane_eval(&product.quotient_high_half_base_field_coordinates()[range]))
        })
    };
    assert_eq!(
        residual(product.reduced_base_field_coordinates()),
        E4::zero()
    );

    let mut tampered = product.reduced_base_field_coordinates().to_vec();
    let nonzero_plane_coefficient = geometry.challenge_subring_dimension() + 7;
    tampered[nonzero_plane_coefficient] += F4::one();
    assert_ne!(
        residual(&tampered),
        E4::zero(),
        "the E[Y] oracle must detect a mutation outside the base extension plane"
    );
}

#[test]
fn e_events_split_planes_at_d_boundaries_without_changing_exponents() {
    let fixture = fixture::<F, E>(
        SisModulusProfileId::Q64Offset59,
        256,
        64,
        128,
        2,
        2,
        9,
        1,
        1,
    );
    let semantics = prepare(&fixture, E::from_u64(17));
    let continued_plane_chunks = semantics
        .relation_events()
        .events()
        .iter()
        .filter(|event| event.alpha_exponent_start() == 64)
        .collect::<Vec<_>>();
    assert_eq!(
        continued_plane_chunks.len(),
        fixture.params.open().digits.num_digits * 2
    );
    for event in continued_plane_chunks {
        assert_eq!(event.physical_coefficients().len(), 64);
    }
}

#[test]
fn semantic_events_match_an_independent_dense_accumulation() {
    let fixture = fixture::<F, E>(
        SisModulusProfileId::Q64Offset59,
        256,
        128,
        64,
        6,
        4,
        11,
        2,
        1,
    );
    let alpha = E::from_u64(19);
    let semantics = prepare(&fixture, alpha);
    let got = materialize_events(semantics.relation_events());
    let mut expected = vec![E::zero(); got.len()];
    let geometry = semantics.geometry();
    let s = geometry.challenge_subring_dimension();
    let powers = scalar_powers(alpha, s);
    let basis = [
        E::from_base_slice(&[F::one(), F::zero()]),
        E::from_base_slice(&[F::zero(), F::one()]),
    ];
    let challenges = match fixture.relation.group_opening_view(0).unwrap() {
        RingRelationGroupOpeningView::SubringCoefficientPacking {
            canonical_subring_challenges,
            ..
        } => canonical_subring_challenges,
        RingRelationGroupOpeningView::EvaluationTrace { .. } => panic!("method was erased"),
    };
    let consistency_weight = relation_row_weight(
        fixture.relation_plan.consistency_row_index(0).unwrap(),
        &fixture.tau1,
    )
    .unwrap();
    let opening_gadget = gadget_row_scalars::<F>(
        fixture.params.open().digits.num_digits,
        fixture.params.open().digits.log_basis,
    );
    let d_d = fixture.params.role_dims().d_d();
    let claims = fixture.opening_batch.num_total_polynomials();
    for claim in 0..claims {
        for unit in fixture
            .relation_plan
            .witness_layout()
            .units_for_group(0)
            .unwrap()
        {
            for block in unit.global_block_range() {
                let challenge = challenges
                    .eval_at_pows::<F, E>(
                        claim * fixture.params.blocks().live_blocks + block,
                        &powers,
                    )
                    .unwrap();
                for (digit, &gadget) in opening_gadget.iter().enumerate() {
                    for (plane, &basis_element) in basis.iter().enumerate() {
                        for (coefficient, &power) in powers.iter().enumerate() {
                            let flat = plane * s + coefficient;
                            let physical = unit
                                .e_coefficient_index(
                                    d_d,
                                    claims,
                                    fixture.params.open().digits.num_digits,
                                    claim,
                                    block,
                                    flat / d_d,
                                    digit,
                                    flat % d_d,
                                )
                                .unwrap();
                            expected[physical] += consistency_weight
                                * challenge
                                * E::lift_base(gadget)
                                * basis_element
                                * power;
                        }
                    }
                }
            }
        }
    }
    let denominator = powers.last().copied().unwrap() * alpha + E::one();
    let quotient_gadget = gadget_row_scalars::<F>(
        fixture
            .relation_plan
            .witness_layout()
            .quotient_depth()
            .expect("quotient-lift fixture"),
        fixture.params.open().digits.log_basis,
    );
    let row = fixture.relation_plan.consistency_row_index(0).unwrap();
    for (digit, &gadget) in quotient_gadget.iter().enumerate() {
        for (plane, &basis_element) in basis.iter().enumerate() {
            for (coefficient, &power) in powers.iter().enumerate() {
                let physical = fixture
                    .relation_plan
                    .witness_layout()
                    .r_coefficient_index(row, digit, plane, coefficient)
                    .unwrap();
                expected[physical] -=
                    consistency_weight * E::lift_base(gadget) * basis_element * denominator * power;
            }
        }
    }
    assert_eq!(got, expected);
}

#[test]
fn malformed_authorities_and_exact_overlap_dispatch_by_method() {
    assert!(akita_error::checked::product([usize::MAX, 2]).is_none());
    let fixture = fixture::<F, E>(
        SisModulusProfileId::Q64Offset59,
        256,
        128,
        64,
        6,
        4,
        11,
        2,
        1,
    );
    let mut short_claims = fixture.claim_coefficients.clone();
    short_claims.pop();
    assert!(
        prepare_coefficient_packing_group_semantics(CoefficientPackingGroupSemanticInputs {
            level_params: &fixture.params,
            opening_batch: &fixture.opening_batch,
            relation_plan: &fixture.relation_plan,
            relation: &fixture.relation,
            group_index: 0,
            prepared_point: &fixture.prepared_point,
            alpha: E::from_u64(3),
            tau1: &fixture.tau1,
            claim_coefficients: &short_claims,
        })
        .is_err()
    );
    let mut short_tau = fixture.tau1.clone();
    short_tau.pop();
    assert!(
        prepare_coefficient_packing_group_semantics(CoefficientPackingGroupSemanticInputs {
            level_params: &fixture.params,
            opening_batch: &fixture.opening_batch,
            relation_plan: &fixture.relation_plan,
            relation: &fixture.relation,
            group_index: 0,
            prepared_point: &fixture.prepared_point,
            alpha: E::from_u64(3),
            tau1: &short_tau,
            claim_coefficients: &fixture.claim_coefficients,
        })
        .is_err()
    );
    assert!(
        prepare_coefficient_packing_group_semantics(CoefficientPackingGroupSemanticInputs {
            level_params: &fixture.params,
            opening_batch: &fixture.opening_batch,
            relation_plan: &fixture.relation_plan,
            relation: &fixture.relation,
            group_index: 1,
            prepared_point: &fixture.prepared_point,
            alpha: E::from_u64(3),
            tau1: &fixture.tau1,
            claim_coefficients: &fixture.claim_coefficients,
        })
        .is_err()
    );
    let wrong_arity_point = PreparedSubringCoefficientPackingPoint::new(
        fixture.prepared_point.geometry(),
        BasisMode::Lagrange,
        4,
        4,
        10,
        &[E::from_u64(2); 10],
    )
    .unwrap();
    assert!(
        prepare_coefficient_packing_group_semantics(CoefficientPackingGroupSemanticInputs {
            level_params: &fixture.params,
            opening_batch: &fixture.opening_batch,
            relation_plan: &fixture.relation_plan,
            relation: &fixture.relation,
            group_index: 0,
            prepared_point: &wrong_arity_point,
            alpha: E::from_u64(3),
            tau1: &fixture.tau1,
            claim_coefficients: &fixture.claim_coefficients,
        })
        .is_err()
    );

    let canonical_challenges = match fixture.relation.group_opening_view(0).unwrap() {
        RingRelationGroupOpeningView::SubringCoefficientPacking {
            canonical_subring_challenges,
            ..
        } => canonical_subring_challenges.clone(),
        RingRelationGroupOpeningView::EvaluationTrace { .. } => panic!("method was erased"),
    };
    let trace_point = RingMultiplierOpeningPoint::from_base(&RingOpeningPoint {
        position_weights: vec![F::zero(); fixture.params.blocks().positions_per_block],
        live_block_weights: vec![F::zero(); fixture.params.blocks().live_blocks],
    });
    let trace_relation = RingRelationInstance::new(
        vec![RingRelationGroupOpening::evaluation_trace(
            canonical_challenges,
            trace_point,
        )],
        fixture.relation.extension_degree(),
        fixture.opening_batch.clone(),
        fixture.relation.gamma().to_vec(),
        fixture.relation.row_coefficient_rings().clone(),
        fixture.relation.rhs().clone(),
        fixture.relation.v().clone(),
        fixture.relation.role_dims(),
    )
    .unwrap();
    assert!(
        prepare_coefficient_packing_group_semantics(CoefficientPackingGroupSemanticInputs {
            level_params: &fixture.params,
            opening_batch: &fixture.opening_batch,
            relation_plan: &fixture.relation_plan,
            relation: &trace_relation,
            group_index: 0,
            prepared_point: &fixture.prepared_point,
            alpha: E::from_u64(3),
            tau1: &fixture.tau1,
            claim_coefficients: &fixture.claim_coefficients,
        })
        .is_err()
    );

    match fixture.relation.group_opening_view(0).unwrap() {
        RingRelationGroupOpeningView::SubringCoefficientPacking { geometry, .. } => {
            assert_eq!(geometry.partial_base_field_width(), 128);
        }
        RingRelationGroupOpeningView::EvaluationTrace { .. } => panic!("method was erased"),
    }

    for mutate in [
        |params: &mut CommittedGroupParams| params.own_group_mut().opening.log_basis_open = 128,
        |params: &mut CommittedGroupParams| {
            params.own_group_mut().profile.inner.digits.log_basis = 128
        },
    ] {
        let mut malformed = fixture.params.clone();
        mutate(&mut malformed);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            prepare_coefficient_packing_group_semantics(CoefficientPackingGroupSemanticInputs {
                level_params: &malformed,
                opening_batch: &fixture.opening_batch,
                relation_plan: &fixture.relation_plan,
                relation: &fixture.relation,
                group_index: 0,
                prepared_point: &fixture.prepared_point,
                alpha: E::from_u64(3),
                tau1: &fixture.tau1,
                claim_coefficients: &fixture.claim_coefficients,
            })
        }));
        assert!(matches!(outcome, Ok(Err(_))));
    }
}

#[test]
fn structured_stage2_terms_match_independent_dense_tables() {
    type F4 = Prime32Offset99;
    type E4 = FpExt4<F4>;

    let fixture = fixture::<F4, E4>(
        SisModulusProfileId::Q32Offset99,
        1024,
        128,
        64,
        6,
        4,
        13,
        2,
        2,
    );
    let alpha = E4::from_u64(41);
    let semantics = prepare(&fixture, alpha);
    let terms = semantics.stage2_terms();
    let direct_got =
        materialize_stage2_source(terms, CoefficientPackingStage2Source::DirectOpening);
    let packing_z_got = materialize_stage2_source(terms, CoefficientPackingStage2Source::PackingZ);
    let mut direct_expected = vec![E4::zero(); terms.physical_field_len()];
    let mut packing_z_expected = vec![E4::zero(); terms.physical_field_len()];

    let geometry = semantics.geometry();
    assert_eq!(geometry.extension_degree(), 4);
    assert_eq!(geometry.packing_factor(), 4);
    assert_eq!(fixture.prepared_point.num_live_blocks(), 2);
    assert_eq!(
        fixture
            .relation_plan
            .witness_layout()
            .units_for_group(0)
            .unwrap()
            .count(),
        2
    );
    let basis = canonical_extension_basis::<F4, E4>(4).unwrap();
    let opening_gadget = gadget_row_scalars::<F4>(
        fixture.params.open().digits.num_digits,
        fixture.params.open().digits.log_basis,
    );
    let witness_gadget = gadget_row_scalars::<F4>(
        fixture.params.inner().digits.num_digits,
        fixture.params.inner().digits.log_basis,
    );
    let fold_gadget = gadget_row_scalars::<F4>(
        fixture.params.num_digits_fold(),
        fixture.params.open().digits.log_basis,
    );
    let consistency_weight = relation_row_weight(
        fixture.relation_plan.consistency_row_index(0).unwrap(),
        &fixture.tau1,
    )
    .unwrap();
    let scalar_weight = relation_row_weight(
        fixture.relation_plan.scalar_opening_row_index().unwrap(),
        &fixture.tau1,
    )
    .unwrap();
    let d_d = fixture.params.role_dims().d_d();
    let s = geometry.challenge_subring_dimension();
    let kh = geometry.subring_embedding_stride();
    let alpha_powers = scalar_powers(alpha, s);
    let units = fixture
        .relation_plan
        .witness_layout()
        .units_for_group(0)
        .unwrap()
        .collect::<Vec<_>>();

    for claim in 0..fixture.opening_batch.num_total_polynomials() {
        for unit in &units {
            for block in unit.global_block_range() {
                let block_weight = fixture.prepared_point.live_block_weights()[block];
                for (digit, &gadget) in opening_gadget.iter().enumerate() {
                    for (plane, &basis_element) in basis.iter().enumerate() {
                        for (coefficient, &tail_weight) in
                            fixture.prepared_point.tail_weights().iter().enumerate()
                        {
                            let flat = plane * s + coefficient;
                            let physical = unit
                                .e_coefficient_index(
                                    d_d,
                                    fixture.opening_batch.num_total_polynomials(),
                                    fixture.params.open().digits.num_digits,
                                    claim,
                                    block,
                                    flat / d_d,
                                    digit,
                                    flat % d_d,
                                )
                                .unwrap();
                            direct_expected[physical] += scalar_weight
                                * fixture.claim_coefficients[claim]
                                * block_weight
                                * E4::lift_base(gadget)
                                * basis_element
                                * tail_weight;
                        }
                    }
                }
            }
        }
    }

    for unit in &units {
        for (position, &position_weight) in
            fixture.prepared_point.position_weights().iter().enumerate()
        {
            for (witness_digit, &witness_weight) in witness_gadget.iter().enumerate() {
                for (fold_digit, &fold_weight) in fold_gadget.iter().enumerate() {
                    let factor = -(consistency_weight
                        * position_weight
                        * E4::lift_base(witness_weight)
                        * E4::lift_base(fold_weight));
                    for coefficient in 0..geometry.a_ring_dimension() {
                        let low_index = coefficient % kh;
                        let subring_index = coefficient / kh;
                        let source_weight = fixture.prepared_point.packing_weights()[low_index]
                            * alpha_powers[subring_index];
                        let physical = unit
                            .z_coefficient_index(
                                geometry.a_ring_dimension(),
                                fixture.params.blocks().positions_per_block,
                                fixture.params.inner().digits.num_digits,
                                fixture.params.num_digits_fold(),
                                position,
                                witness_digit,
                                fold_digit,
                                coefficient,
                            )
                            .unwrap();
                        packing_z_expected[physical] += factor * source_weight;
                    }
                }
            }
        }
    }

    assert_eq!(direct_got, direct_expected);
    assert_eq!(packing_z_got, packing_z_expected);
    assert_eq!(
        direct_got
            .iter()
            .zip(&packing_z_got)
            .map(|(&direct, &packing_z)| direct + packing_z)
            .collect::<Vec<_>>(),
        direct_expected
            .iter()
            .zip(&packing_z_expected)
            .map(|(&direct, &packing_z)| direct + packing_z)
            .collect::<Vec<_>>()
    );
}

#[test]
fn production_extension_degrees_include_exact_overlap_and_h_greater_than_one() {
    type F1 = Prime128OffsetA7F7;
    let overlap = fixture::<F1, F1>(
        SisModulusProfileId::Q128OffsetA7F7,
        64,
        64,
        64,
        1,
        1,
        6,
        1,
        1,
    );
    let overlap_semantics = prepare(&overlap, F1::from_u64(29));
    assert_eq!(overlap_semantics.geometry().extension_degree(), 1);
    assert_eq!(overlap_semantics.geometry().packing_factor(), 1);
    assert_eq!(
        overlap_semantics.geometry().partial_base_field_width(),
        overlap.params.role_dims().d_a()
    );
    assert!(matches!(
        overlap.relation.group_opening_view(0).unwrap(),
        RingRelationGroupOpeningView::SubringCoefficientPacking { .. }
    ));

    type F4 = Prime32Offset99;
    type E4 = FpExt4<F4>;
    let four_planes = fixture::<F4, E4>(
        SisModulusProfileId::Q32Offset99,
        1024,
        128,
        64,
        2,
        2,
        11,
        1,
        1,
    );
    let four_plane_semantics = prepare(&four_planes, E4::from_u64(31));
    assert_eq!(four_plane_semantics.geometry().extension_degree(), 4);
    assert_eq!(four_plane_semantics.geometry().packing_factor(), 4);
    let q_events = four_planes
        .relation_plan
        .witness_layout()
        .quotient_depth()
        .expect("quotient-lift fixture")
        * 4;
    let events = four_plane_semantics.relation_events().events();
    for event in &events[events.len() - q_events..] {
        assert_eq!(event.physical_coefficients().len(), 64);
        assert_eq!(event.alpha_exponent_start(), 0);
    }
}

#[path = "coefficient_packing_relation_authority_tests.rs"]
mod authority;

#[path = "coefficient_packing_relation_multigroup_tests.rs"]
mod multigroup;
