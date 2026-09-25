use super::compact::{CoefficientPackingAffineRelationFamily, CoefficientPackingCompactFactors};
use super::*;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::poly::multilinear_eval;
use akita_algebra::ring::scalar_powers;
use akita_types::{
    coefficient_packing_fixture, coefficient_packing_multigroup_fixture,
    prepare_coefficient_packing_batch_semantics, BasisMode, CoefficientPackingFixture,
    CoefficientPackingGroupSemantics, CoefficientPackingStage2Source,
    PreparedSubringCoefficientPackingPoint, RelationWeightEvent, SisModulusProfileId,
};
use jolt_field::{
    Ext2, FpExt4, One, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59, Ring, Zero,
};
use std::sync::Arc;

type F = Prime64Offset59;
type E = Ext2<F>;

/// The prover's expanded relation events and Stage 2 terms, materialized over
/// the padded flat domain. These are the dense oracles for the compact factors.
struct DenseOracles<E> {
    point: Vec<E>,
    relation: Vec<E>,
    stage2: Vec<E>,
}

fn materialize_events<E: Field>(
    events: &[RelationWeightEvent<E>],
    alpha: E,
    padded_len: usize,
) -> Vec<E> {
    let max_exponent = events
        .iter()
        .map(|event| event.alpha_exponent_start() + event.physical_coefficients().len())
        .max()
        .unwrap_or(0);
    let alpha_powers = scalar_powers(alpha, max_exponent);
    let mut dense = vec![E::zero(); padded_len];
    for event in events {
        for (offset, index) in event.physical_coefficients().enumerate() {
            dense[index] += event.scalar() * alpha_powers[event.alpha_exponent_start() + offset];
        }
    }
    dense
}

fn materialize_stage2<E: Field>(
    semantics: &CoefficientPackingGroupSemantics<E>,
    padded_len: usize,
) -> Vec<E> {
    let terms = semantics.stage2_terms();
    let mut dense = vec![E::zero(); padded_len];
    for term in terms.terms() {
        let source = match term.source() {
            CoefficientPackingStage2Source::DirectOpening => terms.direct_opening_source(),
            CoefficientPackingStage2Source::PackingZ => terms.packing_z_source(),
        };
        for segment in &terms.segments()[term.segments()] {
            for (physical, source_index) in segment
                .physical_coefficients()
                .zip(segment.source_coefficients())
            {
                dense[physical] += term.factor() * source[source_index];
            }
        }
    }
    dense
}

/// An extension element with every base coordinate nonzero, so cross terms
/// between extension coordinates cannot vanish.
fn genuine_extension<Base: Field, Extension: ExtField<Base>>(seed: u64) -> Extension {
    Extension::from_base_fn(|coordinate| Base::from_u64(2 + 3 * seed + 5 * coordinate as u64))
}

/// Rebuild `point` over the same geometry with genuine-extension coordinates.
fn genuine_extension_point<Base: Field, Extension: ExtField<Base>>(
    point: &PreparedSubringCoefficientPackingPoint<Extension>,
    basis: BasisMode,
    seed: u64,
) -> PreparedSubringCoefficientPackingPoint<Extension> {
    let values = (0..point.source_num_vars() as u64)
        .map(|index| genuine_extension::<Base, Extension>(seed + index))
        .collect::<Vec<_>>();
    PreparedSubringCoefficientPackingPoint::new(
        point.geometry(),
        basis,
        point.num_live_positions(),
        point.num_positions_per_block(),
        point.source_num_vars(),
        &values,
    )
    .unwrap()
}

fn batch_inputs<'a, Base: Field, Extension: Field>(
    fixture: &'a CoefficientPackingFixture<Base, Extension>,
    points: &'a [(usize, &'a PreparedSubringCoefficientPackingPoint<Extension>)],
    alpha: Extension,
) -> CoefficientPackingBatchSemanticInputs<'a, Base, Extension> {
    CoefficientPackingBatchSemanticInputs {
        level_params: &fixture.params,
        opening_batch: &fixture.opening_batch,
        relation_plan: &fixture.relation_plan,
        relation: &fixture.relation,
        prepared_points: points,
        alpha,
        tau1: &fixture.tau1,
        claim_coefficients: &fixture.claim_coefficients,
    }
}

fn prepare_compact_and_oracles<Base, Extension>(
    fixture: &CoefficientPackingFixture<Base, Extension>,
    alpha: Extension,
    point_seed: u64,
) -> (
    CoefficientPackingVerifierGroupSemantics<Extension>,
    DenseOracles<Extension>,
)
where
    Base: Field + CanonicalEncoding + Ring,
    Extension: ExtField<Base> + FpExtEncoding<Base> + Ring,
{
    let points = [(0, &fixture.prepared_point)];
    let compact =
        prepare_coefficient_packing_verifier_batch_semantics(batch_inputs(fixture, &points, alpha))
            .unwrap()
            .groups()[0]
            .clone();
    let (events, expanded) =
        prepare_coefficient_packing_batch_semantics(batch_inputs(fixture, &points, alpha)).unwrap();
    let semantics = &expanded.groups()[0];
    let padded_len = semantics
        .stage2_terms()
        .physical_field_len()
        .next_power_of_two();
    let point = (0..padded_len.trailing_zeros())
        .map(|index| genuine_extension::<Base, Extension>(point_seed + u64::from(index)))
        .collect();
    let oracles = DenseOracles {
        point,
        relation: materialize_events(&events, alpha, padded_len),
        stage2: materialize_stage2(semantics, padded_len),
    };
    (compact, oracles)
}

fn assert_compact_factors_match_dense<Base, Extension>(
    fixture: &CoefficientPackingFixture<Base, Extension>,
    alphas: [Extension; 3],
    point_seed: u64,
) where
    Base: Field + CanonicalEncoding + Ring,
    Extension: ExtField<Base> + FpExtEncoding<Base> + Ring,
{
    for alpha in alphas {
        let (compact, oracles) = prepare_compact_and_oracles(fixture, alpha, point_seed);
        let point = &oracles.point;
        let factors = compact.compact_factors();
        assert_eq!(
            factors.evaluate_relation_at_point(point).unwrap(),
            multilinear_eval(&oracles.relation, point).unwrap()
        );
        assert_eq!(
            factors.evaluate_stage2_at_point(point).unwrap(),
            multilinear_eval(&oracles.stage2, point).unwrap()
        );
        assert!(factors
            .evaluate_relation_at_point(&point[..point.len() - 1])
            .is_err());
    }
}

#[test]
fn compact_factors_match_dense_event_and_stage2_oracles() {
    let fixture = coefficient_packing_fixture::<F, E>(
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
    assert_compact_factors_match_dense(&fixture, [E::zero(), E::one(), E::from_u64(17)], 23);
}

#[test]
fn compact_factors_match_dense_oracles_under_monomial_basis() {
    let mut fixture = coefficient_packing_fixture::<F, E>(
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
    let lagrange = &fixture.prepared_point;
    let point_values = (0..lagrange.source_num_vars())
        .map(|index| E::from_u64((index + 2) as u64))
        .collect::<Vec<_>>();
    fixture.prepared_point = PreparedSubringCoefficientPackingPoint::new(
        lagrange.geometry(),
        BasisMode::Monomial,
        lagrange.num_live_positions(),
        lagrange.num_positions_per_block(),
        lagrange.source_num_vars(),
        &point_values,
    )
    .unwrap();
    assert_compact_factors_match_dense(&fixture, [E::zero(), E::one(), E::from_u64(17)], 23);
}

/// Replace every verifier-supplied scalar in `fixture` (tau1, claim
/// coefficients, prepared point) with genuine-extension values, then compare
/// the compact factors with the dense oracles under both bases at
/// genuine-extension alphas.
fn assert_compact_factors_match_dense_at_genuine_extension_values<Base, Extension>(
    mut fixture: CoefficientPackingFixture<Base, Extension>,
) where
    Base: Field + CanonicalEncoding + Ring,
    Extension: ExtField<Base> + FpExtEncoding<Base> + Ring,
{
    assert!(Extension::DEGREE > 1);
    for (index, tau) in fixture.tau1.iter_mut().enumerate() {
        *tau = genuine_extension::<Base, Extension>(100 + index as u64);
    }
    for (index, coefficient) in fixture.claim_coefficients.iter_mut().enumerate() {
        *coefficient = genuine_extension::<Base, Extension>(200 + index as u64);
    }
    for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
        fixture.prepared_point =
            genuine_extension_point::<Base, Extension>(&fixture.prepared_point, basis, 300);
        let alphas = [400, 401, 402].map(genuine_extension::<Base, Extension>);
        assert_compact_factors_match_dense(&fixture, alphas, 500);
    }
}

#[test]
fn compact_factors_match_dense_oracles_at_genuine_extension_values() {
    assert_compact_factors_match_dense_at_genuine_extension_values(coefficient_packing_fixture::<
        F,
        E,
    >(
        SisModulusProfileId::Q64Offset59,
        256,
        128,
        64,
        6,
        4,
        11,
        2,
        2,
    ));
    for d_d in [64, 128] {
        assert_compact_factors_match_dense_at_genuine_extension_values(
            coefficient_packing_fixture::<Prime32Offset99, FpExt4<Prime32Offset99>>(
                SisModulusProfileId::Q32Offset99,
                1024,
                d_d,
                64,
                6,
                4,
                13,
                2,
                2,
            ),
        );
    }
}

#[test]
fn compact_factors_cover_overlap_and_fp32_h4_geometries() {
    let overlap = coefficient_packing_fixture::<Prime128OffsetA7F7, Prime128OffsetA7F7>(
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
    let alphas = [0, 1, 19];
    assert_compact_factors_match_dense(&overlap, alphas.map(Prime128OffsetA7F7::from_u64), 29);

    for d_d in [64, 128] {
        type X = FpExt4<Prime32Offset99>;
        let h4 = coefficient_packing_fixture::<Prime32Offset99, X>(
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
        assert_compact_factors_match_dense(&h4, alphas.map(X::from_u64), 29);
    }
    let recursive_role_subcolumns =
        coefficient_packing_fixture::<Prime128OffsetA7F7, Prime128OffsetA7F7>(
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
    assert_compact_factors_match_dense(
        &recursive_role_subcolumns,
        alphas.map(Prime128OffsetA7F7::from_u64),
        29,
    );
    let (compact, _) = prepare_compact_and_oracles(
        &recursive_role_subcolumns,
        Prime128OffsetA7F7::from_u64(19),
        29,
    );
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
    let fixture = coefficient_packing_fixture::<Prime128OffsetA7F7, Prime128OffsetA7F7>(
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
    let alphas = [0, 1, 19].map(Prime128OffsetA7F7::from_u64);
    assert_compact_factors_match_dense(&fixture, alphas, 29);
}

#[test]
fn multi_group_compact_factors_follow_relation_group_order() {
    let mut fixture = coefficient_packing_multigroup_fixture();
    for (index, tau) in fixture.tau1.iter_mut().enumerate() {
        *tau = genuine_extension::<F, E>(100 + index as u64);
    }
    for (index, coefficient) in fixture.claim_coefficients.iter_mut().enumerate() {
        *coefficient = genuine_extension::<F, E>(200 + index as u64);
    }
    for (group, point) in &mut fixture.points {
        *point =
            genuine_extension_point::<F, E>(point, BasisMode::Lagrange, 300 + 50 * *group as u64);
    }
    let point_refs = fixture
        .points
        .iter()
        .map(|(group, point)| (*group, point))
        .collect::<Vec<_>>();
    let alpha = genuine_extension::<F, E>(37);
    let inputs = || CoefficientPackingBatchSemanticInputs {
        level_params: &fixture.params,
        opening_batch: &fixture.opening_batch,
        relation_plan: &fixture.relation_plan,
        relation: &fixture.relation,
        prepared_points: &point_refs,
        alpha,
        tau1: &fixture.tau1,
        claim_coefficients: &fixture.claim_coefficients,
    };
    let compact_batch = prepare_coefficient_packing_verifier_batch_semantics(inputs()).unwrap();
    let (events, expanded) = prepare_coefficient_packing_batch_semantics(inputs()).unwrap();
    assert_eq!(
        compact_batch
            .groups()
            .iter()
            .map(CoefficientPackingVerifierGroupSemantics::group_index)
            .collect::<Vec<_>>(),
        vec![1, 0]
    );
    let padded_len = expanded.groups()[0]
        .stage2_terms()
        .physical_field_len()
        .next_power_of_two();
    let point = (0..u64::from(padded_len.trailing_zeros()))
        .map(|bit| genuine_extension::<F, E>(41 + bit))
        .collect::<Vec<_>>();
    // The E/Q relation is a batch-wide polynomial: the per-group compact
    // relations must sum to the dense evaluation of every group's events.
    let mut relation_sum = E::zero();
    for (compact, semantics) in compact_batch.groups().iter().zip(expanded.groups()) {
        assert_eq!(compact.group_index(), semantics.group_index());
        assert_eq!(
            compact.group_claim_range(),
            semantics.stage2_terms().group_claim_range()
        );
        assert_eq!(
            semantics
                .stage2_terms()
                .physical_field_len()
                .next_power_of_two(),
            padded_len
        );
        relation_sum += compact
            .compact_factors()
            .evaluate_relation_at_point(&point)
            .unwrap();
        assert_eq!(
            compact
                .compact_factors()
                .evaluate_stage2_at_point(&point)
                .unwrap(),
            semantics.stage2_terms().evaluate_at_point(&point).unwrap()
        );
    }
    assert_ne!(relation_sum, E::zero());
    assert_eq!(
        relation_sum,
        multilinear_eval(&materialize_events(&events, alpha, padded_len), &point).unwrap()
    );
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
        coefficient_weights: coefficient_weights.clone().into(),
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
