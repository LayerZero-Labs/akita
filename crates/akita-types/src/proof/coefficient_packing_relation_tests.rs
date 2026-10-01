use super::*;

use akita_algebra::poly::multilinear_eval;
use jolt_field::{
    CanonicalEncoding, Ext2, ExtField, Field, FpExt4, One, Prime128OffsetA7F7, Prime32Offset99,
    Prime64Offset59, Ring, Zero,
};

use crate::{
    relation_claim_from_compressed_rhs_extension, relation_rhs_coeff_len,
    RingMultiplierOpeningPoint, RingRelationGroupOpening, RingVec,
};
use akita_params::InnerCommitMatrixParams;
use akita_params::{BasisMode, CommitmentRingDims, RingOpeningPoint, SisModulusProfileId};

type F = Prime64Offset59;
type E = Ext2<F>;

use super::test_fixtures::{
    coefficient_packing_fixture as fixture, CoefficientPackingFixture as Fixture,
};

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
        akita_params::CommittedSourceEncoding::TensorSubfieldProjection { extension_degree };
    assert!(matches!(
        RelationWitnessGeometry::for_level(
            &fixture.params,
            &fixture.opening_batch,
            extension_degree,
        ),
        Err(AkitaError::InvalidSetup(_))
    ));
}

fn prepare<'a, Base, Extension>(
    fixture: &'a Fixture<Base, Extension>,
    alpha: Extension,
) -> CoefficientPackingGroupSemantics<'a, Extension>
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

#[test]
fn relation_event_consumer_matches_dense_oracle() {
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
        assert!(semantics
            .relation_events()
            .evaluate_at_point(&point[..point.len() - 1])
            .is_err());
    }
}

#[test]
fn relation_event_large_batch_matches_dense_oracle() {
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

    let mut repeated_events = semantics.relation_events().clone();
    let event_work = repeated_events
        .events()
        .iter()
        .map(|event| {
            event.physical_coefficients().len() / repeated_events.relation_coefficient_block_len()
        })
        .sum::<usize>();
    let event_repetitions = 1024usize.div_ceil(event_work);
    let original_events = repeated_events.events.clone();
    repeated_events.events = original_events
        .iter()
        .cloned()
        .cycle()
        .take(original_events.len() * event_repetitions)
        .collect();
    let repeated_event_work = repeated_events
        .events()
        .iter()
        .map(|event| {
            event.physical_coefficients().len() / repeated_events.relation_coefficient_block_len()
        })
        .sum::<usize>();
    assert!(repeated_event_work >= 1024);
    let mut dense_events = materialize_events(&repeated_events);
    dense_events.resize(padded_len, E::zero());
    assert_eq!(
        repeated_events.evaluate_at_point(&point).unwrap(),
        multilinear_eval(&dense_events, &point).unwrap()
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
    assert_eq!(semantics.group_claim_range(), 0..2);
    assert_eq!(
        semantics.group_claim_coefficients(),
        fixture.claim_coefficients
    );
    assert_eq!(
        semantics.scalar_claim_weight(),
        relation_row_weight(
            fixture.relation_plan.scalar_opening_row_index().unwrap(),
            &fixture.tau1,
        )
        .unwrap()
    );
    assert_eq!(
        semantics.consistency_weight(),
        relation_row_weight(
            fixture.relation_plan.consistency_row_index(0).unwrap(),
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

    assert_eq!(semantics.prepared_point(), &fixture.prepared_point);
    assert_eq!(semantics.witness_units().len(), 1);
    assert_eq!(semantics.d_d(), fixture.params.role_dims().d_d());
    assert_eq!(semantics.num_digits_open(), depth_open);
    assert_eq!(
        semantics.num_digits_inner(),
        fixture.params.inner().digits.num_digits
    );
    assert_eq!(
        semantics.num_digits_fold(),
        fixture.params.num_digits_fold()
    );
    assert_eq!(semantics.alpha_powers(), alpha_powers);
    assert_eq!(semantics.basis(), basis);
    assert_eq!(
        semantics.opening_gadget(),
        opening_gadget
            .into_iter()
            .map(E::lift_base)
            .collect::<Vec<_>>()
    );
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
fn packing_metadata_exposes_canonical_backend_factors() {
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

    let geometry = semantics.geometry();
    assert_eq!(geometry.extension_degree(), 4);
    assert_eq!(geometry.packing_factor(), 4);
    assert_eq!(fixture.prepared_point.num_live_blocks(), 2);
    assert_eq!(semantics.witness_units().len(), 2);
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
    let alpha_powers = scalar_powers(alpha, s);
    assert_eq!(
        semantics.group_claim_range(),
        0..fixture.opening_batch.num_total_polynomials()
    );
    assert_eq!(
        semantics.group_claim_coefficients(),
        fixture.claim_coefficients
    );
    assert_eq!(semantics.prepared_point(), &fixture.prepared_point);
    assert_eq!(semantics.consistency_weight(), consistency_weight);
    assert_eq!(semantics.scalar_claim_weight(), scalar_weight);
    assert_eq!(semantics.d_d(), d_d);
    assert_eq!(semantics.num_digits_open(), opening_gadget.len());
    assert_eq!(semantics.num_digits_inner(), witness_gadget.len());
    assert_eq!(semantics.num_digits_fold(), fold_gadget.len());
    assert_eq!(
        semantics.num_positions_per_block(),
        fixture.params.blocks().positions_per_block
    );
    assert_eq!(
        semantics.physical_field_len(),
        fixture.relation_plan.digit_witness_domain().live_len()
    );
    assert_eq!(
        semantics.relation_coefficient_block_len(),
        fixture
            .relation_plan
            .relation_address_geometry()
            .relation_coefficient_block_len()
    );
    assert_eq!(semantics.alpha_powers(), alpha_powers);
    assert_eq!(semantics.basis(), basis);
    assert_eq!(
        semantics.opening_gadget(),
        opening_gadget
            .into_iter()
            .map(E4::lift_base)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        semantics.witness_gadget(),
        witness_gadget
            .into_iter()
            .map(E4::lift_base)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        semantics.fold_gadget(),
        fold_gadget
            .into_iter()
            .map(E4::lift_base)
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
