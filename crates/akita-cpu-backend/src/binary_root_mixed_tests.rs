use super::*;
use akita_algebra::eq_poly::EqPolynomial;
use akita_prover::backend::{NativeRootRelationRequest, RelationCompiler};
use akita_types::native_root_relation::{NativeRootGroupKind, NativeRootRow, NativeRootTerm};
use jolt_field::{One, Zero};
use jolt_poly::UnivariatePolynomial;

type E = Prime128Offset275;

fn compact_fixture() -> Fixture {
    let descriptor = BinarySourceDescriptor::new(
        BinaryIndexMap::new(128, 4, 4, 1).unwrap(),
        vec![[1; 32]],
        BinaryCommitmentProfiles {
            a: [3; 32],
            b: [4; 32],
        },
    )
    .unwrap();
    let matrix = vec![vec![std::array::from_fn(|coefficient| {
        i128::from(Q24_CENTERED_BOUND) - (coefficient * 13 % 200) as i128
    })]];
    let mut source = [0; D];
    source[0] = 1;
    source[10] = -1;
    let remainder = divide(reference(&matrix[0][0], &source)).0;
    let image = remainder
        .map(|value| Q24Image::from_residue(value.rem_euclid(i128::from(Q24)) as u32).unwrap());
    let challenges = SeedDraw
        .draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            162,
            0,
            1,
            1,
            &FoldChallengeFamily::BinaryRoot,
            None,
        )
        .unwrap();
    Fixture {
        descriptor,
        matrix,
        sources: vec![source],
        images: vec![image],
        challenges,
    }
}

/// Ordinary target calculated directly over Z, without a quotient-ring kernel.
fn ordinary_product(left: &[i128; 8], right: &[i128; 8]) -> Vec<E> {
    let mut product = [0i128; 15];
    for (i, &a) in left.iter().enumerate() {
        for (j, &b) in right.iter().enumerate() {
            product[i + j] += a * b;
        }
    }
    for exponent in (8..15).rev() {
        product[exponent - 8] -= product[exponent];
    }
    product[..8].iter().copied().map(E::from_i128).collect()
}

fn mixed_root() -> (NativeRootRelation<E>, Vec<i8>) {
    let input = compact_fixture();
    let fixed = build(&input).unwrap();
    let (binary_plan, binary_digits) = fixed.successor_table(256).unwrap();
    let mut segments = binary_plan.segments().to_vec();
    segments.push(RootRepackingSegment {
        geometry: RelationRowGeometry::native(8).unwrap(),
        elements: 1,
        digits: 1,
    });
    let plan = RootCoefficientRepacking::new(segments, 256).unwrap();
    let mut digits = vec![0; plan.domain_len()];
    let binary_end = binary_plan.segment_range(1).unwrap().end;
    digits[..binary_end].copy_from_slice(&binary_digits[..binary_end]);
    assert_eq!(
        plan.segment_range(0).unwrap(),
        binary_plan.segment_range(0).unwrap()
    );
    assert_eq!(
        plan.segment_range(1).unwrap(),
        binary_plan.segment_range(1).unwrap()
    );
    let ordinary = [-4i128, -3, -2, -1, 0, 1, 2, 3];
    for (coefficient, &value) in ordinary.iter().enumerate() {
        let address = plan.coefficient_address(2, 0, 0, coefficient).unwrap();
        digits[address] = i8::try_from(value).unwrap();
    }
    let multiplier = [2, -1, 1, 0, 0, 0, 0, 1];
    let ordinary_group = NativeRootGroup {
        kind: NativeRootGroupKind::Ordinary(FoldChallengeFamily::pm1_only(4)),
        rows: vec![NativeRootRow {
            polynomial: RelationPolynomial::negacyclic(8).unwrap(),
            terms: vec![NativeRootTerm {
                segment: 2,
                element: 0,
                multiplier: multiplier.iter().copied().map(E::from_i128).collect(),
            }],
            target: ordinary_product(&multiplier, &ordinary),
        }],
    };
    let binary_group = fixed
        .commitment_group::<E>(&input.descriptor, &input.matrix)
        .unwrap();
    let relation = NativeRootRelation::new(plan, vec![binary_group, ordinary_group]).unwrap();
    assert_eq!(relation.row_count(), 2);
    assert_eq!(relation.repacking().domain_len(), 16384);
    (relation, digits)
}

fn shared_weights(
    relation: &NativeRootRelation<E>,
) -> akita_types::native_root_relation::NativeRootWeights<E> {
    let padding_weights = (0..relation.repacking().domain_len())
        .filter(|&address| relation.repacking().is_padding_address(address).unwrap())
        .map(|address| E::from_u64(7 + address as u64))
        .collect::<Vec<_>>();
    relation
        .ring_switch(
            E::from_u64(17),
            &[E::from_u64(3), E::from_u64(5)],
            &padding_weights,
        )
        .unwrap()
}

#[test]
fn mixed_q24_and_ordinary_root_share_all_stage2_rounds_and_verifier_weights() {
    let (relation, digits) = mixed_root();
    let table = digits
        .iter()
        .map(|&digit| E::from_i64(i64::from(digit)))
        .collect::<Vec<_>>();
    relation.repacking().validate_digit_table(&table).unwrap();
    let padding_weights = (0..relation.repacking().domain_len())
        .filter(|&address| relation.repacking().is_padding_address(address).unwrap())
        .map(|address| E::from_u64(7 + address as u64))
        .collect::<Vec<_>>();
    let backend = crate::CpuBackend::<E, E>::for_arithmetic_tests();
    let compiled = backend
        .compile_relation(NativeRootRelationRequest {
            relation: &relation,
            alpha: E::from_u64(17),
            row_weights: &[E::from_u64(3), E::from_u64(5)],
            padding_weights: &padding_weights,
        })
        .unwrap();
    let weights = compiled.weights.clone();
    weights.check_table(&table).unwrap();
    let variables = weights.table_len().trailing_zeros() as usize;
    let stage1_point = (0..variables)
        .map(|coordinate| E::from_u64(19 + coordinate as u64))
        .collect::<Vec<_>>();
    let equality = EqPolynomial::evals(&stage1_point).unwrap();
    let range_image = table
        .iter()
        .zip(&equality)
        .map(|(&digit, &weight)| weight * digit * (digit + E::one()))
        .sum::<E>();
    let batching = E::from_u64(31);
    let mut session = compiled
        .begin_stage2(digits, &stage1_point, range_image, batching)
        .unwrap();
    assert_eq!(session.num_rounds(), variables);
    let mut claim = batching * range_image + weights.claim();
    assert_eq!(session.input_claim(), claim);
    let mut point = Vec::new();
    for round in 0..variables {
        let polynomial = session.round_polynomial(round, claim).unwrap();
        assert!(polynomial.degree() <= 3);
        assert_eq!(
            polynomial.evaluate(E::zero()) + polynomial.evaluate(E::one()),
            claim
        );
        let challenge = E::from_u64(37 + round as u64);
        claim = polynomial.evaluate(challenge);
        session.bind_challenge(round, challenge).unwrap();
        point.push(challenge);
    }
    let terminal = session.finish().unwrap();
    let table_equality = EqPolynomial::evals(&point).unwrap();
    let witness_evaluation = table
        .iter()
        .zip(table_equality)
        .map(|(&digit, weight)| digit * weight)
        .sum::<E>();
    assert_eq!(terminal.witness_evaluation(), witness_evaluation);
    let equality = stage1_point
        .iter()
        .zip(&point)
        .map(|(&left, &right)| (E::one() - left) * (E::one() - right) + left * right)
        .product::<E>();
    let verifier_expected =
        batching * equality * witness_evaluation * (witness_evaluation + E::one())
            + weights.evaluate(&point).unwrap() * witness_evaluation;
    assert_eq!(terminal.final_claim(), verifier_expected);
    assert_eq!(claim, verifier_expected);

    // The shared verifier consumes this same public oracle through its trait.
    let setup = crate::AkitaProverSetup::<E>::generate_with_capacity(
        8,
        1,
        akita_params::SetupMatrixCapacity {
            num_field_elements: 1,
        },
    )
    .unwrap();
    let verifier_weights: &dyn akita_verifier::Stage2RelationWeights<E, E> = &weights;
    assert!(verifier_weights.is_reduced_evaluation());
    assert_eq!(
        verifier_weights
            .evaluate(&point, &setup.expanded, E::from_u64(17), None)
            .unwrap(),
        weights.evaluate(&point).unwrap()
    );
    assert!(verifier_weights
        .evaluate(&point, &setup.expanded, E::from_u64(17), Some(E::zero()))
        .is_err());
}

#[test]
fn mixed_root_scalar_relation_rejects_each_segment_and_padding_mutation() {
    let (relation, digits) = mixed_root();
    let table = digits
        .iter()
        .map(|&digit| E::from_i64(i64::from(digit)))
        .collect::<Vec<_>>();
    let weights = shared_weights(&relation);
    weights.check_table(&table).unwrap();
    for segment in 0..3 {
        let address = relation
            .repacking()
            .coefficient_address(segment, 0, 0, 0)
            .unwrap();
        let mut changed = table.clone();
        let delta = if digits[address] == 3 { -1 } else { 1 };
        changed[address] += E::from_i64(delta);
        relation.repacking().validate_digit_table(&changed).unwrap();
        assert!(weights.check_table(&changed).is_err(), "segment {segment}");
    }
    for address in [648, 1023, relation.repacking().domain_len() - 1] {
        assert!(relation.repacking().is_padding_address(address).unwrap());
        let mut changed = table.clone();
        changed[address] += E::one();
        assert!(weights.check_table(&changed).is_err());
        assert!(relation.repacking().validate_digit_table(&changed).is_err());
    }
    let mut input = compact_fixture();
    input.sources[0][0] += 1;
    assert!(build(&input).is_err());
    input.sources[0][0] -= 1;
    let original = input.images[0][0].centered();
    input.images[0][0] = Q24Image::from_centered(original + 1).unwrap();
    assert!(build(&input).is_err());
}
