use super::*;
use jolt_field::{One, Prime128OffsetA7F7, Zero};

type F = Prime128OffsetA7F7;

fn boolean_weight(address: usize, point: &[F]) -> F {
    point
        .iter()
        .enumerate()
        .fold(F::one(), |product, (bit, &r)| {
            product
                * if address & (1 << bit) == 0 {
                    F::one() - r
                } else {
                    r
                }
        })
}

fn mle(table: &[F], point: &[F]) -> F {
    table
        .iter()
        .enumerate()
        .map(|(address, &value)| value * boolean_weight(address, point))
        .sum()
}

fn input() -> (Vec<i8>, Vec<F>, [F; 3], F, F) {
    let witness = vec![-4, -3, -2, -1, 0, 1, 2, 3];
    let weights = (0..8)
        .map(|index| F::from_u64(19 + index))
        .collect::<Vec<_>>();
    let point = [3, 5, 7].map(F::from_u64);
    let mut range_image = F::zero();
    let mut relation = F::zero();
    for (address, (&digit, &weight)) in witness.iter().zip(&weights).enumerate() {
        let digit = F::from_i64(i64::from(digit));
        range_image += boolean_weight(address, &point) * digit * (digit + F::one());
        relation += digit * weight;
    }
    (witness, weights, point, range_image, relation)
}

fn session() -> CpuStage2SessionHandle<F> {
    let (witness, weights, point, range_image, relation) = input();
    CpuStage2SessionHandle::from_dense_relation(
        witness,
        weights,
        relation,
        &point,
        range_image,
        F::from_u64(23),
    )
    .unwrap()
}

#[test]
fn dense_entry_uses_existing_round_and_terminal_equations() {
    let (witness, weights, stage1_point, range_image, relation) = input();
    let batching = F::from_u64(23);
    let mut prover = session();
    let mut claim = batching * range_image + relation;
    assert_eq!(prover.input_claim(), claim);
    assert_eq!(prover.num_rounds(), 3);
    let challenges = [11, 13, 17].map(F::from_u64);
    for (round, &challenge) in challenges.iter().enumerate() {
        let polynomial = prover.round_polynomial(round, claim).unwrap();
        assert!(polynomial.degree() <= 3);
        assert_eq!(
            polynomial.evaluate(F::zero()) + polynomial.evaluate(F::one()),
            claim
        );
        claim = polynomial.evaluate(challenge);
        prover.bind_challenge(round, challenge).unwrap();
    }
    let final_claims = prover.finish().unwrap();
    let witness = witness
        .into_iter()
        .map(|digit| F::from_i64(i64::from(digit)))
        .collect::<Vec<_>>();
    let witness_evaluation = mle(&witness, &challenges);
    let equality = stage1_point
        .iter()
        .zip(challenges)
        .map(|(&left, right)| (F::one() - left) * (F::one() - right) + left * right)
        .product::<F>();
    let expected = batching * equality * witness_evaluation * (witness_evaluation + F::one())
        + witness_evaluation * mle(&weights, &challenges);
    assert_eq!(final_claims.witness_evaluation(), witness_evaluation);
    assert_eq!(final_claims.final_claim(), claim);
    assert_eq!(claim, expected);
}

#[test]
fn dense_entry_rejects_wrong_round_order_and_early_terminal() {
    let mut prover = session();
    let claim = prover.input_claim();
    assert!(prover.bind_challenge(0, F::one()).is_err());
    assert!(prover.round_polynomial(1, claim).is_err());
    assert!(prover.round_polynomial(0, claim + F::one()).is_err());
    let polynomial = prover.round_polynomial(0, claim).unwrap();
    assert!(prover.round_polynomial(0, claim).is_err());
    assert!(prover.bind_challenge(1, F::one()).is_err());
    prover.bind_challenge(0, F::from_u64(11)).unwrap();
    assert!(prover.bind_challenge(0, F::one()).is_err());
    assert!(prover
        .round_polynomial(1, polynomial.evaluate(F::from_u64(11)) + F::one())
        .is_err());
    assert!(prover.finish().is_err());
    assert!(session().finish().is_err());
}

#[test]
fn dense_entry_checks_lengths_and_digit_bounds_before_construction() {
    for witness in [vec![], vec![0; 3]] {
        assert!(CpuStage2SessionHandle::<F>::from_dense_relation(
            witness,
            vec![],
            F::zero(),
            &[],
            F::zero(),
            F::one(),
        )
        .is_err());
    }
    for invalid_digit in [-128, -5, 4, 127] {
        assert!(CpuStage2SessionHandle::<F>::from_dense_relation(
            vec![invalid_digit, 0],
            vec![F::zero(); 2],
            F::zero(),
            &[F::from_u64(3)],
            F::zero(),
            F::one(),
        )
        .is_err());
    }
    assert!(CpuStage2SessionHandle::<F>::from_dense_relation(
        vec![0; 8],
        vec![F::zero(); 4],
        F::zero(),
        &[F::zero(); 3],
        F::zero(),
        F::one(),
    )
    .is_err());
    assert!(CpuStage2SessionHandle::<F>::from_dense_relation(
        vec![0; 8],
        vec![F::zero(); 8],
        F::zero(),
        &[F::zero(); 2],
        F::zero(),
        F::one(),
    )
    .is_err());
}

#[test]
fn dense_singleton_has_the_same_terminal_equation_without_rounds() {
    let witness = F::from_i64(-4);
    let weight = F::from_u64(7);
    let batching = F::from_u64(11);
    let relation = witness * weight;
    let range_image = witness * (witness + F::one());
    let session = CpuStage2SessionHandle::from_dense_relation(
        vec![-4],
        vec![weight],
        relation,
        &[],
        range_image,
        batching,
    )
    .unwrap();
    assert_eq!(session.num_rounds(), 0);
    let claims = session.finish().unwrap();
    assert_eq!(claims.witness_evaluation(), witness);
    assert_eq!(claims.final_claim(), relation + batching * range_image);
}
