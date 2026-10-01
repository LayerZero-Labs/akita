use super::*;
use crate::sources::packed_digits::PackedSignedDigits;
use jolt_field::{One, Prime128OffsetA7F7 as F, Zero};

// Same four-digit setup as the existing recursive witness session tests.
fn stage1_session() -> DigitRangeSession<F> {
    let domain = akita_params::FlatBooleanDomain::new(4, 2).unwrap();
    let equality = akita_types::DigitRangeEqualityPoint::from_column_then_ring_challenges(
        &[F::from_u64(3), F::from_u64(5)],
        1,
        1,
    )
    .unwrap();
    let prover = DigitRangeProver::from_packed_digits(
        PackedSignedDigits::from_i8_digits_auto(vec![-2, -1, 0, 1]),
        akita_params::DigitRangePlan::new(16).unwrap(),
        domain,
        equality,
    )
    .unwrap();
    DigitRangeSession::new(prover, None).unwrap()
}

fn drive_to_final_public(session: &mut DigitRangeSession<F>) {
    while session.phase != Phase::FinalPublic {
        match session.phase {
            Phase::Running => {
                let step = session.active.as_ref().unwrap().step(session.product_index);
                let round = session.next_round;
                session
                    .round_polynomial(step, round, session.claim)
                    .unwrap();
                session
                    .bind_challenge(step, round, F::from_u64(round as u64 + 7))
                    .unwrap();
            }
            Phase::ProductPublic => {
                session
                    .public_transition(Stage1Step::Product(session.product_index))
                    .unwrap();
            }
            Phase::ProductBatch => {
                session
                    .bind_batch_challenge(
                        Stage1Transition::ProductBatch(session.product_index),
                        F::from_u64(11),
                    )
                    .unwrap();
            }
            _ => panic!("unexpected phase in standalone range session"),
        }
    }
}

fn final_transition(session: &mut DigitRangeSession<F>) -> (F, Vec<F>) {
    let Stage1PublicTransition::Final {
        range_image_evaluation,
        virtual_evaluations,
    } = session.public_transition(Stage1Step::RangeLeaf).unwrap()
    else {
        panic!("expected final public transition");
    };
    (range_image_evaluation, virtual_evaluations)
}

#[test]
fn stage1_final_public_transition_rejected_step_keeps_state() {
    let mut session = stage1_session();
    let mut untouched = stage1_session();
    drive_to_final_public(&mut session);
    drive_to_final_public(&mut untouched);

    assert!(matches!(
        session.public_transition(Stage1Step::Product(0)),
        Err(AkitaError::InvalidInput(_))
    ));
    assert_eq!(
        final_transition(&mut session),
        final_transition(&mut untouched)
    );
    let actual = session.finish().unwrap();
    let expected = untouched.finish().unwrap();
    assert_eq!(actual.point(), expected.point());
    assert_eq!(actual.final_claim(), expected.final_claim());
}

#[test]
fn stage1_bind_challenge_rejected_arguments_keep_state() {
    for (wrong_step, wrong_round) in [(Stage1Step::Product(0), 1), (Stage1Step::Product(1), 0)] {
        let mut session = stage1_session();
        let mut untouched = stage1_session();
        let step = Stage1Step::Product(0);
        session.round_polynomial(step, 0, F::zero()).unwrap();
        untouched.round_polynomial(step, 0, F::zero()).unwrap();

        assert!(matches!(
            session.bind_challenge(wrong_step, wrong_round, F::one()),
            Err(AkitaError::InvalidInput(_))
        ));
        session.bind_challenge(step, 0, F::from_u64(7)).unwrap();
        untouched.bind_challenge(step, 0, F::from_u64(7)).unwrap();
        drive_to_final_public(&mut session);
        drive_to_final_public(&mut untouched);
        assert_eq!(
            final_transition(&mut session),
            final_transition(&mut untouched)
        );
        let actual = session.finish().unwrap();
        let expected = untouched.finish().unwrap();
        assert_eq!(actual.point(), expected.point());
        assert_eq!(actual.final_claim(), expected.final_claim());
    }
}
