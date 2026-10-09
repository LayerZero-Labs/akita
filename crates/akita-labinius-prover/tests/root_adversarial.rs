#![cfg(feature = "labinius")]

mod root_forgery_support;

use akita_error::AkitaError;
use akita_labinius_verifier::lowered::{
    a_row_residual, check_lowered_clear, parity_row_residual, witness_weights_dense,
};
use jolt_field::Zero;
use root_forgery_support::root_reduction_support::{Case, BASES};
use root_forgery_support::{assemble, combined_terminal_matches, Attack};

#[test]
fn honest_sequence_agrees_with_the_lowered_checker_at_transcript_challenges() {
    for base in BASES {
        for fold in [0, 1] {
            let mut case = Case::new(base, fold);
            let evidence = assemble(&mut case, Attack::Honest);
            check_lowered_clear(
                &case.layout,
                &evidence.public,
                &evidence.digits,
                &case.image,
            )
            .unwrap();
            for row in 0..case.layout.n_a() {
                assert!(a_row_residual(
                    &case.layout,
                    &evidence.public,
                    case.admitted.setup(),
                    &case.commitment,
                    &evidence.response,
                    row
                )
                .unwrap()
                .is_zero());
            }
            assert!(
                parity_row_residual(&case.layout, &evidence.public, &evidence.response)
                    .unwrap()
                    .is_zero()
            );
            assert!(combined_terminal_matches(&case, &evidence.proof));
            case.verify(&evidence.proof).unwrap();
            // The independent round construction agrees with the production prover.
            assert_eq!(evidence.proof, case.prove().0);
        }
    }
}

#[test]
fn nonzero_in_alphabet_padding_tail_accepts_the_full_root_proof() {
    for base in BASES {
        let mut case = Case::new(base, 0);
        let evidence = assemble(&mut case, Attack::NonzeroTail);
        let tail = 648 * case.layout.encoding().response().digit_count();
        assert_eq!(evidence.digits[tail], (1 << base.bits()) - 1);
        let weights = witness_weights_dense(&case.layout, &evidence.public).unwrap();
        assert!(
            weights[tail].is_zero(),
            "the selected tail must have zero relation weight"
        );
        check_lowered_clear(
            &case.layout,
            &evidence.public,
            &evidence.digits,
            &case.image,
        )
        .unwrap();
        assert!(combined_terminal_matches(&case, &evidence.proof));
        case.verify(&evidence.proof).unwrap();
    }
}

#[test]
fn invalid_alphabet_is_rejected_by_the_combined_terminal_for_each_base() {
    for base in BASES {
        let mut case = Case::new(base, 1);
        let e = assemble(&mut case, Attack::BadDigit);
        assert_eq!(
            check_lowered_clear(&case.layout, &e.public, &e.digits, &case.image),
            Err(AkitaError::InvalidProof)
        );
        assert!(a_row_residual(
            &case.layout,
            &e.public,
            case.admitted.setup(),
            &case.commitment,
            &e.response,
            0
        )
        .unwrap()
        .is_zero());
        assert!(parity_row_residual(&case.layout, &e.public, &e.response)
            .unwrap()
            .is_zero());
        assert!(
            !combined_terminal_matches(&case, &e.proof),
            "alphabet must fail combined terminal"
        );
        assert_eq!(case.verify(&e.proof), Err(AkitaError::InvalidProof));
    }
}

#[test]
fn individual_rows_are_mandatory_even_with_best_recomputed_quotients() {
    for base in BASES {
        for attack in [Attack::ParityOnly, Attack::AOnly] {
            let mut case = Case::new(base, 1);
            let e = assemble(&mut case, attack);
            let a = a_row_residual(
                &case.layout,
                &e.public,
                case.admitted.setup(),
                &case.commitment,
                &e.response,
                0,
            )
            .unwrap();
            let parity = parity_row_residual(&case.layout, &e.public, &e.response).unwrap();
            assert_eq!(a.is_zero(), attack == Attack::AOnly);
            assert_eq!(parity.is_zero(), attack == Attack::ParityOnly);
            assert_eq!(
                check_lowered_clear(&case.layout, &e.public, &e.digits, &case.image),
                Err(AkitaError::InvalidProof)
            );
            assert!(
                !combined_terminal_matches(&case, &e.proof),
                "row failure must reach combined terminal"
            );
            assert_eq!(case.verify(&e.proof), Err(AkitaError::InvalidProof));
        }
    }
}

#[test]
fn wrong_image_sum_compensated_in_s_fails_the_combined_terminal() {
    for base in BASES {
        let mut case = Case::new(base, 1);
        let e = assemble(&mut case, Attack::WrongY);
        // The tables satisfy the honest lowered relation; only y_Y and hence
        // the compensating s are false, fixed before tau and beta.
        check_lowered_clear(&case.layout, &e.public, &e.digits, &case.image).unwrap();
        assert!(!combined_terminal_matches(&case, &e.proof));
        assert_eq!(case.verify(&e.proof), Err(AkitaError::InvalidProof));
    }
}
