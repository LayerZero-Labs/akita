#![cfg(feature = "labinius")]

mod common;

use akita_algebra::binary::BinaryField128;
use akita_error::AkitaError;
use akita_labinius_pcs::{ImageEvaluation, ImageProver, ImageVerifier, F};
use akita_params::PolynomialGroupLayout;
use akita_types::{Commitment, CommittedGroup, RingVec};
use common::{admitted, fixture, Case, Recording};
use std::panic::{catch_unwind, AssertUnwindSafe};

#[test]
fn invalid_statement_returns_before_any_proof_byte_is_read() {
    let case = Case::<BinaryField128>::new(0);
    let mut wrong_profile = case.output.committed_group.clone();
    wrong_profile.profile.group = PolynomialGroupLayout::singleton(case.layout.image_log_len() + 1);
    let malformed = CommittedGroup::new(
        *case.output.committed_group.profile(),
        Commitment::new(RingVec::from_coeffs(Vec::<F>::new())),
    );
    let larger = admitted(1, 0x31);
    for (root, commitment, evaluation, expected) in [
        (
            &case.root,
            &case.output.committed_group,
            ImageEvaluation {
                point: &case.point[..case.point.len() - 1],
                value: case.value,
            },
            AkitaError::InvalidPointDimension {
                expected: case.point.len(),
                actual: case.point.len() - 1,
            },
        ),
        (
            &larger,
            &case.output.committed_group,
            case.evaluation(),
            AkitaError::InvalidPointDimension {
                expected: case.point.len() + 1,
                actual: case.point.len(),
            },
        ),
        (
            &case.root,
            &wrong_profile,
            case.evaluation(),
            AkitaError::UnsupportedSchedule(String::new()),
        ),
        (
            &case.root,
            &malformed,
            case.evaluation(),
            AkitaError::InvalidProof,
        ),
    ] {
        let state = akita_transcript::new_verifier_channel(b"validation/v1", b"", &[]).unwrap();
        let mut record = Recording::new(state);
        let result = catch_unwind(AssertUnwindSafe(|| {
            case.verifier.verify_on_channel::<BinaryField128, _>(
                root,
                commitment,
                evaluation,
                &mut record,
            )
        }));
        assert!(result.is_ok());
        let error = result.unwrap().unwrap_err();
        assert_eq!(
            std::mem::discriminant(&error),
            std::mem::discriminant(&expected)
        );
        if let AkitaError::InvalidPointDimension { .. } = expected {
            assert_eq!(error, expected);
        }
        assert!(record.public.is_empty());
        assert!(record.messages.is_empty());
        assert_eq!(record.message_calls, 0);
        assert!(record.draws.is_empty());
    }
    // A trusted catalog containing only log 11 cannot resolve the log 10 group.
    let larger = fixture(11);
    let missing = ImageVerifier::new(
        larger.scheme.schedules().clone(),
        larger.verifier_setup.clone(),
    )
    .unwrap();
    let state = akita_transcript::new_verifier_channel(b"validation/v1", b"", &[]).unwrap();
    let mut record = Recording::new(state);
    assert!(matches!(
        missing.verify_on_channel::<BinaryField128, _>(
            &case.root,
            &case.output.committed_group,
            case.evaluation(),
            &mut record
        ),
        Err(AkitaError::UnsupportedSchedule(_))
    ));
    assert!(record.public.is_empty());
    assert!(record.messages.is_empty());
    assert_eq!(record.message_calls, 0);
    assert!(record.draws.is_empty());
}

#[test]
fn foreign_handle_and_mismatched_prepared_matrix_are_rejected() {
    let case = Case::<BinaryField128>::new(0);
    let f = fixture(case.layout.image_log_len());
    let foreign = ImageProver::new(f.scheme.schedules().clone(), f.prover_setup.clone()).unwrap();
    let state = akita_transcript::new_prover_channel(b"validation/v1", b"").unwrap();
    let mut record = Recording::new(state);
    assert!(matches!(
        foreign.open_on_channel::<BinaryField128, _>(
            &case.root,
            &case.output,
            case.evaluation(),
            &mut record
        ),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(record.public.is_empty());
    assert!(record.messages.is_empty());
    assert_eq!(record.message_calls, 0);
    assert!(record.draws.is_empty());
    let changed_root = admitted(0, 0x72);
    let changed_prepared =
        akita_labinius_pcs::PreparedMatrix::prepare(changed_root.setup()).unwrap();
    assert!(matches!(
        case.prover
            .commit::<BinaryField128>(&case.root, &changed_prepared, &case.source),
        Err(AkitaError::InvalidSetup(_))
    ));
    let prepared = akita_labinius_pcs::PreparedMatrix::prepare(case.root.setup()).unwrap();
    for len in [0, case.source.len() - 1, case.source.len() + 1] {
        let source = vec![0u128; len];
        assert!(matches!(
            case.prover
                .commit::<BinaryField128>(&case.root, &prepared, &source),
            Err(AkitaError::InvalidSize { .. })
        ));
    }
    let state = akita_transcript::new_prover_channel(b"validation/v1", b"").unwrap();
    let mut record = Recording::new(state);
    assert!(matches!(
        case.prover.open_on_channel::<BinaryField128, _>(
            &case.root,
            &case.output,
            ImageEvaluation {
                point: &[],
                value: case.value
            },
            &mut record
        ),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(record.public.is_empty());
    assert!(record.messages.is_empty());
    assert_eq!(record.message_calls, 0);
    assert!(record.draws.is_empty());
}

#[test]
fn byte_errors_are_invalid_proof_when_the_native_verifier_supports_it() {
    let case = Case::<BinaryField128>::new(0);
    assert_eq!(case.verify(&[], b""), Err(AkitaError::InvalidProof));
    let (proof, _, _) = case.prove(b"");
    for length in [0u64, u64::MAX] {
        let mut malformed = proof.clone();
        malformed[..8].copy_from_slice(&length.to_le_bytes());
        assert_eq!(case.verify(&malformed, b""), Err(AkitaError::InvalidProof));
    }
    let mut oversized = proof.clone();
    oversized.push(0);
    assert_eq!(case.verify(&oversized, b""), Err(AkitaError::InvalidProof));
    // Other payload failures must reject without reporting an internal error;
    // the native verifier may classify its arithmetic/decode boundary errors.
    let mut malformed = proof;
    malformed[8..].fill(0xff);
    let result = case.verify(&malformed, b"");
    assert!(result.is_err());
    assert!(!matches!(result, Err(AkitaError::Internal(_))));
}
