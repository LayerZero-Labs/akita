#![cfg(feature = "labinius")]

mod common;
#[path = "root/support.rs"]
mod root_support;

use akita_algebra::binary::BinaryField128;
use akita_error::AkitaError;
use akita_labinius_pcs::{
    config::Digits2, ImageEvaluation, ImageProver, ImageVerifier, PreparedMatrix, RootPcsProver,
    RootPcsVerifier, RootSetup,
};
use akita_params::sis::labinius::LabiniusRootProfile;
use akita_types::proof::AkitaSetupSeed;

fn small() -> RootSetup {
    RootSetup::derive(
        LabiniusRootProfile::D648P128Q28BoundedW46Delta16,
        4,
        0,
        128,
        AkitaSetupSeed::shake256_paged_v1([0x31; 32]),
    )
    .unwrap()
}
fn unsupported<T>(result: Result<T, AkitaError>) {
    match result {
        Err(AkitaError::InvalidSetup(message)) => assert_eq!(
            message,
            "small-modulus LaBinius profile is not supported by the image or root PCS"
        ),
        _ => panic!("expected explicit unsupported-profile InvalidSetup"),
    }
}
#[test]
fn root_setup_entry_points_reject_small_modulus() {
    let fixture = <Digits2 as root_support::TestDigits>::fixture(0);
    unsupported(RootPcsProver::<Digits2>::new(
        small(),
        fixture.images.clone(),
        fixture.digits.clone(),
        fixture.prover_setup.clone(),
    ));
    unsupported(RootPcsVerifier::<Digits2>::new(
        small(),
        fixture.images.clone(),
        fixture.digits.clone(),
        fixture.verifier_setup.clone(),
    ));
}
#[test]
fn image_profile_entry_points_reject_before_channel_activity() {
    let case = common::Case::<BinaryField128>::new(0);
    let fixture = common::fixture(10);
    let prover = ImageProver::new(
        fixture.scheme.schedules().clone(),
        fixture.prover_setup.clone(),
    )
    .unwrap();
    let verifier = ImageVerifier::new(
        fixture.scheme.schedules().clone(),
        fixture.verifier_setup.clone(),
    )
    .unwrap();
    let root = small();
    let prepared = PreparedMatrix::prepare(root.setup()).unwrap();
    unsupported(prover.commit::<BinaryField128>(&root, &prepared, &case.source));
    let evaluation = ImageEvaluation {
        point: &[],
        value: Default::default(),
    };
    let state = akita_transcript::new_prover_channel(b"guard-test", b"").unwrap();
    let mut record = common::Recording::new(state);
    unsupported(prover.open_on_channel::<BinaryField128, _>(
        &root,
        &case.output,
        evaluation,
        &mut record,
    ));
    assert!(record.public.is_empty());
    assert_eq!(record.message_calls, 0);
    assert!(record.draws.is_empty());
    let state = akita_transcript::new_verifier_channel(b"guard-test", b"", &[]).unwrap();
    let mut record = common::Recording::new(state);
    unsupported(verifier.verify_on_channel::<BinaryField128, _>(
        &root,
        &case.output.committed_group,
        evaluation,
        &mut record,
    ));
    assert!(record.public.is_empty());
    assert_eq!(record.message_calls, 0);
    assert!(record.draws.is_empty());
}
