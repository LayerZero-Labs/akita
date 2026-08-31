use std::panic::{catch_unwind, AssertUnwindSafe};

use akita_config::{proof_optimized::fp32, CommitmentConfig};
use akita_error::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaBatchedProof, BasisMode, GroupBatchStatement, OpeningClaims, OpeningMethod,
    PolynomialGroupClaims, RingRelationMode,
};
use jolt_field::One;

use super::small_field_drivers::SingleGroupRoundtrip;

type Config = fp32::Dense;
type Proof = AkitaBatchedProof<fp32::Field, fp32::ExtensionField>;

fn verify(
    roundtrip: &SingleGroupRoundtrip<Config>,
    proof: &Proof,
    label: &[u8],
) -> Result<(), AkitaError> {
    let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        roundtrip.point.clone(),
        vec![roundtrip.expected],
        &roundtrip.commitment,
    )?])?;
    let mut transcript = AkitaTranscript::<fp32::Field>::new(label);
    AkitaCommitmentScheme::<Config>::batched_verify(
        proof,
        &roundtrip.verifier_setup,
        &mut transcript,
        GroupBatchStatement::new(roundtrip.selection, claims)?,
        BasisMode::Lagrange,
    )
}

pub(super) fn assert_fp32_dense(
    roundtrip: &SingleGroupRoundtrip<Config>,
    label: &[u8],
    what: &str,
) {
    let row = Config::resolve_schedule_selection(roundtrip.selection)
        .expect("the selected fp32 dense schedule must resolve");
    let (reduced_index, _) = row
        .schedule()
        .recursive_folds
        .iter()
        .enumerate()
        .find(|(_, step)| {
            step.params.ring_relation_mode == RingRelationMode::ReducedEvaluation
                && step.params.opening_method() == OpeningMethod::EvaluationTrace
        })
        .expect("fp32 dense nv=20 must contain a reduced EvaluationTrace fold");
    let reduced_proof = roundtrip
        .proof
        .recursive_folds
        .get(reduced_index)
        .expect("proof must match the selected recursive schedule");
    assert!(
        reduced_proof.extension_opening_reduction.is_some(),
        "the first reduced EvaluationTrace fold must carry EOR"
    );

    let mut missing = roundtrip.proof.clone();
    missing.recursive_folds[reduced_index].extension_opening_reduction = None;
    let outcome = catch_unwind(AssertUnwindSafe(|| verify(roundtrip, &missing, label)));
    assert!(
        matches!(outcome, Ok(Err(_))),
        "{what}: omitting reduced-fold EOR must reject without panicking"
    );

    let mut tampered = roundtrip.proof.clone();
    *tampered.recursive_folds[reduced_index]
        .extension_opening_reduction
        .as_mut()
        .expect("reduced-fold EOR")
        .partials
        .first_mut()
        .expect("reduced-fold EOR partial") += fp32::ExtensionField::one();
    let outcome = catch_unwind(AssertUnwindSafe(|| verify(roundtrip, &tampered, label)));
    assert!(
        matches!(outcome, Ok(Err(_))),
        "{what}: tampered reduced-fold EOR must reject without panicking"
    );
}
