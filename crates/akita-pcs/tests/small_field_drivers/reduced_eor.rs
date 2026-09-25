use std::panic::{catch_unwind, AssertUnwindSafe};

use akita_config::proof_optimized::fp32;
use akita_error::AkitaError;
use akita_types::{
    BasisMode, GroupBatchStatement, OpeningClaims, OpeningMethod, PolynomialGroupClaims,
    RingRelationMode,
};

use super::small_field_drivers::SingleGroupRoundtrip;

type Config = fp32::Dense;
fn verify(
    roundtrip: &SingleGroupRoundtrip<Config>,
    proof: &[u8],
    label: &[u8],
) -> Result<(), AkitaError> {
    let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        roundtrip.point.clone(),
        vec![roundtrip.expected],
        &roundtrip.commitment,
    )?])?;
    roundtrip
        .scheme
        .verifier(roundtrip.verifier_setup.clone())
        .and_then(|verifier| {
            verifier.batched_verify(
                proof,
                label,
                GroupBatchStatement::new(roundtrip.selection, claims)?,
                BasisMode::Lagrange,
            )
        })
}

pub(super) fn assert_fp32_dense(
    roundtrip: &SingleGroupRoundtrip<Config>,
    label: &[u8],
    what: &str,
) {
    let row = roundtrip
        .scheme
        .schedules()
        .resolve_selection(roundtrip.selection)
        .expect("the selected fp32 dense schedule must resolve");
    let (_, _) = row
        .schedule()
        .recursive_folds
        .iter()
        .enumerate()
        .find(|(_, step)| {
            step.params.ring_relation_mode == RingRelationMode::ReducedEvaluation
                && step.params.opening_method() == OpeningMethod::EvaluationTrace
        })
        .expect("fp32 dense nv=20 must contain a reduced EvaluationTrace fold");
    let truncated = &roundtrip.proof[..roundtrip.proof.len() - 1];
    let outcome = catch_unwind(AssertUnwindSafe(|| verify(roundtrip, truncated, label)));
    assert!(
        matches!(outcome, Ok(Err(_))),
        "{what}: truncating a stream containing reduced-fold EOR must reject without panicking"
    );

    for offset in [roundtrip.proof.len() / 3, roundtrip.proof.len() * 2 / 3] {
        let mut tampered = roundtrip.proof.clone();
        tampered[offset] ^= 1;
        let outcome = catch_unwind(AssertUnwindSafe(|| verify(roundtrip, &tampered, label)));
        assert!(
            matches!(outcome, Ok(Err(_))),
            "{what}: tampered reduced-EOR stream must reject without panicking"
        );
    }
}
