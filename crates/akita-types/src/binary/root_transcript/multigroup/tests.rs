use super::*;
use crate::{GrindingReplay, ProverGrinding, VerifierGrinding};
use akita_params::{binary_setup::BinaryFieldProfile, GrindingPlan, GrindingRun};
use akita_transcript::{
    new_prover_channel, new_verifier_channel, public_bytes, ProtocolSiteId, ProverChannel,
};

fn plan(groups: u32, owners: usize, columns: usize) -> BinaryColumnPlan {
    BinaryColumnPlan::new(
        (0..groups)
            .map(|group| BinaryColumnGroupPlan::new(group, owners, columns).unwrap())
            .collect(),
    )
    .unwrap()
}

fn freeze(state: &mut impl ProofChannel, plan: &BinaryColumnPlan) -> BinaryColumnPrefix {
    public_bytes(
        state,
        ProtocolSiteId::default(),
        b"column fixture all D fixed",
    )
    .unwrap();
    public_bytes(
        state,
        ProtocolSiteId::default(),
        &plan.canonical_bytes().unwrap(),
    )
    .unwrap();
    BinaryColumnPrefix::capture(state).unwrap()
}

fn accepted(
    plan: &BinaryColumnPlan,
    retry_middle: bool,
) -> (
    BinaryColumnPrefix,
    Vec<BinaryColumnResponse>,
    Vec<u8>,
    [u8; 32],
) {
    let mut state = new_prover_channel(b"column seam fixture");
    let prefix = freeze(&mut state, plan);
    let responses = plan
        .groups()
        .iter()
        .map(|group| {
            prefix
                .search_group(group, |nonce, _| {
                    Ok(!retry_middle || group.group() != 1 || nonce == 1)
                })
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut grinding = ProverGrinding::new(&mut state, plan.grinding_plan());
    grinding
        .commit_binary_column_responses(&prefix, plan, &responses)
        .unwrap();
    grinding.finish().unwrap();
    let continuation = crate::binary::challenge_block(&mut state).unwrap();
    (
        prefix,
        responses,
        state.narg_string().to_vec(),
        continuation,
    )
}

#[test]
fn group_retry_preserves_other_streams_and_changes_live_continuation() {
    let plan = plan(3, 1, 2);
    let (prefix, baseline, bytes0, continuation0) = accepted(&plan, false);
    let (retry_prefix, retried, bytes1, continuation1) = accepted(&plan, true);
    assert_eq!(prefix, retry_prefix);
    for group in [0, 2] {
        assert_eq!(baseline[group].supports(), retried[group].supports());
    }
    assert_eq!(baseline[1].response_nonce(), 0);
    assert_eq!(retried[1].response_nonce(), 1);
    assert_ne!(baseline[1].supports(), retried[1].supports());
    assert_ne!(continuation0, continuation1);
    // A zero-bit work site contributes no nonce body: exactly one response
    // byte for each of these three accepted groups.
    assert_eq!(bytes0, vec![0, 0, 0]);
    assert_eq!(bytes1, vec![0, 1, 0]);
    assert!(retried
        .iter()
        .all(|response| response.work_nonce().is_none()));

    let mut state = new_verifier_channel(b"column seam fixture", &bytes1);
    let verifier_prefix = freeze(&mut state, &plan);
    let mut grinding = VerifierGrinding::new(&mut state, plan.grinding_plan());
    let mut visited = Vec::new();
    let replayed = grinding
        .read_binary_column_responses(&verifier_prefix, &plan, |group, nonce, supports| {
            visited.push(group);
            Ok(nonce == u32::from(group == 1) && supports == retried[group as usize].supports())
        })
        .unwrap();
    assert_eq!(visited, vec![0, 1, 2]);
    for (expected, actual) in retried.iter().zip(&replayed) {
        assert_eq!(expected.supports(), actual.supports());
    }
    assert_eq!(
        crate::binary::challenge_block(grinding.state_mut()).unwrap(),
        continuation1
    );
    grinding.finish().unwrap().check_eof(state).unwrap();
}

fn rejects_and_poisons(plan: &BinaryColumnPlan, proof: &[u8]) {
    let mut state = new_verifier_channel(b"column seam fixture", proof);
    let prefix = freeze(&mut state, plan);
    let mut grinding = VerifierGrinding::new(&mut state, plan.grinding_plan());
    assert!(matches!(
        grinding.read_binary_column_responses(&prefix, plan, |_, nonce, _| Ok(nonce == 0)),
        Err(AkitaError::InvalidProof)
    ));
    assert!(matches!(grinding.finish(), Err(AkitaError::InvalidProof)));
    assert!(state.check_eof().is_err());
}

#[test]
fn replay_rejects_response_mutation_noncanonical_truncated_and_trailing_streams() {
    let plan = plan(1, 1, 2);
    let (_, _, proof, _) = accepted(&plan, false);
    assert_eq!(proof, vec![0]);
    // Nonce 1 is canonically encoded and in range, but the response callback
    // rejects it. A nonce parser alone cannot enforce response admission.
    rejects_and_poisons(&plan, &[1]);
    rejects_and_poisons(&plan, &[0x80, 0]);
    rejects_and_poisons(&plan, &[0x80]);
    rejects_and_poisons(&plan, &[]);
    rejects_and_poisons(&plan, &[0x80, 0x20]); // 4096 is outside the response domain.

    let mut state = new_verifier_channel(b"column seam fixture", &[0, 0]);
    let prefix = freeze(&mut state, &plan);
    let mut grinding = VerifierGrinding::new(&mut state, plan.grinding_plan());
    grinding
        .read_binary_column_responses(&prefix, &plan, |_, _, _| Ok(true))
        .unwrap();
    assert!(matches!(
        grinding.finish().unwrap().check_eof(state),
        Err(AkitaError::InvalidProof)
    ));
}

#[test]
fn positive_work_nonce_mutation_is_rejected_before_admission() {
    let plan = plan(1, 257, 1);
    let (prefix, responses, proof, _) = accepted(&plan, false);
    assert_eq!(plan.groups()[0].work_run().grind_bits(), 1);
    let accepted_work = responses[0].work_nonce().unwrap();
    let bad_work = (0..256)
        .find(|&nonce| {
            nonce != accepted_work
                && prefix
                    .0
                    .work_binding(MULTIGROUP_COLUMN_DOMAIN, 0, 0, 0, 1, Some(nonce))
                    .is_err()
        })
        .unwrap();
    let mut tampered = vec![0];
    tampered.extend_from_slice(NonceAtom::new(bad_work).wire().as_ref());
    let mut state = new_verifier_channel(b"column seam fixture", &tampered);
    let verifier_prefix = freeze(&mut state, &plan);
    let mut grinding = VerifierGrinding::new(&mut state, plan.grinding_plan());
    let mut admitted = false;
    assert!(matches!(
        grinding.read_binary_column_responses(&verifier_prefix, &plan, |_, _, _| {
            admitted = true;
            Ok(true)
        }),
        Err(AkitaError::InvalidProof)
    ));
    assert!(!admitted);
    assert!(matches!(grinding.finish(), Err(AkitaError::InvalidProof)));
    assert!(state.check_eof().is_err());
    rejects_and_poisons(&plan, &proof[..proof.len() - 1]);
    let mut noncanonical_work = proof.clone();
    *noncanonical_work.last_mut().unwrap() |= 0x80;
    noncanonical_work.push(0);
    rejects_and_poisons(&plan, &noncanonical_work);
}

#[test]
fn owner_cap_work_uses_at_most_three_canonical_bytes_without_large_witnesses() {
    let cap = BinaryColumnGroupPlan::new(0, 4096, 256).unwrap();
    assert_eq!(cap.work_run().grind_bits(), 12);
    let mut state = new_prover_channel(b"column work cap fixture");
    let prefix = BinaryColumnPrefix::capture(&mut state).unwrap();
    let (nonce, binding) = prefix
        .0
        .search_work(MULTIGROUP_COLUMN_DOMAIN, 0, 0, 4095, 12)
        .unwrap();
    let nonce = nonce.unwrap();
    assert!(NonceAtom::new(nonce).wire().as_ref().len() <= 3);
    assert_eq!(
        prefix
            .0
            .work_binding(MULTIGROUP_COLUMN_DOMAIN, 0, 0, 4095, 12, Some(nonce))
            .unwrap(),
        binding
    );
    assert!(matches!(
        prefix
            .0
            .work_binding(MULTIGROUP_COLUMN_DOMAIN, 0, 0, 0, 0, Some(0)),
        Err(AkitaError::InvalidProof)
    ));
}

#[test]
fn response_search_exhausts_exactly_4096_candidates_with_typed_error() {
    let plan = BinaryColumnGroupPlan::new(0, 1, 1).unwrap();
    let mut state = new_prover_channel(b"column exhaustion fixture");
    let prefix = BinaryColumnPrefix::capture(&mut state).unwrap();
    let mut probes = 0;
    assert!(matches!(
        prefix.search_group(&plan, |nonce, supports| {
            assert_eq!(nonce, probes);
            assert_eq!(supports.len(), 1);
            probes += 1;
            Ok(false)
        }),
        Err(AkitaError::InvalidInput(_))
    ));
    assert_eq!(probes, 4096);
    assert!(matches!(
        prefix.replay_group(&plan, 4096, None, |_, _| Ok(true)),
        Err(AkitaError::InvalidProof)
    ));
}

#[test]
fn replay_cursor_and_response_prefix_mismatches_are_typed_and_poisoned() {
    let plan = plan(1, 1, 2);
    let (prefix, responses, proof, _) = accepted(&plan, false);
    let wrong_cursor = GrindingPlan::new(
        vec![GrindingRun::fold_response(0)],
        BinaryFieldProfile::P64.challenge_order().unwrap(),
    )
    .unwrap();
    let mut state = new_prover_channel(b"column seam fixture");
    freeze(&mut state, &plan);
    let mut grinding = ProverGrinding::new(&mut state, &wrong_cursor);
    assert!(matches!(
        grinding.commit_binary_column_responses(&prefix, &plan, &responses),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(grinding.finish().is_err());

    let mut state = new_verifier_channel(b"column seam fixture", &proof);
    let verifier_prefix = freeze(&mut state, &plan);
    let mut grinding = VerifierGrinding::new(&mut state, &wrong_cursor);
    assert!(matches!(
        grinding.read_binary_column_responses(&verifier_prefix, &plan, |_, _, _| Ok(true)),
        Err(AkitaError::InvalidInput(_)) | Err(AkitaError::InvalidProof)
    ));
    assert!(matches!(grinding.finish(), Err(AkitaError::InvalidProof)));
    assert!(state.check_eof().is_err());

    let mut state: ProverChannel = new_prover_channel(b"different prefix fixture");
    let other = freeze(&mut state, &plan);
    let mut grinding = ProverGrinding::new(&mut state, plan.grinding_plan());
    assert!(matches!(
        grinding.commit_binary_column_responses(&other, &plan, &responses),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(grinding.finish().is_err());
}
