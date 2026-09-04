use std::sync::Arc;

use super::{
    payload_primary_strictly_dominates, payload_projection_dominates,
    projection_bound_is_dominated, setup_primary_strictly_dominates, setup_projection_dominates,
    DescriptorOrderContext, ParentAdmissionClass, PayloadScore, Projection, ProjectionOrder,
    SetupScore,
};
use crate::schedule_params::{
    CandidateMetrics, CompleteObjectiveBound, PackedProofCost, SetupPrefixCapacity,
};

const SETUP_FIRST: crate::SelectionPolicyId =
    crate::SelectionPolicyId::MinFirstDirectSetupThenPayloadV2;
const PADDED_ENVELOPE_FIRST: crate::SelectionPolicyId =
    crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3;

fn context(fold_count: usize, first_fold: u8) -> DescriptorOrderContext {
    DescriptorOrderContext {
        fold_count,
        first_fold_descriptor: (fold_count != 0).then(|| Arc::from([first_fold])),
    }
}

fn admission(fold_depth: u8, natural_len: usize) -> ParentAdmissionClass {
    ParentAdmissionClass {
        fold_depth,
        first_direct_setup_capacity: SetupPrefixCapacity::for_natural_len(natural_len),
    }
}

fn order<'a, Score>(
    score: Score,
    descriptor: &'a [u8],
    context: &'a DescriptorOrderContext,
    admission: ParentAdmissionClass,
) -> ProjectionOrder<'a, Score> {
    ProjectionOrder {
        score,
        descriptor,
        context,
        admission,
    }
}

fn setup_score(
    capacity: SetupPrefixCapacity,
    payload_bytes: usize,
    nonce_bits: usize,
    setup_field_elements: usize,
) -> SetupScore {
    SetupScore {
        first_direct_setup_capacity: capacity,
        first_direct_output_witness_len: 0,
        cost: PackedProofCost::new(payload_bytes, nonce_bits).unwrap(),
        setup_field_elements,
    }
}

fn payload_score(
    payload_bytes: usize,
    nonce_bits: usize,
    setup_field_elements: usize,
) -> PayloadScore {
    PayloadScore {
        cost: PackedProofCost::new(payload_bytes, nonce_bits).unwrap(),
        setup_field_elements,
    }
}

#[test]
fn setup_projection_keeps_setup_descriptor_tradeoffs_that_a_parent_can_mask() {
    let smaller_setup = setup_score(SetupPrefixCapacity::for_natural_len(8), 100, 0, 64);
    let smaller_descriptor = setup_score(SetupPrefixCapacity::for_natural_len(8), 100, 0, 128);
    assert!(!setup_projection_dominates(
        SETUP_FIRST,
        order(smaller_setup, &[2], &context(2, 7), admission(2, 8)),
        order(smaller_descriptor, &[1], &context(2, 7), admission(2, 8),),
    ));
    assert!(!setup_projection_dominates(
        SETUP_FIRST,
        order(smaller_descriptor, &[1], &context(2, 7), admission(2, 8),),
        order(smaller_setup, &[2], &context(2, 7), admission(2, 8)),
    ));

    assert!(setup_projection_dominates(
        SETUP_FIRST,
        order(
            setup_score(SetupPrefixCapacity::for_natural_len(4), 100, 0, 256),
            &[9],
            &context(2, 8),
            admission(2, 4),
        ),
        order(smaller_setup, &[2], &context(3, 7), admission(2, 8)),
    ));
    assert!(setup_projection_dominates(
        SETUP_FIRST,
        order(
            setup_score(SetupPrefixCapacity::for_natural_len(8), 99, 0, 256),
            &[9],
            &context(3, 8),
            admission(2, 8),
        ),
        order(smaller_setup, &[2], &context(2, 7), admission(2, 8)),
    ));
}

#[test]
fn payload_projection_keeps_setup_descriptor_tradeoffs_that_a_parent_can_mask() {
    assert!(!payload_projection_dominates(
        SETUP_FIRST,
        order(
            payload_score(100, 0, 64),
            &[2],
            &context(2, 7),
            admission(2, 8)
        ),
        order(
            payload_score(100, 0, 128),
            &[1],
            &context(2, 7),
            admission(2, 8)
        ),
    ));
    assert!(!payload_projection_dominates(
        SETUP_FIRST,
        order(
            payload_score(100, 0, 128),
            &[1],
            &context(2, 7),
            admission(2, 8)
        ),
        order(
            payload_score(100, 0, 64),
            &[2],
            &context(2, 7),
            admission(2, 8)
        ),
    ));
    assert!(payload_projection_dominates(
        SETUP_FIRST,
        order(
            payload_score(99, 0, 256),
            &[9],
            &context(3, 8),
            admission(2, 4)
        ),
        order(
            payload_score(100, 0, 64),
            &[1],
            &context(2, 7),
            admission(2, 8)
        ),
    ));
    assert!(payload_projection_dominates(
        SETUP_FIRST,
        order(
            payload_score(100, 0, 64),
            &[1],
            &context(2, 7),
            admission(2, 8)
        ),
        order(
            payload_score(100, 0, 128),
            &[2],
            &context(2, 7),
            admission(2, 8)
        ),
    ));
}

#[test]
fn envelope_first_projection_preserves_maskable_setup_tradeoffs() {
    let context = context(2, 7);
    let compatible = admission(2, 8);

    assert!(!payload_projection_dominates(
        PADDED_ENVELOPE_FIRST,
        order(payload_score(99, 0, 128), &[1], &context, compatible),
        order(payload_score(100, 0, 64), &[2], &context, compatible),
    ));
    assert!(payload_projection_dominates(
        PADDED_ENVELOPE_FIRST,
        order(payload_score(99, 0, 64), &[2], &context, compatible),
        order(payload_score(100, 0, 128), &[1], &context, compatible),
    ));

    assert!(!setup_projection_dominates(
        PADDED_ENVELOPE_FIRST,
        order(
            setup_score(SetupPrefixCapacity::for_natural_len(16), 99, 0, 64),
            &[1],
            &context,
            admission(2, 16),
        ),
        order(
            setup_score(SetupPrefixCapacity::for_natural_len(8), 100, 0, 128),
            &[2],
            &context,
            compatible,
        ),
    ));
    assert!(setup_projection_dominates(
        PADDED_ENVELOPE_FIRST,
        order(
            setup_score(SetupPrefixCapacity::for_natural_len(4), 101, 0, 64),
            &[2],
            &context,
            admission(2, 4),
        ),
        order(
            setup_score(SetupPrefixCapacity::for_natural_len(8), 100, 0, 128),
            &[1],
            &context,
            compatible,
        ),
    ));
}

#[test]
fn payload_projection_prices_every_nonce_alignment() {
    let admission = admission(2, 8);
    let context = context(2, 7);
    let smaller_payload = order(payload_score(100, 8, 64), &[1], &context, admission);
    let smaller_nonce = order(payload_score(101, 0, 64), &[2], &context, admission);

    assert!(payload_projection_dominates(
        SETUP_FIRST,
        smaller_payload,
        smaller_nonce
    ));
    assert!(!payload_projection_dominates(
        SETUP_FIRST,
        smaller_nonce,
        smaller_payload,
    ));
}

#[test]
fn projection_dominance_preserves_parent_admission_and_descriptor_order() {
    let score = payload_score(100, 0, 64);
    let two_fold = admission(2, 8);

    assert!(!payload_projection_dominates(
        SETUP_FIRST,
        order(
            payload_score(99, 0, 32),
            &[1],
            &context(1, 7),
            admission(1, 8)
        ),
        order(score, &[2], &context(2, 7), two_fold),
    ));
    assert!(!payload_projection_dominates(
        SETUP_FIRST,
        order(
            payload_score(99, 0, 32),
            &[1],
            &context(2, 7),
            admission(2, 16)
        ),
        order(score, &[2], &context(2, 7), two_fold),
    ));
    assert!(!payload_projection_dominates(
        SETUP_FIRST,
        order(score, &[1], &context(2, 8), two_fold),
        order(score, &[2], &context(2, 7), two_fold),
    ));
    assert!(!payload_projection_dominates(
        SETUP_FIRST,
        order(score, &[1], &context(3, 7), two_fold),
        order(score, &[2], &context(2, 7), two_fold),
    ));

    assert!(!admission(0, 8).is_admitted_by(true, false, 16));
    assert!(admission(1, 8).is_admitted_by(true, false, 16));
    assert!(!admission(1, 8).is_admitted_by(false, true, 16));
    assert!(admission(2, 8).is_admitted_by(false, true, 16));
    assert!(!admission(2, 16).is_admitted_by(false, true, 16));
}

#[test]
fn strict_primary_dominance_does_not_consider_maskable_setup_or_ties() {
    let capacity = SetupPrefixCapacity::for_natural_len(8);
    let compatible = admission(2, 8);
    assert!(setup_primary_strictly_dominates(
        SETUP_FIRST,
        setup_score(capacity, 99, 0, 256),
        compatible,
        setup_score(capacity, 100, 0, 64),
        compatible,
    ));
    assert!(!setup_primary_strictly_dominates(
        SETUP_FIRST,
        setup_score(capacity, 100, 0, 64),
        compatible,
        setup_score(capacity, 100, 0, 128),
        compatible,
    ));
    assert!(payload_primary_strictly_dominates(
        SETUP_FIRST,
        payload_score(99, 0, 256),
        compatible,
        payload_score(100, 0, 64),
        compatible,
    ));
    assert!(!payload_primary_strictly_dominates(
        SETUP_FIRST,
        payload_score(100, 0, 64),
        compatible,
        payload_score(100, 0, 128),
        compatible,
    ));
    assert!(!payload_primary_strictly_dominates(
        SETUP_FIRST,
        payload_score(99, 0, 32),
        admission(1, 8),
        payload_score(100, 0, 64),
        compatible,
    ));
}

fn metrics(natural_len: usize, proof_bytes: usize) -> CandidateMetrics {
    CandidateMetrics {
        first_direct_setup_capacity: SetupPrefixCapacity::for_natural_len(natural_len),
        first_direct_output_witness_len: 0,
        cost: PackedProofCost::new(proof_bytes, 0).unwrap(),
        setup_field_elements: 0,
    }
}

#[test]
fn recursive_bound_requires_dominance_in_both_parent_projections() {
    let candidate_admission = admission(2, 16);
    let lower_bound = CompleteObjectiveBound::SetupFirst {
        first_direct_setup_capacity: 16,
        proof_bytes: 10,
        setup_field_elements: 0,
    };
    let setup_winner = (admission(2, 8), metrics(8, 100));

    assert!(!projection_bound_is_dominated(
        Projection::Payload,
        candidate_admission,
        lower_bound,
        [setup_winner],
    ));

    assert!(projection_bound_is_dominated(
        Projection::FirstDirectSetup,
        candidate_admission,
        lower_bound,
        [setup_winner],
    ));
    assert!(projection_bound_is_dominated(
        Projection::Payload,
        candidate_admission,
        lower_bound,
        [(admission(2, 8), metrics(8, 9))],
    ));
    assert!(!projection_bound_is_dominated(
        Projection::Payload,
        candidate_admission,
        lower_bound,
        [(admission(2, 8), metrics(8, 10))],
    ));
}
