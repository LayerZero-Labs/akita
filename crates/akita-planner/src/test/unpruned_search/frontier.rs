use super::*;

#[derive(PartialEq, Eq)]
enum SuccessorKey {
    Recursive {
        descriptor: Vec<u8>,
        output_witness_len: usize,
        fold_count: usize,
        first_direct_setup_field_len: Option<std::num::NonZeroUsize>,
    },
    Terminal {
        descriptor: Vec<u8>,
        first_direct_setup_field_len: Option<std::num::NonZeroUsize>,
    },
}

struct FrontierCandidate {
    schedule: ScheduleCandidate,
    descriptor: Vec<u8>,
}

struct SuccessorBucket {
    key: SuccessorKey,
    candidates: Vec<FrontierCandidate>,
}

/// Oracle suffixes partitioned by the complete successor identity visible to
/// a parent edge.
///
/// Quotient-free cutovers create many descriptor-distinct successors. Keeping
/// those partitions explicit avoids comparing every new suffix with candidates
/// that cannot dominate it, while retaining the oracle's exact dominance rule.
#[derive(Default)]
pub(super) struct OracleFrontier {
    buckets: Vec<SuccessorBucket>,
}

impl OracleFrontier {
    pub(super) fn into_candidates(self) -> Vec<ScheduleCandidate> {
        self.buckets
            .into_iter()
            .flat_map(|bucket| {
                bucket
                    .candidates
                    .into_iter()
                    .map(|candidate| candidate.schedule)
            })
            .collect()
    }
}

fn successor_key(candidate: &ScheduleCandidate) -> SuccessorKey {
    candidate.folds.first().map_or_else(
        || SuccessorKey::Terminal {
            descriptor: candidate.terminal.params.canonical_descriptor_bytes(),
            first_direct_setup_field_len: candidate.first_direct_setup_field_len,
        },
        |fold| SuccessorKey::Recursive {
            descriptor: fold.params.canonical_descriptor_bytes(),
            output_witness_len: fold.output_witness_len,
            fold_count: candidate.folds.len(),
            first_direct_setup_field_len: candidate.first_direct_setup_field_len,
        },
    )
}

fn candidate_dominates(left: &FrontierCandidate, right: &FrontierCandidate) -> bool {
    if left.schedule.cost == right.schedule.cost
        && left.schedule.setup_field_elements == right.schedule.setup_field_elements
        && left.descriptor == right.descriptor
    {
        return true;
    }
    left.schedule.setup_field_elements <= right.schedule.setup_field_elements
        && left.schedule.cost.expanded_query_count() <= right.schedule.cost.expanded_query_count()
        && left
            .schedule
            .cost
            .strictly_better_for_every_parent(right.schedule.cost)
}

pub(super) fn retain(
    frontier: &mut OracleFrontier,
    candidate: ScheduleCandidate,
) -> Result<(), AkitaError> {
    let key = successor_key(&candidate);
    let candidate = FrontierCandidate {
        descriptor: schedule_descriptor_bytes(&candidate)?,
        schedule: candidate,
    };
    let Some(bucket) = frontier.buckets.iter_mut().find(|bucket| bucket.key == key) else {
        frontier.buckets.push(SuccessorBucket {
            key,
            candidates: vec![candidate],
        });
        return Ok(());
    };
    for incumbent in &bucket.candidates {
        if candidate_dominates(incumbent, &candidate) {
            return Ok(());
        }
    }
    let mut retained = Vec::with_capacity(bucket.candidates.len() + 1);
    for incumbent in bucket.candidates.drain(..) {
        if !candidate_dominates(&candidate, &incumbent) {
            retained.push(incumbent);
        }
    }
    retained.push(candidate);
    bucket.candidates = retained;
    Ok(())
}

#[test]
fn oracle_frontier_retains_lower_query_tradeoffs() -> Result<(), AkitaError> {
    let challenge = SparseChallengeConfig::pm1_only(3);
    let mut params = CommittedGroupParams::params_only(
        akita_types::SisModulusProfileId::Q32Offset99,
        64,
        3,
        2,
        8,
        2,
        challenge,
    )
    .with_decomp(1, 64, 2, 2, 2)?;
    let inner = params.inner().matrix;
    params.own_group_mut().profile.inner.matrix =
        akita_types::InnerCommitMatrixParams::new_unchecked(
            inner.security_policy(),
            inner
                .sis_table_key()
                .expect("L infinity matrix")
                .table_digest,
            inner.sis_modulus_profile(),
            inner.output_rank(),
            inner.input_width(),
            4_095,
            inner.ring_dimension(),
        );
    let (terminal_params, linf_cap) =
        akita_types::TerminalFoldParams::try_from_expanded_group(params)?;
    let response_shape = akita_types::TerminalResponseShape::derive(&terminal_params, linf_cap)?;
    let candidate =
        |payload_bytes, expanded_query_count| -> Result<ScheduleCandidate, AkitaError> {
            Ok(ScheduleCandidate {
                first_direct_setup_field_len: std::num::NonZeroUsize::new(1),
                first_direct_output_witness_len: 0,
                cost: PackedProofCost::new(payload_bytes, 0, expanded_query_count)?,
                setup_field_elements: 1,
                folds: CandidateFoldChain::default(),
                terminal: std::sync::Arc::new(CandidateTerminalResponse {
                    params: terminal_params.clone(),
                    sparse_challenge_config: challenge,
                    input_witness_len: 64,
                    estimated_direct_payload_bytes: 0,
                    response_shape: response_shape.clone(),
                    estimated_payload_bytes: 0,
                }),
            })
        };
    let mut frontier = OracleFrontier::default();
    retain(&mut frontier, candidate(10, 95)?)?;
    retain(&mut frontier, candidate(11, 85)?)?;

    let retained = frontier.into_candidates();
    assert_eq!(
        retained.len(),
        2,
        "a proof-better candidate with more queries must not dominate away a lower-query tradeoff"
    );
    assert!(retained
        .iter()
        .any(|candidate| candidate.cost.expanded_query_count() == 85));
    Ok(())
}
