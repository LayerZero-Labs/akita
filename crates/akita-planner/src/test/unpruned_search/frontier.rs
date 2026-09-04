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
