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

fn candidate_dominates(
    left: &ScheduleCandidate,
    right: &ScheduleCandidate,
) -> Result<bool, AkitaError> {
    if successor_key(left) != successor_key(right) {
        return Ok(false);
    }
    if left.cost == right.cost
        && left.setup_field_elements == right.setup_field_elements
        && schedule_descriptor_bytes(left)? == schedule_descriptor_bytes(right)?
    {
        return Ok(true);
    }
    Ok(left.setup_field_elements <= right.setup_field_elements
        && left.cost.strictly_better_for_every_parent(right.cost))
}

pub(super) fn retain(
    frontier: &mut Vec<ScheduleCandidate>,
    candidate: ScheduleCandidate,
) -> Result<(), AkitaError> {
    for incumbent in frontier.iter() {
        if candidate_dominates(incumbent, &candidate)? {
            return Ok(());
        }
    }
    let mut retained = Vec::with_capacity(frontier.len() + 1);
    for incumbent in frontier.drain(..) {
        if !candidate_dominates(&candidate, &incumbent)? {
            retained.push(incumbent);
        }
    }
    retained.push(candidate);
    *frontier = retained;
    Ok(())
}
