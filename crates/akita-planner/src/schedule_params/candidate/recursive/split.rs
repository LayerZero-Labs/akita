use super::*;

impl SplitBoundPolicy {
    pub(super) fn is_enabled(self) -> bool {
        match self {
            Self::Enabled => true,
            #[cfg(all(test, feature = "catalog-gen"))]
            Self::DisabledForOracle => false,
        }
    }
}

fn push_recursive_split_candidate(candidates: &mut Vec<usize>, reduced_vars: usize, p: isize) {
    if p <= 0 || p >= reduced_vars as isize {
        return;
    }
    let r = reduced_vars - p as usize;
    if !candidates.contains(&r) {
        candidates.push(r);
    }
}

const EXHAUSTIVE_SPLIT_VARIABLE_LIMIT: usize = 12;
const LARGE_SPLIT_BALANCE_RADIUS: isize = 2;

fn bounded_recursive_split_candidates(
    num_ring_elems: usize,
    reduced_vars: usize,
    delta_commit: usize,
    delta_open: usize,
    num_chunks: usize,
) -> Vec<usize> {
    if reduced_vars <= EXHAUSTIVE_SPLIT_VARIABLE_LIMIT {
        return (1..reduced_vars).rev().collect();
    }

    let mut candidates = Vec::new();
    push_recursive_split_candidate(&mut candidates, reduced_vars, 1);
    push_recursive_split_candidate(&mut candidates, reduced_vars, reduced_vars as isize - 1);

    let target_num = 2u128
        .saturating_mul(delta_open as u128)
        .saturating_mul(num_ring_elems as u128);
    let target_den = (delta_commit as u128).saturating_mul(num_chunks.max(1) as u128);
    if target_num > 0 && target_den > 0 {
        let mut center = 1usize;
        let mut best_distance: Option<u128> = None;
        for p in 1..reduced_vars {
            let Some(power) = 1u128.checked_shl((2 * p) as u32) else {
                break;
            };
            let scaled = target_den.saturating_mul(power);
            let distance = scaled.abs_diff(target_num);
            if best_distance.is_none_or(|best| distance < best) {
                center = p;
                best_distance = Some(distance);
            }
        }
        for offset in -LARGE_SPLIT_BALANCE_RADIUS..=LARGE_SPLIT_BALANCE_RADIUS {
            push_recursive_split_candidate(&mut candidates, reduced_vars, center as isize + offset);
        }
    }

    candidates.sort_by(|left, right| right.cmp(left));
    candidates
}

/// Return the exact split domain selected by the catalog-bound search policy.
pub(crate) fn recursive_split_search_domain(
    search_policy: crate::RecursiveSplitSearchPolicy,
    num_ring_elems: usize,
    reduced_vars: usize,
    delta_commit: usize,
    delta_open: usize,
    num_chunks: usize,
) -> Vec<usize> {
    match search_policy {
        crate::RecursiveSplitSearchPolicy::Exhaustive => (1..reduced_vars).rev().collect(),
        crate::RecursiveSplitSearchPolicy::BoundedBalancedExtremesV1 => {
            bounded_recursive_split_candidates(
                num_ring_elems,
                reduced_vars,
                delta_commit,
                delta_open,
                num_chunks,
            )
        }
    }
}

/// Inputs shared by conservative recursive split bounds.
#[derive(Clone, Copy)]
pub(in super::super) struct RecursiveSplitLowerBoundInput {
    pub(in super::super) num_ring_elems: usize,
    pub(in super::super) ring_dimension: usize,
    pub(in super::super) opening_width: usize,
    pub(in super::super) reduced_vars: usize,
    pub(in super::super) r: usize,
    pub(in super::super) delta_commit: usize,
    pub(in super::super) delta_open: usize,
    pub(in super::super) num_chunks: usize,
}

pub(super) fn recursive_witness_body_lower_bound(
    input: RecursiveSplitLowerBoundInput,
) -> Option<usize> {
    if input.r == 0 || input.r >= input.reduced_vars {
        return None;
    }
    let p = input.reduced_vars.checked_sub(input.r)?;
    let num_positions_per_block = checked::pow2(p)?;
    let num_live_blocks = input.num_ring_elems.div_ceil(num_positions_per_block);

    let e_hat = num_live_blocks.checked_mul(input.delta_open)?;
    let t_hat_floor = e_hat;
    let z_hat_floor = num_positions_per_block
        .checked_mul(input.delta_commit)?
        .checked_mul(input.num_chunks.max(1))?;
    let physical_width_floor = e_hat
        .checked_mul(input.opening_width)?
        .checked_add(t_hat_floor.checked_mul(input.ring_dimension)?)?
        .checked_add(z_hat_floor.checked_mul(input.ring_dimension)?)?;
    Some(physical_width_floor)
}

/// Lower bound on the final layout score for one recursive split.
///
/// The true score adds challenge and chunk work to the next witness. The next
/// witness itself includes at least the physical Z/E/T body returned above;
/// setup-prefix and relation-tail terms can only increase it.
pub(in super::super) fn recursive_split_lower_bound(
    input: RecursiveSplitLowerBoundInput,
) -> Option<usize> {
    let physical_width_floor = recursive_witness_body_lower_bound(input)?;
    let p = input.reduced_vars.checked_sub(input.r)?;
    let num_positions_per_block = checked::pow2(p)?;
    let num_live_blocks = input.num_ring_elems.div_ceil(num_positions_per_block);
    physical_width_floor
        .checked_add(num_live_blocks)?
        .checked_add(num_live_blocks)
}

pub(in super::super) fn recursive_candidate_order_key(
    score: LayoutCandidateScore,
    block_index_bits: usize,
) -> (LayoutCandidateScore, std::cmp::Reverse<usize>) {
    (score, std::cmp::Reverse(block_index_bits))
}

/// Best materialized splits of one relation mode in layout order, one per
/// fold proof-of-work nonce length.
///
/// The layout score does not see the fold proof of work, whose nonce grows
/// with the live block count. A split with a worse score but fewer live
/// blocks can still send fewer nonce bytes, so a split is dropped only when
/// another has a layout key and fold nonce bytes no larger. The objective
/// pays nonce bytes, so wider nonces of the same byte length tie.
///
/// This is a bounded heuristic under any nonce key: it ignores the setup
/// prefix and downstream work. Keying on nonce bits instead keeps about 2.5x
/// the search and does not find better schedules, so keep the byte key.
#[derive(Default)]
pub(super) struct SplitFrontier {
    entries: Vec<SplitFrontierEntry>,
}

struct SplitFrontierEntry {
    score: LayoutCandidateScore,
    split: usize,
    /// Fewest live blocks at which every split of this frontier sends at
    /// least this entry's fold nonce bytes, whatever its setup prefix.
    nonce_floor_live_blocks: usize,
    nonce_bytes: usize,
    candidate: CommittedGroupParams,
    next_witness_len: usize,
}

impl SplitFrontier {
    /// Whether a split with these bounds can still enter. A split with at
    /// least an entry's floor of live blocks sends at least its nonce bytes,
    /// so an entry with a smaller score than the split's lower bound
    /// dominates all of it.
    pub(super) fn admits(&self, bounds: RecursiveSplitBounds) -> bool {
        bounds.score.is_none_or(|bound| {
            self.entries.iter().all(|entry| {
                entry.nonce_floor_live_blocks > bounds.live_blocks || bound <= entry.score.0
            })
        })
    }

    /// Offer a split whose fold sends `nonce_bytes` of proof-of-work nonce,
    /// as [`fold_nonce_bytes`] prices it.
    pub(super) fn offer(
        &mut self,
        score: LayoutCandidateScore,
        split: usize,
        candidate: CommittedGroupParams,
        next_witness_len: usize,
        nonce_bytes: usize,
    ) -> Result<(), AkitaError> {
        let key = recursive_candidate_order_key(score, split);
        let entry_key =
            |entry: &SplitFrontierEntry| recursive_candidate_order_key(entry.score, entry.split);
        if self
            .entries
            .iter()
            .any(|entry| entry.nonce_bytes <= nonce_bytes && entry_key(entry) <= key)
        {
            return Ok(());
        }
        self.entries
            .retain(|entry| !(nonce_bytes <= entry.nonce_bytes && key <= entry_key(entry)));
        self.entries.push(SplitFrontierEntry {
            score,
            split,
            nonce_floor_live_blocks: nonce_floor_live_blocks(&candidate, nonce_bytes)?,
            nonce_bytes,
            candidate,
            next_witness_len,
        });
        Ok(())
    }

    /// Retained splits as `(split, candidate, next witness length)`, best
    /// layout key first.
    pub(super) fn into_candidates(mut self) -> impl Iterator<Item = BestLinfCandidate> {
        self.entries
            .sort_by_key(|entry| recursive_candidate_order_key(entry.score, entry.split));
        self.entries
            .into_iter()
            .map(|entry| (entry.split, entry.candidate, entry.next_witness_len))
    }
}

/// Fold proof-of-work nonce bytes of `candidate` under its own opening
/// layout, or `None` when its fold cannot be priced, such as when a challenge
/// message exceeds the message limit. Such a split can never be planned.
pub(super) fn fold_nonce_bytes(
    candidate: &CommittedGroupParams,
) -> Result<Option<usize>, AkitaError> {
    let layout = candidate.opening_layout_for_final_group(candidate.group())?;
    match akita_params::FoldChallengeCost::new(candidate, &layout) {
        Ok(cost) => Ok(Some(cost.nonce_max_bytes())),
        Err(AkitaError::InvalidSetup(_)) => Ok(None),
        Err(error) => Err(error),
    }
}

/// Fewest live blocks `L` at which every split of `candidate`'s frontier
/// sends at least `nonce_bytes` of fold nonce, or `usize::MAX` when no live
/// block count guarantees it.
///
/// Every split of one frontier draws its own group's challenges from the
/// same family, one per polynomial and live block, and its fold proof of
/// work covers at least those challenges against at most that family's set
/// size, whatever setup prefix it carries. So the least `L` whose own-group
/// challenges alone need `nonce_bytes` bounds every split with `L` or more
/// live blocks. A count whose proof of work cannot be priced can never be
/// planned, so it counts as reaching any length.
fn nonce_floor_live_blocks(
    candidate: &CommittedGroupParams,
    nonce_bytes: usize,
) -> Result<usize, AkitaError> {
    let polynomials = u64::try_from(candidate.group().num_polynomials())
        .map_err(|_| AkitaError::InvalidSetup("fold polynomial count exceeds u64".into()))?;
    let set_bits = candidate.own_group().fold_challenge_family()?.set_bits();
    let reaches = |live: usize| {
        let Some(challenges) = u64::try_from(live)
            .ok()
            .and_then(|live| live.checked_mul(polynomials))
        else {
            return Ok(true);
        };
        match akita_params::GrindingRun::fold_challenge_work(0, challenges, set_bits) {
            Ok(work) => Ok(work.nonce_max_bytes()? >= nonce_bytes),
            Err(AkitaError::InvalidSetup(_)) => Ok(true),
            Err(error) => Err(error),
        }
    };
    // A split has at most one live block per ring element.
    let max_live_blocks = candidate.blocks().live_ring_elements_per_claim;
    if max_live_blocks == 0 || !reaches(max_live_blocks)? {
        return Ok(usize::MAX);
    }
    let (mut low, mut high) = (1, max_live_blocks);
    while low < high {
        let mid = low + (high - low) / 2;
        if reaches(mid)? {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    Ok(low)
}
