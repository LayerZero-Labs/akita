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
    let num_positions_per_block = 1usize.checked_shl(p as u32)?;
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
    let num_positions_per_block = 1usize.checked_shl(p as u32)?;
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
