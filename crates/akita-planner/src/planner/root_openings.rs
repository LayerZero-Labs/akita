use std::collections::BTreeMap;

use super::*;
use crate::schedule_params::MAX_PRECOMMIT_OPENING_PRODUCTS;

/// Scan every admitted class-opening assignment at a fixed root context.
/// Retain assignments by their smallest exact combined outgoing witness over
/// all allowed splits and slices. Shared subrings receive no reserved slots.
pub(crate) fn smallest_witness_root_candidates(
    key: &ScheduleLookupKey,
    contracts: &[CommittedSourceContract],
    policy: &PlannerPolicy,
    dimensions: CommitmentRingDims,
    open_basis: u32,
    main: &PreparedRootFinalGroup,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    if main.candidates_by_split.iter().all(Vec::is_empty) {
        return Ok(Vec::new());
    }
    let classes = precommitted_group_equivalence_classes(&key.precommitteds, contracts)?;
    let mut domains = Vec::new();
    for indices in &classes {
        let index = indices[0];
        let profile = key.precommitteds[index];
        let mut admitted = Vec::new();
        for opening in PlannerOpeningCandidate::coefficient_packing_domain(
            policy.claim_ext_degree,
            CommitmentRingDims {
                inner: profile.inner.matrix.ring_dimension(),
                outer: profile.outer.matrix.ring_dimension(),
                opening: dimensions.d_d(),
            },
        )? {
            if let Some(group) = materialize_precommitted_group_for_open_basis(
                &(profile, contracts[index]),
                policy,
                opening,
                dimensions.d_d(),
                open_basis,
            )? {
                let width = group.d_segment_width(policy.claim_ext_degree, dimensions.d_d())?;
                admitted.push((group, width));
            }
        }
        if admitted.is_empty() {
            return Ok(Vec::new());
        }
        domains.push(admitted);
    }

    // Stream the product with an odometer; its cardinality need not fit usize.
    // Only the best 256 assignments and their root candidates remain allocated.
    let mut selected = BTreeMap::new();
    let mut assignment = vec![0; domains.len()];
    loop {
        let producers = PreparedRootProducers::from_class_openings(
            &classes,
            domains
                .iter()
                .zip(&assignment)
                .map(|(domain, &index)| domain[index]),
            key.precommitteds.len(),
        )?;
        let candidates = main.candidates(policy, &producers, false)?;
        if let Some(score) = candidates.iter().map(|(_, len)| *len).min() {
            let rank = (score, assignment.clone());
            if selected.len() < MAX_PRECOMMIT_OPENING_PRODUCTS
                || selected
                    .last_key_value()
                    .is_some_and(|(worst, _)| &rank < worst)
            {
                selected.insert(rank, candidates);
                if selected.len() > MAX_PRECOMMIT_OPENING_PRODUCTS {
                    selected.pop_last();
                }
            }
        }
        let mut cursor = assignment.len();
        loop {
            let Some(index) = cursor.checked_sub(1) else {
                return Ok(selected.into_values().flatten().collect());
            };
            cursor = index;
            assignment[index] += 1;
            if assignment[index] < domains[index].len() {
                break;
            }
            assignment[index] = 0;
        }
    }
}

#[cfg(all(test, feature = "catalog-gen"))]
#[path = "../test/root_witness_ranking.rs"]
mod tests;
