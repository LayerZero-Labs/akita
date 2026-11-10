use akita_error::AkitaError;
use akita_params::{CommitmentRingDims, ScheduleLookupKey};
use jolt_algebra::checked;

use crate::{planner::precommitted_group_equivalence_classes, PlannerPolicy};

use super::super::PlannerOpeningCandidate;

/// Maximum number of root precommit opening assignments per opening dimension.
/// Class multiplicity does not increase the product size.
pub(crate) const MAX_PRECOMMIT_OPENING_PRODUCTS: usize = 256;

/// Enumerate small root opening products in their original order.
/// `None` defers an oversized product until bases and root layouts are fixed,
/// so it can be ranked by the exact combined outgoing witness length.
pub(crate) fn packing_precommit_opening_products(
    policy: &PlannerPolicy,
    dimensions: CommitmentRingDims,
    key: &ScheduleLookupKey,
    precommitted_source_contracts: &[akita_params::sis::CommittedSourceContract],
) -> Result<Option<Vec<Vec<PlannerOpeningCandidate>>>, AkitaError> {
    if !super::super::precommitted_groups_support_opening_dimension(
        key.precommitteds.iter(),
        dimensions.d_d(),
    ) {
        return Ok(Some(Vec::new()));
    }
    let classes =
        precommitted_group_equivalence_classes(&key.precommitteds, precommitted_source_contracts)?;
    let class_domains = classes
        .iter()
        .map(|indices| {
            let profile = &key.precommitteds[indices[0]];
            PlannerOpeningCandidate::coefficient_packing_domain(
                policy.claim_ext_degree,
                CommitmentRingDims {
                    inner: profile.inner.matrix.ring_dimension(),
                    outer: profile.outer.matrix.ring_dimension(),
                    opening: dimensions.d_d(),
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if class_domains.iter().any(Vec::is_empty) {
        return Ok(Some(Vec::new()));
    }
    if checked::product(class_domains.iter().map(Vec::len))
        .is_some_and(|count| count <= MAX_PRECOMMIT_OPENING_PRODUCTS)
    {
        let mut products = vec![Vec::with_capacity(class_domains.len())];
        for domain in &class_domains {
            products = products
                .iter()
                .flat_map(|product| {
                    domain.iter().map(move |&opening| {
                        let mut extended = product.clone();
                        extended.push(opening);
                        extended
                    })
                })
                .collect();
        }
        return Ok(Some(products));
    }

    Ok(None)
}

#[cfg(test)]
#[path = "../../test/precommit_openings.rs"]
mod tests;
