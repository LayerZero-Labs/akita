use akita_error::{checked, AkitaError};
use akita_params::{CommitmentRingDims, ScheduleLookupKey};

use crate::{planner::precommitted_group_equivalence_classes, PlannerPolicy};

use super::super::PlannerOpeningCandidate;

/// Maximum number of root precommit opening assignments per opening dimension.
/// Class multiplicity does not increase the product size.
pub(crate) const MAX_PRECOMMIT_OPENING_PRODUCTS: usize = 256;

/// Enumerate one opening per interchangeable precommitted class, in class order.
///
/// Products of at most 256 retain their exhaustive order. Larger products keep
/// every shared-subring assignment first, then the cheapest mixed assignments
/// by total D segment width per opening digit, including class multiplicity.
/// This geometry score estimates setup and proof size; it does not replace
/// response sizing, security admission, or complete-schedule selection.
pub(crate) fn packing_precommit_opening_products(
    policy: &PlannerPolicy,
    dimensions: CommitmentRingDims,
    key: &ScheduleLookupKey,
    precommitted_source_contracts: &[akita_params::sis::CommittedSourceContract],
) -> Result<Vec<Vec<PlannerOpeningCandidate>>, AkitaError> {
    if !super::super::precommitted_groups_support_opening_dimension(
        key.precommitteds.iter(),
        dimensions.d_d(),
    ) {
        return Ok(Vec::new());
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
        return Ok(Vec::new());
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
        return Ok(products);
    }

    let mut products = Vec::new();
    for opening in &class_domains[0] {
        let shared = class_domains
            .iter()
            .map(|domain| {
                domain
                    .iter()
                    .copied()
                    .find(|candidate| candidate.method() == opening.method())
            })
            .collect::<Option<Vec<_>>>();
        if let Some(shared) = shared {
            products.push(shared);
        }
    }

    // An additive score lets each prefix retain only its 256 cheapest choices:
    // a discarded prefix has 256 no-worse prefixes with the same continuations.
    // This avoids allocating the full product, including when its count overflows.
    // Domain indices break equal-width ties in the original enumeration order.
    let mut ranked = vec![(0usize, Vec::<usize>::new())];
    for (indices, domain) in classes.iter().zip(&class_domains) {
        let profile = &key.precommitteds[indices[0]];
        let mut extended = Vec::new();
        for (opening_index, opening) in domain.iter().enumerate() {
            let width = akita_params::opening_d_segment_width(
                opening.method(),
                policy.claim_ext_degree,
                profile.inner.matrix.ring_dimension(),
                dimensions.d_d(),
                1,
                profile.blocks.live_blocks,
                profile.group.num_polynomials(),
            )?;
            for (score, assignment) in &ranked {
                let score = checked::mul_add(width, indices.len(), *score).ok_or_else(|| {
                    AkitaError::InvalidSetup("root opening assignment D width overflow".into())
                })?;
                let mut assignment = assignment.clone();
                assignment.push(opening_index);
                extended.push((score, assignment));
            }
        }
        extended.sort_unstable();
        extended.truncate(MAX_PRECOMMIT_OPENING_PRODUCTS);
        ranked = extended;
    }
    for (_, assignment) in ranked {
        let product = assignment
            .into_iter()
            .enumerate()
            .map(|(class, opening)| class_domains[class][opening])
            .collect::<Vec<_>>();
        if !products.contains(&product) {
            products.push(product);
        }
        if products.len() == MAX_PRECOMMIT_OPENING_PRODUCTS {
            break;
        }
    }
    Ok(products)
}

#[cfg(test)]
#[path = "../../test/precommit_openings.rs"]
mod tests;
