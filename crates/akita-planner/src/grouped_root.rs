//! Grouped capacity pricing through the unchanged singleton compiler.

use super::{checked, price_binary_root_planes, AkitaError, BinaryPrimePlanePrice};
use akita_params::binary_setup::{BinaryFieldProfile, SetupStreamRole};
use akita_types::binary_source::GroupedNativeRootPlan;

/// Planning inputs; selected resources are bound by `GroupedTypedPrefix::new`.
#[derive(Clone, Debug)]
pub struct GroupedRootPrice {
    pub binary: Vec<BinaryPrimePlanePrice>,
    pub outgoing_physical_cells: usize,
    pub outgoing_domain_len: usize,
    pub outgoing_ring_elements: usize,
    pub typed_prefix_natural_len: usize,
    pub typed_prefix_domain_len: usize,
}

/// Require an available Q64 table cell for each complete D digit stream.
/// Signed preimages are already compiled in the common plan, not priced as small sources.
pub fn price_grouped_root(plan: &GroupedNativeRootPlan) -> Result<GroupedRootPrice, AkitaError> {
    let mut binary = Vec::new();
    binary
        .try_reserve_exact(plan.binary().len())
        .map_err(|_| AkitaError::InvalidProof)?;
    for group in plan.binary() {
        let source = group.source();
        let price = price_binary_root_planes(
            &source.manifest,
            source.word,
            &source.prime,
            plan.repacking().destination_ring_dimension(),
            None,
        )?;
        validate_d_capacity(price.d_columns, price.d_rank)?;
        binary.push(price);
    }
    let typed_prefix_natural_len =
        checked::sum(binary.iter().map(|price| price.typed_prefix.natural_len))
            .ok_or(AkitaError::InvalidProof)?;
    let outgoing_physical_cells = checked::sum(
        (0..plan.repacking().segments().len())
            .map(|i| plan.repacking().segment_range(i).map(|range| range.len()))
            .collect::<Result<Vec<_>, _>>()?,
    )
    .ok_or(AkitaError::InvalidProof)?;
    Ok(GroupedRootPrice {
        binary,
        outgoing_physical_cells,
        outgoing_domain_len: plan.repacking().domain_len(),
        outgoing_ring_elements: checked::exact_div(
            plan.repacking().domain_len(),
            plan.repacking().destination_ring_dimension(),
        )
        .ok_or(AkitaError::InvalidProof)?,
        typed_prefix_natural_len,
        typed_prefix_domain_len: checked::ceil_log2(typed_prefix_natural_len)
            .and_then(checked::pow2)
            .ok_or(AkitaError::InvalidProof)?,
    })
}

fn validate_d_capacity(columns: usize, rank: usize) -> Result<(), AkitaError> {
    let minimum = BinaryFieldProfile::P64
        .digit_commitment_rank(SetupStreamRole::WitnessCommitment, columns)?;
    if rank < minimum {
        return Err(AkitaError::InvalidSetup(
            "grouped D rank below table minimum".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_types::binary_source::{grouped_binding_fixture, grouped_sources_fixture};

    #[test]
    fn grouped_rejects_missing_sis_entry() {
        let error = validate_d_capacity(usize::MAX, 20).unwrap_err();
        assert!(error.to_string().contains("binary digit width exceeds admitted ranks"));
        assert!(validate_d_capacity(1, 0).is_err());
        validate_d_capacity(1, 1).unwrap();
    }

    #[test]
    fn grouped_price_matches_bound_staged_capacity() {
        let (binary, ordinary) = grouped_sources_fixture();
        let plan = GroupedNativeRootPlan::new(&binary, &ordinary, 128).unwrap();
        let price = price_grouped_root(&plan).unwrap();
        let typed = grouped_binding_fixture(&plan);
        assert_eq!(price.binary.len(), 4);
        assert_eq!(price.typed_prefix_natural_len, typed.natural_len());
        assert_eq!(
            price.typed_prefix_domain_len,
            typed.first().n_prefix().unwrap()
        );
        assert_eq!(price.outgoing_domain_len, typed.outgoing().domain_len());
        assert_eq!(price.outgoing_ring_elements * 128, price.outgoing_domain_len);
        assert!(price.outgoing_physical_cells <= price.outgoing_domain_len);
        for (source, actual) in binary.iter().zip(&price.binary) {
            let singleton = price_binary_root_planes(
                &source.manifest,
                source.word,
                &source.prime,
                128,
                None,
            )
            .unwrap();
            assert_eq!(*actual, singleton);
            assert_eq!(actual.d_table_min_rank, Some(actual.d_rank));
            assert_eq!(actual.planes, 4);
            assert_eq!(actual.d_live_digits, (1 + 4 * 22) * 162 * 2);
        }
    }
}
