//! Research-only catalog accounting and final grouped setup binding.

use std::collections::BTreeMap;

use akita_error::AkitaError;
use akita_params::{OpeningScheduleSelection, SetupPrefixSlotId, SisModulusProfileId};
use akita_serialization::Valid;
use akita_types::binary_source::{GroupedPrefixCapacity, GroupedTypedPrefix};
use akita_types::AkitaSetupDescriptor;

use crate::{
    required_native_setup_prefix_slot_ids_for_row, required_setup_prefix_slot_ids_for_schedule,
    CommitmentConfig, SetupRequirements, TrustedScheduleCatalog,
};

/// Complete setup requirements plus the exact native rows they bind.
///
/// Construction requires one capacity binding per eligible native row, in catalog
/// digest order. Ordinary rows retain ordinary accounting. After sizing, use
/// [`Self::bind_setup`] to bind the final descriptor and complete slot inventory.
/// This does not admit native opening or ordinary artifact serialization.
#[derive(Debug)]
pub struct NativeSetupRequirements<F> {
    requirements: SetupRequirements<F>,
    bindings: Vec<(OpeningScheduleSelection, GroupedPrefixCapacity)>,
}

impl<F> NativeSetupRequirements<F> {
    /// Require the exact ordered list of in-capacity native row selections.
    /// Source manifests remain caller-trusted inputs; the catalog authenticates
    /// schedule geometry, not the native equations or source claims.
    pub fn from_catalog<Cfg: CommitmentConfig<Field = F>>(
        catalog: &TrustedScheduleCatalog<Cfg>,
        max_num_vars: usize,
        max_num_batched_polys: usize,
        bindings: &[(OpeningScheduleSelection, GroupedPrefixCapacity)],
    ) -> Result<Self, AkitaError> {
        crate::validate_setup_capacity_metadata(max_num_vars, max_num_batched_polys)?;
        if !Cfg::recursive_setup_planning()
            || Cfg::sis_modulus_profile() != SisModulusProfileId::Q64Offset59
        {
            return Err(AkitaError::InvalidSetup(
                "native setup requires the recursive p64 policy".into(),
            ));
        }
        let rows = catalog.rows_within_setup_capacity(max_num_vars, max_num_batched_polys)?;
        let mut expected = rows.iter().filter(|row| row.is_native_root());
        let mut native_slots = BTreeMap::<SetupPrefixSlotId, &GroupedPrefixCapacity>::new();
        for (selection, capacity) in bindings {
            catalog.resolve_selection(*selection)?;
            if expected.next().map(|row| row.selection()) != Some(*selection) {
                return Err(AkitaError::InvalidSetup(
                    "native setup bindings must match eligible rows in catalog order".into(),
                ));
            }
            if let Some(previous) = native_slots.insert(capacity.first().clone(), capacity) {
                if previous.plan().identity() != capacity.plan().identity()
                    || previous.successor() != capacity.successor()
                {
                    return Err(AkitaError::InvalidSetup(
                        "native setup slot has conflicting grouped streams".into(),
                    ));
                }
            }
            for profile in capacity
                .plan()
                .binary()
                .iter()
                .map(|binary| &binary.source().carrier)
                .chain(
                    capacity
                        .plan()
                        .ordinary()
                        .into_iter()
                        .flat_map(|batch| batch.plans().iter().map(|plan| plan.profile())),
                )
            {
                if profile.group.num_vars() > max_num_vars
                    || profile.group.num_polynomials() > max_num_batched_polys
                {
                    return Err(AkitaError::InvalidSetup(
                        "native source exceeds setup polynomial capacity".into(),
                    ));
                }
            }
        }
        if bindings.is_empty() || expected.next().is_some() {
            return Err(AkitaError::InvalidSetup(
                "native setup bindings omit an eligible row".into(),
            ));
        }
        let requirements = SetupRequirements::from_catalog_with_prefixes(
            catalog,
            max_num_vars,
            max_num_batched_polys,
            |row| {
                let (ids, source_capacity, first) = if row.is_native_root() {
                    let (_, capacity) = bindings
                        .iter()
                        .find(|(selection, _)| *selection == row.selection())
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("missing native setup binding".into())
                        })?;
                    (
                        required_native_setup_prefix_slot_ids_for_row(row, capacity)?,
                        capacity.matrix_capacity().num_field_elements,
                        Some(capacity.first()),
                    )
                } else {
                    (
                        required_setup_prefix_slot_ids_for_schedule(
                            row.schedule(),
                            &row.profiles().opening_layout()?,
                        )?,
                        0,
                        None,
                    )
                };
                if ids
                    .iter()
                    .any(|slot| native_slots.contains_key(slot) && Some(slot) != first)
                {
                    return Err(AkitaError::InvalidSetup(
                        "ordinary prefix aliases a native setup stream".into(),
                    ));
                }
                Ok((ids, source_capacity))
            },
        )?;
        Ok(Self {
            requirements,
            bindings: bindings.to_vec(),
        })
    }

    /// Validated matrix sizing and inventory; native preparation remains explicit.
    pub fn requirements(&self) -> &SetupRequirements<F> {
        &self.requirements
    }

    /// Check the entire envelope and exact sorted inventory before binding seeds.
    /// The result retains catalog row order; backend authentication is still required.
    pub fn bind_setup(
        &self,
        setup: AkitaSetupDescriptor,
        slots: &[SetupPrefixSlotId],
    ) -> Result<Vec<(OpeningScheduleSelection, GroupedTypedPrefix)>, AkitaError> {
        setup.check().map_err(|_| AkitaError::InvalidProof)?;
        if setup.max_num_vars != self.requirements.max_num_vars()
            || setup.max_num_batched_polys != self.requirements.max_num_batched_polys()
            || setup.num_field_elements < self.requirements.matrix_capacity().num_field_elements
            || slots != self.requirements.prefix_slot_ids()
        {
            return Err(AkitaError::InvalidSetup(
                "native setup descriptor or exact slot inventory mismatch".into(),
            ));
        }
        self.bindings
            .iter()
            .map(|(selection, capacity)| {
                Ok((
                    *selection,
                    GroupedTypedPrefix::bind_capacity(capacity.clone(), setup.clone())?,
                ))
            })
            .collect()
    }
}
