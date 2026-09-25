//! Setup requirements derived in one pass over a validated trusted catalog.

use crate::CommitmentConfig;
use akita_error::AkitaError;
use akita_types::{
    setup_matrix_capacity_for_schedule, AkitaScheduleLookupKey, FoldSchedule, SetupMatrixCapacity,
};

/// Running maximum over every setup matrix a sizing request can reach.
///
/// Observing a shape is the only way to raise the envelope, so a reachable
/// shape can never be priced without also marking the request supported.
struct SetupCapacityScan {
    supported: bool,
    capacity: SetupMatrixCapacity,
}

impl SetupCapacityScan {
    fn new() -> Self {
        Self {
            supported: false,
            capacity: SetupMatrixCapacity::minimum(),
        }
    }

    fn observe(&mut self, field_elements: usize) {
        self.supported = true;
        self.capacity.num_field_elements = self.capacity.num_field_elements.max(field_elements);
    }

    fn observe_schedule(&mut self, schedule: &FoldSchedule) -> Result<(), AkitaError> {
        self.observe(setup_matrix_capacity_for_schedule(schedule)?.num_field_elements);
        Ok(())
    }

    fn finish(self, max_num_vars: usize) -> Result<SetupMatrixCapacity, AkitaError> {
        if !self.supported {
            return Err(AkitaError::InvalidSetup(format!(
                "setup matrix sizing found no admitted schedules for max_num_vars={max_num_vars}"
            )));
        }
        Ok(self.capacity)
    }
}

/// Matrix capacity and exact prefix commitments required at one capacity bound.
///
/// Requirements computed from several catalogs at the same bound combine with
/// [`SetupRequirements::union`], so one setup can serve every combined family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupRequirements {
    /// Maximum polynomial variable count the requirements were computed at.
    pub max_num_vars: usize,
    /// Maximum polynomial count per opening batch the requirements were computed at.
    pub max_num_batched_polys: usize,
    /// Shared public matrix envelope, including independently reachable precommits.
    pub matrix_capacity: SetupMatrixCapacity,
    /// Canonical ordered set of prefix commitments for eligible schedule rows.
    pub prefix_slot_ids: Vec<akita_types::SetupPrefixSlotId>,
}

impl SetupRequirements {
    /// Size the shared setup matrix from one validated trusted catalog.
    ///
    /// Every admitted row is already expanded and audited. Setup sizing therefore
    /// scans those exact rows instead of consulting any compiled schedule table.
    pub fn from_catalog<Cfg: CommitmentConfig>(
        catalog: &crate::TrustedScheduleCatalog<Cfg>,
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<Self, AkitaError> {
        validate_setup_capacity_metadata(max_num_vars, max_num_batched_polys)?;

        let mut scan = SetupCapacityScan::new();
        let mut prefix_slot_ids = std::collections::BTreeSet::new();
        for row in catalog.rows() {
            for profile in &row.profiles().precommitteds {
                if profile.group.num_vars() <= max_num_vars
                    && profile.group.num_polynomials() <= max_num_batched_polys
                {
                    scan.observe(akita_types::commit_only_setup_field_elements(
                        &profile.inner.matrix,
                        &profile.outer.matrix,
                        profile.outer_slice_count,
                    )?);
                }
            }

            let key = AkitaScheduleLookupKey {
                final_group: row.profiles().final_group.group,
                precommitteds: row.profiles().precommitteds.clone(),
            };
            if key.fits_setup_capacity(max_num_vars, max_num_batched_polys)? {
                scan.observe_schedule(row.schedule())?;
                if Cfg::recursive_setup_planning() {
                    prefix_slot_ids.extend(
                        crate::setup_prefix_slots::required_setup_prefix_slot_ids_for_schedule(
                            row.schedule(),
                            &key.opening_layout()?,
                        )?,
                    );
                }
            }
        }
        Ok(Self {
            max_num_vars,
            max_num_batched_polys,
            matrix_capacity: scan.finish(max_num_vars)?,
            prefix_slot_ids: prefix_slot_ids.into_iter().collect(),
        })
    }

    /// Combine requirements computed at the same capacity bound.
    ///
    /// The result covers the larger matrix envelope and the sorted union of
    /// both prefix-commitment sets.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the two requirements were
    /// computed at different `max_num_vars` or `max_num_batched_polys`.
    pub fn union(self, other: Self) -> Result<Self, AkitaError> {
        if (self.max_num_vars, self.max_num_batched_polys)
            != (other.max_num_vars, other.max_num_batched_polys)
        {
            return Err(AkitaError::InvalidSetup(format!(
                "setup requirements at ({} vars, {} polynomials) cannot combine with \
                 requirements at ({} vars, {} polynomials)",
                self.max_num_vars,
                self.max_num_batched_polys,
                other.max_num_vars,
                other.max_num_batched_polys
            )));
        }
        let prefix_slot_ids: std::collections::BTreeSet<_> = self
            .prefix_slot_ids
            .into_iter()
            .chain(other.prefix_slot_ids)
            .collect();
        Ok(Self {
            max_num_vars: self.max_num_vars,
            max_num_batched_polys: self.max_num_batched_polys,
            matrix_capacity: SetupMatrixCapacity {
                num_field_elements: self
                    .matrix_capacity
                    .num_field_elements
                    .max(other.matrix_capacity.num_field_elements),
            },
            prefix_slot_ids: prefix_slot_ids.into_iter().collect(),
        })
    }
}

/// Validate setup-capacity metadata shared by sizing and setup-prefix planning.
pub fn validate_setup_capacity_metadata(
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<(), AkitaError> {
    if max_num_batched_polys == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    if max_num_vars >= usize::BITS as usize {
        return Err(AkitaError::InvalidSetup(format!(
            "verifier setup capacity ({max_num_vars} vars, {max_num_batched_polys} polynomials) \
             exceeds preprocessing limits"
        )));
    }
    Ok(())
}
