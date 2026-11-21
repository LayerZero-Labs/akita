//! Matrix requirements and setup capacity checks for admitted schedules.

use crate::CommitmentConfig;
use akita_error::AkitaError;
use akita_params::{
    setup_matrix_capacity_for_schedule, setup_matrix_field_elements_for_schedule,
    verifier_setup_matrix_capacity_for_schedule, FoldSchedule, OpeningClaimsLayout,
    SetupMatrixCapacity, SetupPrefixSlotId,
};
use akita_types::AkitaExpandedSetup;
use std::collections::BTreeSet;
use std::marker::PhantomData;

/// Reject a concrete schedule whose exact matrix footprint exceeds setup.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when sizing overflows or the setup's
/// materialized shared matrix is too short for `schedule` and `layout`.
pub fn ensure_prover_schedule_fits_setup<Cfg>(
    setup: &AkitaExpandedSetup<Cfg::Field>,
    schedule: &FoldSchedule,
    layout: &OpeningClaimsLayout,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
{
    // `setup_matrix_field_elements_for_schedule` already maxes over the root
    // level's A/B/D matrices, every frozen precommitted group, the compression maps,
    // and the fold tail, so it dominates any per-level recomputation here.
    schedule.root.params.validate_opening_batch(layout)?;
    let required_field_elements = setup_matrix_field_elements_for_schedule(schedule)?;
    let available_field_elements = setup.shared_matrix.as_field_slice().len();
    if required_field_elements <= available_field_elements {
        return Ok(());
    }
    Err(AkitaError::InvalidSetup(format!(
        "schedule requires {required_field_elements} physical setup field elements, but setup \
         provides {available_field_elements}"
    )))
}

/// Whether a concrete schedule's direct verifier matrix uses fit setup.
///
/// Offloaded producer edges are covered by verifier-visible setup-prefix
/// commitments and do not require their full committed source prefixes here.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when sizing `schedule` overflows.
pub fn verifier_schedule_fits_setup(
    setup: &AkitaExpandedSetup<impl jolt_field::Field>,
    schedule: &FoldSchedule,
    layout: &OpeningClaimsLayout,
    mode: akita_params::SetupMode,
) -> Result<bool, AkitaError> {
    let required = verifier_setup_matrix_capacity_for_schedule(schedule, layout, mode)?;
    Ok(required.num_field_elements <= setup.shared_matrix.as_field_slice().len())
}

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

/// Matrix capacity and exact prefix commitments required at one capacity bound,
/// for setups over the field `F`.
///
/// Requirements are only produced by [`SetupRequirements::from_catalog`] and
/// [`SetupRequirements::union`], so every value carries validated capacity
/// metadata, a non-empty matrix envelope, and a strictly increasing set of
/// prefix slot ids. Requirements computed from several catalogs over the same
/// field at the same bound combine with [`SetupRequirements::union`], so one
/// setup can serve every combined family.
pub struct SetupRequirements<F> {
    max_num_vars: usize,
    max_num_batched_polys: usize,
    matrix_capacity: SetupMatrixCapacity,
    prefix_slot_ids: Vec<SetupPrefixSlotId>,
    field: PhantomData<fn() -> F>,
}

impl<F> std::fmt::Debug for SetupRequirements<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SetupRequirements")
            .field("max_num_vars", &self.max_num_vars)
            .field("max_num_batched_polys", &self.max_num_batched_polys)
            .field("matrix_capacity", &self.matrix_capacity)
            .field("prefix_slot_ids", &self.prefix_slot_ids)
            .finish()
    }
}

impl<F> Clone for SetupRequirements<F> {
    fn clone(&self) -> Self {
        Self {
            max_num_vars: self.max_num_vars,
            max_num_batched_polys: self.max_num_batched_polys,
            matrix_capacity: self.matrix_capacity,
            prefix_slot_ids: self.prefix_slot_ids.clone(),
            field: PhantomData,
        }
    }
}

impl<F> PartialEq for SetupRequirements<F> {
    fn eq(&self, other: &Self) -> bool {
        (
            self.max_num_vars,
            self.max_num_batched_polys,
            self.matrix_capacity,
            &self.prefix_slot_ids,
        ) == (
            other.max_num_vars,
            other.max_num_batched_polys,
            other.matrix_capacity,
            &other.prefix_slot_ids,
        )
    }
}

impl<F> Eq for SetupRequirements<F> {}

impl<F> SetupRequirements<F> {
    /// Size the shared setup matrix from one validated trusted catalog.
    ///
    /// Every admitted row is already expanded and audited. Setup sizing therefore
    /// scans those exact rows instead of consulting any compiled schedule table.
    pub fn from_catalog<Cfg: CommitmentConfig<Field = F>>(
        catalog: &crate::TrustedScheduleCatalog<Cfg>,
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<Self, AkitaError> {
        catalog.validate_complete()?;
        validate_setup_capacity_metadata(max_num_vars, max_num_batched_polys)?;

        let mut scan = SetupCapacityScan::new();
        let mut prefix_slot_ids = BTreeSet::new();
        for row in catalog.rows() {
            for profile in &row.profiles().precommitteds {
                if profile.group.num_vars() <= max_num_vars
                    && profile.group.num_polynomials() <= max_num_batched_polys
                {
                    scan.observe(akita_params::commit_only_setup_field_elements(
                        &profile.inner.matrix,
                        &profile.outer.matrix,
                        profile.outer_slice_count,
                    )?);
                }
            }
        }
        for row in catalog.rows_within_setup_capacity(max_num_vars, max_num_batched_polys)? {
            scan.observe_schedule(row.schedule())?;
            if Cfg::recursive_setup_planning() {
                prefix_slot_ids.extend(
                    crate::setup_prefix_slots::required_setup_prefix_slot_ids_for_schedule(
                        row.schedule(),
                        &row.profiles().opening_layout()?,
                    )?,
                );
            }
        }
        Ok(Self {
            max_num_vars,
            max_num_batched_polys,
            matrix_capacity: scan.finish(max_num_vars)?,
            prefix_slot_ids: prefix_slot_ids.into_iter().collect(),
            field: PhantomData,
        })
    }

    /// Maximum polynomial variable count the requirements were computed at.
    pub fn max_num_vars(&self) -> usize {
        self.max_num_vars
    }

    /// Maximum polynomial count per opening batch the requirements were computed at.
    pub fn max_num_batched_polys(&self) -> usize {
        self.max_num_batched_polys
    }

    /// Shared public matrix envelope, including independently reachable precommits.
    pub fn matrix_capacity(&self) -> SetupMatrixCapacity {
        self.matrix_capacity
    }

    /// Strictly increasing prefix commitments for eligible schedule rows.
    pub fn prefix_slot_ids(&self) -> &[SetupPrefixSlotId] {
        &self.prefix_slot_ids
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
        let prefix_slot_ids: BTreeSet<_> = self
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
            field: PhantomData,
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
