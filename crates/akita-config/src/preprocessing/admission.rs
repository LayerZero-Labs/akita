//! Row admission: which rows a loader admits and how large a setup it derives.
//!
//! - **A1** admits rows against the set limits, or against a program's bounds
//!   for a set with program rows, and fixes the catalog identity.
//! - **A2** sizes the matrix from the admitted rows only.
//! - **A3** derives that prefix and describes the setup with the admitted
//!   bounds. With program rows, the variable bound is the larger of the trace
//!   bound and the widest precommitted group of the merged catalog.
//! - **A4** checks that the resulting verifier setup admits exactly the A1
//!   rows (or the selected row).

use crate::setup_prefix_slots::required_setup_prefix_slot_ids_for_schedule;
use crate::{policy_of, CommitmentConfig, SetupRequirements, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_params::{
    verifier_setup_matrix_capacity_for_schedule, CommittedGroupBatchProfile, FoldSchedule,
    GroupCommitPhaseParams, OpeningScheduleSelection, ScheduleLookupKey, SetupMatrixCapacity,
    SetupPrefixSlotId,
};
use akita_schedules::{ResolvedScheduleRow, ValidatedScheduleCatalog};
use akita_types::preprocessing::{profile_key, ProfileKey, SetRecord};
use akita_types::{AkitaExpandedSetup, AkitaSetupDescriptor, AkitaSetupSeed};
use std::collections::BTreeSet;

/// The rows one program planned for itself, authenticated by that program's
/// preprocessing digest. They are the one loader input that is not a release
/// file.
#[derive(Debug, Clone)]
pub struct ProgramRows {
    /// Program trace bound on polynomial variables. Trace length is not known
    /// before execution, so rows of every trace length within it are
    /// admitted.
    pub max_num_vars: usize,
    /// Program bound on polynomials per opening batch; at most the set limit.
    pub max_num_batched_polys: usize,
    /// Grouped rows planned for this program, in any order.
    pub rows: Vec<(CommittedGroupBatchProfile, FoldSchedule)>,
    /// Frozen root profiles of the anchored producer sets. Every precommitted
    /// group of a program row must equal one of them.
    pub producer_profiles: Vec<GroupCommitPhaseParams>,
    /// Merged catalog digest stored in the program preprocessing, once it
    /// exists. Every loader must reproduce it.
    pub merged_catalog_digest: Option<[u8; 32]>,
}

/// Admitted rows of one parameter set and the catalog identity they fix.
#[derive(Debug, Clone)]
pub struct Admission<Cfg: CommitmentConfig> {
    catalog: TrustedScheduleCatalog<Cfg>,
    rows: Vec<OpeningScheduleSelection>,
    max_num_vars: usize,
    max_num_batched_polys: usize,
    allowed_profiles: BTreeSet<ProfileKey>,
}

impl<Cfg: CommitmentConfig> Admission<Cfg> {
    /// A1 over a complete base catalog that already matched the set record.
    ///
    /// Without program rows the base rows within the set limits are admitted
    /// and the base catalog is the catalog identity. With program rows the
    /// catalog identity is the merged catalog over every base row and every
    /// program row (ordered by row digest, so independent of the order rows
    /// are supplied in), and its rows within the program bounds are
    /// admitted.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when no row is admitted, a program
    /// bound exceeds a set limit, an admitted row uses a profile outside the
    /// set's universe, a program row names an unanchored precommitted
    /// profile, or the merged catalog does not have the stored digest.
    pub fn admit(
        record: &SetRecord,
        base: TrustedScheduleCatalog<Cfg>,
        program: Option<&ProgramRows>,
    ) -> Result<Self, AkitaError> {
        let set_limits = set_limits(record)?;
        let Some(program) = program else {
            let rows = base
                .rows_within_setup_capacity(set_limits.0, set_limits.1)?
                .into_iter()
                .map(ResolvedScheduleRow::selection)
                .collect::<Vec<_>>();
            return Self::finish(
                record,
                base,
                rows,
                set_limits,
                record.catalog_profiles.iter().copied().collect(),
            );
        };
        require_program_rows(record, program, set_limits)?;
        for (profiles, _) in &program.rows {
            check_producer_profiles(program, profiles)?;
        }
        let chunking = base.policy().witness_chunk;
        let merged = ValidatedScheduleCatalog::try_new(
            Cfg::schedule_family_name(),
            base.rows()
                .map(|row| (row.profiles().clone(), row.schedule().clone()))
                .chain(program.rows.iter().cloned()),
            &policy_of::<Cfg>(chunking),
            Cfg::ring_challenge_config,
        )?;
        let merged = TrustedScheduleCatalog::<Cfg>::new(merged, chunking)?;
        check_merged_digest(program, &merged)?;
        let bounds = program_bounds(record, program, set_limits, &merged)?;
        let rows = merged
            .rows_within_setup_capacity(bounds.0, bounds.1)?
            .into_iter()
            .map(ResolvedScheduleRow::selection)
            .collect();
        Self::finish(record, merged, rows, bounds, universe(record))
    }

    /// A1 for a single-row load from an authenticated selected-row view, as
    /// a guest loads.
    ///
    /// The view must reproduce the catalog identity: the set's base catalog
    /// digest, or the program's merged catalog digest. The selection must be
    /// carried by the view and fit the limits.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the view has another digest
    /// or the selection is outside A1.
    pub fn admit_view(
        record: &SetRecord,
        view: TrustedScheduleCatalog<Cfg>,
        selection: OpeningScheduleSelection,
        program: Option<&ProgramRows>,
    ) -> Result<Self, AkitaError> {
        let set_limits = set_limits(record)?;
        let expected_digest = match program {
            None => record.catalog_digest,
            Some(program) => {
                require_program_rows(record, program, set_limits)?;
                program.merged_catalog_digest.ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "a single-row load with program rows needs the stored merged catalog digest"
                            .into(),
                    )
                })?
            }
        };
        if view.catalog_digest() != expected_digest {
            return Err(AkitaError::InvalidSetup(
                "catalog view does not reproduce the catalog identity".into(),
            ));
        }
        let (bounds, allowed) = match program {
            None => (
                set_limits,
                record.catalog_profiles.iter().copied().collect(),
            ),
            Some(program) => (
                program_bounds(record, program, set_limits, &view)?,
                universe(record),
            ),
        };
        let row = view.resolve_selection(selection)?;
        if !row_fits(row.profiles(), bounds)? {
            return Err(AkitaError::InvalidSetup(
                "selected row is outside the admitted rows".into(),
            ));
        }
        Self::finish(record, view, vec![selection], bounds, allowed)
    }

    fn finish(
        record: &SetRecord,
        catalog: TrustedScheduleCatalog<Cfg>,
        mut rows: Vec<OpeningScheduleSelection>,
        bounds: (usize, usize),
        allowed_profiles: BTreeSet<ProfileKey>,
    ) -> Result<Self, AkitaError> {
        if rows.is_empty() {
            return Err(AkitaError::InvalidSetup(format!(
                "parameter set {:?} admits no schedule row",
                record.name
            )));
        }
        rows.sort_unstable();
        let admission = Self {
            catalog,
            rows,
            max_num_vars: bounds.0,
            max_num_batched_polys: bounds.1,
            allowed_profiles,
        };
        admission.required_slot_ids(None)?;
        Ok(admission)
    }

    /// The catalog identity: the base catalog, the merged catalog, or the
    /// selected-row view of either.
    pub fn catalog(&self) -> &TrustedScheduleCatalog<Cfg> {
        &self.catalog
    }

    /// The A1 rows, in digest order.
    pub fn rows(&self) -> &[OpeningScheduleSelection] {
        &self.rows
    }

    /// Variable and polynomial bounds of the setup descriptor.
    pub fn bounds(&self) -> (usize, usize) {
        (self.max_num_vars, self.max_num_batched_polys)
    }

    /// Frozen root profiles of the admitted rows, which program rows of
    /// another set may name as precommitted groups.
    pub fn root_profiles(&self) -> Vec<GroupCommitPhaseParams> {
        self.resolved(None)
            .map(|rows| {
                rows.into_iter()
                    .map(|row| row.profiles().final_group)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The rows a verifier will verify: all A1 rows, or the selected row,
    /// which must be one of them.
    fn resolved(
        &self,
        selection: Option<OpeningScheduleSelection>,
    ) -> Result<Vec<&ResolvedScheduleRow>, AkitaError> {
        match selection {
            None => self
                .rows
                .iter()
                .map(|row| self.catalog.resolve_selection(*row))
                .collect(),
            Some(selection) if self.rows.binary_search(&selection).is_ok() => {
                Ok(vec![self.catalog.resolve_selection(selection)?])
            }
            Some(_) => Err(AkitaError::InvalidSetup(
                "selected row is outside the admitted rows".into(),
            )),
        }
    }

    /// A2 for a verifier: the largest verifier matrix capacity over the rows
    /// it will verify.
    ///
    /// # Errors
    ///
    /// Returns an error when the selection is not admitted or sizing fails.
    pub fn verifier_matrix_capacity(
        &self,
        selection: Option<OpeningScheduleSelection>,
    ) -> Result<SetupMatrixCapacity, AkitaError> {
        let mut capacity = SetupMatrixCapacity::minimum();
        for row in self.resolved(selection)? {
            let needed = verifier_setup_matrix_capacity_for_schedule(
                row.schedule(),
                &row.profiles().opening_layout()?,
            )?;
            capacity.num_field_elements =
                capacity.num_field_elements.max(needed.num_field_elements);
        }
        Ok(capacity)
    }

    /// A2 for a prover: matrix capacity and prefix slots over every row it
    /// may prove.
    ///
    /// # Errors
    ///
    /// Returns an error for a selected-row view or when sizing fails.
    pub fn prover_requirements(&self) -> Result<SetupRequirements<Cfg::Field>, AkitaError> {
        SetupRequirements::from_catalog(
            &self.catalog,
            self.max_num_vars,
            self.max_num_batched_polys,
        )
    }

    /// Setup-prefix slots the rows require, each derived from its audited
    /// schedule and required to use an allowed profile.
    ///
    /// # Errors
    ///
    /// Returns an error when a slot's profile is not allowed for the set.
    pub fn required_slot_ids(
        &self,
        selection: Option<OpeningScheduleSelection>,
    ) -> Result<Vec<SetupPrefixSlotId>, AkitaError> {
        let mut ids = BTreeSet::new();
        for row in self.resolved(selection)? {
            for id in required_setup_prefix_slot_ids_for_schedule(
                row.schedule(),
                &row.profiles().opening_layout()?,
            )? {
                if !self
                    .allowed_profiles
                    .contains(&profile_key(&id.commitment_profile))
                {
                    return Err(AkitaError::InvalidSetup(
                        "an admitted row uses a setup-prefix profile outside its set".into(),
                    ));
                }
                ids.insert(id);
            }
        }
        Ok(ids.into_iter().collect())
    }

    /// A3: the descriptor of a setup over `capacity` field elements of the
    /// stream `setup_seed`.
    pub fn setup_descriptor(
        &self,
        setup_seed: &AkitaSetupSeed,
        capacity: SetupMatrixCapacity,
    ) -> AkitaSetupDescriptor {
        AkitaSetupDescriptor {
            max_num_vars: self.max_num_vars,
            max_num_batched_polys: self.max_num_batched_polys,
            num_field_elements: capacity.num_field_elements,
            setup_seed: setup_seed.clone(),
        }
    }

    /// A4: a full load's setup admits exactly the A1 rows, and a single-row
    /// load's setup admits its selection.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when admission disagrees.
    pub fn check_verifier_setup(
        &self,
        setup: &AkitaExpandedSetup<Cfg::Field>,
        selection: Option<OpeningScheduleSelection>,
    ) -> Result<(), AkitaError> {
        let agrees = match selection {
            None => {
                self.catalog
                    .verifier_admitted_rows(setup)?
                    .into_iter()
                    .map(ResolvedScheduleRow::selection)
                    .collect::<Vec<_>>()
                    == self.rows
            }
            Some(selection) => {
                let row = self.resolved(Some(selection))?[0];
                TrustedScheduleCatalog::<Cfg>::verifier_admits(setup, row)?
            }
        };
        if agrees {
            Ok(())
        } else {
            Err(AkitaError::InvalidSetup(
                "derived verifier setup does not admit exactly the admitted rows".into(),
            ))
        }
    }

    /// The catalog identity's selected-row view of `selection`, for a
    /// single-row load.
    ///
    /// # Errors
    ///
    /// Returns an error when the selection is not admitted.
    pub fn selected_view(
        &self,
        selection: OpeningScheduleSelection,
    ) -> Result<TrustedScheduleCatalog<Cfg>, AkitaError> {
        self.resolved(Some(selection))?;
        let bytes = self.catalog.to_verifier_view(&[selection])?;
        TrustedScheduleCatalog::from_verifier_view(&bytes, self.catalog.policy().witness_chunk)
    }
}

fn set_limits(record: &SetRecord) -> Result<(usize, usize), AkitaError> {
    let max_num_vars = usize::try_from(record.max_num_vars)
        .map_err(|_| AkitaError::InvalidSetup("set variable limit overflows usize".into()))?;
    let max_num_batched_polys = usize::try_from(record.max_num_batched_polys)
        .map_err(|_| AkitaError::InvalidSetup("set polynomial limit overflows usize".into()))?;
    crate::validate_setup_capacity_metadata(max_num_vars, max_num_batched_polys)?;
    Ok((max_num_vars, max_num_batched_polys))
}

fn universe(record: &SetRecord) -> BTreeSet<ProfileKey> {
    record
        .catalog_profiles
        .iter()
        .chain(&record.universe)
        .copied()
        .collect()
}

fn require_program_rows(
    record: &SetRecord,
    program: &ProgramRows,
    set_limits: (usize, usize),
) -> Result<(), AkitaError> {
    if !record.admits_program_rows {
        return Err(AkitaError::InvalidSetup(format!(
            "parameter set {:?} does not admit program rows",
            record.name
        )));
    }
    if program.max_num_vars > set_limits.0 || program.max_num_batched_polys > set_limits.1 {
        return Err(AkitaError::InvalidSetup(
            "program bounds exceed the set limits".into(),
        ));
    }
    crate::validate_setup_capacity_metadata(program.max_num_vars, program.max_num_batched_polys)
}

/// Program bounds widened to the widest precommitted group of the catalog
/// identity, so no row with a precommitted group wider than the trace is
/// dropped.
fn program_bounds<Cfg: CommitmentConfig>(
    record: &SetRecord,
    program: &ProgramRows,
    set_limits: (usize, usize),
    identity: &TrustedScheduleCatalog<Cfg>,
) -> Result<(usize, usize), AkitaError> {
    let widest_precommitted = identity
        .rows()
        .flat_map(|row| &row.profiles().precommitteds)
        .map(|profile| profile.group.num_vars())
        .max()
        .unwrap_or(0);
    let max_num_vars = program.max_num_vars.max(widest_precommitted);
    if max_num_vars > set_limits.0 {
        return Err(AkitaError::InvalidSetup(format!(
            "a precommitted group of set {:?} is wider than its variable limit",
            record.name
        )));
    }
    Ok((max_num_vars, program.max_num_batched_polys))
}

fn row_fits(
    profiles: &CommittedGroupBatchProfile,
    bounds: (usize, usize),
) -> Result<bool, AkitaError> {
    ScheduleLookupKey {
        final_group: profiles.final_group.group,
        precommitteds: profiles.precommitteds.clone(),
    }
    .fits_setup_capacity(bounds.0, bounds.1)
}

fn check_producer_profiles(
    program: &ProgramRows,
    profiles: &CommittedGroupBatchProfile,
) -> Result<(), AkitaError> {
    if profiles
        .precommitteds
        .iter()
        .all(|profile| program.producer_profiles.contains(profile))
    {
        Ok(())
    } else {
        Err(AkitaError::InvalidSetup(
            "program row names a precommitted profile outside its anchored producer sets".into(),
        ))
    }
}

fn check_merged_digest<Cfg: CommitmentConfig>(
    program: &ProgramRows,
    merged: &TrustedScheduleCatalog<Cfg>,
) -> Result<(), AkitaError> {
    match program.merged_catalog_digest {
        Some(digest) if digest != merged.catalog_digest() => Err(AkitaError::InvalidSetup(
            "merged catalog does not have the digest stored in the program preprocessing".into(),
        )),
        _ => Ok(()),
    }
}
