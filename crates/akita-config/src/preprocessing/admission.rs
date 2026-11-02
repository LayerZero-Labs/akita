//! Row admission: which rows a loader admits and how large a setup it derives.
//!
//! - **A1** admits rows against the set limits, or against a program's bounds,
//!   and fixes the catalog identity. A program row's precommitted groups must
//!   carry the frozen profiles of their anchored producer sets.
//! - **A2** sizes the matrix from the admitted rows only.
//! - **A3** derives that prefix and describes the setup with the admitted
//!   bounds. With program rows, the variable bound is the larger of the trace
//!   bound and the widest precommitted group of the merged catalog.
//! - **A4** checks that the resulting verifier setup admits exactly the A1
//!   rows (or the selected row).

use super::{load_set_catalog, open_set, set_chunking};
use crate::setup_prefix_slots::required_setup_prefix_slot_ids_for_schedule;
use crate::{
    policy_of, row_fits_setup_capacity, CommitmentConfig, SetupRequirements, TrustedScheduleCatalog,
};
use akita_error::AkitaError;
use akita_params::{
    verifier_setup_matrix_capacity_for_schedule, CommittedGroupBatchProfile, FoldSchedule,
    GroupCommitPhaseParams, OpeningScheduleSelection, PolynomialGroupLayout, SetupMatrixCapacity,
    SetupPrefixSlotId,
};
use akita_schedules::{ResolvedScheduleRow, ValidatedScheduleCatalog};
use akita_types::preprocessing::{
    profile_key, AnchoredManifest, BundleId, ProfileKey, ReleaseFileSource, SetRecord,
};
use akita_types::{AkitaExpandedSetup, AkitaSetupDescriptor, AkitaSetupSeed};
use std::collections::{BTreeMap, BTreeSet};

/// What one program brings to a loader: its bounds, which select the rows a
/// loader admits, and the rows it planned for itself. Both are authenticated
/// by that program's preprocessing digest; they are the one loader input
/// that is not a release file.
///
/// A set without program rows takes only the bounds (`rows` empty), so a
/// small program never derives the setup of the set's largest row.
#[derive(Debug, Clone)]
pub struct ProgramRows {
    /// Program trace bound on polynomial variables. Trace length is not known
    /// before execution, so rows of every trace length within it are
    /// admitted.
    pub max_num_vars: usize,
    /// Program bound on polynomials per opening batch; at most the set limit.
    pub max_num_batched_polys: usize,
    /// Grouped rows planned for this program, in any order.
    pub rows: Vec<ProgramRow>,
    /// Merged catalog digest stored in the program preprocessing, once it
    /// exists. Every loader must reproduce it.
    pub merged_catalog_digest: Option<[u8; 32]>,
    /// Frozen profiles of the anchored producer sets the rows name.
    pub producers: ProducerProfiles,
}

/// One grouped row a program planned for itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramRow {
    /// The row's committed group profiles.
    pub profiles: CommittedGroupBatchProfile,
    /// The row's fold schedule.
    pub schedule: FoldSchedule,
    /// The anchored producer set of each precommitted group, in the order
    /// of `profiles.precommitteds`.
    pub producer_sets: Vec<String>,
}

/// The frozen group profiles of anchored producer sets: for each set loaded
/// with [`Self::load`], the final-group profile of every single-group row
/// the set admits, by group layout.
///
/// A program row's precommitted group must carry exactly the frozen profile
/// of its layout in the producer set the row names, so a program cannot
/// introduce a precommitted profile that no anchored set publishes.
#[derive(Debug, Clone)]
pub struct ProducerProfiles {
    bundle_id: BundleId,
    sets: BTreeMap<String, BTreeMap<(usize, usize), GroupCommitPhaseParams>>,
}

impl ProducerProfiles {
    /// No producer set yet, for producer sets of `release`.
    pub fn new(release: &AnchoredManifest) -> Self {
        Self {
            bundle_id: release.bundle_id(),
            sets: BTreeMap::new(),
        }
    }

    /// Load the producer set `set_name` of the same release: open it for
    /// `Cfg`, authenticate its base catalog, admit its rows within the set
    /// limits, and record the final-group profile of each single-group row.
    ///
    /// # Errors
    ///
    /// Returns an error when the release differs, the set does not open for
    /// `Cfg`, its catalog does not match, or no row is admitted.
    pub fn load<Cfg: CommitmentConfig>(
        &mut self,
        release: &AnchoredManifest,
        files: &dyn ReleaseFileSource,
        set_name: &str,
    ) -> Result<(), AkitaError> {
        if release.bundle_id() != self.bundle_id {
            return Err(AkitaError::InvalidSetup(
                "every producer set of one program comes from one release".into(),
            ));
        }
        let (record, _) = open_set::<Cfg>(release, set_name)?;
        let admission = Admission::admit(
            release,
            record,
            load_set_catalog::<Cfg>(release, record, files)?,
            None,
        )?;
        let mut profiles = BTreeMap::new();
        for selection in &admission.rows {
            let row = admission.catalog.resolve_selection(*selection)?;
            if row.profiles().precommitteds.is_empty() {
                let profile = row.profiles().final_group;
                profiles.insert(layout_key(&profile), profile);
            }
        }
        self.sets.insert(set_name.to_owned(), profiles);
        Ok(())
    }

    /// The frozen profile of `layout` in the loaded producer set `set_name`,
    /// as a program planning its rows must use it.
    ///
    /// # Errors
    ///
    /// Returns an error when the set was not loaded or admits no
    /// single-group row of that layout.
    pub fn frozen_profile(
        &self,
        set_name: &str,
        layout: PolynomialGroupLayout,
    ) -> Result<GroupCommitPhaseParams, AkitaError> {
        self.sets
            .get(set_name)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!("producer set {set_name:?} was not loaded"))
            })?
            .get(&(layout.num_vars(), layout.num_polynomials()))
            .copied()
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "producer set {set_name:?} admits no group of {} variables and {} polynomials",
                    layout.num_vars(),
                    layout.num_polynomials()
                ))
            })
    }

    /// Require every precommitted group of every row to carry the frozen
    /// profile of its layout in the producer set the row names.
    fn check(&self, release: &AnchoredManifest, rows: &[ProgramRow]) -> Result<(), AkitaError> {
        if release.bundle_id() != self.bundle_id {
            return Err(AkitaError::InvalidSetup(
                "producer sets come from another release".into(),
            ));
        }
        for row in rows {
            if row.producer_sets.len() != row.profiles.precommitteds.len() {
                return Err(AkitaError::InvalidSetup(
                    "a program row names one producer set per precommitted group".into(),
                ));
            }
            for (set, profile) in row.producer_sets.iter().zip(&row.profiles.precommitteds) {
                let frozen = self
                    .sets
                    .get(set)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(format!(
                            "program row names producer set {set:?}, which was not loaded"
                        ))
                    })?
                    .get(&layout_key(profile));
                if frozen != Some(profile) {
                    return Err(AkitaError::InvalidSetup(format!(
                        "a precommitted group profile differs from the frozen profile of its layout in producer set {set:?}"
                    )));
                }
            }
        }
        Ok(())
    }
}

fn layout_key(profile: &GroupCommitPhaseParams) -> (usize, usize) {
    (profile.group.num_vars(), profile.group.num_polynomials())
}

/// Admitted rows of one parameter set and the catalog identity they fix.
#[derive(Debug, Clone)]
pub struct Admission<Cfg: CommitmentConfig> {
    catalog: TrustedScheduleCatalog<Cfg>,
    rows: Vec<OpeningScheduleSelection>,
    max_num_vars: usize,
    max_num_batched_polys: usize,
    /// Setup-prefix profiles admitted rows may use; `None` for local profile
    /// derivation, where the caller derives every commitment itself.
    allowed_profiles: Option<BTreeSet<ProfileKey>>,
}

impl<Cfg: CommitmentConfig> Admission<Cfg> {
    /// A1 over a complete base catalog of `release` that already matched the
    /// set record.
    ///
    /// Without program rows the base catalog is the catalog identity, and its
    /// rows within the set limits, or within the program bounds when given,
    /// are admitted. With program rows the catalog identity is the merged
    /// catalog over every base row and every program row (ordered by row
    /// digest, so independent of the order rows are supplied in), and its
    /// rows within the program bounds are admitted. Every precommitted group
    /// of a program row must carry the frozen profile of its producer set.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when no row is admitted, a program
    /// bound exceeds a set limit, a set without program rows is given some, a
    /// precommitted profile differs from its producer set's, an admitted row
    /// uses a profile outside the set's universe, or the catalog identity
    /// does not have the stored digest.
    pub fn admit(
        release: &AnchoredManifest,
        record: &SetRecord,
        base: TrustedScheduleCatalog<Cfg>,
        program: Option<&ProgramRows>,
    ) -> Result<Self, AkitaError> {
        let mut admission = Self::admit_any_profile(release, record, base, program)?;
        admission.allowed_profiles = Some(if program.is_some() && record.admits_program_rows {
            universe(record)
        } else {
            record.catalog_profiles.iter().copied().collect()
        });
        admission.required_slot_ids(None)?;
        Ok(admission)
    }

    /// A1 for local profile derivation: as [`Self::admit`], except that
    /// admitted rows may use setup-prefix profiles outside the set's
    /// universe. The release publishes no commitment for those profiles, so
    /// no loader issues a setup from this admission; only a host that derives
    /// every commitment itself uses it (`LocallyDerivedVerifierSetup` in
    /// `akita-verifier`).
    ///
    /// # Errors
    ///
    /// As [`Self::admit`], except for the universe check.
    pub fn admit_any_profile(
        release: &AnchoredManifest,
        record: &SetRecord,
        base: TrustedScheduleCatalog<Cfg>,
        program: Option<&ProgramRows>,
    ) -> Result<Self, AkitaError> {
        let set_limits = set_limits(record)?;
        let Some(program) = program else {
            return Self::admit_base(record, base, set_limits);
        };
        check_program_bounds(program, set_limits)?;
        if !record.admits_program_rows {
            require_no_program_rows(record, program)?;
            if program
                .merged_catalog_digest
                .is_some_and(|digest| digest != base.catalog_digest())
            {
                return Err(AkitaError::InvalidSetup(
                    "base catalog does not have the digest stored in the program preprocessing"
                        .into(),
                ));
            }
            return Self::admit_base(
                record,
                base,
                (program.max_num_vars, program.max_num_batched_polys),
            );
        }
        program.producers.check(release, &program.rows)?;
        let chunking = set_chunking(record)?;
        let merged = ValidatedScheduleCatalog::try_new(
            Cfg::schedule_family_name(),
            base.rows()
                .map(|row| (row.profiles().clone(), row.schedule().clone()))
                .chain(
                    program
                        .rows
                        .iter()
                        .map(|row| (row.profiles.clone(), row.schedule.clone())),
                ),
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
        Self::finish(record, merged, rows, bounds, None)
    }

    /// The base catalog as the catalog identity, with its rows within
    /// `bounds` admitted.
    fn admit_base(
        record: &SetRecord,
        base: TrustedScheduleCatalog<Cfg>,
        bounds: (usize, usize),
    ) -> Result<Self, AkitaError> {
        let rows = base
            .rows_within_setup_capacity(bounds.0, bounds.1)?
            .into_iter()
            .map(ResolvedScheduleRow::selection)
            .collect();
        Self::finish(record, base, rows, bounds, None)
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
        if let Some(program) = program {
            check_program_bounds(program, set_limits)?;
            if !record.admits_program_rows {
                require_no_program_rows(record, program)?;
            }
        }
        let expected_digest = match program {
            Some(program) if record.admits_program_rows => {
                program.merged_catalog_digest.ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "a single-row load with program rows needs the stored merged catalog digest"
                            .into(),
                    )
                })?
            }
            _ => record.catalog_digest,
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
            Some(program) if !record.admits_program_rows => (
                (program.max_num_vars, program.max_num_batched_polys),
                record.catalog_profiles.iter().copied().collect(),
            ),
            Some(program) => (
                program_bounds(record, program, set_limits, &view)?,
                universe(record),
            ),
        };
        let row = view.resolve_selection(selection)?;
        if !row_fits_setup_capacity(row, bounds.0, bounds.1)? {
            return Err(AkitaError::InvalidSetup(
                "selected row is outside the admitted rows".into(),
            ));
        }
        Self::finish(record, view, vec![selection], bounds, Some(allowed))
    }

    fn finish(
        record: &SetRecord,
        catalog: TrustedScheduleCatalog<Cfg>,
        mut rows: Vec<OpeningScheduleSelection>,
        bounds: (usize, usize),
        allowed_profiles: Option<BTreeSet<ProfileKey>>,
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

    /// The admitted row `selection`.
    fn admitted_row(
        &self,
        selection: OpeningScheduleSelection,
    ) -> Result<&ResolvedScheduleRow, AkitaError> {
        if self.rows.binary_search(&selection).is_err() {
            return Err(AkitaError::InvalidSetup(
                "selected row is outside the admitted rows".into(),
            ));
        }
        self.catalog.resolve_selection(selection)
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
            Some(selection) => Ok(vec![self.admitted_row(selection)?]),
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
                if self
                    .allowed_profiles
                    .as_ref()
                    .is_some_and(|allowed| !allowed.contains(&profile_key(&id.commitment_profile)))
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
            Some(selection) => TrustedScheduleCatalog::<Cfg>::verifier_admits(
                setup,
                self.admitted_row(selection)?,
            )?,
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
        self.admitted_row(selection)?;
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

fn require_no_program_rows(record: &SetRecord, program: &ProgramRows) -> Result<(), AkitaError> {
    if program.rows.is_empty() {
        Ok(())
    } else {
        Err(AkitaError::InvalidSetup(format!(
            "parameter set {:?} does not admit program rows",
            record.name
        )))
    }
}

fn check_program_bounds(
    program: &ProgramRows,
    set_limits: (usize, usize),
) -> Result<(), AkitaError> {
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
