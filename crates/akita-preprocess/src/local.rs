//! Local profile derivation (opt-in, host only).
//!
//! A custom deployment whose program rows need setup-prefix profiles outside
//! every anchored universe can have both sides compute the missing
//! commitments themselves. This module is the verifier's producer of a
//! [`LocallyDerivedVerifierSetup`]: it admits rows exactly as a loader does,
//! except for the universe check, and derives the matrix and every
//! commitment from the release's seed in this process. Unlike a loader's
//! seed-derived mode, there is no manifest digest to compare a commitment
//! with, so the result is a distinct type that only
//! [`akita_verifier::AkitaVerifier::from_locally_derived`] accepts.

use akita_config::preprocessing::{
    load_set_catalog, open_set, Admission, PrefixCommitmentDeriver, ProgramRows,
};
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_cpu_backend::CpuPrefixDeriver;
use akita_error::AkitaError;
use akita_serialization::Valid;
use akita_types::preprocessing::{AnchoredManifest, ReleaseFileSource};
use akita_types::{
    AkitaExpandedSetup, AkitaVerifierSetup, SetupPrefixVerifierRegistry, SetupPrefixVerifierSlot,
};
use akita_verifier::preprocessing::LocallyDerivedVerifierSetup;
use jolt_field::{CanonicalEncoding, Unreduced, WithCommitAccumulator};
use std::sync::Arc;

/// Derive the verifier setup of an anchored set's rows, including program
/// rows whose profiles lie outside the set's universe, and the catalog
/// identity it verifies.
///
/// Rows are admitted by [`Admission::admit_any_profile`] (A1), the matrix is
/// sized from them (A2) and derived from the release's seed (A3), every
/// commitment the rows use is computed from the seed by the CPU reference
/// path, and the result must admit exactly the admitted rows (A4).
///
/// # Errors
///
/// Returns an error when the set does not open for `Cfg`, its catalog does
/// not match the manifest, admission fails, or a commitment cannot be
/// derived.
pub fn derive_local_verifier_setup<Cfg>(
    release: &AnchoredManifest,
    files: &dyn ReleaseFileSource,
    set_name: &str,
    program: Option<&ProgramRows>,
) -> Result<
    (
        LocallyDerivedVerifierSetup<Cfg::Field>,
        TrustedScheduleCatalog<Cfg>,
    ),
    AkitaError,
>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding + Unreduced + WithCommitAccumulator + Valid + 'static,
{
    let (record, stream) = open_set::<Cfg>(release, set_name)?;
    let admission = Admission::admit_any_profile(
        release,
        record,
        load_set_catalog::<Cfg>(release, record, files)?,
        program,
    )?;
    let setup_seed = stream.identity.setup_seed();
    let ids = admission.required_slot_ids(None)?;
    let commitments = CpuPrefixDeriver.derive_prefix_commitments(setup_seed, &ids)?;
    let mut registry = SetupPrefixVerifierRegistry::new(setup_seed.clone());
    for (id, commitment) in ids.into_iter().zip(commitments) {
        registry.insert(SetupPrefixVerifierSlot { id, commitment })?;
    }
    let expanded = AkitaExpandedSetup::derive(admission.setup_descriptor(
        setup_seed,
        admission.verifier_matrix_capacity(None, akita_params::SetupMode::Check)?,
    ))?;
    // SAFETY: the matrix and every commitment were derived here from the
    // release's seed.
    let setup = unsafe { AkitaVerifierSetup::from_parts(Arc::new(expanded), registry)? };
    admission.check_verifier_setup(setup.expanded(), None, akita_params::SetupMode::Check)?;
    // SAFETY: as above; no commitment was imported.
    let setup = unsafe { LocallyDerivedVerifierSetup::assume_locally_derived(setup) };
    Ok((setup, admission.catalog().clone()))
}
