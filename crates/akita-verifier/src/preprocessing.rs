//! Verifier loader: builds pinned verifier state from an anchored release.
//!
//! A production verifier is built only from what this loader issues
//! ([`PinnedVerifierSetup`], [`PinnedTerminalCache`]). The loader opens the
//! manifest under the binary's trust anchor (see
//! [`akita_types::preprocessing::open_anchored_manifest`]), compares the set
//! record with the compiled configuration, admits rows (A1), sizes and
//! derives the matrix (A2, A3), loads the commitments the admitted rows use,
//! checks admission again on the result (A4), and issues the pinned value.

use crate::prepared_cache::TrustedTerminalCache;
use akita_config::preprocessing::{
    check_set_record_code, load_prefix_commitments, load_set_catalog, load_set_catalog_view,
    set_chunking, set_stream, Admission, CommitmentSource, ProgramRows,
};
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_params::{FlatMatrix, OpeningScheduleSelection, ScheduleRowDigest};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, TrustedBytes, Valid};
use akita_types::preprocessing::{
    profile_key, read_release_file, terminal_cache_file_name, AnchoredManifest, BundleId,
    ReleaseFileSource, SetRecord,
};
use akita_types::{
    AkitaExpandedSetup, AkitaVerifierSetup, SetupPrefixVerifierRegistry, SetupPrefixVerifierSlot,
    MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS, PREPARED_VERIFIER_NTT_CACHE_MAX_BYTES,
};
use std::sync::Arc;

/// Verifier setup and catalog issued by the loader from an anchored release.
///
/// Its fields are private and it has no public constructor and no
/// deserializer: every commitment in it matched the anchored manifest, its
/// matrix was derived from the release's seed (or, in a guest, embedded in an
/// approved image), and its catalog is the catalog identity of the admitted
/// rows, or that identity's selected-row view for a single-row load.
///
/// ```compile_fail
/// use akita_verifier::PinnedVerifierSetup;
/// use akita_config::proof_optimized::fp128::OneHot;
/// fn forge(
///     setup: akita_types::AkitaVerifierSetup<akita_config::proof_optimized::fp128::Field>,
///     catalog: akita_config::TrustedScheduleCatalog<OneHot>,
///     bundle_id: akita_types::preprocessing::BundleId,
/// ) -> PinnedVerifierSetup<OneHot> {
///     PinnedVerifierSetup { setup, catalog, selection: None, bundle_id, set_name: String::new() }
/// }
/// ```
///
/// ```compile_fail
/// use akita_verifier::AkitaVerifier;
/// use akita_config::proof_optimized::fp128::OneHot;
/// fn build(setup: akita_types::AkitaVerifierSetup<akita_config::proof_optimized::fp128::Field>) {
///     let _ = AkitaVerifier::<OneHot>::from_pinned(setup);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct PinnedVerifierSetup<Cfg: CommitmentConfig> {
    setup: AkitaVerifierSetup<Cfg::Field>,
    catalog: TrustedScheduleCatalog<Cfg>,
    selection: Option<OpeningScheduleSelection>,
    bundle_id: BundleId,
    set_name: String,
}

impl<Cfg: CommitmentConfig> PinnedVerifierSetup<Cfg> {
    /// The verifier setup.
    pub fn setup(&self) -> &AkitaVerifierSetup<Cfg::Field> {
        &self.setup
    }

    /// The catalog identity, or its selected-row view for a single-row load.
    pub fn catalog(&self) -> &TrustedScheduleCatalog<Cfg> {
        &self.catalog
    }

    /// The selected row of a single-row load.
    pub fn selection(&self) -> Option<OpeningScheduleSelection> {
        self.selection
    }

    /// The release this setup was loaded from.
    pub fn bundle_id(&self) -> BundleId {
        self.bundle_id
    }

    /// The anchored parameter set this setup was loaded for.
    pub fn set_name(&self) -> &str {
        &self.set_name
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        AkitaVerifierSetup<Cfg::Field>,
        TrustedScheduleCatalog<Cfg>,
        Option<OpeningScheduleSelection>,
    ) {
        (self.setup, self.catalog, self.selection)
    }
}

/// A terminal NTT cache issued by the loader for one pinned single-row
/// setup: a release file checked against the manifest, or guest-embedded
/// bytes under [`Self::embedded_unchecked`].
#[derive(Debug, Clone)]
pub struct PinnedTerminalCache {
    bytes: TerminalCacheBytes,
    row: ScheduleRowDigest,
}

#[derive(Debug, Clone)]
enum TerminalCacheBytes {
    Owned(Vec<u8>),
    Embedded(&'static [u8]),
}

impl PinnedTerminalCache {
    /// Issue an embedded cache for `row` without hashing it, used in place.
    ///
    /// Guest-only: hashing a large embedded file may cost too many cycles.
    /// Integrations deny this method outside their guest crate with
    /// `clippy::disallowed_methods`.
    ///
    /// # Safety
    ///
    /// Soundness contract: `bytes` are embedded in an approved guest image
    /// and either match the release manifest (a release file) or equal the
    /// guest build step's derivation from the anchored release (a program
    /// row's cache). The body must be aligned for in-place use.
    #[must_use]
    pub unsafe fn embedded_unchecked(row: ScheduleRowDigest, bytes: &'static [u8]) -> Self {
        Self {
            bytes: TerminalCacheBytes::Embedded(bytes),
            row,
        }
    }

    /// The schedule row this cache serves.
    pub fn row(&self) -> ScheduleRowDigest {
        self.row
    }

    pub(crate) fn as_trusted(&self) -> TrustedTerminalCache<'_> {
        match &self.bytes {
            TerminalCacheBytes::Owned(bytes) => TrustedTerminalCache::Decode(bytes),
            TerminalCacheBytes::Embedded(bytes) => TrustedTerminalCache::View(bytes),
        }
    }
}

/// A public matrix prefix embedded in a guest image, which a guest loader
/// takes instead of deriving it from the seed.
#[derive(Debug, Clone, Copy)]
pub struct PinnedMatrixPrefix {
    bytes: TrustedBytes,
}

impl PinnedMatrixPrefix {
    /// Issue an embedded matrix prefix: an uncompressed `FlatMatrix`
    /// encoding, viewed in place.
    ///
    /// Guest-only, like [`PinnedTerminalCache::embedded_unchecked`].
    ///
    /// # Safety
    ///
    /// Soundness contract: the bytes are embedded in an approved guest image
    /// and are the prefix of the public matrix derived from the release's
    /// seed, checked at approval against the derivation or `page-hashes`.
    #[must_use]
    pub unsafe fn embedded_unchecked(bytes: TrustedBytes) -> Self {
        Self { bytes }
    }
}

/// How a guest takes its single-row catalog view.
#[derive(Debug, Clone, Copy)]
pub enum GuestCatalogView<'a> {
    /// The release's `catalog-view/<set>/<row>` file.
    Release,
    /// The build step's selected-row view of a program's merged catalog,
    /// checked against the merged catalog digest of the program rows.
    Merged(&'a [u8]),
}

/// Full load: a verifier for every row the set admits.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when the set record does not match
/// `Cfg`, a file or commitment does not match the manifest, no row is
/// admitted, or the derived setup does not admit exactly the admitted rows.
#[tracing::instrument(skip_all, name = "load_verifier_setup", fields(set = set_name))]
pub fn load_verifier_setup<Cfg>(
    release: &AnchoredManifest,
    files: &dyn ReleaseFileSource,
    set_name: &str,
    program: Option<&ProgramRows>,
    source: CommitmentSource<'_, Cfg::Field>,
) -> Result<PinnedVerifierSetup<Cfg>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Valid + AkitaSerialize + AkitaDeserialize<Context = ()>,
{
    let record = checked_record::<Cfg>(release, set_name)?;
    let admission = Admission::admit(
        record,
        load_set_catalog::<Cfg>(release, record, files)?,
        program,
    )?;
    issue(release, record, files, &admission, None, source, None)
}

/// Single-row load on a host: a verifier for `selection` only, which must
/// be one of the set's admitted rows. Its catalog is the selected-row view
/// of the catalog identity.
///
/// # Errors
///
/// As [`load_verifier_setup`], and when `selection` is not admitted.
#[tracing::instrument(skip_all, name = "load_selected_verifier_setup", fields(set = set_name))]
pub fn load_selected_verifier_setup<Cfg>(
    release: &AnchoredManifest,
    files: &dyn ReleaseFileSource,
    set_name: &str,
    selection: OpeningScheduleSelection,
    program: Option<&ProgramRows>,
    source: CommitmentSource<'_, Cfg::Field>,
) -> Result<PinnedVerifierSetup<Cfg>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Valid + AkitaSerialize + AkitaDeserialize<Context = ()>,
{
    let record = checked_record::<Cfg>(release, set_name)?;
    let admission = Admission::admit(
        record,
        load_set_catalog::<Cfg>(release, record, files)?,
        program,
    )?;
    issue(
        release,
        record,
        files,
        &admission,
        Some(selection),
        source,
        None,
    )
}

/// Single-row load in a guest, from small embedded files.
///
/// The catalog view and registry entries are checked like any release file
/// (registry mode only; a guest never derives commitments), and A3 takes the
/// embedded matrix prefix, whose shape must be exactly the admitted row's
/// verifier capacity.
///
/// # Errors
///
/// As [`load_selected_verifier_setup`], and when the view or matrix does not
/// match.
pub fn load_guest_verifier_setup<Cfg>(
    release: &AnchoredManifest,
    files: &dyn ReleaseFileSource,
    set_name: &str,
    selection: OpeningScheduleSelection,
    program: Option<&ProgramRows>,
    view: GuestCatalogView<'_>,
    matrix: PinnedMatrixPrefix,
) -> Result<PinnedVerifierSetup<Cfg>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Valid + AkitaSerialize + AkitaDeserialize<Context = ()> + 'static,
{
    let record = checked_record::<Cfg>(release, set_name)?;
    let view = match (view, program) {
        (GuestCatalogView::Release, None) => {
            load_set_catalog_view::<Cfg>(release, record, files, selection.row_digest)?
        }
        (GuestCatalogView::Merged(bytes), Some(_)) => {
            TrustedScheduleCatalog::<Cfg>::from_verifier_view(bytes, set_chunking(record)?)?
        }
        _ => {
            return Err(AkitaError::InvalidSetup(
                "a guest with program rows takes the merged view, and only then".into(),
            ))
        }
    };
    let admission = Admission::admit_view(record, view, selection, program)?;
    issue(
        release,
        record,
        files,
        &admission,
        Some(selection),
        CommitmentSource::Registry,
        Some(matrix),
    )
}

/// Load the release's terminal cache for a pinned single-row setup and the
/// guest target `target`.
///
/// # Errors
///
/// Returns an error when the setup is not a single-row load from this
/// release, the release has no such target or file, or the file does not
/// match the manifest.
pub fn load_terminal_cache<Cfg: CommitmentConfig>(
    release: &AnchoredManifest,
    files: &dyn ReleaseFileSource,
    pinned: &PinnedVerifierSetup<Cfg>,
    target: &str,
) -> Result<PinnedTerminalCache, AkitaError> {
    let selection = pinned.selection.ok_or_else(|| {
        AkitaError::InvalidSetup("a terminal cache serves a single-row load".into())
    })?;
    if release.bundle_id() != pinned.bundle_id {
        return Err(AkitaError::InvalidSetup(
            "every file of one verifier comes from one release".into(),
        ));
    }
    if !release
        .manifest()
        .targets
        .iter()
        .any(|known| known == target)
    {
        return Err(AkitaError::InvalidSetup(format!(
            "release provides no files for target {target:?}"
        )));
    }
    let record = release.set(&pinned.set_name)?;
    let name = terminal_cache_file_name(
        &record.stream_id,
        &selection.row_digest,
        target,
        pinned.setup.expanded().descriptor().num_field_elements,
    );
    Ok(PinnedTerminalCache {
        bytes: TerminalCacheBytes::Owned(read_release_file(
            files,
            release.manifest(),
            &name,
            PREPARED_VERIFIER_NTT_CACHE_MAX_BYTES,
        )?),
        row: selection.row_digest,
    })
}

fn checked_record<'a, Cfg: CommitmentConfig>(
    release: &'a AnchoredManifest,
    set_name: &str,
) -> Result<&'a SetRecord, AkitaError> {
    let record = release.set(set_name)?;
    check_set_record_code::<Cfg>(record)?;
    Ok(record)
}

fn issue<Cfg>(
    release: &AnchoredManifest,
    record: &SetRecord,
    files: &dyn ReleaseFileSource,
    admission: &Admission<Cfg>,
    selection: Option<OpeningScheduleSelection>,
    source: CommitmentSource<'_, Cfg::Field>,
    embedded_matrix: Option<PinnedMatrixPrefix>,
) -> Result<PinnedVerifierSetup<Cfg>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Valid + AkitaSerialize + AkitaDeserialize<Context = ()>,
{
    let stream = set_stream::<Cfg>(release, record)?;
    let setup_seed = stream.identity.setup_seed();
    let descriptor =
        admission.setup_descriptor(setup_seed, admission.verifier_matrix_capacity(selection)?);
    let expanded = match embedded_matrix {
        None => AkitaExpandedSetup::derive(descriptor)?,
        Some(matrix) => {
            let (shared_matrix, rest) = FlatMatrix::borrow_trusted_with_expected_shape(
                matrix.bytes,
                descriptor.num_field_elements,
                MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
            )
            .map_err(|error| AkitaError::InvalidSetup(format!("embedded matrix: {error}")))?;
            if !rest.bytes().is_empty() {
                return Err(AkitaError::InvalidSetup(
                    "embedded matrix is longer than the admitted row's capacity".into(),
                ));
            }
            // SAFETY: the contract of `PinnedMatrixPrefix::embedded_unchecked`:
            // approval checked the bytes against the seed's derivation.
            unsafe {
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    descriptor,
                    shared_matrix,
                )
            }
        }
    };
    let ids = admission.required_slot_ids(selection)?;
    let commitments = load_prefix_commitments::<Cfg>(release, record, files, &ids, source)?;
    let mut registry = SetupPrefixVerifierRegistry::new(setup_seed.clone());
    for id in ids {
        let commitment = commitments[&profile_key(&id.commitment_profile)].clone();
        registry.insert(SetupPrefixVerifierSlot { id, commitment })?;
    }
    // SAFETY: the matrix is derived from the release's seed (or embedded in
    // an approved image), and every commitment matched the anchored
    // manifest's prefix commitment digest.
    let setup = unsafe { AkitaVerifierSetup::from_parts(Arc::new(expanded), registry)? };
    admission.check_verifier_setup(setup.expanded(), selection)?;
    let catalog = match selection {
        Some(selection) if admission.catalog().validate_complete().is_ok() => {
            admission.selected_view(selection)?
        }
        _ => admission.catalog().clone(),
    };
    Ok(PinnedVerifierSetup {
        setup,
        catalog,
        selection,
        bundle_id: release.bundle_id(),
        set_name: record.name.clone(),
    })
}

/// A verifier setup whose commitments the host computed itself, for a
/// custom deployment whose rows need profiles outside every anchored
/// universe. There is no manifest digest to compare with.
///
/// Produced only by a host-only crate downstream of `akita-cpu-backend` and
/// `akita-planner`, and accepted only by the `unsafe`
/// [`crate::AkitaVerifier::from_locally_derived`]. Never available in a
/// guest.
#[derive(Debug, Clone)]
pub struct LocallyDerivedVerifierSetup<F: jolt_field::Field> {
    setup: AkitaVerifierSetup<F>,
}

impl<F: jolt_field::Field> LocallyDerivedVerifierSetup<F> {
    /// Wrap a setup derived in this process.
    ///
    /// # Safety
    ///
    /// Soundness contract: the caller derived the matrix from its seed and
    /// computed every commitment from that matrix itself; it imported no
    /// commitment from a prover or a file.
    #[doc(hidden)]
    #[must_use]
    pub unsafe fn assume_locally_derived(setup: AkitaVerifierSetup<F>) -> Self {
        Self { setup }
    }

    pub(crate) fn into_setup(self) -> AkitaVerifierSetup<F> {
        self.setup
    }

    /// The verifier setup.
    pub fn setup(&self) -> &AkitaVerifierSetup<F> {
        &self.setup
    }
}
