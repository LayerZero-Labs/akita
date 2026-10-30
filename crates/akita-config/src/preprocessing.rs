//! Release checks shared by the prover and verifier loaders.
//!
//! A loader opens a manifest with
//! [`akita_types::preprocessing::open_anchored_manifest`], opens a set
//! against its compiled configuration with [`open_set`], admits rows with
//! [`Admission`], and loads the commitments the admitted rows use with
//! [`load_prefix_commitments`], from registry files or derived from the seed.
//! Prover and verifier run exactly this code, so they size their setups from
//! the same rows.

mod admission;
mod commitments;

pub use admission::{Admission, ProgramRows};
pub use commitments::{
    check_prefix_commitment, load_prefix_commitments, PrefixCommitmentDeriver,
    PrefixCommitmentSource,
};

use crate::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_params::descriptor_bytes::sis_modulus_profile_tag;
use akita_params::sis::SisTableDigest;
use akita_params::{ChunkedWitnessCfg, SisL2TableDigest};
use akita_types::decomposition_descriptor_bytes;
use akita_types::preprocessing::{
    catalog_file_name, read_release_file, AlgorithmIdentity, AnchoredManifest, ReleaseFileSource,
    SetRecord, StreamEntry, StreamIdentity,
};

/// Set-record fields that are a function of the compiled configuration.
///
/// A loader compares them with its own configuration before it uses any file
/// (Invariant 5); limits, catalog identity, and profile lists are release data
/// authenticated through the manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetRecordCodeFields {
    /// `Cfg::schedule_family_name()`.
    pub config_family: String,
    /// `SetupSection` decomposition bytes.
    pub decomposition: Vec<u8>,
    /// SIS modulus profile tag.
    pub sis_modulus_profile_tag: u8,
    /// L-infinity SIS table digest.
    pub sis_table_digest: [u8; 32],
    /// L2 SIS table digest.
    pub sis_l2_table_digest: [u8; 32],
    /// Compiled commitment algorithm.
    pub algorithm_identity: AlgorithmIdentity,
}

/// The code-derived set-record fields of `Cfg`.
pub fn set_record_code_fields<Cfg: CommitmentConfig>() -> SetRecordCodeFields {
    SetRecordCodeFields {
        config_family: Cfg::schedule_family_name().to_owned(),
        decomposition: decomposition_descriptor_bytes(&Cfg::decomposition()),
        sis_modulus_profile_tag: sis_modulus_profile_tag(Cfg::sis_modulus_profile()),
        sis_table_digest: SisTableDigest::CURRENT.0,
        sis_l2_table_digest: SisL2TableDigest::CURRENT.0,
        algorithm_identity: AlgorithmIdentity::compiled(Cfg::sis_modulus_profile()),
    }
}

/// Open an anchored set for `Cfg`: its record, with the code-derived fields
/// required to equal `Cfg`'s (Invariant 5), and its stream, required to be a
/// stream of `Cfg`'s base field.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when the set was not anchored, a
/// code-derived field differs, or the stream is unknown or over another
/// field.
pub fn open_set<'a, Cfg: CommitmentConfig>(
    release: &'a AnchoredManifest,
    set_name: &str,
) -> Result<(&'a SetRecord, &'a StreamEntry), AkitaError> {
    let record = release.set(set_name)?;
    check_set_record_code::<Cfg>(record)?;
    Ok((record, set_stream::<Cfg>(release, record)?))
}

fn check_set_record_code<Cfg: CommitmentConfig>(record: &SetRecord) -> Result<(), AkitaError> {
    let code = set_record_code_fields::<Cfg>();
    let mismatch = if record.config_family != code.config_family {
        Some("config family")
    } else if record.decomposition != code.decomposition {
        Some("decomposition")
    } else if record.sis_modulus_profile_tag != code.sis_modulus_profile_tag {
        Some("SIS modulus profile")
    } else if record.sis_table_digest != code.sis_table_digest {
        Some("SIS table digest")
    } else if record.sis_l2_table_digest != code.sis_l2_table_digest {
        Some("SIS L2 table digest")
    } else if record.algorithm_identity != code.algorithm_identity {
        Some("commitment algorithm")
    } else {
        None
    };
    match mismatch {
        None => Ok(()),
        Some(field) => Err(AkitaError::InvalidSetup(format!(
            "parameter set {:?} does not match the compiled configuration: {field} differs",
            record.name
        ))),
    }
}

/// The witness chunking a set record declares.
///
/// # Errors
///
/// Returns an error when a count does not fit `usize` or the layout is
/// invalid.
pub fn set_chunking(record: &SetRecord) -> Result<ChunkedWitnessCfg, AkitaError> {
    let chunking = ChunkedWitnessCfg {
        num_chunks: usize::try_from(record.num_chunks)
            .map_err(|_| AkitaError::InvalidSetup("set chunk count overflows usize".into()))?,
        num_activated_levels: usize::try_from(record.num_activated_levels).map_err(|_| {
            AkitaError::InvalidSetup("set activated level count overflows usize".into())
        })?,
    };
    chunking.validate()?;
    Ok(chunking)
}

fn set_stream<'a, Cfg: CommitmentConfig>(
    release: &'a AnchoredManifest,
    record: &SetRecord,
) -> Result<&'a StreamEntry, AkitaError> {
    let stream = release
        .manifest()
        .stream(&record.stream_id)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!("set {:?} names an unknown stream", record.name))
        })?;
    let expected =
        StreamIdentity::for_field::<Cfg::Field>(stream.identity.setup_seed())?.to_bytes();
    if stream.identity.to_bytes() != expected {
        return Err(AkitaError::InvalidSetup(format!(
            "set {:?} names a stream over another field",
            record.name
        )));
    }
    Ok(stream)
}

/// Load and authenticate a set's base catalog.
///
/// The catalog file is checked against the manifest, decoded against the
/// compiled configuration and the set's chunking, and required to have the
/// catalog digest the set record names.
///
/// # Errors
///
/// Returns an error when the file or its digest does not match.
pub fn load_set_catalog<Cfg: CommitmentConfig>(
    release: &AnchoredManifest,
    record: &SetRecord,
    files: &dyn ReleaseFileSource,
) -> Result<TrustedScheduleCatalog<Cfg>, AkitaError> {
    let bytes = read_release_file(
        files,
        release.manifest(),
        &catalog_file_name(&record.name),
        crate::MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES,
    )?;
    let catalog =
        TrustedScheduleCatalog::<Cfg>::from_artifact_bytes(&bytes, set_chunking(record)?)?;
    if catalog.catalog_digest() != record.catalog_digest {
        return Err(AkitaError::InvalidSetup(format!(
            "catalog of set {:?} does not have the digest its set record names",
            record.name
        )));
    }
    Ok(catalog)
}

/// Load a set's selected-row verifier view of `row`, as a guest does instead
/// of loading the complete catalog. [`Admission::admit_view`] checks that it
/// reproduces the catalog identity.
///
/// # Errors
///
/// Returns an error when the file does not match the manifest.
pub fn load_set_catalog_view<Cfg: CommitmentConfig>(
    release: &AnchoredManifest,
    record: &SetRecord,
    files: &dyn ReleaseFileSource,
    row: akita_params::ScheduleRowDigest,
) -> Result<TrustedScheduleCatalog<Cfg>, AkitaError> {
    let name = akita_types::preprocessing::catalog_view_file_name(&record.name, &row);
    let bytes = read_release_file(
        files,
        release.manifest(),
        &name,
        crate::MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES,
    )?;
    TrustedScheduleCatalog::<Cfg>::from_verifier_view(&bytes, set_chunking(record)?)
}
