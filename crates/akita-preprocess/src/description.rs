//! The hand-written description a release is generated from.

use akita_error::AkitaError;
use akita_types::preprocessing::{parse_digest_hex, validate_name, ReleaseKind};
use akita_types::AkitaSetupSeed;
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// The only description format this generator reads.
pub const DESCRIPTION_FORMAT_VERSION: u32 = 1;

/// File name of a description inside `params/<release>/`.
pub const DESCRIPTION_FILE: &str = "description.json";

/// File name of the checked-in manifest inside `params/<release>/`.
pub const MANIFEST_FILE: &str = "manifest";

/// A release description, as written under `params/<release>/`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseDescription {
    /// Must be [`DESCRIPTION_FORMAT_VERSION`].
    pub format_version: u32,
    /// Release name; the directory under `params/` carries the same name.
    pub name: String,
    /// `"akita"` or `"jolt"`.
    pub kind: String,
    /// Generator identity recorded in the manifest.
    pub generator_version: String,
    /// 64 lowercase hex digits: the `shake256-paged-v1` seed of the matrix.
    pub seed: String,
    /// Guest targets the release provides files for.
    #[serde(default)]
    pub targets: Vec<String>,
    /// Largest stored part of one file.
    pub max_file_part: u64,
    /// Publish prover hints for the CPU reference prover.
    #[serde(default)]
    pub hints: bool,
    /// Publish the page hashes of every stream.
    #[serde(default)]
    pub page_hashes: bool,
    /// Complete page ranges `[first, last]` of the matrix to publish.
    #[serde(default)]
    pub matrix_pages: Vec<[u64; 2]>,
    /// Parameter sets.
    pub sets: Vec<SetDescription>,
}

/// One parameter set.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetDescription {
    /// Anchored set name.
    pub name: String,
    /// `CommitmentConfig::schedule_family_name()` of a compiled config.
    pub config: String,
    /// Catalog artifact, relative to the repository root.
    pub catalog: PathBuf,
    /// Witness chunk count.
    pub num_chunks: usize,
    /// Leading fold levels priced under the chunked layout.
    pub num_activated_levels: usize,
    /// Largest supported polynomial variable count.
    pub max_num_vars: usize,
    /// Largest supported polynomial count per opening batch.
    pub max_num_batched_polys: usize,
    /// Present when the set admits program rows.
    #[serde(default)]
    pub program_rows: Option<ProgramRowsDescription>,
}

/// The profile universe of a set that admits program rows.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramRowsDescription {
    /// Files, relative to the repository root, listing one setup-prefix
    /// slot id per line as hex of its uncompressed encoding. Their profiles
    /// join the catalog's in the universe.
    #[serde(default)]
    pub universe_slots: Vec<PathBuf>,
}

impl ReleaseDescription {
    /// Read and validate `params/<release>/description.json`.
    ///
    /// # Errors
    ///
    /// Returns an error for unreadable, malformed, or invalid descriptions.
    pub fn read(path: &Path) -> Result<Self, AkitaError> {
        let bytes = std::fs::read(path).map_err(|error| {
            AkitaError::InvalidInput(format!("cannot read {}: {error}", path.display()))
        })?;
        let description: Self = serde_json::from_slice(&bytes).map_err(|error| {
            AkitaError::InvalidInput(format!("invalid description {}: {error}", path.display()))
        })?;
        description.validate()?;
        Ok(description)
    }

    fn validate(&self) -> Result<(), AkitaError> {
        if self.format_version != DESCRIPTION_FORMAT_VERSION {
            return Err(AkitaError::InvalidInput(format!(
                "unsupported description format {}",
                self.format_version
            )));
        }
        validate_name(&self.name)?;
        self.kind()?;
        self.setup_seed()?;
        for target in &self.targets {
            validate_name(target)?;
        }
        if self.sets.is_empty() {
            return Err(AkitaError::InvalidInput(
                "a release has at least one set".into(),
            ));
        }
        for set in &self.sets {
            validate_name(&set.name)?;
        }
        Ok(())
    }

    /// The release kind.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown kind.
    pub fn kind(&self) -> Result<ReleaseKind, AkitaError> {
        match self.kind.as_str() {
            "akita" => Ok(ReleaseKind::Akita),
            "jolt" => Ok(ReleaseKind::Jolt),
            other => Err(AkitaError::InvalidInput(format!(
                "unknown release kind {other:?}"
            ))),
        }
    }

    /// The seed of every stream of the release.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed hex.
    pub fn setup_seed(&self) -> Result<AkitaSetupSeed, AkitaError> {
        Ok(AkitaSetupSeed::shake256_paged_v1(parse_digest_hex(
            &self.seed,
        )?))
    }
}
