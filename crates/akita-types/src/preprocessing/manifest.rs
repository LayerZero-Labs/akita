//! Release manifest: set records, commitment registry, and file hashes.

use super::encoding::{malformed, push_bytes, push_str, push_u32, push_u64, push_usize, Reader};
use super::files::{validate_file_name, validate_name, MAX_FILE_NAME_BYTES, MAX_NAME_BYTES};
use super::identity::{
    AlgorithmIdentity, BundleId, FileHash, PrefixCommitmentDigest, ProfileKey, StreamId,
    StreamIdentity,
};
use akita_error::AkitaError;
use akita_params::descriptor_bytes::DigestHash;
use akita_params::DescriptorHasher;

const MANIFEST_TAG: &str = "akita/preprocessing/manifest/v1";
const BUNDLE_TAG: &str = "akita/preprocessing/bundle/v1";

/// The only manifest format this code reads and writes.
pub const MANIFEST_FORMAT_VERSION: u32 = 1;

/// Largest manifest accepted, checked before hashing or decoding.
///
/// The cap leaves headroom over the largest known release; exceeding it is a
/// code transition.
pub const MAX_MANIFEST_BYTES: usize = 16 << 20;

const MAX_LIST_ITEMS: usize = 1 << 20;
const MAX_VERSION_BYTES: usize = 256;
const MAX_DECOMPOSITION_BYTES: usize = 64;

const fn digest_hash_tag(hash: DigestHash) -> u8 {
    match hash {
        DigestHash::Sha3_256 => 0,
        DigestHash::Keccak256 => 1,
    }
}

fn digest_hash_from_tag(tag: u8) -> Result<DigestHash, AkitaError> {
    match tag {
        0 => Ok(DigestHash::Sha3_256),
        1 => Ok(DigestHash::Keccak256),
        _ => Err(malformed("unknown digest hash")),
    }
}

/// Which integration a release is published for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseKind {
    /// Standalone Akita.
    Akita,
    /// Jolt's Akita backend.
    Jolt,
}

impl ReleaseKind {
    const fn tag(self) -> u8 {
        match self {
            Self::Akita => 0,
            Self::Jolt => 1,
        }
    }

    fn from_tag(tag: u8) -> Result<Self, AkitaError> {
        match tag {
            0 => Ok(Self::Akita),
            1 => Ok(Self::Jolt),
            _ => Err(malformed("unknown release kind")),
        }
    }
}

/// One public field stream covered by the release.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamEntry {
    /// Field, seed, and page size.
    pub identity: StreamIdentity,
    /// Field elements of the stream the release covers: the largest
    /// capacity any set or profile needs.
    pub element_count: u64,
}

/// One published setup-prefix commitment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegistryEntry {
    /// Stream the commitment is computed over.
    pub stream_id: StreamId,
    /// Profile the commitment is computed for.
    pub profile_key: ProfileKey,
    /// Digest binding payload, stream, algorithm, and profile.
    pub prefix_commitment_digest: PrefixCommitmentDigest,
}

/// One release file, named by hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    /// File name relative to the release root.
    pub name: String,
    /// Exact byte length.
    pub length: u64,
    /// `H(tag || str(name) || bytes(content))`.
    pub file_hash: FileHash,
}

/// Everything a loader needs to accept or reject one parameter set before it
/// reads any other file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetRecord {
    /// Set name, as anchored.
    pub name: String,
    /// `CommitmentConfig::schedule_family_name()`.
    pub config_family: String,
    /// The bytes `SetupSection` encodes for the decomposition.
    pub decomposition: Vec<u8>,
    /// SIS modulus profile tag.
    pub sis_modulus_profile_tag: u8,
    /// L-infinity SIS table digest.
    pub sis_table_digest: [u8; 32],
    /// L2 SIS table digest.
    pub sis_l2_table_digest: [u8; 32],
    /// Commitment algorithm of every registry entry the set uses.
    pub algorithm_identity: AlgorithmIdentity,
    /// The set's public field stream.
    pub stream_id: StreamId,
    /// Witness chunk count.
    pub num_chunks: u64,
    /// Leading fold levels priced under the chunked layout.
    pub num_activated_levels: u64,
    /// Largest supported polynomial variable count.
    pub max_num_vars: u64,
    /// Largest supported polynomial count per opening batch.
    pub max_num_batched_polys: u64,
    /// `catalog_digest()` of the set's base catalog.
    pub catalog_digest: [u8; 32],
    /// Profiles the base catalog's rows within the limits use, sorted.
    pub catalog_profiles: Vec<ProfileKey>,
    /// Whether the set admits program rows.
    pub admits_program_rows: bool,
    /// Profiles program rows may use, sorted; empty without program rows.
    pub universe: Vec<ProfileKey>,
}

impl SetRecord {
    /// Encode the record.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_str(&mut out, &self.name);
        push_str(&mut out, &self.config_family);
        push_bytes(&mut out, &self.decomposition);
        out.push(self.sis_modulus_profile_tag);
        out.extend_from_slice(&self.sis_table_digest);
        out.extend_from_slice(&self.sis_l2_table_digest);
        out.extend_from_slice(&self.algorithm_identity.to_bytes());
        out.extend_from_slice(self.stream_id.as_bytes());
        push_u64(&mut out, self.num_chunks);
        push_u64(&mut out, self.num_activated_levels);
        push_u64(&mut out, self.max_num_vars);
        push_u64(&mut out, self.max_num_batched_polys);
        out.extend_from_slice(&self.catalog_digest);
        push_profiles(&mut out, &self.catalog_profiles);
        out.push(u8::from(self.admits_program_rows));
        push_profiles(&mut out, &self.universe);
        out
    }

    fn decode(bytes: &[u8]) -> Result<Self, AkitaError> {
        let mut reader = Reader::new(bytes);
        let name = reader.str(MAX_NAME_BYTES, "set name")?.to_owned();
        let config_family = reader.str(MAX_NAME_BYTES, "config family")?.to_owned();
        let decomposition = reader
            .bytes(MAX_DECOMPOSITION_BYTES, "decomposition")?
            .to_vec();
        let sis_modulus_profile_tag = reader.u8("SIS modulus profile")?;
        let sis_table_digest = reader.digest("SIS table digest")?;
        let sis_l2_table_digest = reader.digest("SIS L2 table digest")?;
        let algorithm_identity = AlgorithmIdentity::decode(&mut reader)?;
        let stream_id = StreamId::from_bytes(reader.digest("set stream")?);
        let num_chunks = reader.u64("num_chunks")?;
        let num_activated_levels = reader.u64("num_activated_levels")?;
        let max_num_vars = reader.u64("max_num_vars")?;
        let max_num_batched_polys = reader.u64("max_num_batched_polys")?;
        let catalog_digest = reader.digest("catalog digest")?;
        let catalog_profiles = read_profiles(&mut reader, "catalog profiles")?;
        let admits_program_rows = reader.flag("program-row flag")?;
        let universe = read_profiles(&mut reader, "profile universe")?;
        reader.finish("set record")?;
        Ok(Self {
            name,
            config_family,
            decomposition,
            sis_modulus_profile_tag,
            sis_table_digest,
            sis_l2_table_digest,
            algorithm_identity,
            stream_id,
            num_chunks,
            num_activated_levels,
            max_num_vars,
            max_num_batched_polys,
            catalog_digest,
            catalog_profiles,
            admits_program_rows,
            universe,
        })
    }
}

fn push_profiles(out: &mut Vec<u8>, profiles: &[ProfileKey]) {
    push_usize(out, profiles.len());
    for profile in profiles {
        out.extend_from_slice(profile.as_bytes());
    }
}

fn read_profiles(reader: &mut Reader<'_>, what: &str) -> Result<Vec<ProfileKey>, AkitaError> {
    let count = reader.count(MAX_LIST_ITEMS, 32, what)?;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(ProfileKey::from_bytes(reader.digest(what)?));
    }
    Ok(out)
}

fn require_strictly_sorted<T, K: Ord>(
    items: &[T],
    key: impl Fn(&T) -> K,
    what: &str,
) -> Result<(), AkitaError> {
    if items.windows(2).all(|pair| key(&pair[0]) < key(&pair[1])) {
        Ok(())
    } else {
        Err(malformed(&format!("{what} are not strictly sorted")))
    }
}

/// The decoded manifest of one release.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseManifest {
    /// Hash of every identity in the release: SHA3-256 or Keccak-256. A
    /// release is valid for one selection only.
    pub digest_hash: DigestHash,
    /// Release name.
    pub release_name: String,
    /// Integration the release is published for.
    pub kind: ReleaseKind,
    /// Generator that produced the release.
    pub generator_version: String,
    /// Guest targets the release provides files for, sorted.
    pub targets: Vec<String>,
    /// Largest stored part of one file, in bytes.
    pub max_file_part: u64,
    /// Streams, sorted by stream id.
    pub streams: Vec<StreamEntry>,
    /// Commitments, sorted by `(stream_id, profile_key)`.
    pub registry: Vec<RegistryEntry>,
    /// Parameter sets, sorted by name.
    pub sets: Vec<SetRecord>,
    /// Files, sorted by name. The manifest does not list itself.
    pub files: Vec<FileEntry>,
}

impl ReleaseManifest {
    /// Encode the manifest, rejecting unsorted or duplicate entries.
    ///
    /// # Errors
    ///
    /// Returns an error when a list is not strictly sorted, a name is invalid,
    /// or the encoding exceeds [`MAX_MANIFEST_BYTES`].
    pub fn to_bytes(&self) -> Result<Vec<u8>, AkitaError> {
        self.validate()?;
        let mut out = Vec::new();
        push_str(&mut out, MANIFEST_TAG);
        push_u32(&mut out, MANIFEST_FORMAT_VERSION);
        out.push(digest_hash_tag(self.digest_hash));
        push_str(&mut out, &self.release_name);
        out.push(self.kind.tag());
        push_str(&mut out, &self.generator_version);
        push_usize(&mut out, self.targets.len());
        for target in &self.targets {
            push_str(&mut out, target);
        }
        push_u64(&mut out, self.max_file_part);
        push_usize(&mut out, self.streams.len());
        for stream in &self.streams {
            out.extend_from_slice(&stream.identity.to_bytes());
            push_u64(&mut out, stream.element_count);
        }
        push_usize(&mut out, self.registry.len());
        for entry in &self.registry {
            out.extend_from_slice(entry.stream_id.as_bytes());
            out.extend_from_slice(entry.profile_key.as_bytes());
            out.extend_from_slice(entry.prefix_commitment_digest.as_bytes());
        }
        push_usize(&mut out, self.sets.len());
        for set in &self.sets {
            push_bytes(&mut out, &set.to_bytes());
        }
        push_usize(&mut out, self.files.len());
        for file in &self.files {
            push_str(&mut out, &file.name);
            push_u64(&mut out, file.length);
            out.extend_from_slice(file.file_hash.as_bytes());
        }
        if out.len() > MAX_MANIFEST_BYTES {
            return Err(AkitaError::InvalidSetup(format!(
                "preprocessing manifest of {} bytes exceeds the {MAX_MANIFEST_BYTES}-byte cap",
                out.len()
            )));
        }
        Ok(out)
    }

    /// Decode a manifest. Decoding is total and bounded: any input that is
    /// over the cap, malformed, unsorted, or not canonical is rejected.
    ///
    /// Call this only on bytes whose [`bundle_id`] an anchor already accepted.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] for any invalid input.
    pub fn decode(bytes: &[u8]) -> Result<Self, AkitaError> {
        if bytes.len() > MAX_MANIFEST_BYTES {
            return Err(malformed("manifest exceeds the length cap"));
        }
        let mut reader = Reader::new(bytes);
        if reader.str(MANIFEST_TAG.len(), "manifest tag")? != MANIFEST_TAG {
            return Err(malformed("manifest tag"));
        }
        if reader.u32("format version")? != MANIFEST_FORMAT_VERSION {
            return Err(malformed("unknown format version"));
        }
        let digest_hash = digest_hash_from_tag(reader.u8("digest hash")?)?;
        let release_name = reader.str(MAX_NAME_BYTES, "release name")?.to_owned();
        let kind = ReleaseKind::from_tag(reader.u8("release kind")?)?;
        let generator_version = reader
            .str(MAX_VERSION_BYTES, "generator version")?
            .to_owned();
        let target_count = reader.count(MAX_LIST_ITEMS, 9, "targets")?;
        let mut targets = Vec::with_capacity(target_count);
        for _ in 0..target_count {
            targets.push(reader.str(MAX_NAME_BYTES, "target")?.to_owned());
        }
        let max_file_part = reader.u64("max file part")?;
        let stream_count =
            reader.count(MAX_LIST_ITEMS, StreamIdentity::ENCODED_LEN + 8, "streams")?;
        let mut streams = Vec::with_capacity(stream_count);
        for _ in 0..stream_count {
            streams.push(StreamEntry {
                identity: StreamIdentity::decode(&mut reader)?,
                element_count: reader.u64("stream element count")?,
            });
        }
        let registry_count = reader.count(MAX_LIST_ITEMS, 96, "registry")?;
        let mut registry = Vec::with_capacity(registry_count);
        for _ in 0..registry_count {
            registry.push(RegistryEntry {
                stream_id: StreamId::from_bytes(reader.digest("registry stream")?),
                profile_key: ProfileKey::from_bytes(reader.digest("registry profile")?),
                prefix_commitment_digest: PrefixCommitmentDigest::from_bytes(
                    reader.digest("registry commitment")?,
                ),
            });
        }
        let set_count = reader.count(MAX_LIST_ITEMS, 8, "sets")?;
        let mut sets = Vec::new();
        for _ in 0..set_count {
            sets.push(SetRecord::decode(
                reader.bytes(MAX_MANIFEST_BYTES, "set record")?,
            )?);
        }
        let file_count = reader.count(MAX_LIST_ITEMS, 9 + 8 + 32, "files")?;
        let mut files = Vec::with_capacity(file_count);
        for _ in 0..file_count {
            files.push(FileEntry {
                name: reader.str(MAX_FILE_NAME_BYTES, "file name")?.to_owned(),
                length: reader.u64("file length")?,
                file_hash: FileHash::from_bytes(reader.digest("file hash")?),
            });
        }
        reader.finish("manifest")?;
        let manifest = Self {
            digest_hash,
            release_name,
            kind,
            generator_version,
            targets,
            max_file_part,
            streams,
            registry,
            sets,
            files,
        };
        if manifest.to_bytes()? != bytes {
            return Err(malformed("manifest is not canonical"));
        }
        Ok(manifest)
    }

    fn validate(&self) -> Result<(), AkitaError> {
        validate_name(&self.release_name)?;
        if self.generator_version.is_empty() || self.generator_version.len() > MAX_VERSION_BYTES {
            return Err(malformed("generator version"));
        }
        for target in &self.targets {
            validate_name(target)?;
        }
        if self.max_file_part == 0 {
            return Err(malformed("max file part is zero"));
        }
        require_strictly_sorted(&self.targets, |target| target.clone(), "targets")?;
        require_strictly_sorted(&self.streams, |entry| entry.identity.stream_id(), "streams")?;
        require_strictly_sorted(
            &self.registry,
            |entry| (entry.stream_id, entry.profile_key),
            "registry entries",
        )?;
        require_strictly_sorted(&self.sets, |set| set.name.clone(), "set records")?;
        require_strictly_sorted(&self.files, |file| file.name.clone(), "files")?;
        for set in &self.sets {
            validate_name(&set.name)?;
            if set.config_family.is_empty() || set.config_family.len() > MAX_NAME_BYTES {
                return Err(malformed("config family"));
            }
            require_strictly_sorted(&set.catalog_profiles, |key| *key, "catalog profiles")?;
            require_strictly_sorted(&set.universe, |key| *key, "profile universe")?;
            if !set.admits_program_rows && !set.universe.is_empty() {
                return Err(malformed("profile universe without program rows"));
            }
        }
        for file in &self.files {
            validate_file_name(&file.name)?;
        }
        Ok(())
    }

    /// The set record named `name`.
    #[must_use]
    pub fn set(&self, name: &str) -> Option<&SetRecord> {
        self.sets
            .binary_search_by(|set| set.name.as_str().cmp(name))
            .ok()
            .map(|index| &self.sets[index])
    }

    /// The file entry named `name`.
    #[must_use]
    pub fn file(&self, name: &str) -> Option<&FileEntry> {
        self.files
            .binary_search_by(|file| file.name.as_str().cmp(name))
            .ok()
            .map(|index| &self.files[index])
    }

    /// The stream with id `stream_id`.
    #[must_use]
    pub fn stream(&self, stream_id: &StreamId) -> Option<&StreamEntry> {
        self.streams
            .iter()
            .find(|entry| &entry.identity.stream_id() == stream_id)
    }

    /// The registry entry for `(stream_id, profile_key)`.
    #[must_use]
    pub fn registry_entry(
        &self,
        stream_id: &StreamId,
        profile_key: &ProfileKey,
    ) -> Option<&RegistryEntry> {
        self.registry
            .binary_search_by(|entry| {
                (entry.stream_id, entry.profile_key).cmp(&(*stream_id, *profile_key))
            })
            .ok()
            .map(|index| &self.registry[index])
    }
}

/// `H(tag("akita/preprocessing/bundle/v1") || manifest)`, over the exact
/// manifest bytes. Computed before the manifest is decoded.
#[must_use]
pub fn bundle_id(manifest: &[u8]) -> BundleId {
    let mut tag = Vec::with_capacity(BUNDLE_TAG.len() + 8);
    push_str(&mut tag, BUNDLE_TAG);
    let mut hasher = DescriptorHasher::new();
    hasher.update(&tag);
    hasher.update(manifest);
    BundleId::from_bytes(hasher.finalize())
}
