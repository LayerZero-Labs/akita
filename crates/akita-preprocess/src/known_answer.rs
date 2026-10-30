//! Known-answer corpus: pins the behaviour of the compiled commitment code.
//!
//! The corpus is a versioned list of small setup-prefix profiles, one per
//! corpus class, checked in at `corpus/known-answer-v1.txt`. Committing each
//! over a fixed test stream and hashing the results gives the per-field
//! known-answer digest compiled into
//! [`akita_types::preprocessing::known_answer_digest`]. A kernel or encoding
//! change that alters any output without a version bump changes the digest;
//! extending the corpus is a code transition.

use akita_config::preprocessing::PrefixCommitmentDeriver;
use akita_cpu_backend::CpuPrefixDeriver;
use akita_error::AkitaError;
use akita_params::descriptor_bytes::sis_modulus_profile_tag;
use akita_params::sis::{InnerCommitSecurityRoute, SisModulusProfileId};
use akita_params::{ChunkedWitnessCfg, SetupPrefixSlotId};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
use akita_types::preprocessing::{
    hex, known_answer_corpus_digest, parse_hex, profile_descriptor, setup_prefix_payload,
    StreamIdentity,
};
use akita_types::AkitaSetupSeed;
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// The checked-in corpus: `<SIS modulus profile tag> <hex slot id>` lines.
pub const CORPUS: &str = include_str!("../corpus/known-answer-v1.txt");

/// Seed of the fixed test stream every corpus entry is committed over.
pub const TEST_SEED: [u8; 32] = *b"akita/preprocessing/known-answer";

/// Catalog families whose setup-prefix profiles the corpus covers.
pub const CORPUS_FAMILIES: &[&str] = &[
    "fp32_dense_recursive",
    "fp64_dense_recursive",
    "fp128_dense_recursive",
    "fp128_onehot_recursive",
    "fp128_onehot_recursive_multi_chunk_w2r3",
    "fp128_onehot_recursive_multi_chunk_w4r3",
    "fp128_onehot_recursive_multi_chunk_w8r2",
    "fp128_onehot_recursive_multi_chunk_w8r3",
];

/// What a corpus entry stands for: every feature that selects a distinct
/// commitment code path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CorpusClass {
    /// SIS modulus profile tag (the field).
    pub field: u8,
    /// A ring dimension.
    pub a_ring_dimension: usize,
    /// B ring dimension.
    pub b_ring_dimension: usize,
    /// Inner route: 0 for L-infinity, 1 for L2. The outer route is always
    /// L-infinity.
    pub inner_route: u8,
    /// Logical B inputs per physical B matrix.
    pub slice_count: usize,
    /// Compression maps in the outer compression chain.
    pub compression_chain_len: usize,
    /// Whether the source spans several blocks.
    pub multi_block: bool,
}

/// The corpus class of a setup-prefix slot.
///
/// # Errors
///
/// Returns an error when the profile has no compression chain.
pub fn corpus_class(id: &SetupPrefixSlotId) -> Result<CorpusClass, AkitaError> {
    let profile = &id.commitment_profile;
    Ok(CorpusClass {
        field: sis_modulus_profile_tag(profile.inner.matrix.sis_modulus_profile()),
        a_ring_dimension: profile.inner.matrix.ring_dimension(),
        b_ring_dimension: profile.outer.matrix.ring_dimension(),
        inner_route: match profile.inner.matrix.security_route() {
            InnerCommitSecurityRoute::Linf(_) => 0,
            InnerCommitSecurityRoute::L2 { .. } => 1,
        },
        slice_count: profile.outer_slice_count.get(),
        compression_chain_len: profile.outer_compression_plan()?.maps().len(),
        multi_block: profile.blocks.live_blocks > 1,
    })
}

/// The decoded corpus, in file order.
///
/// # Errors
///
/// Returns an error for a malformed line.
pub fn corpus_entries() -> Result<Vec<(SisModulusProfileId, SetupPrefixSlotId)>, AkitaError> {
    CORPUS
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let (tag, slot) = line
                .split_once(' ')
                .ok_or_else(|| AkitaError::InvalidInput(format!("bad corpus line {line:?}")))?;
            let field = match tag {
                "0" => SisModulusProfileId::Q32Offset99,
                "1" => SisModulusProfileId::Q64Offset59,
                "2" => SisModulusProfileId::Q128OffsetA7F7,
                _ => {
                    return Err(AkitaError::InvalidInput(format!(
                        "bad corpus field {tag:?}"
                    )))
                }
            };
            let id = SetupPrefixSlotId::deserialize_uncompressed_exact(&parse_hex(slot)?, &())
                .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
            id.check()
                .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
            Ok((field, id))
        })
        .collect()
}

/// Recompute the known-answer digest of `field` from the corpus with the
/// compiled commitment code.
///
/// # Errors
///
/// Returns an error when a corpus entry cannot be committed.
pub fn compute_known_answer_digest(field: SisModulusProfileId) -> Result<[u8; 32], AkitaError> {
    let ids = corpus_entries()?
        .into_iter()
        .filter(|(entry_field, _)| *entry_field == field)
        .map(|(_, id)| id)
        .collect::<Vec<_>>();
    crate::dispatch_field!(field, |F| digest_over::<F>(&ids))
}

fn digest_over<F>(ids: &[SetupPrefixSlotId]) -> Result<[u8; 32], AkitaError>
where
    F: Field
        + CanonicalEncoding
        + Unreduced
        + WithCommitAccumulator
        + Valid
        + AkitaSerialize
        + 'static,
{
    let seed = AkitaSetupSeed::shake256_paged_v1(TEST_SEED);
    let mut entries = Vec::with_capacity(ids.len());
    for id in ids {
        // One at a time, so every entry is committed independently of the
        // others' export order.
        let commitment = PrefixCommitmentDeriver::<F>::derive_prefix_commitments(
            &CpuPrefixDeriver,
            &seed,
            std::slice::from_ref(id),
        )?
        .pop()
        .ok_or_else(|| AkitaError::Internal("corpus commitment is missing".into()))?;
        entries.push((
            profile_descriptor(&id.commitment_profile),
            setup_prefix_payload(&commitment)?,
        ));
    }
    Ok(known_answer_corpus_digest(
        &StreamIdentity::for_field::<F>(&seed)?,
        &entries,
    ))
}

/// Build a corpus from the checked-in catalogs of [`CORPUS_FAMILIES`]: per
/// class, the slot with the smallest prefix.
///
/// # Errors
///
/// Returns an error when a catalog cannot be read.
pub fn build_corpus(repo_root: &Path) -> Result<String, AkitaError> {
    let mut by_class = BTreeMap::<CorpusClass, SetupPrefixSlotId>::new();
    for family in CORPUS_FAMILIES {
        let path = repo_root
            .join("artifacts/schedules")
            .join(format!("{family}.aks"));
        let bytes = std::fs::read(&path).map_err(|error| {
            AkitaError::InvalidInput(format!("cannot read {}: {error}", path.display()))
        })?;
        let chunking = family_chunking(family);
        let ids =
            crate::dispatch_config!(*family, |Cfg| catalog_slot_ids::<Cfg>(&bytes, chunking))?;
        for id in ids {
            let class = corpus_class(&id)?;
            let smaller = by_class.get(&class).is_none_or(|kept| {
                (id.n_prefix().unwrap_or(usize::MAX), &id)
                    < (kept.n_prefix().unwrap_or(usize::MAX), kept)
            });
            if smaller {
                by_class.insert(class, id);
            }
        }
    }
    let mut out = String::from(
        "# Known-answer corpus v1: <SIS modulus profile tag> <uncompressed slot id hex>.\n\
         # Generated by `akita-preprocess known-answer --write-corpus`; one slot per class.\n",
    );
    for (class, id) in by_class {
        let mut encoded = Vec::new();
        id.serialize_uncompressed(&mut encoded)
            .map_err(|error| AkitaError::Internal(error.to_string()))?;
        out.push_str(&format!("{} {}\n", class.field, hex(&encoded)));
    }
    Ok(out)
}

/// Witness chunking of a catalog family, from its `w<chunks>r<levels>` name.
fn family_chunking(family: &str) -> ChunkedWitnessCfg {
    if let Some(profile) = family
        .rsplit('_')
        .next()
        .filter(|tail| tail.starts_with('w'))
    {
        if let Some((chunks, levels)) = profile[1..].split_once('r') {
            if let (Ok(num_chunks), Ok(num_activated_levels)) = (chunks.parse(), levels.parse()) {
                return ChunkedWitnessCfg {
                    num_chunks,
                    num_activated_levels,
                };
            }
        }
    }
    if family.contains("multi_chunk") {
        return ChunkedWitnessCfg::d64_production();
    }
    ChunkedWitnessCfg::default_non_chunked()
}

fn catalog_slot_ids<Cfg: akita_config::CommitmentConfig>(
    bytes: &[u8],
    chunking: ChunkedWitnessCfg,
) -> Result<Vec<SetupPrefixSlotId>, AkitaError> {
    let catalog =
        akita_config::TrustedScheduleCatalog::<Cfg>::from_artifact_bytes(bytes, chunking)?;
    let mut ids = Vec::new();
    for row in catalog.rows() {
        ids.extend(akita_config::required_setup_prefix_slot_ids_for_schedule(
            row.schedule(),
            &row.profiles().opening_layout()?,
        )?);
    }
    Ok(ids)
}

/// Require every profile of every release under `params_dir` to fall in a
/// corpus class.
///
/// # Errors
///
/// Returns an error naming the first profile with no corpus class.
pub fn check_corpus_coverage(repo_root: &Path, params_dir: &Path) -> Result<(), AkitaError> {
    let classes = corpus_entries()?
        .iter()
        .map(|(_, id)| corpus_class(id))
        .collect::<Result<BTreeSet<_>, _>>()?;
    require_classes(&classes, &crate::release_profiles(repo_root, params_dir)?)
}

fn require_classes(
    classes: &BTreeSet<CorpusClass>,
    ids: &[SetupPrefixSlotId],
) -> Result<(), AkitaError> {
    for id in ids {
        let class = corpus_class(id)?;
        if !classes.contains(&class) {
            return Err(AkitaError::InvalidSetup(format!(
                "known-answer corpus has no class {class:?}; extend the corpus (a code transition)"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on_large_stack(test: fn()) {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(test)
            .expect("spawn")
            .join()
            .expect("test thread");
    }

    #[test]
    fn known_answer_digests_match_the_compiled_constants() {
        on_large_stack(|| {
            for field in [
                SisModulusProfileId::Q32Offset99,
                SisModulusProfileId::Q64Offset59,
                SisModulusProfileId::Q128OffsetA7F7,
            ] {
                assert_eq!(
                    hex(&compute_known_answer_digest(field).unwrap()),
                    hex(&akita_types::preprocessing::known_answer_digest(field)),
                    "{field:?}: the commitment code changed without a version bump"
                );
            }
        });
    }

    #[test]
    fn corpus_holds_one_entry_per_class_and_covers_every_anchored_profile() {
        on_large_stack(|| {
            let entries = corpus_entries().unwrap();
            let classes = entries
                .iter()
                .map(|(_, id)| corpus_class(id).unwrap())
                .collect::<BTreeSet<_>>();
            assert_eq!(classes.len(), entries.len());
            let root = crate::tests::repo_root();
            let profiles = crate::release_profiles(&root, &root.join("params")).unwrap();
            assert!(!profiles.is_empty());
            require_classes(&classes, &profiles).unwrap();
            assert!(require_classes(&BTreeSet::new(), &profiles).is_err());
        });
    }

    #[test]
    fn checked_in_corpus_is_the_one_the_catalogs_build() {
        on_large_stack(|| {
            assert_eq!(build_corpus(&crate::tests::repo_root()).unwrap(), CORPUS);
        });
    }
}
