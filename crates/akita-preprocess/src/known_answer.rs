//! Known-answer corpus: pins the behaviour of the compiled commitment code.
//!
//! The corpus is a versioned list of small setup-prefix slot ids, one per
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
use akita_params::sis::SisModulusProfileId;
use akita_params::SetupPrefixSlotId;
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::preprocessing::{
    hex, known_answer_corpus_digest, profile_descriptor, profile_route, setup_prefix_payload,
    StreamIdentity,
};
use akita_types::AkitaSetupSeed;
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::collections::{BTreeMap, BTreeSet};

/// Where the corpus is checked in, relative to the repository root.
const CORPUS_PATH: &str = "crates/akita-preprocess/corpus/known-answer-v1.txt";

/// The checked-in corpus: one hex-encoded uncompressed slot id per line.
const CORPUS: &str = include_str!("../corpus/known-answer-v1.txt");

/// Seed of the fixed test stream every corpus entry is committed over.
const TEST_SEED: [u8; 32] = *b"akita/preprocessing/known-answer";

/// What a corpus entry stands for: every feature that selects a distinct
/// commitment code path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CorpusClass {
    /// SIS modulus profile tag (the field).
    field: u8,
    a_ring_dimension: usize,
    b_ring_dimension: usize,
    /// Inner security route; the outer route is always L-infinity.
    inner_route: u8,
    /// Logical B inputs per physical B matrix.
    slice_count: usize,
    /// Compression maps in the outer compression chain.
    compression_chain_len: usize,
    /// Whether the source spans several blocks.
    multi_block: bool,
}

fn corpus_class(id: &SetupPrefixSlotId) -> Result<CorpusClass, AkitaError> {
    let profile = &id.commitment_profile;
    Ok(CorpusClass {
        field: sis_modulus_profile_tag(profile.inner.matrix.sis_modulus_profile()),
        a_ring_dimension: profile.inner.matrix.ring_dimension(),
        b_ring_dimension: profile.outer.matrix.ring_dimension(),
        inner_route: profile_route(profile),
        slice_count: profile.outer_slice_count.get(),
        compression_chain_len: profile.outer_compression_plan()?.maps().len(),
        multi_block: profile.blocks.live_blocks > 1,
    })
}

fn corpus_ids() -> Result<Vec<SetupPrefixSlotId>, AkitaError> {
    crate::parse_slot_ids(CORPUS, CORPUS_PATH)
}

/// Recompute the known-answer digest of `field` from the corpus with the
/// compiled commitment code.
///
/// # Errors
///
/// Returns an error when a corpus entry cannot be committed.
pub fn compute_known_answer_digest(field: SisModulusProfileId) -> Result<[u8; 32], AkitaError> {
    let ids = corpus_ids()?
        .into_iter()
        .filter(|id| id.commitment_profile.inner.matrix.sis_modulus_profile() == field)
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

/// Require every slot id in `ids` to fall in a corpus class.
///
/// # Errors
///
/// Returns an error listing, for each missing class, the corpus line of its
/// slot id with the smallest prefix.
pub fn check_corpus_coverage(ids: &[SetupPrefixSlotId]) -> Result<(), AkitaError> {
    let classes = corpus_ids()?
        .iter()
        .map(corpus_class)
        .collect::<Result<BTreeSet<_>, _>>()?;
    let missing = missing_corpus_lines(&classes, ids)?;
    if missing.is_empty() {
        return Ok(());
    }
    Err(AkitaError::InvalidSetup(format!(
        "the known-answer corpus misses {} classes; append these lines to {CORPUS_PATH} \
         and update the known-answer digests (a code transition):\n{}",
        missing.len(),
        missing.join("\n")
    )))
}

fn missing_corpus_lines(
    classes: &BTreeSet<CorpusClass>,
    ids: &[SetupPrefixSlotId],
) -> Result<Vec<String>, AkitaError> {
    let size = |id: &SetupPrefixSlotId| id.n_prefix().unwrap_or(usize::MAX);
    let mut missing = BTreeMap::<CorpusClass, &SetupPrefixSlotId>::new();
    for id in ids {
        let class = corpus_class(id)?;
        if classes.contains(&class) {
            continue;
        }
        let kept = missing.entry(class).or_insert(id);
        if (size(id), id) < (size(kept), *kept) {
            *kept = id;
        }
    }
    missing
        .into_values()
        .map(|id| {
            let mut encoded = Vec::new();
            id.serialize_uncompressed(&mut encoded)
                .map_err(|error| AkitaError::Internal(error.to_string()))?;
            Ok(hex(&encoded))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::on_large_stack;

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
    fn coverage_prints_the_corpus_line_of_each_missing_class() {
        let ids = corpus_ids().unwrap();
        let classes = ids
            .iter()
            .map(|id| corpus_class(id).unwrap())
            .collect::<BTreeSet<_>>();
        assert_eq!(classes.len(), ids.len(), "one corpus entry per class");
        assert!(missing_corpus_lines(&classes, &ids).unwrap().is_empty());
        let lines = missing_corpus_lines(&BTreeSet::new(), &ids).unwrap();
        let reparsed = crate::parse_slot_ids(&lines.join("\n"), "missing lines").unwrap();
        assert_eq!(
            reparsed.into_iter().collect::<BTreeSet<_>>(),
            ids.into_iter().collect::<BTreeSet<_>>()
        );
    }
}
