use super::support::{checked_in_artifact_bytes, checked_in_catalog};
use akita_config::{policy_of, proof_optimized::fp128, CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_params::{ChunkedWitnessCfg, MultiChunkProfileId};
use akita_schedules::{policy_digest, ValidatedScheduleCatalog};

#[test]
fn catalog_construction_rejects_rows_with_different_chunk_width_or_depth() {
    let single = checked_in_catalog::<fp128::OneHot>(ChunkedWitnessCfg::default_non_chunked());
    let chunked =
        checked_in_catalog::<fp128::OneHotMultiChunkW2R3>(MultiChunkProfileId::W2R3.cfg());
    for (source, expected, position) in [
        (
            single.catalog(),
            MultiChunkProfileId::W2R3.cfg(),
            "root final group",
        ),
        (
            chunked.catalog(),
            MultiChunkProfileId::W4R3.cfg(),
            "root final group",
        ),
        (
            chunked.catalog(),
            MultiChunkProfileId::W2R2.cfg(),
            "root final group",
        ),
        (
            chunked.catalog(),
            ChunkedWitnessCfg {
                num_chunks: 2,
                num_activated_levels: 4,
            },
            "root final group",
        ),
    ] {
        let error = ValidatedScheduleCatalog::try_new(
            fp128::OneHot::schedule_family_name(),
            source
                .rows()
                .map(|row| (row.profiles().clone(), row.schedule().clone())),
            &policy_of::<fp128::OneHot>(expected),
            fp128::OneHot::ring_challenge_config,
        )
        .expect_err("rows must execute the declared chunk geometry");
        assert!(
            matches!(error, AkitaError::InvalidSetup(ref message)
            if message.starts_with(position) && message.contains("witness chunks")),
            "{error}"
        );
    }
}

#[test]
fn artifact_rejects_single_chunk_rows_under_a_matching_multi_chunk_policy_digest() {
    let bytes = checked_in_artifact_bytes::<fp128::OneHot>();
    let expected = MultiChunkProfileId::W2R3.cfg();
    let policy = policy_of::<fp128::OneHot>(expected);
    let mut artifact = String::from_utf8(bytes).unwrap();
    let start =
        super::json_mutation::json_value_start(artifact.as_bytes(), b"\"policy_digest\"", 0);
    let end = start + artifact[start..].find(']').unwrap() + 1;
    let digest = serde_json::to_string_pretty(&policy_digest(&policy))
        .unwrap()
        .replace("\n", "\n  ");
    // Change only the policy digest, preserving canonical encoding and row
    // order. The row audit must reject even though the expected digest matches.
    artifact.replace_range(start..end, &digest);
    let error =
        TrustedScheduleCatalog::<fp128::OneHot>::from_artifact_bytes(artifact.as_bytes(), expected)
            .expect_err("the policy digest must not conceal stale row geometry");
    assert!(
        matches!(error, AkitaError::InvalidSetup(ref message)
        if message.starts_with("root final group") && message.contains("witness chunks")),
        "{error}"
    );
}

#[test]
fn catalog_rejects_noncanonical_per_fold_activation_depth() {
    let chunking = MultiChunkProfileId::W2R3.cfg();
    let catalog = checked_in_catalog::<fp128::OneHotMultiChunkW2R3>(chunking);
    let row = catalog.rows().next().unwrap();
    for level in [0, 1] {
        let mut schedule = row.schedule().clone();
        let params = if level == 0 {
            &mut schedule.root.params
        } else {
            &mut schedule.recursive_folds[0].params
        };
        params.witness_chunk.num_activated_levels = 7;
        let error = ValidatedScheduleCatalog::try_new(
            catalog.family_name(),
            [(row.profiles().clone(), schedule)],
            catalog.policy(),
            fp128::OneHotMultiChunkW2R3::ring_challenge_config,
        )
        .expect_err("each fold must carry the canonical activation depth");
        assert!(
            matches!(error, AkitaError::InvalidSetup(ref message)
            if message.contains("witness chunks") && message.contains("policy geometry")),
            "{error}"
        );
    }
}
