use super::support::checked_in_catalog;
use akita_config::{
    proof_optimized::fp128, CommitmentConfig, RecursiveCommitmentConfig, TrustedScheduleCatalog,
    ValidatedScheduleCatalog,
};
use akita_serialization::AkitaSerialize;

fn serialized_slot_ids<Cfg: CommitmentConfig>(
    chunking: akita_params::ChunkedWitnessCfg,
) -> Vec<String> {
    akita_config::SetupRequirements::from_catalog::<Cfg>(
        &checked_in_catalog::<Cfg>(chunking),
        50,
        16,
    )
    .map(|requirements| requirements.prefix_slot_ids().to_vec())
    .expect("derive recursive setup-prefix slots")
    .into_iter()
    .map(|slot| {
        let mut bytes = Vec::new();
        slot.serialize_compressed(&mut bytes)
            .expect("serialize setup-prefix slot id");
        bytes
            .into_iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    })
    .collect()
}

fn recursive_prefix_fixture<Cfg: CommitmentConfig>(
    chunking: akita_params::ChunkedWitnessCfg,
) -> (usize, String) {
    let slots = serialized_slot_ids::<Cfg>(chunking);
    let digest = akita_params::digest_descriptor_bytes(slots.join("\n").as_bytes());
    let digest = digest
        .into_iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    (slots.len(), digest)
}

#[test]
fn recursive_prefix_slot_id_fixture() {
    // These setup-prefix identities bind the generated schedule and policy.
    let onehot = recursive_prefix_fixture::<RecursiveCommitmentConfig<fp128::OneHot>>(
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    );
    let multichunk = recursive_prefix_fixture::<RecursiveCommitmentConfig<fp128::OneHotMultiChunk>>(
        akita_params::MultiChunkProfileId::W8R2.cfg(),
    );
    assert_eq!(
        onehot,
        (
            3,
            "8d6a55cc3a2796c4c02e9b8eb1c5915abce5905da7af78035b100a57dd80bf2a".to_string(),
        )
    );
    assert_eq!(
        multichunk,
        (
            2,
            "e8fb7d64db50b5a11e20995b8f8b1dcf5f6c59deb40c69429534ab36e41ce7a7".to_string(),
        )
    );
}

#[test]
fn setup_prefix_planning_rejects_invalid_capacity_metadata() {
    let dense =
        checked_in_catalog::<fp128::Dense>(akita_params::ChunkedWitnessCfg::default_non_chunked());
    let zero_batch = akita_config::SetupRequirements::from_catalog::<fp128::Dense>(&dense, 14, 0)
        .map(|requirements| requirements.prefix_slot_ids().to_vec())
        .expect_err("zero-batch setup metadata must reject for nonrecursive configs");
    assert!(format!("{zero_batch}").contains("at least 1"));

    let recursive = checked_in_catalog::<RecursiveCommitmentConfig<fp128::OneHot>>(
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    );
    let oversized_vars = akita_config::SetupRequirements::from_catalog::<
        RecursiveCommitmentConfig<fp128::OneHot>,
    >(&recursive, usize::BITS as usize, 1)
    .map(|requirements| requirements.prefix_slot_ids().to_vec())
    .expect_err("oversized setup metadata must reject for recursive configs");
    assert!(format!("{oversized_vars}").contains("exceeds preprocessing limits"));
}

#[test]
fn setup_requirements_keep_precommits_when_the_grouped_row_does_not_fit() {
    type Cfg = fp128::Dense;
    let catalog = checked_in_catalog::<Cfg>(akita_params::ChunkedWitnessCfg::default_non_chunked());
    let row = catalog
        .rows()
        .find(|row| !row.profiles().precommitteds.is_empty())
        .expect("dense grouped row");
    let profile = row.profiles().precommitteds[0];
    assert!(row.profiles().final_group.group.num_vars() > profile.group.num_vars());
    let grouped_only = ValidatedScheduleCatalog::try_new(
        Cfg::schedule_family_name(),
        [(row.profiles().clone(), row.schedule().clone())],
        catalog.policy(),
        Cfg::ring_challenge_config,
    )
    .unwrap();
    let grouped_only =
        TrustedScheduleCatalog::<Cfg>::new(grouped_only, catalog.policy().witness_chunk).unwrap();
    let required = akita_config::SetupRequirements::from_catalog::<Cfg>(
        &grouped_only,
        profile.group.num_vars(),
        profile.group.num_polynomials(),
    )
    .expect("independent precommit remains supported");
    let expected = akita_params::commit_only_setup_field_elements(
        &profile.inner.matrix,
        &profile.outer.matrix,
        profile.outer_slice_count,
    )
    .unwrap();
    assert_eq!(required.matrix_capacity().num_field_elements, expected);
    assert!(required.prefix_slot_ids().is_empty());
}

#[test]
fn setup_requirements_union_covers_both_families_at_one_bound() {
    type OneHot = RecursiveCommitmentConfig<fp128::OneHot>;
    type MultiChunk = RecursiveCommitmentConfig<fp128::OneHotMultiChunk>;
    let requirements_at = |max_num_vars, max_num_batched_polys| {
        (
            akita_config::SetupRequirements::from_catalog::<OneHot>(
                &checked_in_catalog::<OneHot>(
                    akita_params::ChunkedWitnessCfg::default_non_chunked(),
                ),
                max_num_vars,
                max_num_batched_polys,
            )
            .expect("one-hot requirements"),
            akita_config::SetupRequirements::from_catalog::<MultiChunk>(
                &checked_in_catalog::<MultiChunk>(akita_params::MultiChunkProfileId::W8R2.cfg()),
                max_num_vars,
                max_num_batched_polys,
            )
            .expect("multichunk requirements"),
        )
    };
    let (onehot, multichunk) = requirements_at(50, 16);
    let combined = onehot
        .clone()
        .union(multichunk.clone())
        .expect("same-bound requirements combine");

    assert_eq!(
        (combined.max_num_vars(), combined.max_num_batched_polys()),
        (50, 16)
    );
    assert_eq!(
        combined.matrix_capacity().num_field_elements,
        onehot
            .matrix_capacity()
            .num_field_elements
            .max(multichunk.matrix_capacity().num_field_elements)
    );
    assert!(combined
        .prefix_slot_ids()
        .windows(2)
        .all(|pair| pair[0] < pair[1]));
    for slot in onehot
        .prefix_slot_ids()
        .iter()
        .chain(multichunk.prefix_slot_ids())
    {
        assert!(combined.prefix_slot_ids().contains(slot));
    }
    assert!(
        combined.prefix_slot_ids().len()
            <= onehot.prefix_slot_ids().len() + multichunk.prefix_slot_ids().len()
    );
    assert_eq!(
        combined
            .clone()
            .union(combined.clone())
            .expect("idempotent"),
        combined
    );

    let (_, smaller_bound) = requirements_at(40, 16);
    let error = onehot
        .union(smaller_bound)
        .expect_err("requirements at different bounds must not combine");
    assert!(error.to_string().contains("cannot combine"));
}
