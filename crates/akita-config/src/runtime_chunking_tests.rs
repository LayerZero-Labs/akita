use crate::proof_optimized::fp128::{OneHot, OneHotMultiChunkW2R3, OneHotMultiChunkW4R3};
use crate::{policy_of, CommitmentConfig, TrustedScheduleCatalog};
use akita_params::ChunkedWitnessCfg;
use akita_schedules::ValidatedScheduleCatalog;

fn rebind<Cfg: CommitmentConfig>(chunking: ChunkedWitnessCfg) {
    let source = crate::test_support::workspace_schedule_catalog::<Cfg>(chunking).unwrap();
    let policy = policy_of::<OneHot>(chunking);
    let catalog = ValidatedScheduleCatalog::try_new(
        OneHot::schedule_family_name(),
        source
            .rows()
            .map(|row| (row.profiles().clone(), row.schedule().clone())),
        &policy,
        OneHot::ring_challenge_config,
    )
    .unwrap();
    let bytes = catalog.to_artifact_bytes().unwrap();
    let admitted = TrustedScheduleCatalog::<OneHot>::from_artifact_bytes(&bytes, chunking).unwrap();
    assert_eq!(admitted.policy().witness_chunk, chunking);
    assert_eq!(admitted.to_artifact_bytes().unwrap(), bytes);
    assert!(TrustedScheduleCatalog::<OneHot>::from_artifact_bytes(
        &bytes,
        akita_params::ChunkedWitnessCfg::default_non_chunked()
    )
    .is_err());
    let wrong_depth = ChunkedWitnessCfg {
        num_activated_levels: 2,
        ..chunking
    };
    assert!(TrustedScheduleCatalog::<OneHot>::from_artifact_bytes(&bytes, wrong_depth).is_err());
    let wrong_width = ChunkedWitnessCfg {
        num_chunks: 8,
        ..chunking
    };
    assert!(TrustedScheduleCatalog::<OneHot>::new(catalog, wrong_width).is_err());
}

#[test]
fn one_config_admits_distinct_runtime_geometries() {
    rebind::<OneHotMultiChunkW2R3>(akita_params::ChunkedWitnessCfg {
        num_chunks: 2,
        num_activated_levels: 3,
    });
    rebind::<OneHotMultiChunkW4R3>(akita_params::ChunkedWitnessCfg {
        num_chunks: 4,
        num_activated_levels: 3,
    });
}

#[test]
fn runtime_policy_rejects_invalid_geometry() {
    for (num_chunks, num_activated_levels) in
        [(0, 2), (3, 2), (128, 2), (1, 2), (2, 0), (2, usize::MAX)]
    {
        assert!(crate::validate_config_policy::<OneHot>(ChunkedWitnessCfg {
            num_chunks,
            num_activated_levels
        })
        .is_err());
    }
}
