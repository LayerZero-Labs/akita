use crate::cache::tests::watchdog;
use crate::opaque::backend::SetupPrefixCache;
use akita_config::proof_optimized::fp128;
use akita_error::AkitaError;
use akita_params::SetupPrefixSlotId;
use std::sync::{mpsc, Arc};
use std::time::Duration;

fn prefix_id() -> SetupPrefixSlotId {
    type Cfg = fp128::Dense;
    const NV: usize = 14;
    let catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>(
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .unwrap();
    let row = catalog
        .resolve_key(&akita_params::ScheduleLookupKey::single(
            akita_params::PolynomialGroupLayout::new(NV, 1),
        ))
        .unwrap();
    let params = &row.schedule().root.params;
    let n_prefix = (params.d_a() * params.outer_slice_count().get()).next_power_of_two();
    let prefix = akita_params::setup_prefix_precommitted_params(params, n_prefix).unwrap();
    akita_params::scheduled_setup_prefix(n_prefix, prefix)
        .slot_id()
        .unwrap()
}

#[test]
fn setup_prefix_same_key_reentrant_build_completes() {
    watchdog(Duration::from_secs(30), || {
        let cache = SetupPrefixCache::<u64>::default();
        let id = prefix_id();
        let attempts = std::cell::Cell::new(0);
        let inner = std::cell::RefCell::new(None);
        let selected = cache
            .get_or_build(&id, || {
                attempts.set(attempts.get() + 1);
                let published = cache.get_or_build(&id, || {
                    attempts.set(attempts.get() + 1);
                    Ok(11)
                })?;
                *inner.borrow_mut() = Some(published);
                Ok(7)
            })
            .unwrap();
        assert_eq!(attempts.get(), 2);
        assert_eq!(cache.len().unwrap(), 1);
        assert_eq!(*selected, 11);
        assert!(Arc::ptr_eq(&selected, inner.borrow().as_ref().unwrap()));
    });
}

#[test]
fn failed_prefix_cache_generations_are_removed_and_retriable() {
    let id = prefix_id();
    let failed = SetupPrefixCache::<u64>::default();
    for offset in 0..32 {
        let mut failed_id = id.clone();
        failed_id.natural_len = failed_id.natural_len.checked_add(offset).unwrap();
        assert!(failed
            .get_or_build(&failed_id, || {
                Err(AkitaError::InvalidSetup("injected prefix failure".into()))
            })
            .is_err());
    }
    assert_eq!(failed.len().unwrap(), 0);
    assert!(failed
        .get_or_build(&id, || {
            Err(AkitaError::InvalidSetup("injected prefix failure".into()))
        })
        .is_err());
    assert_eq!(failed.len().unwrap(), 0);
    assert_eq!(*failed.get_or_build(&id, || Ok(17u64)).unwrap(), 17);
    assert_eq!(failed.len().unwrap(), 1);
}

#[test]
fn concurrent_prefix_cache_failure_preserves_published_success() {
    watchdog(Duration::from_secs(30), || {
        for failure_first in [false, true] {
            let cache = Arc::new(SetupPrefixCache::<u64>::default());
            let id = prefix_id();
            let (started_tx, started_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel();
            let mut failing = Some({
                let cache = Arc::clone(&cache);
                let id = id.clone();
                std::thread::spawn(move || {
                    cache.get_or_build(&id, || {
                        started_tx.send(()).unwrap();
                        release_rx.recv().unwrap();
                        Err(AkitaError::InvalidSetup("injected prefix failure".into()))
                    })
                })
            });
            started_rx.recv().unwrap();
            let failed = if failure_first {
                release_tx.send(()).unwrap();
                Some(failing.take().unwrap().join().unwrap())
            } else {
                None
            };
            let published = cache.get_or_build(&id, || Ok(23)).unwrap();
            let failed = if let Some(failed) = failed {
                failed
            } else {
                release_tx.send(()).unwrap();
                failing.take().unwrap().join().unwrap()
            };
            assert_eq!(
                failed.unwrap_err(),
                AkitaError::InvalidSetup("injected prefix failure".into())
            );
            assert_eq!(*published, 23);
            assert_eq!(cache.len().unwrap(), 1);
            for _ in 0..7 {
                let selected = cache
                    .get_or_build(&id, || panic!("published success must remain cached"))
                    .unwrap();
                assert!(Arc::ptr_eq(&published, &selected));
            }
        }
    });
}

#[test]
fn panicking_prefix_derivation_does_not_block_or_remove_success() {
    watchdog(Duration::from_secs(30), || {
        let cache = Arc::new(SetupPrefixCache::<u64>::default());
        let id = prefix_id();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let deriver = {
            let cache = Arc::clone(&cache);
            let id = id.clone();
            std::thread::spawn(move || {
                cache.get_or_build(&id, || {
                    started_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    panic!("injected prefix derivation panic");
                })
            })
        };
        started_rx.recv().unwrap();
        let published = cache.get_or_build(&id, || Ok(29)).unwrap();
        assert_eq!(*published, 29);
        release_tx.send(()).unwrap();
        assert!(deriver.join().is_err());
        assert_eq!(cache.len().unwrap(), 1);
        let selected = cache
            .get_or_build(&id, || panic!("published success must remain cached"))
            .unwrap();
        assert!(Arc::ptr_eq(&published, &selected));

        let lone = SetupPrefixCache::<u64>::default();
        assert!(std::panic::catch_unwind(|| {
            let _ = lone.get_or_build(&id, || panic!("injected lone prefix panic"));
        })
        .is_err());
        assert_eq!(lone.len().unwrap(), 0);
        assert_eq!(*lone.get_or_build(&id, || Ok(31)).unwrap(), 31);
        assert_eq!(lone.len().unwrap(), 1);
    });
}

/// Checks completion and value agreement under cold contention. The reentry
/// tests are the deterministic deadlock regressions.
#[cfg(feature = "parallel")]
#[test]
fn cold_setup_prefix_rayon_acquisitions_complete() {
    use crate::CpuBackend;
    use akita_params::test_fixtures::{retarget_group_role_dims_wide, sample_level_params};
    use akita_types::{NttCacheKey, NttTransformDomain};
    use jolt_field::Prime32Offset99 as F;
    use rayon::prelude::*;

    watchdog(Duration::from_secs(120), || {
        // q32 compression starts at D=128, so the prefix B image must be D=128.
        // This retargets the audited fixture used by commitment/setup_prefix_tests.
        const D: usize = 128;
        let mut params = sample_level_params();
        retarget_group_role_dims_wide(&mut params, D, D, 64);
        let prefix = akita_params::setup_prefix_precommitted_params(&params, D).unwrap();
        let id = akita_params::scheduled_setup_prefix(D, prefix)
            .slot_id()
            .unwrap();
        let capacity = akita_params::SetupMatrixCapacity {
            num_field_elements: akita_params::setup_prefix_slot_field_elements(&id).unwrap(),
        };
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        for n in [4, 14] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap();
            for _ in 0..16 {
                let setup =
                    crate::AkitaProverSetup::<F>::generate_with_capacity(8, 1, capacity).unwrap();
                let reference = CpuBackend::<F, F>::new(setup.expanded.clone()).unwrap();
                let expected = serial.install(|| reference.prepare_setup_prefix(&id).unwrap());
                let backend = CpuBackend::<F, F>::new(setup.expanded).unwrap();
                let prepared = backend.prepared().unwrap();
                assert_eq!(backend.setup_prefix_cache_len().unwrap(), 0);
                assert!(prepared.shared_ntt_cache_metrics().unwrap().is_empty());
                pool.install(|| {
                    (0..2 * n).into_par_iter().for_each(|task| {
                        // Setup-prefix slots are exact identities. Overlapping smaller
                        // prefixes contend in the NTT cache used by their real builder.
                        let key = NttCacheKey {
                            ring_d: D,
                            num_ring_elements: 1 + task % 3,
                            domain: NttTransformDomain::Negacyclic,
                        };
                        prepared.with_shared_ntt::<D, _>(key, |_| Ok(())).unwrap();
                        let actual = backend.prepare_setup_prefix(&id).unwrap();
                        assert_eq!(actual.public, expected.public);
                        assert_eq!(
                            actual.commitment_handle.committed.retained,
                            expected.commitment_handle.committed.retained
                        );
                    });
                });
                assert_eq!(backend.setup_prefix_cache_len().unwrap(), 1);
                let published = backend.prepare_setup_prefix(&id).unwrap();
                assert_eq!(published.public, expected.public);
                let metrics = prepared.shared_ntt_cache_metrics().unwrap();
                let covering: Vec<_> = metrics
                    .iter()
                    .filter(|metric| {
                        metric.key.ring_d == D
                            && metric.key.domain == NttTransformDomain::Negacyclic
                    })
                    .collect();
                assert_eq!(covering.len(), 1);
                assert!(covering[0].key.num_ring_elements >= 3);
            }
        }
    });
}
