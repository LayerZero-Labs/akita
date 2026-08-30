use super::{CpuBackend, CpuPreparedSetup};
use crate::backend::RingSwitchRelationView;
use crate::compute::backend::{ComputeBackendSetup, DigitRowsComputeBackend};
use crate::compute::{RingSwitchRelationKernel, RingSwitchRelationPlan};
use crate::AkitaProverSetup;
use akita_types::MAX_I8_LOG_BASIS;
use akita_types::{NttCacheKey, NttTransformDomain, SetupMatrixCapacity};
use jolt_field::Prime64Offset59;
use std::sync::atomic::Ordering;
use std::sync::Arc;

pub(super) type F = Prime64Offset59;
pub(super) const D: usize = 64;

fn setup_capacity(num_ring_elements: usize) -> SetupMatrixCapacity {
    SetupMatrixCapacity {
        num_field_elements: num_ring_elements * D,
    }
}

pub(super) fn prepared() -> CpuPreparedSetup<F> {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
    CpuBackend::DEFAULT.prepare_setup(&setup).unwrap()
}

#[test]
fn cpu_prepared_setup_identity_rejects_mismatched_setup() {
    let setup_a = AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
    let setup_b = AkitaProverSetup::<F>::generate_with_capacity(9, 1, setup_capacity(D)).unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup_a).unwrap();

    CpuBackend::DEFAULT
        .validate_prepared_setup(&prepared, setup_a.expanded.as_ref())
        .expect("matching setup");
    assert!(
        CpuBackend::DEFAULT
            .validate_prepared_setup(&prepared, setup_b.expanded.as_ref())
            .is_err(),
        "prepared context must stay bound to the setup used to create it"
    );
}

#[test]
fn cpu_prepared_setup_identity_accepts_equivalent_setup() {
    let setup_a = AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
    let setup_b = AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
    assert!(!Arc::ptr_eq(&setup_a.expanded, &setup_b.expanded));

    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup_a).unwrap();

    CpuBackend::DEFAULT
        .validate_prepared_setup(&prepared, setup_b.expanded.as_ref())
        .expect("equivalent deterministic setup should validate");
}

#[test]
fn cpu_prepared_setup_reports_checked_crt_capacity_profile() {
    let prepared = prepared();
    CpuBackend::DEFAULT
        .digit_rows::<D>(&prepared, 1, &[&[[1i8; D]]], 2)
        .expect("build exact NTT prefix");
    let profile = prepared.shared_ntt_profile(D).expect("profile");

    assert_eq!(profile.profile_id, "Q64/3xi32");
    assert_eq!(profile.num_primes, 3);
    assert_eq!(profile.limb_bits, 32);
    assert_eq!(profile.max_i8_log_basis, MAX_I8_LOG_BASIS);
    assert!(profile.balanced_digit_safe_width > 0);
    assert!(profile.raw_i8_safe_width > 0);
}

#[test]
fn prepare_setup_starts_with_empty_ntt_cache() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
    assert_eq!(prepared.shared_ntt_cache_bytes(), 0);
    assert!(prepared.shared_ntt.lock().unwrap().is_empty());
}

#[test]
fn cpu_prepared_setup_builds_only_requested_ntt_slots() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
    let partial_key = NttCacheKey {
        ring_d: D,
        num_ring_elements: 1,
        domain: NttTransformDomain::Negacyclic,
    };
    CpuBackend::DEFAULT
        .ensure_ntt_slot(&prepared, partial_key)
        .expect("warm partial slot");
    assert!(prepared.shared_ntt_cache_bytes() > 0);
    let cache = prepared.shared_ntt.lock().unwrap();
    assert!(cache.contains_key(&partial_key));
    assert_eq!(cache.len(), 1);
    drop(cache);
    let miss = NttCacheKey {
        ring_d: D,
        num_ring_elements: 99_999,
        domain: NttTransformDomain::Negacyclic,
    };
    assert!(!prepared.shared_ntt.lock().unwrap().contains_key(&miss));
}

#[test]
fn concurrent_same_key_ntt_warm_builds_once() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
    let prepared = CpuBackend::DEFAULT
        .prepare_expanded(setup.expanded.clone())
        .expect("empty prepared setup");
    let key = NttCacheKey {
        ring_d: D,
        num_ring_elements: 2,
        domain: NttTransformDomain::Negacyclic,
    };

    std::thread::scope(|scope| {
        for _ in 0..8 {
            let prepared = &prepared;
            scope.spawn(move || {
                CpuBackend::DEFAULT
                    .ensure_ntt_slot(prepared, key)
                    .expect("warm shared NTT slot");
            });
        }
    });
    CpuBackend::DEFAULT
        .ensure_ntt_slot(&prepared, key)
        .expect("repeated warm is a no-op");

    assert_eq!(prepared.ntt_slot_build_count.load(Ordering::Relaxed), 1);
    assert!(prepared.shared_ntt_cache_bytes() > 0);
}

#[test]
fn larger_initialized_prefix_covers_smaller_request() {
    let prepared = prepared();
    let covering_key = NttCacheKey {
        ring_d: D,
        num_ring_elements: 8,
        domain: NttTransformDomain::Negacyclic,
    };
    CpuBackend::DEFAULT
        .ensure_ntt_slot(&prepared, covering_key)
        .expect("warm covering prefix");

    prepared
        .with_shared_ntt::<D, _>(
            NttCacheKey::from_matrix_shape(D, 1, 3, NttTransformDomain::Negacyclic).unwrap(),
            |_ntt| Ok(()),
        )
        .expect("reuse covering prefix");

    assert_eq!(prepared.ntt_slot_build_count.load(Ordering::Relaxed), 1);
    let cache = prepared.shared_ntt.lock().unwrap();
    assert_eq!(cache.len(), 1);
    assert!(cache.contains_key(&covering_key));
}

#[test]
fn larger_request_replaces_smaller_cached_prefix() {
    let prepared = prepared();
    let small = NttCacheKey {
        ring_d: D,
        num_ring_elements: 3,
        domain: NttTransformDomain::Negacyclic,
    };
    let large = NttCacheKey {
        ring_d: D,
        num_ring_elements: 8,
        domain: NttTransformDomain::Negacyclic,
    };
    CpuBackend::DEFAULT
        .ensure_ntt_slot(&prepared, small)
        .expect("warm small prefix");
    CpuBackend::DEFAULT
        .ensure_ntt_slot(&prepared, large)
        .expect("grow to larger prefix");

    let cache = prepared.shared_ntt.lock().unwrap();
    assert_eq!(cache.len(), 1);
    assert!(cache.contains_key(&large));
}

#[test]
fn failed_growth_retains_smaller_cached_prefix() {
    let prepared = prepared();
    let small = NttCacheKey {
        ring_d: D,
        num_ring_elements: 3,
        domain: NttTransformDomain::Negacyclic,
    };
    let oversized = NttCacheKey {
        ring_d: D,
        num_ring_elements: D + 1,
        domain: NttTransformDomain::Negacyclic,
    };

    CpuBackend::DEFAULT
        .ensure_ntt_slot(&prepared, small)
        .expect("warm small prefix");
    assert!(CpuBackend::DEFAULT
        .ensure_ntt_slot(&prepared, oversized)
        .is_err());
    CpuBackend::DEFAULT
        .ensure_ntt_slot(&prepared, small)
        .expect("failed growth must leave the smaller prefix usable");

    let cache = prepared.shared_ntt.lock().unwrap();
    assert_eq!(cache.len(), 1);
    assert!(cache.contains_key(&small));
}

#[test]
fn planned_cache_bytes_match_max_joined_resident_state() {
    let prepared = prepared();
    let keys = [
        NttCacheKey {
            ring_d: D,
            num_ring_elements: 3,
            domain: NttTransformDomain::Negacyclic,
        },
        NttCacheKey {
            ring_d: D,
            num_ring_elements: 8,
            domain: NttTransformDomain::Negacyclic,
        },
        NttCacheKey {
            ring_d: D,
            num_ring_elements: 2,
            domain: NttTransformDomain::Cyclic,
        },
    ];
    let planned = prepared
        .planned_shared_ntt_cache_bytes(keys)
        .expect("planned bytes");
    for key in keys {
        CpuBackend::DEFAULT
            .ensure_ntt_slot(&prepared, key)
            .expect("prewarm exact requirement");
    }

    assert_eq!(prepared.shared_ntt_cache_bytes(), planned);
    assert_eq!(prepared.shared_ntt_cache_metrics().unwrap().len(), 2);
}

#[test]
fn concurrent_prefix_growth_retains_only_the_maximum() {
    let prepared = prepared();
    std::thread::scope(|scope| {
        for num_ring_elements in [2, 5, 3, 8, 4, 7] {
            let prepared = &prepared;
            scope.spawn(move || {
                CpuBackend::DEFAULT
                    .ensure_ntt_slot(
                        prepared,
                        NttCacheKey {
                            ring_d: D,
                            num_ring_elements,
                            domain: NttTransformDomain::Cyclic,
                        },
                    )
                    .expect("grow shared NTT prefix");
            });
        }
    });

    let cache = prepared.shared_ntt.lock().unwrap();
    assert_eq!(cache.len(), 1);
    assert!(cache.contains_key(&NttCacheKey {
        ring_d: D,
        num_ring_elements: 8,
        domain: NttTransformDomain::Cyclic,
    }));
}

#[test]
fn failed_oversized_warm_does_not_cover_valid_request() {
    let prepared = prepared();
    let oversized = NttCacheKey {
        ring_d: D,
        num_ring_elements: D + 1,
        domain: NttTransformDomain::Negacyclic,
    };
    let valid = NttCacheKey {
        ring_d: D,
        num_ring_elements: 3,
        domain: NttTransformDomain::Negacyclic,
    };

    assert!(CpuBackend::DEFAULT
        .ensure_ntt_slot(&prepared, oversized)
        .is_err());
    assert!(prepared.shared_ntt.lock().unwrap().is_empty());
    CpuBackend::DEFAULT
        .ensure_ntt_slot(&prepared, valid)
        .expect("failed oversized warm must not poison a valid prefix");

    let cache = prepared.shared_ntt.lock().unwrap();
    assert_eq!(cache.len(), 1);
    assert!(cache.contains_key(&valid));
    assert_eq!(prepared.ntt_slot_build_count.load(Ordering::Relaxed), 2);
}

#[test]
fn concurrent_failed_growth_leaves_valid_prefix_recoverable() {
    let prepared = prepared();
    let oversized = NttCacheKey {
        ring_d: D,
        num_ring_elements: D + 1,
        domain: NttTransformDomain::Cyclic,
    };
    let valid = NttCacheKey {
        ring_d: D,
        num_ring_elements: 8,
        domain: NttTransformDomain::Cyclic,
    };

    std::thread::scope(|scope| {
        let failed = scope.spawn(|| CpuBackend::DEFAULT.ensure_ntt_slot(&prepared, oversized));
        let warmed = scope.spawn(|| CpuBackend::DEFAULT.ensure_ntt_slot(&prepared, valid));
        assert!(failed.join().expect("oversized warm thread").is_err());
        warmed
            .join()
            .expect("valid warm thread")
            .expect("valid warm must retry a failed covering entry");
    });
    CpuBackend::DEFAULT
        .ensure_ntt_slot(&prepared, valid)
        .expect("valid prefix remains available after failed growth");

    let cache = prepared.shared_ntt.lock().unwrap();
    assert_eq!(cache.len(), 1);
    assert!(cache.contains_key(&valid));
}

#[test]
fn ring_switch_d_role_prepares_both_domains_at_exact_extent() {
    let prepared = prepared();
    let e_hat = vec![[1i8; D]; 5];
    let t_hat = vec![[1i8; D]; 3];
    let z_segment = vec![[1i32; D]; 2];

    CpuBackend::DEFAULT
        .relation_rows(
            &prepared,
            RingSwitchRelationView {
                e_hat: &e_hat,
                t_hat: &t_hat,
                z_segment: &z_segment,
                z_folded_centered_inf_norm: 1,
            },
            RingSwitchRelationPlan {
                n_d: 2,
                n_b: 1,
                n_a: 1,
                log_basis_open: 2,
                log_basis_outer: 2,
            },
        )
        .expect("ring-switch rows");

    let cache = prepared.shared_ntt.lock().unwrap();
    assert!(cache.contains_key(&NttCacheKey {
        ring_d: D,
        num_ring_elements: 10,
        domain: NttTransformDomain::Cyclic,
    }));
    assert!(cache.contains_key(&NttCacheKey {
        ring_d: D,
        num_ring_elements: 10,
        domain: NttTransformDomain::Negacyclic,
    }));
    assert_eq!(cache.len(), 2);
}

#[test]
fn cyclic_only_ring_switch_rows_do_not_prepare_negacyclic_state() {
    let prepared = prepared();
    let t_hat = vec![[1i8; D]; 3];

    let rows = CpuBackend::DEFAULT
        .relation_rows(
            &prepared,
            RingSwitchRelationView {
                e_hat: &[],
                t_hat: &t_hat,
                z_segment: &[],
                z_folded_centered_inf_norm: 0,
            },
            RingSwitchRelationPlan {
                n_d: 0,
                n_b: 2,
                n_a: 0,
                log_basis_open: 2,
                log_basis_outer: 2,
            },
        )
        .expect("B-only ring-switch rows");

    assert_eq!(rows.b_cyclic.len(), 2);
    assert!(rows.d_negacyclic.is_empty());
    assert!(rows.d_cyclic.is_empty());
    assert!(rows.a_quotients.is_empty());
    let cache = prepared.shared_ntt.lock().unwrap();
    assert!(cache.contains_key(&NttCacheKey {
        ring_d: D,
        num_ring_elements: 6,
        domain: NttTransformDomain::Cyclic,
    }));
    assert_eq!(cache.len(), 1);
}
