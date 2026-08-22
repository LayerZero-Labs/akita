//! Streamed ring-switch kernels against their cached-transform equivalents,
//! and the built-NTT-slot release/rebuild lifecycle.

use super::*;
use crate::backend::RingSwitchRelationView;
use crate::compute::backend::ComputeBackendSetup;
use crate::compute::requirements::NttOperationCluster;
use crate::compute::{RingSwitchRelationKernel, RingSwitchRelationPlan};
use crate::kernels::linear::{
    digit_relation_rows_cached_prover_bounds, digit_relation_rows_streamed_prover_bounds,
    fused_quotient_matrix_extent, fused_split_eq_quotients_prover_bounds,
    fused_split_eq_quotients_streamed_prover_bounds,
};
use crate::AkitaProverSetup;
use akita_error::AkitaError;
use akita_types::{NttCacheKey, NttTransformDomain, SetupMatrixCapacity};
use jolt_field::{Prime128Offset275, Prime32Offset99, Prime64Offset59};
use std::sync::atomic::Ordering;

type F = Prime64Offset59;
const D: usize = 64;

fn setup_capacity(num_ring_elements: usize) -> SetupMatrixCapacity {
    SetupMatrixCapacity {
        num_field_elements: num_ring_elements * D,
    }
}

fn prepared() -> CpuPreparedSetup<F> {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
    CpuBackend::DEFAULT.prepare_setup(&setup).unwrap()
}

fn cyclic_key(extent: usize) -> NttCacheKey {
    NttCacheKey::from_matrix_shape(D, 1, extent, NttTransformDomain::Cyclic).unwrap()
}

fn negacyclic_key(extent: usize) -> NttCacheKey {
    NttCacheKey::from_matrix_shape(D, 1, extent, NttTransformDomain::Negacyclic).unwrap()
}

#[test]
fn cpu_resource_limits_have_checked_defaults_and_boundaries() {
    let default = CpuBackend::default();
    assert_eq!(
        default.max_cached_ring_switch_elements(),
        CpuBackend::DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS
    );
    assert_eq!(
        default.commit_scratch_bytes_per_worker(),
        CpuBackend::DEFAULT_COMMIT_SCRATCH_BYTES_PER_WORKER
    );

    let stream_all =
        CpuBackend::with_resource_limits(0, CpuBackend::DEFAULT_COMMIT_SCRATCH_BYTES_PER_WORKER)
            .unwrap();
    assert!(!stream_all.ntt_operation_uses_cache(NttOperationCluster::RingSwitch, 1));
    assert!(stream_all.ntt_operation_uses_cache(NttOperationCluster::Commit, usize::MAX));

    let retain_all = CpuBackend::with_resource_limits(
        usize::MAX,
        CpuBackend::DEFAULT_COMMIT_SCRATCH_BYTES_PER_WORKER,
    )
    .unwrap();
    assert!(retain_all.ntt_operation_uses_cache(NttOperationCluster::RingSwitch, usize::MAX));
    assert!(CpuBackend::with_resource_limits(1, 0).is_err());
}

#[test]
fn configured_ring_switch_routes_preserve_relation_rows() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
    let cached_backend = CpuBackend::with_resource_limits(
        usize::MAX,
        CpuBackend::DEFAULT_COMMIT_SCRATCH_BYTES_PER_WORKER,
    )
    .unwrap();
    let streamed_backend =
        CpuBackend::with_resource_limits(0, CpuBackend::DEFAULT_COMMIT_SCRATCH_BYTES_PER_WORKER)
            .unwrap();
    let cached_prepared = cached_backend.prepare_setup(&setup).unwrap();
    let streamed_prepared = streamed_backend.prepare_setup(&setup).unwrap();
    let e_hat = vec![[1i8; D], [-1i8; D], [1i8; D]];
    let t_hat = vec![[-1i8; D], [3i8; D]];
    let z_segment = vec![[1i32; D], [-2i32; D], [3i32; D], [5i32; D]];
    let source = RingSwitchRelationView {
        e_hat: &e_hat,
        t_hat: &t_hat,
        z_segment: &z_segment,
        z_folded_centered_inf_norm: 5,
    };
    let plan = RingSwitchRelationPlan {
        n_d: 2,
        n_b: 2,
        n_a: 1,
        log_basis_open: 2,
        log_basis_outer: 3,
    };

    let cached = cached_backend
        .relation_rows(&cached_prepared, source, plan)
        .expect("cached relation rows");
    let streamed = streamed_backend
        .relation_rows(&streamed_prepared, source, plan)
        .expect("streamed relation rows");

    assert_eq!(streamed, cached);
    assert_eq!(cached.d_negacyclic.len(), plan.n_d);
    assert!(!cached_prepared
        .shared_ntt_cache_metrics()
        .unwrap()
        .is_empty());
    assert!(streamed_prepared
        .shared_ntt_cache_metrics()
        .unwrap()
        .is_empty());
}

#[test]
fn configured_ring_switch_routes_reject_malformed_active_roles() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
    let cached_backend = CpuBackend::with_resource_limits(
        usize::MAX,
        CpuBackend::DEFAULT_COMMIT_SCRATCH_BYTES_PER_WORKER,
    )
    .unwrap();
    let streamed_backend =
        CpuBackend::with_resource_limits(0, CpuBackend::DEFAULT_COMMIT_SCRATCH_BYTES_PER_WORKER)
            .unwrap();
    let cached_prepared = cached_backend.prepare_setup(&setup).unwrap();
    let streamed_prepared = streamed_backend.prepare_setup(&setup).unwrap();
    let digits = [[1i8; D]];
    let centered = [[1i32; D]];

    let malformed = [
        (
            RingSwitchRelationView {
                e_hat: &[],
                t_hat: &digits,
                z_segment: &[],
                z_folded_centered_inf_norm: 0,
            },
            RingSwitchRelationPlan {
                n_d: 1,
                n_b: 1,
                n_a: 0,
                log_basis_open: 2,
                log_basis_outer: 2,
            },
        ),
        (
            RingSwitchRelationView {
                e_hat: &digits,
                t_hat: &[],
                z_segment: &[],
                z_folded_centered_inf_norm: 0,
            },
            RingSwitchRelationPlan {
                n_d: 1,
                n_b: 1,
                n_a: 0,
                log_basis_open: 2,
                log_basis_outer: 2,
            },
        ),
        (
            RingSwitchRelationView {
                e_hat: &digits,
                t_hat: &[],
                z_segment: &[],
                z_folded_centered_inf_norm: 1,
            },
            RingSwitchRelationPlan {
                n_d: 1,
                n_b: 0,
                n_a: 1,
                log_basis_open: 2,
                log_basis_outer: 2,
            },
        ),
    ];

    for (source, plan) in malformed {
        let streamed = streamed_backend.relation_rows(&streamed_prepared, source, plan);
        let cached = cached_backend.relation_rows(&cached_prepared, source, plan);
        assert!(matches!(streamed, Err(AkitaError::InvalidInput(_))));
        assert!(matches!(cached, Err(AkitaError::InvalidInput(_))));
    }

    let valid_a = RingSwitchRelationView {
        e_hat: &digits,
        t_hat: &digits,
        z_segment: &centered,
        z_folded_centered_inf_norm: 1,
    };
    let valid_plan = RingSwitchRelationPlan {
        n_d: 1,
        n_b: 0,
        n_a: 1,
        log_basis_open: 2,
        log_basis_outer: 2,
    };
    assert_eq!(
        streamed_backend
            .relation_rows(&streamed_prepared, valid_a, valid_plan)
            .unwrap(),
        cached_backend
            .relation_rows(&cached_prepared, valid_a, valid_plan)
            .unwrap()
    );
}

#[test]
fn cached_and_streamed_routes_share_acceptance_across_crt_bounds() {
    type F32 = Prime32Offset99;
    let setup = AkitaProverSetup::<F32>::generate_with_capacity(
        8,
        1,
        SetupMatrixCapacity {
            num_field_elements: 64 * D,
        },
    )
    .unwrap();
    let streamed_backend =
        CpuBackend::with_resource_limits(0, CpuBackend::DEFAULT_COMMIT_SCRATCH_BYTES_PER_WORKER)
            .unwrap();
    let cached_backend = CpuBackend::with_resource_limits(
        usize::MAX,
        CpuBackend::DEFAULT_COMMIT_SCRATCH_BYTES_PER_WORKER,
    )
    .unwrap();
    let streamed_prepared = streamed_backend.prepare_setup(&setup).unwrap();
    let cached_prepared = cached_backend.prepare_setup(&setup).unwrap();
    let z_segment = vec![[1i32; D]; 64];
    let plan = RingSwitchRelationPlan {
        n_d: 0,
        n_b: 0,
        n_a: 1,
        log_basis_open: 1,
        log_basis_outer: 1,
    };
    for centered_bound in [1, 5, u16::MAX as u32, u32::MAX] {
        let source = RingSwitchRelationView {
            e_hat: &[],
            t_hat: &[],
            z_segment: &z_segment,
            z_folded_centered_inf_norm: centered_bound,
        };
        let streamed = streamed_backend.relation_rows(&streamed_prepared, source, plan);
        let cached = cached_backend.relation_rows(&cached_prepared, source, plan);
        assert_eq!(
            streamed.is_ok(),
            cached.is_ok(),
            "route acceptance differs at centered bound {centered_bound}"
        );
        assert_eq!(streamed.unwrap(), cached.unwrap());
    }
    assert!(streamed_prepared
        .shared_ntt_cache_metrics()
        .unwrap()
        .is_empty());
    assert!(!cached_prepared
        .shared_ntt_cache_metrics()
        .unwrap()
        .is_empty());
}

#[test]
fn streamed_relation_rows_match_cached_kernel() {
    let prepared = prepared();
    let t_hat = vec![[-1i8; D], [3i8; D]];
    let z_segment = vec![[1i32; D], [-2i32; D], [3i32; D], [5i32; D]];
    let extent = 2usize.saturating_mul(t_hat.len()).max(z_segment.len());
    let view = prepared
        .expanded
        .shared_matrix()
        .ring_view::<D>(1, extent)
        .expect("field view");
    let streamed = fused_split_eq_quotients_streamed_prover_bounds(
        view.as_slice(),
        2,
        1,
        &t_hat,
        &z_segment,
        5,
        3,
    )
    .expect("streamed rows");
    let cached = prepared
        .with_shared_ntt::<D, _>(cyclic_key(extent), |cyclic_ntt| {
            prepared.with_shared_ntt::<D, _>(negacyclic_key(extent), |negacyclic_ntt| {
                fused_split_eq_quotients_prover_bounds(
                    negacyclic_ntt,
                    cyclic_ntt,
                    2,
                    1,
                    &t_hat,
                    &z_segment,
                    5,
                    3,
                )
            })
        })
        .expect("cached rows");
    assert_eq!(streamed, cached);
}

#[test]
fn streamed_relation_rows_match_cached_q32_kernel() {
    type F32 = Prime32Offset99;
    let setup = AkitaProverSetup::<F32>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let t_hat = vec![[-1i8; D], [3i8; D]];
    let z_segment = vec![[1i32; D], [-2i32; D], [3i32; D], [5i32; D]];
    let extent = 2usize.saturating_mul(t_hat.len()).max(z_segment.len());
    let view = prepared
        .expanded
        .shared_matrix()
        .ring_view::<D>(1, extent)
        .expect("field view");
    let streamed = fused_split_eq_quotients_streamed_prover_bounds(
        view.as_slice(),
        2,
        1,
        &t_hat,
        &z_segment,
        5,
        3,
    )
    .expect("streamed rows");
    let cached = prepared
        .with_shared_ntt::<D, _>(cyclic_key(extent), |cyclic_ntt| {
            prepared.with_shared_ntt::<D, _>(negacyclic_key(extent), |negacyclic_ntt| {
                fused_split_eq_quotients_prover_bounds(
                    negacyclic_ntt,
                    cyclic_ntt,
                    2,
                    1,
                    &t_hat,
                    &z_segment,
                    5,
                    3,
                )
            })
        })
        .expect("cached rows");
    assert_eq!(streamed, cached);
}

#[test]
fn cached_and_streamed_reject_the_same_short_matrix_shape() {
    assert!(fused_quotient_matrix_extent(usize::MAX, 2, 0, 0).is_err());

    let prepared = prepared();
    let e_hat = vec![[1i8; D], [-1i8; D]];
    let view = prepared
        .expanded
        .shared_matrix()
        .ring_view::<D>(1, 1)
        .expect("one-element field view");
    let streamed = digit_relation_rows_streamed_prover_bounds(view.as_slice(), 1, &e_hat, 2);
    assert!(streamed.is_err());

    let cached = prepared.with_shared_ntt::<D, _>(cyclic_key(1), |cyclic_ntt| {
        digit_relation_rows_cached_prover_bounds::<F, D>(cyclic_ntt, cyclic_ntt, 1, &e_hat, 2)
    });
    assert!(cached.is_err());
}

#[test]
fn streamed_chunked_z_quotient_matches_cached_kernel() {
    let prepared = prepared();
    // A capacity bound sized so the safe CRT chunk width lands strictly
    // between 1 and z_len, forcing the chunked path in both the cached
    // and streamed kernels.
    let z_bound = 1u32 << 17;
    let z_segment: Vec<[i32; D]> = (0..64).map(|i| [(i % 23) - 11; D]).collect();
    let extent = z_segment.len();
    let view = prepared
        .expanded
        .shared_matrix()
        .ring_view::<D>(1, extent)
        .expect("field view");
    let streamed = fused_split_eq_quotients_streamed_prover_bounds(
        view.as_slice(),
        0,
        1,
        &[][..],
        &z_segment,
        z_bound,
        1,
    )
    .expect("streamed rows");
    let cached = prepared
        .with_shared_ntt::<D, _>(cyclic_key(extent), |cyclic_ntt| {
            prepared.with_shared_ntt::<D, _>(negacyclic_key(extent), |negacyclic_ntt| {
                fused_split_eq_quotients_prover_bounds(
                    negacyclic_ntt,
                    cyclic_ntt,
                    0,
                    1,
                    &[][..],
                    &z_segment,
                    z_bound,
                    1,
                )
            })
        })
        .expect("cached rows");
    assert_eq!(streamed, cached);
}

#[test]
fn streamed_chunked_t_rows_match_cached_kernel() {
    const T_LEN: usize = 512;
    const D128: usize = 64;
    let setup = AkitaProverSetup::<Prime128Offset275>::generate_with_capacity(
        8,
        1,
        SetupMatrixCapacity {
            num_field_elements: T_LEN * D128,
        },
    )
    .unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let t_hat = vec![[1i8; D128]; T_LEN];
    let z_segment = vec![[1i32; D128], [-2i32; D128], [3i32; D128], [1i32; D128]];
    let view = prepared
        .expanded
        .shared_matrix()
        .ring_view::<D128>(1, T_LEN)
        .expect("field view");
    let streamed = fused_split_eq_quotients_streamed_prover_bounds(
        view.as_slice(),
        1,
        1,
        &t_hat,
        &z_segment,
        3,
        8,
    )
    .expect("streamed rows");
    let cached = prepared
        .with_shared_ntt::<D128, _>(
            NttCacheKey::from_matrix_shape(D128, 1, T_LEN, NttTransformDomain::Cyclic).unwrap(),
            |cyclic_ntt| {
                prepared.with_shared_ntt::<D128, _>(
                    NttCacheKey::from_matrix_shape(D128, 1, T_LEN, NttTransformDomain::Negacyclic)
                        .unwrap(),
                    |negacyclic_ntt| {
                        fused_split_eq_quotients_prover_bounds(
                            negacyclic_ntt,
                            cyclic_ntt,
                            1,
                            1,
                            &t_hat,
                            &z_segment,
                            3,
                            8,
                        )
                    },
                )
            },
        )
        .expect("cached rows");
    assert_eq!(streamed, cached);
}

#[test]
fn drop_built_ntt_slots_frees_and_rebuilds() {
    let prepared = prepared();
    let key = cyclic_key(D);
    prepared
        .with_shared_ntt::<D, _>(key, |_| Ok(()))
        .expect("build slot");
    assert!(prepared.shared_ntt_cache_bytes() > 0);
    let freed = prepared.drop_built_ntt_slots().unwrap();
    assert!(freed > 0);
    assert_eq!(prepared.shared_ntt_cache_bytes(), 0);
    prepared
        .with_shared_ntt::<D, _>(key, |_| Ok(()))
        .expect("slot rebuilds after release");
    assert!(prepared.shared_ntt_cache_bytes() > 0);
}

#[test]
fn released_large_prefix_does_not_cover_smaller_rebuild() {
    let prepared = prepared();
    let large = cyclic_key(32);
    let small = cyclic_key(3);

    CpuBackend::DEFAULT
        .ensure_ntt_slot(&prepared, large)
        .expect("warm large prefix");
    let large_bytes = prepared.shared_ntt_cache_bytes();
    assert_eq!(prepared.drop_built_ntt_slots().unwrap(), large_bytes);
    assert!(prepared.shared_ntt.lock().unwrap().is_empty());

    CpuBackend::DEFAULT
        .ensure_ntt_slot(&prepared, small)
        .expect("rebuild exact smaller prefix");

    let metrics = prepared.shared_ntt_cache_metrics().unwrap();
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].key, small);
    assert!(metrics[0].cache_bytes < large_bytes);
}

#[test]
fn dropping_built_slots_does_not_invalidate_active_reader() {
    let prepared = prepared();
    let key = cyclic_key(D);
    prepared
        .with_shared_ntt::<D, _>(key, |_| Ok(()))
        .expect("build slot");
    let bytes = prepared.shared_ntt_cache_bytes();
    let builds_before = prepared.ntt_slot_build_count.load(Ordering::Relaxed);
    let entered = std::sync::Barrier::new(2);
    let released = std::sync::Barrier::new(2);

    std::thread::scope(|scope| {
        let reader = scope.spawn(|| {
            prepared
                .with_shared_ntt::<D, _>(key, |ntt| {
                    entered.wait();
                    released.wait();
                    assert_eq!(ntt.cache_bytes(), bytes);
                    Ok(())
                })
                .expect("active reader keeps its cache alive");
        });
        entered.wait();
        assert_eq!(prepared.drop_built_ntt_slots().unwrap(), bytes);
        assert_eq!(prepared.shared_ntt_cache_bytes(), 0);
        released.wait();
        reader.join().expect("reader thread");
    });

    prepared
        .with_shared_ntt::<D, _>(key, |_| Ok(()))
        .expect("released slot rebuilds");
    assert_eq!(
        prepared.ntt_slot_build_count.load(Ordering::Relaxed),
        builds_before + 1
    );
}
