//! Streamed ring-switch kernels against their cached-transform equivalents,
//! and the built-NTT-slot release/rebuild lifecycle.

use super::*;
use crate::compute::backend::ComputeBackendSetup;
use crate::AkitaProverSetup;
use akita_field::{Prime128Offset275, Prime64Offset59};
use akita_types::SetupMatrixCapacity;

type F = Prime64Offset59;
const D: usize = 64;

fn setup_capacity(num_ring_elements: usize) -> SetupMatrixCapacity {
    SetupMatrixCapacity {
        num_field_elements: num_ring_elements * D,
    }
}

fn prepared() -> CpuPreparedSetup<F> {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
    CpuBackend.prepare_setup(&setup).unwrap()
}

fn params_key() -> NttCacheKey {
    NttCacheKey::from_matrix_shape(D, 1, 1, NttTransformDomain::Cyclic).unwrap()
}

fn cyclic_key(extent: usize) -> NttCacheKey {
    NttCacheKey::from_matrix_shape(D, 1, extent, NttTransformDomain::Cyclic).unwrap()
}

fn negacyclic_key(extent: usize) -> NttCacheKey {
    NttCacheKey::from_matrix_shape(D, 1, extent, NttTransformDomain::Negacyclic).unwrap()
}

#[test]
fn streamed_relation_rows_match_cached_kernel() {
    let prepared = prepared();
    let e_hat = vec![[1i8; D], [-1i8; D], [1i8; D]];
    let t_hat = vec![[-1i8; D], [3i8; D]];
    let z_segment = vec![[1i32; D], [-2i32; D], [3i32; D], [5i32; D]];
    let extent = 2usize
        .saturating_mul(e_hat.len())
        .max(2usize.saturating_mul(t_hat.len()))
        .max(z_segment.len());
    let view = prepared
        .expanded
        .shared_matrix()
        .ring_view::<D>(1, extent)
        .expect("field view");
    let source = StreamedASource::Flat(view.as_slice());
    let streamed = prepared
        .with_shared_ntt::<D, _>(params_key(), |ntt| {
            fused_split_eq_quotients_streamed_prover_bounds(
                ntt, &source, 2, 2, 1, &e_hat, &t_hat, &z_segment, 5, 2, 3,
            )
        })
        .expect("streamed rows")
        .expect("shape is one-shot safe");
    let cached = prepared
        .with_shared_ntt::<D, _>(cyclic_key(extent), |cyclic_ntt| {
            prepared.with_shared_ntt::<D, _>(negacyclic_key(extent), |negacyclic_ntt| {
                fused_split_eq_quotients_prover_bounds(
                    negacyclic_ntt,
                    cyclic_ntt,
                    2,
                    2,
                    1,
                    &e_hat,
                    &t_hat,
                    &z_segment,
                    5,
                    2,
                    3,
                )
            })
        })
        .expect("cached rows");
    assert_eq!(streamed, cached);
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
    let source = StreamedASource::Flat(view.as_slice());
    let streamed = prepared
        .with_shared_ntt::<D, _>(params_key(), |ntt| {
            fused_split_eq_quotients_streamed_prover_bounds(
                ntt,
                &source,
                0,
                0,
                1,
                &[][..],
                &[][..],
                &z_segment,
                z_bound,
                1,
                1,
            )
        })
        .expect("streamed rows")
        .expect("chunked z path streams");
    let cached = prepared
        .with_shared_ntt::<D, _>(cyclic_key(extent), |cyclic_ntt| {
            prepared.with_shared_ntt::<D, _>(negacyclic_key(extent), |negacyclic_ntt| {
                fused_split_eq_quotients_prover_bounds(
                    negacyclic_ntt,
                    cyclic_ntt,
                    0,
                    0,
                    1,
                    &[][..],
                    &[][..],
                    &z_segment,
                    z_bound,
                    1,
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
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let e_hat = vec![[1i8; D128], [-1i8; D128], [1i8; D128]];
    let t_hat = vec![[1i8; D128]; T_LEN];
    let z_segment = vec![[1i32; D128], [-2i32; D128], [3i32; D128], [1i32; D128]];
    let view = prepared
        .expanded
        .shared_matrix()
        .ring_view::<D128>(1, T_LEN)
        .expect("field view");
    let flat_source = StreamedASource::Flat(view.as_slice());
    let streamed = prepared
        .with_shared_ntt::<D128, _>(
            NttCacheKey::from_matrix_shape(D128, 1, 1, NttTransformDomain::Cyclic).unwrap(),
            |ntt| {
                fused_split_eq_quotients_streamed_prover_bounds(
                    ntt,
                    &flat_source,
                    1,
                    1,
                    1,
                    &e_hat,
                    &t_hat,
                    &z_segment,
                    3,
                    2,
                    8,
                )
            },
        )
        .expect("streamed rows")
        .expect("chunked t path streams");
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
                            1,
                            &e_hat,
                            &t_hat,
                            &z_segment,
                            3,
                            2,
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
    let freed = prepared.drop_built_ntt_slots();
    assert!(freed > 0);
    assert_eq!(prepared.shared_ntt_cache_bytes(), 0);
    prepared
        .with_shared_ntt::<D, _>(key, |_| Ok(()))
        .expect("slot rebuilds after release");
    assert!(prepared.shared_ntt_cache_bytes() > 0);
}
