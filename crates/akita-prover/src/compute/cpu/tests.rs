use super::*;
use crate::compute::backend::{
    ComputeBackendSetup, CyclicRowsComputeBackend, DigitRowsComputeBackend,
    RingSwitchComputeBackend,
};
use crate::compute::plans::RingSwitchRelationRowsPlan;
use crate::kernels::linear::{
    fused_split_eq_quotients, mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic,
};
use crate::AkitaProverSetup;
use akita_field::{Prime128Offset275, Prime64Offset59};
use akita_types::SetupMatrixEnvelope;
use std::sync::Arc;

type F = Prime64Offset59;
const D: usize = 64;

fn setup_envelope(max_setup_len: usize) -> SetupMatrixEnvelope {
    SetupMatrixEnvelope { max_setup_len }
}

fn prepared() -> CpuPreparedSetup<F> {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(D)).unwrap();
    CpuBackend.prepare_setup(&setup).unwrap()
}

#[test]
fn cpu_prepared_setup_identity_rejects_mismatched_setup() {
    let setup_a =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(D)).unwrap();
    let setup_b =
        AkitaProverSetup::<F>::generate_with_capacity(9, 1, D, setup_envelope(D)).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup_a).unwrap();

    CpuBackend
        .validate_prepared_setup(&prepared, setup_a.expanded.as_ref())
        .expect("matching setup");
    assert!(
        CpuBackend
            .validate_prepared_setup(&prepared, setup_b.expanded.as_ref())
            .is_err(),
        "prepared context must stay bound to the setup used to create it"
    );
}

#[test]
fn cpu_prepared_setup_identity_accepts_equivalent_setup() {
    let setup_a =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(D)).unwrap();
    let setup_b =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(D)).unwrap();
    assert!(!Arc::ptr_eq(&setup_a.expanded, &setup_b.expanded));

    let prepared = CpuBackend.prepare_setup(&setup_a).unwrap();

    CpuBackend
        .validate_prepared_setup(&prepared, setup_b.expanded.as_ref())
        .expect("equivalent deterministic setup should validate");
}

#[test]
fn cpu_prepared_setup_reports_checked_crt_capacity_profile() {
    let prepared = prepared();
    let profile = prepared.shared_ntt_profile::<D>().expect("profile");

    assert_eq!(profile.profile_id, "Q64/3xi32");
    assert_eq!(profile.num_primes, 3);
    assert_eq!(profile.limb_bits, 32);
    assert_eq!(profile.max_i8_log_basis, MAX_I8_LOG_BASIS);
    assert!(profile.balanced_digit_safe_width > 0);
    assert!(profile.raw_i8_safe_width > 0);
}

#[test]
fn prepare_setup_registers_envelope_ntt_contract() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(D)).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared");
    // Registration reserves the slot without transforming the matrix; the
    // build is deferred to first use so the transformed A does not sit in
    // memory across stages that never touch it.
    assert_eq!(prepared.shared_ntt_cache_bytes(), 0);
    let envelope_key =
        NttCacheKey::from_envelope(setup.expanded.as_ref(), D).expect("envelope key");
    assert!(prepared
        .shared_ntt
        .lock()
        .unwrap()
        .contains_key(&envelope_key));
    assert_eq!(prepared.ntt_slot_build_count.load(Ordering::Relaxed), 0);

    prepared
        .with_shared_ntt::<D, _>(4, |_slot| Ok(()))
        .expect("first use builds a slot sized to the request");
    let sized_bytes = prepared.shared_ntt_cache_bytes();
    assert!(sized_bytes > 0);
    assert_eq!(prepared.ntt_slot_build_count.load(Ordering::Relaxed), 1);
    // The envelope cell stays reserved: the sized build must be smaller.
    let full_key = NttCacheKey {
        ring_d: D,
        num_ring_elements: envelope_key.num_ring_elements,
    };
    CpuBackend
        .ensure_ntt_slot(&prepared, full_key)
        .expect("explicit envelope warm still builds the full slot");
    assert!(prepared.shared_ntt_cache_bytes() > sized_bytes);

    prepared
        .with_shared_ntt::<D, _>(4, |_slot| Ok(()))
        .expect("subsequent uses hit a built covering slot");
    assert_eq!(prepared.ntt_slot_build_count.load(Ordering::Relaxed), 2);
}

#[test]
fn prepare_expanded_with_envelope_ntt_builds_envelope_slot() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(D)).unwrap();
    let prepared = CpuBackend
        .prepare_expanded_with_envelope_ntt::<D>(setup.expanded.clone())
        .expect("prepared");
    assert!(prepared.shared_ntt_cache_bytes() > 0);
    let envelope_key =
        NttCacheKey::from_envelope(setup.expanded.as_ref(), D).expect("envelope key");
    assert!(prepared
        .shared_ntt
        .lock()
        .unwrap()
        .contains_key(&envelope_key));
}

#[test]
fn cpu_prepared_setup_warms_multiple_ntt_slots() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(D)).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared");
    let envelope_key =
        NttCacheKey::from_envelope(setup.expanded.as_ref(), D).expect("envelope key");
    let partial_key = NttCacheKey {
        ring_d: D,
        num_ring_elements: 1,
    };
    CpuBackend
        .ensure_ntt_slot(&prepared, partial_key)
        .expect("warm partial slot");
    assert!(prepared.shared_ntt_cache_bytes() > 0);
    let cache = prepared.shared_ntt.lock().unwrap();
    assert!(cache.contains_key(&envelope_key));
    assert!(cache.contains_key(&partial_key));
    drop(cache);
    let miss = NttCacheKey {
        ring_d: D,
        num_ring_elements: 99_999,
    };
    assert!(!prepared.shared_ntt.lock().unwrap().contains_key(&miss));
}

#[test]
fn concurrent_same_key_ntt_warm_builds_once() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(D)).unwrap();
    let prepared = CpuBackend
        .prepare_expanded::<D>(setup.expanded.clone())
        .expect("empty prepared setup");
    let key = NttCacheKey::from_envelope(setup.expanded.as_ref(), D).expect("envelope key");

    std::thread::scope(|scope| {
        for _ in 0..8 {
            let prepared = &prepared;
            scope.spawn(move || {
                CpuBackend
                    .ensure_ntt_slot(prepared, key)
                    .expect("warm shared NTT slot");
            });
        }
    });
    CpuBackend
        .ensure_ntt_slot(&prepared, key)
        .expect("repeated warm is a no-op");

    assert_eq!(prepared.ntt_slot_build_count.load(Ordering::Relaxed), 1);
    assert!(prepared.shared_ntt_cache_bytes() > 0);
}

#[test]
fn cpu_digit_rows_match_direct_kernel() {
    let prepared = prepared();
    let digits = vec![[1i8; D], [-1i8; D], [2i8; D]];
    let log_basis = 3;
    let via_backend = CpuBackend
        .digit_rows::<D>(&prepared, 2, &digits, log_basis)
        .expect("backend digit rows");
    let direct = prepared
        .with_shared_ntt::<D, _>(2 * digits.len(), |ntt| {
            mat_vec_mul_ntt_single_i8(ntt, 2, digits.len(), &digits, log_basis)
        })
        .expect("direct digit rows");
    assert_eq!(via_backend, direct);
}

#[test]
fn cpu_digit_rows_cache_omits_cyclic_transform() {
    let prepared = prepared();
    let digits = vec![[1i8; D], [-1i8; D], [2i8; D]];

    CpuBackend
        .digit_rows::<D>(&prepared, 2, &digits, 3)
        .expect("digit rows");

    let cache = prepared.shared_neg_ntt.lock().unwrap();
    let (key, slot) = cache
        .iter()
        .find_map(|(key, cell)| cell.get().map(|slot| (key, slot)))
        .expect("built negacyclic slot");
    let slot = slot.as_ref().expect("valid negacyclic slot");
    let typed = slot
        .cache
        .downcast_ref::<PreparedNttCache<D>>()
        .expect("typed cache");
    match typed {
        PreparedNttCache::Q64 { cyc, .. } => assert!(cyc.is_none()),
        _ => panic!("Prime64Offset59 must use Q64"),
    }
    assert_eq!(
        slot.cache_bytes,
        key.num_ring_elements * D * 3 * size_of::<i32>()
    );
}

#[test]
fn cpu_q128_digit_rows_cache_is_one_transform_per_ring() {
    let setup = AkitaProverSetup::<Prime128Offset275>::generate_with_capacity(
        8,
        1,
        D,
        setup_envelope(8 * D),
    )
    .unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let digits = vec![[1i8; D], [-1i8; D], [2i8; D]];

    CpuBackend
        .digit_rows::<D>(&prepared, 2, &digits, 3)
        .expect("digit rows");

    let cache = prepared.shared_neg_ntt.lock().unwrap();
    let (key, slot) = cache
        .iter()
        .find_map(|(key, cell)| cell.get().map(|slot| (key, slot)))
        .expect("built negacyclic slot");
    let slot = slot.as_ref().expect("valid negacyclic slot");
    let typed = slot
        .cache
        .downcast_ref::<PreparedNttCache<D>>()
        .expect("typed cache");
    match typed {
        PreparedNttCache::Q128 { cyc, tail, .. } => {
            assert!(cyc.is_none());
            assert!(tail.is_none());
        }
        _ => panic!("Prime128Offset275 must use Q128"),
    }
    assert_eq!(
        slot.cache_bytes,
        key.num_ring_elements * D * 5 * size_of::<i32>()
    );
}

#[test]
fn cpu_digit_rows_reuses_both_transform_slot() {
    let prepared = prepared();
    let key = NttCacheKey::from_envelope(prepared.expanded.as_ref(), D).unwrap();
    CpuBackend.ensure_ntt_slot(&prepared, key).unwrap();
    let builds_before = prepared.ntt_slot_build_count.load(Ordering::Relaxed);

    CpuBackend
        .digit_rows::<D>(&prepared, 2, &[[1i8; D], [-1i8; D], [2i8; D]], 3)
        .expect("digit rows");

    assert!(prepared.shared_neg_ntt.lock().unwrap().is_empty());
    assert_eq!(
        prepared.ntt_slot_build_count.load(Ordering::Relaxed),
        builds_before
    );
}

#[test]
fn cpu_digit_rows_accept_logical_input_longer_than_stride() {
    let prepared = prepared();
    let digits = vec![[1i8; D]; 12];
    let log_basis = 3;
    let via_backend = CpuBackend
        .digit_rows::<D>(&prepared, 2, &digits, log_basis)
        .expect("backend digit rows");
    let direct = prepared
        .with_shared_ntt::<D, _>(2 * digits.len(), |ntt| {
            mat_vec_mul_ntt_single_i8(ntt, 2, digits.len(), &digits, log_basis)
        })
        .expect("direct digit rows");
    assert_eq!(via_backend, direct);
}

#[test]
fn recursive_commit_ignores_commitment_padding_blocks() {
    let prepared = prepared();
    let coeffs = vec![[1i8; D]; 6];
    let rows = CpuBackend
        .recursive_witness_commit_rows(
            &prepared,
            RecursiveWitnessCommitRowsPlan {
                coeffs: &coeffs,
                n_rows: 1,
                num_positions_per_block: 2,
                num_live_blocks: 2,
                num_digits_inner: 1,
                log_basis_inner: 3,
                known_balanced_log_basis: Some(3),
            },
        )
        .expect("recursive commit rows");

    assert_eq!(rows.len(), 2);
}

#[test]
fn cpu_cyclic_digit_rows_match_direct_kernel() {
    let prepared = prepared();
    let digits = vec![[1i8; D], [0i8; D], [-2i8; D], [3i8; D]];
    let log_basis = 3;
    let via_backend = CpuBackend
        .cyclic_digit_rows::<D>(&prepared, 2, &digits, log_basis)
        .expect("backend cyclic digit rows");
    let direct = prepared
        .with_shared_ntt::<D, _>(2 * digits.len(), |ntt| {
            mat_vec_mul_ntt_single_i8_cyclic(ntt, 2, digits.len(), &digits, log_basis)
        })
        .expect("direct cyclic digit rows");
    assert_eq!(via_backend, direct);
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
    let matrix = prepared.expanded.shared_matrix().full();
    let source = StreamedASource::Flat(
        matrix
            .ring_view::<D>(1, extent)
            .expect("field view")
            .as_slice(),
    );
    let streamed = prepared
        .with_shared_ntt::<D, _>(1, |ntt| {
            fused_split_eq_quotients_streamed_prover_bounds(
                ntt, &source, 2, 2, 1, &e_hat, &t_hat, &z_segment, 5, 2, 3,
            )
        })
        .expect("streamed rows")
        .expect("shape is one-shot safe");
    let cached = prepared
        .with_shared_ntt::<D, _>(extent, |ntt| {
            fused_split_eq_quotients_prover_bounds(
                ntt, 2, 2, 1, &e_hat, &t_hat, &z_segment, 5, 2, 3,
            )
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
    let matrix = prepared.expanded.shared_matrix().full();
    let source = StreamedASource::Flat(
        matrix
            .ring_view::<D>(1, extent)
            .expect("field view")
            .as_slice(),
    );
    let streamed = prepared
        .with_shared_ntt::<D, _>(1, |ntt| {
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
        .with_shared_ntt::<D, _>(extent, |ntt| {
            fused_split_eq_quotients_prover_bounds(
                ntt,
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
        .expect("cached rows");
    assert_eq!(streamed, cached);
}

#[test]
fn streamed_chunked_t_rows_match_cached_kernel() {
    const T_LEN: usize = 512;
    let setup = AkitaProverSetup::<Prime128Offset275>::generate_with_capacity(
        8,
        1,
        D,
        setup_envelope(T_LEN * D),
    )
    .unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let e_hat = vec![[1i8; D], [-1i8; D], [1i8; D]];
    let t_hat = vec![[1i8; D]; T_LEN];
    let z_segment = vec![[1i32; D], [-2i32; D], [3i32; D], [1i32; D]];
    let matrix = prepared.expanded.shared_matrix().full();
    let flat_source = StreamedASource::Flat(
        matrix
            .ring_view::<D>(1, T_LEN)
            .expect("field view")
            .as_slice(),
    );
    let deriver = prepared.expanded.shared_matrix().element_deriver();
    let seed_source = StreamedASource::Seed {
        deriver: &deriver,
        len: T_LEN,
    };
    let run = |source: &StreamedASource<'_, Prime128Offset275, D>| {
        prepared
            .with_shared_ntt::<D, _>(1, |ntt| {
                fused_split_eq_quotients_streamed_prover_bounds(
                    ntt, source, 1, 1, 1, &e_hat, &t_hat, &z_segment, 3, 2, 8,
                )
            })
            .expect("streamed rows")
            .expect("chunked t path streams")
    };
    let cached = prepared
        .with_shared_ntt::<D, _>(T_LEN, |ntt| {
            fused_split_eq_quotients_prover_bounds(
                ntt, 1, 1, 1, &e_hat, &t_hat, &z_segment, 3, 2, 8,
            )
        })
        .expect("cached rows");
    assert_eq!(run(&flat_source), cached);
    assert_eq!(run(&seed_source), cached);
}

#[test]
fn seed_derived_elements_match_materialized_matrix() {
    let prepared = prepared();
    let shared = prepared.expanded.shared_matrix();
    let matrix = shared.full();
    let deriver = shared.element_deriver();
    let mut coeffs = [F::zero(); D];
    for idx in [0usize, 1, 7, matrix.total_ring_elements() - 1] {
        deriver.entry_coeffs(idx, &mut coeffs);
        assert_eq!(
            coeffs.as_slice(),
            &matrix.as_field_slice()[idx * D..(idx + 1) * D],
            "seed-derived entry {idx} disagrees with the materialized matrix"
        );
    }
}

#[test]
fn seed_source_relation_rows_match_flat_source() {
    let prepared = prepared();
    let e_hat = vec![[1i8; D], [-1i8; D], [1i8; D]];
    let t_hat = vec![[-1i8; D], [3i8; D]];
    let z_segment = vec![[1i32; D], [-2i32; D], [3i32; D], [5i32; D]];
    let extent = 2usize
        .saturating_mul(e_hat.len())
        .max(2usize.saturating_mul(t_hat.len()))
        .max(z_segment.len());
    let shared = prepared.expanded.shared_matrix();
    let matrix = shared.full();
    let deriver = shared.element_deriver();
    let flat_source =
        StreamedASource::Flat(matrix.ring_view::<D>(1, extent).expect("view").as_slice());
    let seed_source = StreamedASource::Seed {
        deriver: &deriver,
        len: extent,
    };
    let run = |source: &StreamedASource<'_, F, D>| {
        prepared
            .with_shared_ntt::<D, _>(1, |ntt| {
                fused_split_eq_quotients_streamed_prover_bounds(
                    ntt, source, 2, 2, 1, &e_hat, &t_hat, &z_segment, 5, 2, 3,
                )
            })
            .expect("streamed rows")
            .expect("one-shot safe")
    };
    assert_eq!(run(&flat_source), run(&seed_source));
}

#[test]
fn released_matrix_serves_prefix_and_rederives_beyond() {
    let prepared = prepared();
    let shared = prepared.expanded.shared_matrix();
    let full = shared.total_ring_elements();
    let matrix_before = shared.full();
    let freed = prepared.release_setup_matrix_to_prefix(2);
    assert!(freed > 0);
    assert_eq!(
        shared.total_ring_elements(),
        full,
        "metadata must not shrink"
    );
    // Within the prefix: served without derivation, identical contents.
    let prefix = shared.covering_at_dyn(2, D).expect("prefix");
    assert_eq!(
        &prefix.as_field_slice()[..2 * D],
        &matrix_before.as_field_slice()[..2 * D]
    );
    // Beyond the prefix: re-derived, still identical to the original.
    let rederived = shared.covering_at_dyn(full, D).expect("rederived");
    assert_eq!(rederived.as_field_slice(), matrix_before.as_field_slice());
    // Backend paths keep working post-release (slot build reads a prefix).
    let digits = vec![[1i8; D], [-1i8; D]];
    let rows = CpuBackend
        .digit_rows::<D>(&prepared, 1, &digits, 3)
        .expect("digit rows post-release");
    assert_eq!(rows.len(), 1);
}

#[test]
fn cpu_ring_switch_relation_rows_use_distinct_open_and_outer_bases() {
    let prepared = prepared();
    let e_hat = vec![[1i8; D], [-1i8; D]];
    let t_hat = vec![[-1i8; D], [3i8; D]];
    let z_segment = vec![[1i32; D], [-2i32; D], [3i32; D]];
    let via_backend = CpuBackend
        .ring_switch_relation_rows::<D>(
            &prepared,
            RingSwitchRelationRowsPlan {
                n_d: 1,
                n_b: 1,
                n_a: 1,
                e_hat: &e_hat,
                t_hat: &t_hat,
                z_segment: &z_segment,
                z_folded_centered_inf_norm: 3,
                log_basis_open: 2,
                log_basis_outer: 3,
            },
        )
        .expect("backend ring-switch relation rows");
    let direct = prepared
        .with_shared_ntt::<D, _>(z_segment.len(), |ntt| {
            fused_split_eq_quotients(ntt, 1, 1, 1, &e_hat, &t_hat, &z_segment, 3)
        })
        .expect("direct fused split-eq rows");
    assert_eq!(
        (
            via_backend.d_cyclic,
            via_backend.b_cyclic,
            via_backend.a_quotients
        ),
        direct
    );
}
