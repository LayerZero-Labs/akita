#[cfg(test)]
mod tests {
    use crate::ntt_cache_requires_exactness_tail;
    use jolt_field::{Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};

    #[test]
    fn reachable_negative_binary_widths_need_no_exactness_tail() {
        use crate::prepare_compression_ntt_cache;
        use akita_algebra::CyclotomicRing;
        use akita_params::FlatMatrix;

        // First-map dimensions sit in both the protocol NTT band and the compression ladder.
        assert!(!ntt_cache_requires_exactness_tail::<Prime128OffsetA7F7, 16>(4_096, 1).unwrap());
        assert!(!ntt_cache_requires_exactness_tail::<Prime64Offset59, 32>(2_048, 1).unwrap());
        assert!(!ntt_cache_requires_exactness_tail::<Prime32Offset99, 64>(1_024, 1).unwrap());

        // Compression-only dims must use the purpose-aware prep path.
        let q128_d8 = FlatMatrix::from_ring_slice(&vec![
            CyclotomicRing::<Prime128OffsetA7F7, 8>::zero();
            256
        ]);
        let q128_cache =
            prepare_compression_ntt_cache(q128_d8.ring_view::<8>(1, 256).expect("view"))
                .expect("q128/D8 cache");
        assert!(!q128_cache.has_exactness_tail());
        assert!(q128_cache.has_cyclic());

        let q64_d16 =
            FlatMatrix::from_ring_slice(&vec![CyclotomicRing::<Prime64Offset59, 16>::zero(); 128]);
        let q64_cache =
            prepare_compression_ntt_cache(q64_d16.ring_view::<16>(1, 128).expect("view"))
                .expect("q64/D16 cache");
        assert!(!q64_cache.has_exactness_tail());
        assert!(q64_cache.has_cyclic());

        let q32_d32 =
            FlatMatrix::from_ring_slice(&vec![CyclotomicRing::<Prime32Offset99, 32>::zero(); 64]);
        let q32_cache =
            prepare_compression_ntt_cache(q32_d32.ring_view::<32>(1, 64).expect("view"))
                .expect("q32/D32 cache");
        assert!(!q32_cache.has_exactness_tail());
        assert!(q32_cache.has_cyclic());
    }
}
