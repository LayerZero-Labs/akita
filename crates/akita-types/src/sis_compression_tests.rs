#[cfg(test)]
mod tests {
    use crate::ntt_cache_requires_exactness_tail;
    use jolt_field::{Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};

    #[test]
    fn reachable_negative_binary_widths_need_no_exactness_tail() {
        use crate::prepare_compression_ntt_cache;
        use akita_algebra::CyclotomicRing;
        use akita_params::FlatMatrix;

        // First maps at their complete 8 KiB source width.
        assert!(!ntt_cache_requires_exactness_tail::<Prime128OffsetA7F7, 32>(2_048, 1).unwrap());
        assert!(!ntt_cache_requires_exactness_tail::<Prime64Offset59, 64>(1_024, 1).unwrap());
        assert!(!ntt_cache_requires_exactness_tail::<Prime32Offset99, 128>(512, 1).unwrap());

        // Terminal maps at the width of the first map's complete output.
        let q128_d16 = FlatMatrix::from_ring_slice(&vec![
            CyclotomicRing::<Prime128OffsetA7F7, 16>::zero();
            256
        ]);
        let q128_cache =
            prepare_compression_ntt_cache(q128_d16.ring_view::<16>(1, 256).expect("view"))
                .expect("q128/D16 cache");
        assert!(!q128_cache.has_exactness_tail());
        assert!(q128_cache.has_cyclic());

        let q64_d32 =
            FlatMatrix::from_ring_slice(&vec![CyclotomicRing::<Prime64Offset59, 32>::zero(); 128]);
        let q64_cache =
            prepare_compression_ntt_cache(q64_d32.ring_view::<32>(1, 128).expect("view"))
                .expect("q64/D32 cache");
        assert!(!q64_cache.has_exactness_tail());
        assert!(q64_cache.has_cyclic());

        let q32_d64 =
            FlatMatrix::from_ring_slice(&vec![CyclotomicRing::<Prime32Offset99, 64>::zero(); 64]);
        let q32_cache =
            prepare_compression_ntt_cache(q32_d64.ring_view::<64>(1, 64).expect("view"))
                .expect("q32/D64 cache");
        assert!(!q32_cache.has_exactness_tail());
        assert!(q32_cache.has_cyclic());
    }
}
