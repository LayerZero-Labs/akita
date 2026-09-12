//! Runtime-dispatched x86 AES-CTR backends.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

const ROUND_KEYS: usize = 11;
const AESNI_BATCH_BLOCKS: usize = 8;
const VAES256_BATCH_VECTORS: usize = 8;

type RoundKeys = [__m128i; ROUND_KEYS];
type RoundKeys256 = [__m256i; ROUND_KEYS];
type RoundKeys512 = [__m512i; ROUND_KEYS];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HardwareKind {
    AesNi,
    Vaes256,
    Vaes512,
}

const fn select_backend(
    aes: bool,
    sse2: bool,
    avx2: bool,
    avx512f: bool,
    vaes: bool,
) -> Option<HardwareKind> {
    if !aes || !sse2 {
        None
    } else if avx512f && vaes {
        Some(HardwareKind::Vaes512)
    } else if avx2 && vaes {
        Some(HardwareKind::Vaes256)
    } else {
        Some(HardwareKind::AesNi)
    }
}

#[allow(
    clippy::large_enum_variant,
    reason = "inline round keys keep verifier-reachable setup allocation-free"
)]
enum Backend {
    AesNi(RoundKeys),
    Vaes256(RoundKeys256),
    Vaes512x4(RoundKeys512),
    #[cfg(test)]
    Vaes512x8(RoundKeys512),
    #[cfg(test)]
    Vaes512x16(RoundKeys512),
}

pub(super) struct HardwareBackend(Backend);

impl HardwareBackend {
    pub(super) fn detect(key: &[u8; 16]) -> Option<Self> {
        let kind = select_backend(
            is_x86_feature_detected!("aes"),
            is_x86_feature_detected!("sse2"),
            is_x86_feature_detected!("avx2"),
            is_x86_feature_detected!("avx512f"),
            is_x86_feature_detected!("vaes"),
        )?;
        // SAFETY: AES and SSE2 support were detected above.
        let keys = unsafe { expand_key(key) };
        match kind {
            HardwareKind::AesNi => Some(Self(Backend::AesNi(keys))),
            // SAFETY: AVX2 and VAES support were detected by `select_backend`.
            HardwareKind::Vaes256 => Some(Self(Backend::Vaes256(unsafe { broadcast256(&keys) }))),
            // SAFETY: AVX-512F and VAES support were detected by `select_backend`.
            HardwareKind::Vaes512 => Some(Self(Backend::Vaes512x4(unsafe { broadcast512(&keys) }))),
        }
    }

    #[cfg(test)]
    pub(super) fn supported_for_tests(key: &[u8; 16]) -> Vec<Self> {
        if !is_x86_feature_detected!("aes") || !is_x86_feature_detected!("sse2") {
            return Vec::new();
        }
        // SAFETY: AES support was detected above.
        let keys = unsafe { expand_key(key) };
        let mut backends = vec![Self(Backend::AesNi(keys))];
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("vaes") {
            // SAFETY: AVX2 support was detected above.
            backends.push(Self(Backend::Vaes256(unsafe { broadcast256(&keys) })));
        }
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("vaes") {
            // SAFETY: AVX-512F support was detected above.
            let keys512 = unsafe { broadcast512(&keys) };
            backends.push(Self(Backend::Vaes512x4(keys512)));
            backends.push(Self(Backend::Vaes512x8(keys512)));
            backends.push(Self(Backend::Vaes512x16(keys512)));
        }
        backends
    }

    pub(super) fn fill(&self, base_low: u64, nonce: u64, output: &mut [u8]) {
        // Every constructor checks the complete feature set required by its backend.
        unsafe {
            match &self.0 {
                Backend::AesNi(keys) => fill_aesni(keys, base_low, nonce, output),
                Backend::Vaes256(keys) => fill_vaes256(keys, base_low, nonce, output),
                Backend::Vaes512x4(keys) => fill_vaes512::<4>(keys, base_low, nonce, output),
                #[cfg(test)]
                Backend::Vaes512x8(keys) => fill_vaes512::<8>(keys, base_low, nonce, output),
                #[cfg(test)]
                Backend::Vaes512x16(keys) => fill_vaes512::<16>(keys, base_low, nonce, output),
            }
        }
    }

    #[cfg(test)]
    pub(super) const fn name(&self) -> &'static str {
        match &self.0 {
            Backend::AesNi(_) => "AES-NI",
            Backend::Vaes256(_) => "VAES256",
            Backend::Vaes512x4(_) => "VAES512x4",
            #[cfg(test)]
            Backend::Vaes512x8(_) => "VAES512x8",
            #[cfg(test)]
            Backend::Vaes512x16(_) => "VAES512x16",
        }
    }

    #[cfg(test)]
    pub(super) fn rebuild(&self, key: &[u8; 16]) -> Self {
        // The test constructors only enumerate backends whose features were detected.
        let keys = unsafe { expand_key(key) };
        Self(match &self.0 {
            Backend::AesNi(_) => Backend::AesNi(keys),
            Backend::Vaes256(_) => Backend::Vaes256(unsafe { broadcast256(&keys) }),
            Backend::Vaes512x4(_) => Backend::Vaes512x4(unsafe { broadcast512(&keys) }),
            #[cfg(test)]
            Backend::Vaes512x8(_) => Backend::Vaes512x8(unsafe { broadcast512(&keys) }),
            #[cfg(test)]
            Backend::Vaes512x16(_) => Backend::Vaes512x16(unsafe { broadcast512(&keys) }),
        })
    }
}

#[target_feature(enable = "avx2")]
unsafe fn broadcast256(keys: &RoundKeys) -> RoundKeys256 {
    keys.map(|key| _mm256_broadcastsi128_si256(key))
}

#[target_feature(enable = "avx512f")]
unsafe fn broadcast512(keys: &RoundKeys) -> RoundKeys512 {
    keys.map(|key| _mm512_broadcast_i32x4(key))
}

#[target_feature(enable = "aes,sse2")]
unsafe fn expand_key(key: &[u8; 16]) -> RoundKeys {
    #[target_feature(enable = "aes,sse2")]
    unsafe fn expand_round<const RCON: i32>(keys: &mut RoundKeys, position: usize) {
        let mut current = keys[position - 1];
        let mut assist = _mm_aeskeygenassist_si128(current, RCON);
        assist = _mm_shuffle_epi32(assist, 0xff);
        let mut shifted = _mm_slli_si128(current, 4);
        current = _mm_xor_si128(current, shifted);
        shifted = _mm_slli_si128(shifted, 4);
        current = _mm_xor_si128(current, shifted);
        shifted = _mm_slli_si128(shifted, 4);
        current = _mm_xor_si128(current, shifted);
        keys[position] = _mm_xor_si128(current, assist);
    }

    let mut keys = [_mm_setzero_si128(); ROUND_KEYS];
    keys[0] = _mm_loadu_si128(key.as_ptr().cast());
    expand_round::<0x01>(&mut keys, 1);
    expand_round::<0x02>(&mut keys, 2);
    expand_round::<0x04>(&mut keys, 3);
    expand_round::<0x08>(&mut keys, 4);
    expand_round::<0x10>(&mut keys, 5);
    expand_round::<0x20>(&mut keys, 6);
    expand_round::<0x40>(&mut keys, 7);
    expand_round::<0x80>(&mut keys, 8);
    expand_round::<0x1b>(&mut keys, 9);
    expand_round::<0x36>(&mut keys, 10);
    keys
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn counter_block(low: u64, nonce: u64) -> __m128i {
    _mm_set_epi64x(nonce as i64, low as i64)
}

#[target_feature(enable = "aes,sse2")]
unsafe fn encrypt_aesni(keys: &RoundKeys, mut block: __m128i) -> __m128i {
    block = _mm_xor_si128(block, keys[0]);
    for key in &keys[1..ROUND_KEYS - 1] {
        block = _mm_aesenc_si128(block, *key);
    }
    _mm_aesenclast_si128(block, keys[ROUND_KEYS - 1])
}

#[target_feature(enable = "aes,sse2")]
unsafe fn fill_aesni(keys: &RoundKeys, base_low: u64, nonce: u64, output: &mut [u8]) {
    let full_len = output.len() / 16 * 16;
    let (blocks, tail) = output.split_at_mut(full_len);
    let mut block_index = 0usize;
    let mut chunks = blocks.chunks_exact_mut(AESNI_BATCH_BLOCKS * 16);
    for chunk in &mut chunks {
        let mut batch = [_mm_setzero_si128(); AESNI_BATCH_BLOCKS];
        batch[0] = counter_block(base_low.wrapping_add(block_index as u64), nonce);
        let increment = _mm_set_epi64x(0, 1);
        for offset in 1..AESNI_BATCH_BLOCKS {
            batch[offset] = _mm_add_epi64(batch[offset - 1], increment);
        }
        for (round, key) in keys[..ROUND_KEYS - 1].iter().enumerate() {
            for block in &mut batch {
                *block = if round == 0 {
                    _mm_xor_si128(*block, *key)
                } else {
                    _mm_aesenc_si128(*block, *key)
                };
            }
        }
        for block in &mut batch {
            *block = _mm_aesenclast_si128(*block, keys[ROUND_KEYS - 1]);
        }
        for (offset, block) in batch.iter().enumerate() {
            _mm_storeu_si128(chunk.as_mut_ptr().add(offset * 16).cast(), *block);
        }
        block_index += AESNI_BATCH_BLOCKS;
    }
    for chunk in chunks.into_remainder().chunks_exact_mut(16) {
        let block = encrypt_aesni(
            keys,
            counter_block(base_low.wrapping_add(block_index as u64), nonce),
        );
        _mm_storeu_si128(chunk.as_mut_ptr().cast(), block);
        block_index += 1;
    }
    if !tail.is_empty() {
        let block = encrypt_aesni(
            keys,
            counter_block(base_low.wrapping_add(block_index as u64), nonce),
        );
        let mut final_block = [0u8; 16];
        _mm_storeu_si128(final_block.as_mut_ptr().cast(), block);
        tail.copy_from_slice(&final_block[..tail.len()]);
    }
}

#[target_feature(enable = "aes,avx2,sse2,vaes")]
unsafe fn fill_vaes256(keys: &RoundKeys256, base_low: u64, nonce: u64, output: &mut [u8]) {
    const BATCH_BLOCKS: usize = VAES256_BATCH_VECTORS * 2;
    let mut chunks = output.chunks_exact_mut(BATCH_BLOCKS * 16);
    let mut block_index = 0usize;
    for chunk in &mut chunks {
        let mut batch = [_mm256_setzero_si256(); VAES256_BATCH_VECTORS];
        let low = base_low.wrapping_add(block_index as u64);
        batch[0] = _mm256_set_epi64x(
            nonce as i64,
            low.wrapping_add(1) as i64,
            nonce as i64,
            low as i64,
        );
        let increment = _mm256_set_epi64x(0, 2, 0, 2);
        for offset in 1..VAES256_BATCH_VECTORS {
            batch[offset] = _mm256_add_epi64(batch[offset - 1], increment);
        }
        for (round, key) in keys.iter().enumerate() {
            for vector in &mut batch {
                *vector = if round == 0 {
                    _mm256_xor_si256(*vector, *key)
                } else if round == ROUND_KEYS - 1 {
                    _mm256_aesenclast_epi128(*vector, *key)
                } else {
                    _mm256_aesenc_epi128(*vector, *key)
                };
            }
        }
        for (offset, vector) in batch.iter().enumerate() {
            _mm256_storeu_si256(chunk.as_mut_ptr().add(offset * 32).cast(), *vector);
        }
        block_index += BATCH_BLOCKS;
    }
    let remainder = chunks.into_remainder();
    if !remainder.is_empty() {
        fill_aesni(
            &keys.map(|key| _mm256_castsi256_si128(key)),
            base_low.wrapping_add(block_index as u64),
            nonce,
            remainder,
        );
    }
}

#[target_feature(enable = "aes,avx512f,sse2,vaes")]
unsafe fn fill_vaes512<const BATCH_VECTORS: usize>(
    keys: &RoundKeys512,
    base_low: u64,
    nonce: u64,
    output: &mut [u8],
) {
    let batch_blocks = BATCH_VECTORS * 4;
    let mut chunks = output.chunks_exact_mut(batch_blocks * 16);
    let mut block_index = 0usize;
    for chunk in &mut chunks {
        let mut batch = [_mm512_setzero_si512(); BATCH_VECTORS];
        let low = base_low.wrapping_add(block_index as u64);
        batch[0] = _mm512_set_epi64(
            nonce as i64,
            low.wrapping_add(3) as i64,
            nonce as i64,
            low.wrapping_add(2) as i64,
            nonce as i64,
            low.wrapping_add(1) as i64,
            nonce as i64,
            low as i64,
        );
        let increment = _mm512_set_epi64(0, 4, 0, 4, 0, 4, 0, 4);
        for offset in 1..BATCH_VECTORS {
            batch[offset] = _mm512_add_epi64(batch[offset - 1], increment);
        }
        for (round, key) in keys.iter().enumerate() {
            for vector in &mut batch {
                *vector = if round == 0 {
                    _mm512_xor_si512(*vector, *key)
                } else if round == ROUND_KEYS - 1 {
                    _mm512_aesenclast_epi128(*vector, *key)
                } else {
                    _mm512_aesenc_epi128(*vector, *key)
                };
            }
        }
        for (offset, vector) in batch.iter().enumerate() {
            _mm512_storeu_si512(chunk.as_mut_ptr().add(offset * 64).cast(), *vector);
        }
        block_index += batch_blocks;
    }
    let remainder = chunks.into_remainder();
    if !remainder.is_empty() {
        fill_aesni(
            &keys.map(|key| _mm512_castsi512_si128(key)),
            base_low.wrapping_add(block_index as u64),
            nonce,
            remainder,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_selection_requires_each_backend_feature() {
        assert_eq!(select_backend(false, true, true, true, true), None);
        assert_eq!(select_backend(true, false, true, true, true), None);
        assert_eq!(
            select_backend(true, true, false, false, true),
            Some(HardwareKind::AesNi)
        );
        assert_eq!(
            select_backend(true, true, true, true, false),
            Some(HardwareKind::AesNi)
        );
        assert_eq!(
            select_backend(true, true, true, false, true),
            Some(HardwareKind::Vaes256)
        );
        assert_eq!(
            select_backend(true, true, true, true, true),
            Some(HardwareKind::Vaes512)
        );
    }
}
