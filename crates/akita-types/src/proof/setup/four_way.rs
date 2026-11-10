//! Four independent SHAKE256 pages. The permutation follows FIPS 202,
//! sections 3.2 and 3.3; no external implementation is embedded here.
use super::{
    absorb_page_prefix, field_modulus_bytes, AkitaSetupSeed, CanonicalEncoding, Field,
    SetupPageReader, SetupSeedPageXof, SETUP_XOF_RATE,
};
use core::arch::x86_64::*;

pub(super) fn available() -> bool {
    !cfg!(feature = "setup-force-scalar") && std::is_x86_feature_detected!("avx2")
}

// Scratch is bounded by four pages of 32-byte field draws, rounded to rate
// blocks. If sampling consumes more (including rejections), continue each
// lane independently from its saved state, without truncation or replay.
struct PageBlocks {
    blocks: std::vec::IntoIter<[u8; SETUP_XOF_RATE]>,
    state: [u64; 25],
}

impl SetupPageReader for PageBlocks {
    fn read_block(&mut self, block: &mut [u8; SETUP_XOF_RATE]) {
        if let Some(next) = self.blocks.next() {
            *block = next;
        } else {
            keccak::Keccak::new().with_f1600(|permute| permute(&mut self.state));
            for (bytes, word) in block.chunks_exact_mut(8).zip(self.state.iter()) {
                bytes.copy_from_slice(&word.to_le_bytes());
            }
        }
    }
}

/// Caller must check AVX2 and supply exactly four full pages.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn fill_pages<F: Field + CanonicalEncoding>(
    seed: &AkitaSetupSeed,
    first_page: usize,
    output: &mut [F],
) {
    let width = field_modulus_bytes::<F>()
        .iter()
        .skip_while(|&&byte| byte == 0)
        .count();
    let blocks = (seed.derivation.page_field_elements() * width).div_ceil(SETUP_XOF_RATE);
    // SAFETY: this function's AVX2 precondition is checked by range::fill_pages.
    let readers = unsafe { page_streams::<F>(seed, first_page, blocks) };
    for (page, reader) in output
        .chunks_mut(seed.derivation.page_field_elements())
        .zip(readers)
    {
        let mut rng = SetupSeedPageXof {
            reader,
            buffer: [0; SETUP_XOF_RATE],
            position: SETUP_XOF_RATE,
        };
        for value in page {
            *value = F::random(&mut rng);
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn page_streams<F: Field + CanonicalEncoding>(
    seed: &AkitaSetupSeed,
    first_page: usize,
    block_count: usize,
) -> [PageBlocks; 4] {
    let prefixes: [Vec<u8>; 4] = std::array::from_fn(|lane| {
        let mut bytes = Vec::new();
        absorb_page_prefix::<F>(seed, first_page + lane, |part| {
            bytes.extend_from_slice(part)
        });
        bytes
    });
    let mut state = [[0u64; 4]; 25];
    let mut position = 0;
    // Prefix lengths are identical: page indices are fixed-width u64 values.
    for index in 0..prefixes[0].len() {
        for (lane, prefix) in prefixes.iter().enumerate() {
            state[position / 8][lane] ^= u64::from(prefix[index]) << (8 * (position % 8));
        }
        position += 1;
        if position == SETUP_XOF_RATE {
            // SAFETY: the caller checked AVX2 before entering this function.
            unsafe { permute(&mut state) };
            position = 0;
        }
    }
    for word in &mut state[position / 8] {
        *word ^= 0x1f << (8 * (position % 8));
    }
    for word in &mut state[SETUP_XOF_RATE / 8 - 1] {
        *word ^= 1 << 63;
    }
    let mut blocks: [Vec<[u8; SETUP_XOF_RATE]>; 4] =
        std::array::from_fn(|_| Vec::with_capacity(block_count));
    for _ in 0..block_count {
        // SAFETY: the caller checked AVX2 before entering this function.
        unsafe { permute(&mut state) };
        for (lane, output) in blocks.iter_mut().enumerate() {
            let mut block = [0; SETUP_XOF_RATE];
            for (bytes, word) in block.chunks_exact_mut(8).zip(state.iter()) {
                bytes.copy_from_slice(&word[lane].to_le_bytes());
            }
            output.push(block);
        }
    }
    let mut lane = 0;
    blocks.map(|blocks| {
        let reader = PageBlocks {
            blocks: blocks.into_iter(),
            state: state.map(|word| word[lane]),
        };
        lane += 1;
        reader
    })
}

// State layout is x + 5*y; each vector contains that word from four sponges.
#[target_feature(enable = "avx2")]
unsafe fn permute(state: &mut [[u64; 4]; 25]) {
    const ROTATION: [i64; 25] = [
        0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56,
        14,
    ];
    const ROUND: [u64; 24] = [
        0x0000000000000001,
        0x0000000000008082,
        0x800000000000808a,
        0x8000000080008000,
        0x000000000000808b,
        0x0000000080000001,
        0x8000000080008081,
        0x8000000000008009,
        0x000000000000008a,
        0x0000000000000088,
        0x0000000080008009,
        0x000000008000000a,
        0x000000008000808b,
        0x800000000000008b,
        0x8000000000008089,
        0x8000000000008003,
        0x8000000000008002,
        0x8000000000000080,
        0x000000000000800a,
        0x800000008000000a,
        0x8000000080008081,
        0x8000000000008080,
        0x0000000080000001,
        0x8000000080008008,
    ];
    let mut a: [__m256i; 25] =
        state.map(|w| _mm256_set_epi64x(w[3] as i64, w[2] as i64, w[1] as i64, w[0] as i64));
    for round in ROUND {
        let mut c = [_mm256_setzero_si256(); 5];
        for x in 0..5 {
            for y in 0..5 {
                c[x] = _mm256_xor_si256(c[x], a[x + 5 * y]);
            }
        }
        for x in 0..5 {
            let next = c[(x + 1) % 5];
            let rotated =
                _mm256_or_si256(_mm256_slli_epi64::<1>(next), _mm256_srli_epi64::<63>(next));
            let d = _mm256_xor_si256(c[(x + 4) % 5], rotated);
            for y in 0..5 {
                a[x + 5 * y] = _mm256_xor_si256(a[x + 5 * y], d);
            }
        }
        let mut b = [_mm256_setzero_si256(); 25];
        for x in 0..5 {
            for y in 0..5 {
                let i = x + 5 * y;
                // AVX2 variable shifts by 64 yield zero, so rotation zero works.
                b[y + 5 * ((2 * x + 3 * y) % 5)] = _mm256_or_si256(
                    _mm256_sllv_epi64(a[i], _mm256_set1_epi64x(ROTATION[i])),
                    _mm256_srlv_epi64(a[i], _mm256_set1_epi64x(64 - ROTATION[i])),
                );
            }
        }
        for x in 0..5 {
            for y in 0..5 {
                a[x + 5 * y] = _mm256_xor_si256(
                    b[x + 5 * y],
                    _mm256_andnot_si256(b[(x + 1) % 5 + 5 * y], b[(x + 2) % 5 + 5 * y]),
                );
            }
        }
        a[0] = _mm256_xor_si256(a[0], _mm256_set1_epi64x(round as i64));
    }
    for (word, vector) in state.iter_mut().zip(a) {
        // SAFETY: AVX2 is a caller precondition; word is a writable 32-byte
        // array and storeu does not require vector alignment.
        unsafe { _mm256_storeu_si256(word.as_mut_ptr().cast(), vector) };
    }
}

#[cfg(test)]
mod tests;
