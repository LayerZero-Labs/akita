//! Four independent SHAKE256 pages. The permutation follows FIPS 202,
//! sections 3.2 and 3.3; no external implementation is embedded here.
use super::{
    absorb_page_prefix, field_modulus_bytes, AkitaSetupSeed, CanonicalEncoding, Field,
    SetupPageReader, SetupSeedPageXof, SETUP_XOF_RATE,
};
use akita_params::keccak_four_way::permute;

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

#[cfg(test)]
mod tests;
