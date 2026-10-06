//! Streaming XOF cursor used by the signed-sparse fold-challenge sampler.
//!
//! Every fold coordinate gets a fresh indexed SHAKE256 stream. A sampler may
//! reset this cursor between coordinates and consume rate lanes directly.
//!
//! The cursor's `next_*` helpers use bitmask rejection sampling, so every
//! returned value is uniform over the requested range with no modulo bias.

const SHAKE256_RATE: usize = 136;
const SHAKE_DOMAIN_SUFFIX: u8 = 0x1f;
const GROUP_ROOT_LEN: usize = 32;
const COORDINATE_INPUT_LEN: usize = GROUP_ROOT_LEN + size_of::<u64>();

/// SHAKE256 state after absorbing one dedicated group root. Cloning this state
/// gives each coordinate a fresh XOF without repeating the root absorption.
#[derive(Clone)]
pub(crate) struct IndexedXofPrefix {
    state: [u64; 25],
}

impl IndexedXofPrefix {
    pub(crate) fn new(seed: &[u8]) -> Result<Self, &'static str> {
        if seed.len() != GROUP_ROOT_LEN {
            return Err("indexed sparse challenge group root must be exactly 32 bytes");
        }
        let mut state = [0u64; 25];
        absorb_bytes(&mut state, 0, seed);
        Ok(Self { state })
    }

    fn reader(&self, coordinate_index: u64) -> IndexedShakeReader {
        let mut state = self.state;
        absorb_bytes(&mut state, GROUP_ROOT_LEN, &coordinate_index.to_le_bytes());
        xor_state_byte(&mut state, COORDINATE_INPUT_LEN, SHAKE_DOMAIN_SUFFIX);
        xor_state_byte(&mut state, SHAKE256_RATE - 1, 0x80);
        f1600(&mut state);
        IndexedShakeReader { state, pos: 0 }
    }
}

/// Keccak-f[1600]: the guest's Keccak inline when enabled, else the `keccak`
/// crate.
#[inline(always)]
fn f1600(state: &mut [u64; 25]) {
    #[cfg(all(feature = "keccak-inline", target_arch = "riscv64"))]
    {
        // The inline XORs one 136-byte rate block into the state before the
        // permutation; an all-zero block leaves the state unchanged first.
        #[repr(align(8))]
        struct Block([u8; 136]);
        static ZERO: Block = Block([0; 136]);
        // SAFETY: `state` is 25 aligned, writable words; `ZERO` is 136
        // aligned, readable bytes; the two do not overlap.
        unsafe {
            jolt_inlines_keccak256::keccak256_absorb_permute(state.as_mut_ptr(), ZERO.0.as_ptr());
        }
    }
    #[cfg(not(all(feature = "keccak-inline", target_arch = "riscv64")))]
    keccak::f1600(state);
}

fn absorb_bytes(state: &mut [u64; 25], offset: usize, bytes: &[u8]) {
    for (index, &byte) in bytes.iter().enumerate() {
        xor_state_byte(state, offset + index, byte);
    }
}

fn xor_state_byte(state: &mut [u64; 25], index: usize, byte: u8) {
    state[index / 8] ^= u64::from(byte) << (8 * (index % 8));
}

struct IndexedShakeReader {
    state: [u64; 25],
    pos: usize,
}

impl IndexedShakeReader {
    fn read(&mut self, out: &mut [u8]) {
        let mut written = 0;
        while written < out.len() {
            if self.pos == SHAKE256_RATE {
                f1600(&mut self.state);
                self.pos = 0;
            }
            let available = SHAKE256_RATE - self.pos;
            let take = available.min(out.len() - written);
            let end = self.pos + take;
            while self.pos < end {
                let lane = self.state[self.pos / 8].to_le_bytes();
                let lane_offset = self.pos % 8;
                let lane_take = (8 - lane_offset).min(end - self.pos);
                out[written..written + lane_take]
                    .copy_from_slice(&lane[lane_offset..lane_offset + lane_take]);
                self.pos += lane_take;
                written += lane_take;
            }
        }
    }
}

/// Streaming cursor reading directly from the SHAKE256 rate lanes.
pub(crate) struct XofCursor {
    reader: IndexedShakeReader,
}

impl XofCursor {
    /// Allocate reusable cursor storage before its first indexed reset.
    pub(crate) fn new() -> Self {
        Self {
            reader: IndexedShakeReader {
                state: [0u64; 25],
                pos: SHAKE256_RATE,
            },
        }
    }

    /// Build the canonical stream for one claim-major fold coordinate.
    #[cfg(test)]
    pub(crate) fn from_indexed_prefix(prefix: &IndexedXofPrefix, coordinate_index: u64) -> Self {
        Self {
            reader: prefix.reader(coordinate_index),
        }
    }

    /// Reset to another coordinate stream without allocating.
    pub(crate) fn reset_indexed_prefix(
        &mut self,
        prefix: &IndexedXofPrefix,
        coordinate_index: u64,
    ) {
        self.reader = prefix.reader(coordinate_index);
    }

    #[inline]
    fn next_u8(&mut self) -> u8 {
        if self.reader.pos == SHAKE256_RATE {
            f1600(&mut self.reader.state);
            self.reader.pos = 0;
        }
        let pos = self.reader.pos;
        self.reader.pos += 1;
        (self.reader.state[pos / 8] >> (8 * (pos % 8))) as u8
    }

    /// Copy the next stream bytes, permuting only at a rate boundary.
    #[inline]
    pub(crate) fn fill_bytes(&mut self, out: &mut [u8]) {
        self.reader.read(out);
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        u32::from_le_bytes(std::array::from_fn(|_| self.next_u8()))
    }

    /// Uniformly sample from `0..modulus` using bitmask rejection sampling
    /// with minimal XOF consumption. Uses 1-byte reads when the modulus
    /// fits in 8 bits, 2-byte reads for 16 bits, else 4 bytes.
    #[inline]
    pub(crate) fn next_usize_mod(&mut self, modulus: usize) -> usize {
        debug_assert!(modulus > 0);
        if modulus == 1 {
            return 0;
        }
        let bits = usize::BITS - (modulus - 1).leading_zeros();
        if bits <= 8 {
            let mask = ((1u16 << bits) - 1) as u8;
            loop {
                let val = (self.next_u8() & mask) as usize;
                if val < modulus {
                    return val;
                }
            }
        } else if bits <= 16 {
            let mask = (1usize << bits) - 1;
            loop {
                let lo = self.next_u8() as usize;
                let hi = self.next_u8() as usize;
                let val = (lo | (hi << 8)) & mask;
                if val < modulus {
                    return val;
                }
            }
        } else {
            let mask: usize = (1 << bits) - 1;
            loop {
                let val = (self.next_u32() as usize) & mask;
                if val < modulus {
                    return val;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shake::digest::{ExtendableOutput, Update, XofReader};
    use shake::Shake256;

    #[test]
    fn indexed_cursor_uses_the_canonical_coordinate_input() {
        let seed = [0x5au8; 32];
        let index = 0x0102_0304_0506_0708u64;
        let mut expected_xof = Shake256::default();
        expected_xof.update(&seed);
        expected_xof.update(&index.to_le_bytes());
        let mut expected_reader = expected_xof.finalize_xof();
        let mut expected = [0u8; 384];
        expected_reader.read(&mut expected);

        let prefix = IndexedXofPrefix::new(&seed).unwrap();
        let mut cursor = XofCursor::from_indexed_prefix(&prefix, index);
        let mut actual = [0u8; 384];
        cursor.fill_bytes(&mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn indexed_prefix_requires_the_canonical_root_width() {
        assert_eq!(
            IndexedXofPrefix::new(&[0u8; GROUP_ROOT_LEN - 1]).err(),
            Some("indexed sparse challenge group root must be exactly 32 bytes")
        );
        assert_eq!(
            IndexedXofPrefix::new(&[0u8; GROUP_ROOT_LEN + 1]).err(),
            Some("indexed sparse challenge group root must be exactly 32 bytes")
        );
    }

    #[test]
    fn resetting_an_indexed_cursor_matches_a_fresh_cursor() {
        let seed = [0xabu8; 32];
        let prefix = IndexedXofPrefix::new(&seed).unwrap();
        let mut reused = XofCursor::from_indexed_prefix(&prefix, 0);
        reused.reset_indexed_prefix(&prefix, u64::MAX);
        let mut fresh = XofCursor::from_indexed_prefix(&prefix, u64::MAX);
        let mut reused_bytes = [0u8; 96];
        let mut fresh_bytes = [0u8; 96];
        reused.fill_bytes(&mut reused_bytes);
        fresh.fill_bytes(&mut fresh_bytes);
        assert_eq!(reused_bytes, fresh_bytes);
    }

    #[test]
    fn next_u32_preserves_stream_bytes_across_a_rate_boundary() {
        let prefix = IndexedXofPrefix::new(&[0x3cu8; GROUP_ROOT_LEN]).unwrap();
        let mut expected_cursor = XofCursor::from_indexed_prefix(&prefix, 17);
        let mut expected = [0u8; SHAKE256_RATE + 4];
        expected_cursor.fill_bytes(&mut expected);

        let mut cursor = XofCursor::from_indexed_prefix(&prefix, 17);
        let mut prefix_bytes = [0u8; SHAKE256_RATE - 2];
        cursor.fill_bytes(&mut prefix_bytes);
        assert_eq!(prefix_bytes, expected[..SHAKE256_RATE - 2]);
        assert_eq!(
            cursor.next_u32(),
            u32::from_le_bytes(
                expected[SHAKE256_RATE - 2..SHAKE256_RATE + 2]
                    .try_into()
                    .unwrap()
            )
        );
    }
}
