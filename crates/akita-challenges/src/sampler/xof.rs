//! Streaming XOF cursor used by the signed-sparse fold-challenge sampler.
//!
//! Every fold coordinate gets a fresh indexed SHAKE256 stream. A sampler may
//! reset this cursor between coordinates and reuse its small squeeze buffer.
//!
//! The cursor's `next_*` helpers use bitmask rejection sampling, so every
//! returned value is uniform over the requested range with no modulo bias.

/// Domain separator absorbed into the SHAKE256 instance before the
/// transcript-derived seed. Distinct from any transcript-layer domain tag so
/// that the PRG output cannot be mistaken for a transcript challenge.
const SPARSE_PRG_DOMAIN: &[u8] = b"akita/sparse-challenge-prg";
const FOLD_CHALLENGE_COORDINATE_DOMAIN: &[u8] = b"akita/fold-challenge-coordinate/v1";

const SHAKE256_RATE: usize = 136;
const SHAKE_DOMAIN_SUFFIX: u8 = 0x1f;

/// SHAKE256 state after absorbing the fixed domains and one group root.
/// Cloning this state gives each coordinate a fresh XOF without repeating the
/// common prefix absorption.
#[derive(Clone)]
pub(crate) struct IndexedXofPrefix {
    state: [u64; 25],
    coordinate_offset: usize,
    padding_offset: usize,
}

impl IndexedXofPrefix {
    pub(crate) fn new(seed: &[u8]) -> Result<Self, &'static str> {
        let coordinate_offset = SPARSE_PRG_DOMAIN
            .len()
            .checked_add(FOLD_CHALLENGE_COORDINATE_DOMAIN.len())
            .and_then(|length| length.checked_add(seed.len()))
            .ok_or("indexed sparse challenge prefix length overflow")?;
        let padding_offset = coordinate_offset
            .checked_add(size_of::<u64>())
            .ok_or("indexed sparse challenge input length overflow")?;
        if padding_offset >= SHAKE256_RATE {
            return Err("indexed sparse challenge input exceeds one SHAKE256 rate block");
        }
        let mut state = [0u64; 25];
        absorb_bytes(&mut state, 0, SPARSE_PRG_DOMAIN);
        absorb_bytes(
            &mut state,
            SPARSE_PRG_DOMAIN.len(),
            FOLD_CHALLENGE_COORDINATE_DOMAIN,
        );
        absorb_bytes(
            &mut state,
            SPARSE_PRG_DOMAIN.len() + FOLD_CHALLENGE_COORDINATE_DOMAIN.len(),
            seed,
        );
        Ok(Self {
            state,
            coordinate_offset,
            padding_offset,
        })
    }

    fn reader(&self, coordinate_index: u64) -> IndexedShakeReader {
        let mut state = self.state;
        absorb_bytes(
            &mut state,
            self.coordinate_offset,
            &coordinate_index.to_le_bytes(),
        );
        xor_state_byte(&mut state, self.padding_offset, SHAKE_DOMAIN_SUFFIX);
        xor_state_byte(&mut state, SHAKE256_RATE - 1, 0x80);
        keccak::f1600(&mut state);
        IndexedShakeReader { state, pos: 0 }
    }
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
                keccak::f1600(&mut self.state);
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

/// One coordinate normally consumes well below one SHAKE256 rate block. Fill
/// a bounded buffer so resetting a short coordinate does not squeeze bytes
/// that will be discarded.
const XOF_BUFFER_SIZE: usize = 128;

/// Streaming cursor backed by a SHAKE256 XOF and a reusable sub-rate buffer.
pub(crate) struct XofCursor {
    reader: IndexedShakeReader,
    buf: [u8; XOF_BUFFER_SIZE],
    pos: usize,
    len: usize,
}

impl XofCursor {
    /// Allocate reusable cursor storage before its first indexed reset.
    pub(crate) fn new() -> Self {
        Self {
            reader: IndexedShakeReader {
                state: [0u64; 25],
                pos: SHAKE256_RATE,
            },
            buf: [0u8; XOF_BUFFER_SIZE],
            pos: 0,
            len: 0,
        }
    }

    /// Build the canonical stream for one claim-major fold coordinate.
    #[cfg(test)]
    pub(crate) fn from_indexed_prefix(prefix: &IndexedXofPrefix, coordinate_index: u64) -> Self {
        let mut xof = prefix.reader(coordinate_index);
        let mut buf = [0u8; XOF_BUFFER_SIZE];
        xof.read(&mut buf);
        Self {
            reader: xof,
            buf,
            pos: 0,
            len: XOF_BUFFER_SIZE,
        }
    }

    /// Reset to another coordinate stream without reallocating the buffer.
    pub(crate) fn reset_indexed_prefix(
        &mut self,
        prefix: &IndexedXofPrefix,
        coordinate_index: u64,
    ) {
        self.reader = prefix.reader(coordinate_index);
        self.pos = 0;
        self.len = 0;
        self.refill();
    }

    #[inline]
    fn refill(&mut self) {
        self.reader.read(&mut self.buf);
        self.pos = 0;
        self.len = XOF_BUFFER_SIZE;
    }

    #[inline]
    fn next_u8(&mut self) -> u8 {
        if self.pos >= self.len {
            self.refill();
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        b
    }

    /// Copy `out.len()` bytes from the buffered XOF stream in one pass.
    #[inline]
    pub(crate) fn fill_bytes(&mut self, out: &mut [u8]) {
        let mut off = 0;
        while off < out.len() {
            if self.pos >= self.len {
                self.refill();
            }
            let avail = self.len - self.pos;
            let take = avail.min(out.len() - off);
            out[off..off + take].copy_from_slice(&self.buf[self.pos..self.pos + take]);
            self.pos += take;
            off += take;
        }
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.pos + 4 <= self.len {
            let val = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
            self.pos += 4;
            val
        } else {
            let mut tmp = [0u8; 4];
            for b in &mut tmp {
                *b = self.next_u8();
            }
            u32::from_le_bytes(tmp)
        }
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
    use sha3::digest::{ExtendableOutput, Update, XofReader};
    use sha3::Shake256;

    #[test]
    fn indexed_cursor_uses_the_canonical_coordinate_input() {
        let seed = [0x5au8; 32];
        let index = 0x0102_0304_0506_0708u64;
        let mut expected_xof = Shake256::default();
        expected_xof.update(b"akita/sparse-challenge-prg");
        expected_xof.update(b"akita/fold-challenge-coordinate/v1");
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
    fn indexed_prefix_rejects_inputs_that_need_a_second_absorb_block() {
        let fixed_len =
            SPARSE_PRG_DOMAIN.len() + FOLD_CHALLENGE_COORDINATE_DOMAIN.len() + size_of::<u64>();
        let largest_one_block_seed = vec![0u8; SHAKE256_RATE - fixed_len - 1];
        let index = u64::MAX;
        let mut expected_xof = Shake256::default();
        expected_xof.update(SPARSE_PRG_DOMAIN);
        expected_xof.update(FOLD_CHALLENGE_COORDINATE_DOMAIN);
        expected_xof.update(&largest_one_block_seed);
        expected_xof.update(&index.to_le_bytes());
        let mut expected_reader = expected_xof.finalize_xof();
        let mut expected = [0u8; 256];
        expected_reader.read(&mut expected);

        let prefix = IndexedXofPrefix::new(&largest_one_block_seed).unwrap();
        let mut cursor = XofCursor::from_indexed_prefix(&prefix, index);
        let mut actual = [0u8; 256];
        cursor.fill_bytes(&mut actual);
        assert_eq!(actual, expected);

        let oversized_seed = vec![0u8; SHAKE256_RATE - fixed_len];
        assert_eq!(
            IndexedXofPrefix::new(&oversized_seed).err(),
            Some("indexed sparse challenge input exceeds one SHAKE256 rate block")
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
}
