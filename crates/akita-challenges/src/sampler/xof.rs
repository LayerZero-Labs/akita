//! Streaming XOF cursor used by the signed-sparse fold-challenge sampler.
//!
//! Every fold coordinate gets a fresh indexed SHAKE256 stream. A sampler may
//! reset this cursor between coordinates to reuse its 4 KiB allocation.
//!
//! The cursor's `next_*` helpers use bitmask rejection sampling, so every
//! returned value is uniform over the requested range with no modulo bias.

use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;

/// Domain separator absorbed into the SHAKE256 instance before the
/// transcript-derived seed. Distinct from any transcript-layer domain tag so
/// that the PRG output cannot be mistaken for a transcript challenge.
const SPARSE_PRG_DOMAIN: &[u8] = b"akita/sparse-challenge-prg";
const FOLD_CHALLENGE_COORDINATE_DOMAIN: &[u8] = b"akita/fold-challenge-coordinate/v1";

type ShakeReader = <Shake256 as ExtendableOutput>::Reader;

/// SHAKE256 state after absorbing the fixed domains and one group root.
/// Cloning this state gives each coordinate a fresh XOF without repeating the
/// common prefix absorption.
pub(crate) struct IndexedXofPrefix(Shake256);

impl IndexedXofPrefix {
    pub(crate) fn new(seed: &[u8]) -> Self {
        let mut xof = Shake256::default();
        xof.update(SPARSE_PRG_DOMAIN);
        xof.update(FOLD_CHALLENGE_COORDINATE_DOMAIN);
        xof.update(seed);
        Self(xof)
    }

    fn reader(&self, coordinate_index: u64) -> ShakeReader {
        let mut xof = self.0.clone();
        xof.update(&coordinate_index.to_le_bytes());
        xof.finalize_xof()
    }
}

/// Internal buffer size (~30 SHAKE256 rate blocks) used to amortise XOF
/// squeezes across many small reads.
const XOF_BUF_SIZE: usize = 4096;
/// One coordinate normally consumes well below one SHAKE256 rate block. Fill
/// the reusable allocation in bounded chunks so resetting a short coordinate
/// does not squeeze 4 KiB that will be discarded.
const XOF_REFILL_SIZE: usize = 128;

/// Streaming cursor backed by a SHAKE256 XOF with a 4 KB internal buffer
/// (~30 rate blocks) to amortize squeeze calls.
pub(crate) struct XofCursor {
    reader: ShakeReader,
    buf: Box<[u8; XOF_BUF_SIZE]>,
    pos: usize,
    len: usize,
}

impl XofCursor {
    /// Build a cursor by absorbing the static domain separator followed by the
    /// transcript-derived `seed` into a fresh SHAKE256 instance.
    #[cfg(test)]
    pub(crate) fn from_seed(seed: &[u8]) -> Self {
        Self::from_seed_parts(seed, &[])
    }

    /// Build the canonical stream for one claim-major fold coordinate.
    pub(crate) fn from_indexed_prefix(prefix: &IndexedXofPrefix, coordinate_index: u64) -> Self {
        let mut xof = prefix.reader(coordinate_index);
        let mut buf = Box::new([0u8; XOF_BUF_SIZE]);
        xof.read(&mut buf[..XOF_REFILL_SIZE]);
        Self {
            reader: xof,
            buf,
            pos: 0,
            len: XOF_REFILL_SIZE,
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

    #[cfg(test)]
    fn from_seed_parts(seed: &[u8], suffixes: &[&[u8]]) -> Self {
        let mut xof = Shake256::default();
        xof.update(SPARSE_PRG_DOMAIN);
        xof.update(seed);
        for suffix in suffixes {
            xof.update(suffix);
        }
        let mut cursor = Self {
            reader: xof.finalize_xof(),
            buf: Box::new([0u8; XOF_BUF_SIZE]),
            pos: 0,
            len: 0,
        };
        cursor.refill();
        cursor
    }

    #[inline]
    fn refill(&mut self) {
        self.reader.read(&mut self.buf[..XOF_REFILL_SIZE]);
        self.pos = 0;
        self.len = XOF_REFILL_SIZE;
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
        let mut expected = [0u8; 96];
        expected_reader.read(&mut expected);

        let prefix = IndexedXofPrefix::new(&seed);
        let mut cursor = XofCursor::from_indexed_prefix(&prefix, index);
        let mut actual = [0u8; 96];
        cursor.fill_bytes(&mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn resetting_an_indexed_cursor_matches_a_fresh_cursor() {
        let seed = [0xabu8; 32];
        let prefix = IndexedXofPrefix::new(&seed);
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
