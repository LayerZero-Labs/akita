//! Streaming XOF cursor used by the signed-sparse fold-challenge sampler.
//!
//! Every per-challenge draw consumes randomness from the same SHAKE256-backed
//! cursor. Centralising the cursor here means the PRG choice, the buffer size,
//! and the bias-free drawing primitives can be swapped or audited in one place.
//!
//! The cursor's `next_*` helpers use bitmask rejection sampling, so every
//! returned value is uniform over the requested range with no modulo bias.

use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;

/// Domain separator absorbed into the SHAKE256 instance before the
/// transcript-derived seed. Distinct from any transcript-layer domain tag so
/// that the PRG output cannot be mistaken for a transcript challenge.
const SPARSE_PRG_DOMAIN: &[u8] = b"akita/sparse-challenge-prg";

type ShakeReader = <Shake256 as ExtendableOutput>::Reader;

/// Internal buffer size (~30 SHAKE256 rate blocks) used to amortise XOF
/// squeezes across many small reads.
const XOF_BUF_SIZE: usize = 4096;

/// Streaming cursor backed by a SHAKE256 XOF with a 4 KB internal buffer
/// (~30 rate blocks) to amortize squeeze calls.
pub(crate) struct XofCursor {
    reader: ShakeReader,
    buf: Box<[u8; XOF_BUF_SIZE]>,
    pos: usize,
}

impl XofCursor {
    /// Build a cursor by absorbing the static domain separator followed by the
    /// transcript-derived `seed` into a fresh SHAKE256 instance.
    pub(crate) fn from_seed(seed: &[u8]) -> Self {
        let mut xof = Shake256::default();
        xof.update(SPARSE_PRG_DOMAIN);
        xof.update(seed);
        let mut cursor = Self {
            reader: xof.finalize_xof(),
            buf: Box::new([0u8; XOF_BUF_SIZE]),
            pos: XOF_BUF_SIZE,
        };
        cursor.refill();
        cursor
    }

    #[inline]
    fn refill(&mut self) {
        self.reader.read(self.buf.as_mut());
        self.pos = 0;
    }

    #[inline]
    fn next_u8(&mut self) -> u8 {
        if self.pos >= XOF_BUF_SIZE {
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
            if self.pos >= XOF_BUF_SIZE {
                self.refill();
            }
            let avail = XOF_BUF_SIZE - self.pos;
            let take = avail.min(out.len() - off);
            out[off..off + take].copy_from_slice(&self.buf[self.pos..self.pos + take]);
            self.pos += take;
            off += take;
        }
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.pos + 4 <= XOF_BUF_SIZE {
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
