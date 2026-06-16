//! Streaming XOF cursor used by every sparse-challenge sampler.
//!
//! All three sampling families ([`crate::SparseChallengeConfig::Uniform`],
//! [`crate::SparseChallengeConfig::ExactShell`],
//! [`crate::SparseChallengeConfig::BoundedL1Norm`]) consume their per-challenge
//! randomness from the same SHAKE256-backed cursor. Centralising the cursor
//! here means the PRG choice, the buffer size, and the bias-free drawing
//! primitives can be swapped or audited in one place.
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
    /// Build a cursor by absorbing the sparse-challenge domain separator
    /// followed by the transcript-derived `seed` into a fresh SHAKE256 instance.
    pub(crate) fn from_seed(seed: &[u8]) -> Self {
        Self::from_seed_with_domain(SPARSE_PRG_DOMAIN, seed)
    }

    /// Build a cursor by absorbing an explicit `domain` separator followed by
    /// the transcript-derived `seed`.
    ///
    /// Callers outside the sparse-challenge family (e.g. the JL projection
    /// sampler) pass their own domain so their PRG stream is separated from the
    /// sparse-challenge stream and cannot collide on a shared seed.
    pub(crate) fn from_seed_with_domain(domain: &[u8], seed: &[u8]) -> Self {
        let mut xof = Shake256::default();
        xof.update(domain);
        xof.update(seed);
        let mut cursor = Self {
            reader: xof.finalize_xof(),
            buf: Box::new([0u8; XOF_BUF_SIZE]),
            pos: XOF_BUF_SIZE,
        };
        cursor.refill();
        cursor
    }

    /// Fill `out` with the next bytes from the XOF stream.
    ///
    /// Used by callers that consume raw PRG bytes directly (e.g. packing a
    /// ternary projection row two bits per entry) rather than via the typed
    /// `next_*` draws.
    pub(crate) fn fill_bytes(&mut self, out: &mut [u8]) {
        for slot in out.iter_mut() {
            *slot = self.next_u8();
        }
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

    /// Read 16 little-endian bytes from the XOF and interpret them as an
    /// unsigned 128-bit integer.
    ///
    /// This is the canonical top-level draw for the truncated-`2^128`
    /// bounded-`L1` sampler. There is no rejection loop and no modulo
    /// reduction: the realized distribution is uniform over `[0, 2^128)`,
    /// matching `read_u128_le` in `specs/bounded-l1-sparse-challenge.md`.
    #[inline]
    pub(crate) fn next_u128_le(&mut self) -> u128 {
        if self.pos + 16 <= XOF_BUF_SIZE {
            let val = u128::from_le_bytes(self.buf[self.pos..self.pos + 16].try_into().unwrap());
            self.pos += 16;
            val
        } else {
            let mut bytes = [0u8; 16];
            for slot in bytes.iter_mut() {
                *slot = self.next_u8();
            }
            u128::from_le_bytes(bytes)
        }
    }

    /// Draw a uniformly random sign in `{-1, +1}`.
    #[inline]
    pub(crate) fn next_sign(&mut self) -> i8 {
        if (self.next_u8() & 1) == 0 {
            1
        } else {
            -1
        }
    }
}
