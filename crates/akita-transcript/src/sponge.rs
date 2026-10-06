//! Spongefish backend selected by Akita's transcript feature.

/// Sponge backend selected by the active transcript feature.
///
/// Exactly one transcript backend feature must be active in the complete PCS graph.
#[cfg(all(
    feature = "transcript-blake2b",
    not(all(feature = "blake2-inline", target_arch = "riscv64"))
))]
pub type TranscriptSponge = spongefish::instantiations::Blake2b512;

/// Sponge backend selected by the active transcript feature, over the inline
/// hasher. Host builds keep spongefish's sponge; only a RISC-V guest has the
/// inline.
#[cfg(all(
    feature = "transcript-blake2b",
    feature = "blake2-inline",
    target_arch = "riscv64"
))]
pub type TranscriptSponge = inline::Blake2bSponge;

/// The inline-hasher sponge, exported so a host with `jolt-inlines-blake2`'s
/// software path can pin it against spongefish's `Blake2b512` byte for byte.
#[cfg(all(feature = "transcript-blake2b", feature = "blake2-inline"))]
pub use inline::Blake2bSponge as InlineBlake2bSponge;

/// Sponge backend selected by the active transcript feature.
#[cfg(feature = "transcript-keccak")]
pub type TranscriptSponge = spongefish::instantiations::Keccak;

#[cfg(all(feature = "transcript-blake2b", feature = "blake2-inline"))]
mod inline {
    use jolt_inlines_blake2::{Blake2b, BLOCK_INPUT_SIZE_IN_BYTES};
    use spongefish::DuplexSpongeInterface;

    const DIGEST: usize = 64;

    /// `spongefish::instantiations::Blake2b512` over the inline hasher, with
    /// the same output bytes.
    ///
    /// Every absorb phase, squeeze phase, and squeeze-end digest of that sponge
    /// opens with a constant 128-byte mask block. This sponge compresses each
    /// mask once at construction and clones the prepared hasher, saving one
    /// compression per phase. The chaining value always follows the mask, so
    /// the mask is never the final block that BLAKE2b flags.
    #[derive(Clone)]
    pub struct Blake2bSponge {
        hasher: Blake2b,
        cv: [u8; DIGEST],
        mode: Mode,
        leftovers: [u8; DIGEST],
        leftover_start: usize,
        absorb_prefix: Blake2b,
        squeeze_prefix: Blake2b,
        squeeze_end_prefix: Blake2b,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Mode {
        Start,
        Absorb,
        /// Squeezing, with the index of the next output digest.
        Squeeze(u64),
    }

    fn masked(last: u8) -> Blake2b {
        let mut block = [0; BLOCK_INPUT_SIZE_IN_BYTES];
        block[BLOCK_INPUT_SIZE_IN_BYTES - 1] = last;
        let mut hasher = Blake2b::new();
        hasher.update_block_eager(&block);
        hasher
    }

    impl Default for Blake2bSponge {
        fn default() -> Self {
            Self {
                hasher: Blake2b::new(),
                cv: [0; DIGEST],
                mode: Mode::Start,
                leftovers: [0; DIGEST],
                leftover_start: DIGEST,
                absorb_prefix: masked(0x00),
                squeeze_prefix: masked(0x01),
                squeeze_end_prefix: masked(0x02),
            }
        }
    }

    impl Blake2bSponge {
        fn leftover_len(&self) -> usize {
            DIGEST - self.leftover_start
        }

        fn squeeze_end(&mut self) {
            if let Mode::Squeeze(count) = self.mode {
                let byte_count = count * DIGEST as u64 - self.leftover_len() as u64;
                let mut hasher = self.squeeze_end_prefix.clone();
                hasher.update(&self.cv);
                hasher.update(&byte_count.to_be_bytes());
                self.cv = hasher.finalize();
                self.hasher = Blake2b::new();
                self.mode = Mode::Start;
                self.leftover_start = DIGEST;
            }
        }
    }

    impl DuplexSpongeInterface for Blake2bSponge {
        type U = u8;

        fn absorb(&mut self, input: &[u8]) -> &mut Self {
            self.squeeze_end();
            if self.mode == Mode::Start {
                self.mode = Mode::Absorb;
                self.hasher = self.absorb_prefix.clone();
                self.hasher.update(&self.cv);
            }
            self.hasher.update(input);
            self
        }

        fn ratchet(&mut self) -> &mut Self {
            self.squeeze_end();
            let inner = core::mem::replace(&mut self.hasher, Blake2b::new()).finalize();
            self.cv = Blake2b::digest(&inner);
            self.leftover_start = DIGEST;
            self.mode = Mode::Start;
            self
        }

        fn squeeze(&mut self, mut output: &mut [u8]) -> &mut Self {
            let mut index = match self.mode {
                Mode::Squeeze(index) => index,
                Mode::Start | Mode::Absorb => {
                    if self.mode == Mode::Absorb {
                        self.ratchet();
                    }
                    self.hasher = self.squeeze_prefix.clone();
                    self.hasher.update(&self.cv);
                    0
                }
            };
            while !output.is_empty() {
                if self.leftover_len() == 0 {
                    let mut hasher = self.hasher.clone();
                    hasher.update(&index.to_be_bytes());
                    self.leftovers = hasher.finalize();
                    self.leftover_start = 0;
                    index += 1;
                }
                let len = output.len().min(self.leftover_len());
                let (head, rest) = output.split_at_mut(len);
                head.copy_from_slice(
                    &self.leftovers[self.leftover_start..self.leftover_start + len],
                );
                self.leftover_start += len;
                output = rest;
            }
            self.mode = Mode::Squeeze(index);
            self
        }
    }
}
