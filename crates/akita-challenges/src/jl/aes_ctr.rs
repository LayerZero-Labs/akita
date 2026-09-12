//! Canonical AES-128 counter expansion for JL matrix bitplanes.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

use aes::Aes128;
use akita_error::AkitaError;
use ctr::cipher::{KeyIvInit, StreamCipher};

type Aes128Ctr = ctr::Ctr64LE<Aes128>;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const X86_HARDWARE_MIN_BYTES: usize = 256;

pub(super) struct Aes128CtrExpander {
    key: [u8; 16],
    base_low: u64,
    base_high: u64,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    hardware: Option<x86::HardwareBackend>,
}

impl Aes128CtrExpander {
    pub(super) fn new(key: &[u8; 16], base_block: [u8; 16]) -> Self {
        let base_low = u64::from_le_bytes(std::array::from_fn(|index| base_block[index]));
        let base_high = u64::from_le_bytes(std::array::from_fn(|index| base_block[index + 8]));
        Self {
            key: *key,
            base_low,
            base_high,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            hardware: x86::HardwareBackend::detect(key),
        }
    }

    pub(super) fn fill_stream(&self, stream: u64, output: &mut [u8]) -> Result<(), AkitaError> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if output.len() >= X86_HARDWARE_MIN_BYTES {
            if let Some(hardware) = &self.hardware {
                // An addressable slice contains fewer than `u64::MAX` AES blocks, so
                // the Ctr64LE position cannot exhaust. The low counter itself wraps.
                hardware.fill(self.base_low, self.base_high ^ stream, output);
                return Ok(());
            }
        }

        let mut iv = [0u8; 16];
        iv[..8].copy_from_slice(&self.base_low.to_le_bytes());
        iv[8..].copy_from_slice(&(self.base_high ^ stream).to_le_bytes());
        let mut cipher = Aes128Ctr::new((&self.key).into(), (&iv).into());
        cipher
            .try_write_keystream(output)
            .map_err(|_| AkitaError::InvalidInput("JL AES counter stream exhausted".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ctr::cipher::{KeyIvInit, StreamCipher};
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use std::{hint::black_box, time::Instant};

    type ReferenceAes128Ctr = ctr::Ctr64LE<Aes128>;

    fn reference_fill(key: &[u8; 16], base_block: [u8; 16], stream: u64, output: &mut [u8]) {
        let mut iv = base_block;
        let base_high = u64::from_le_bytes(iv[8..].try_into().unwrap());
        iv[8..].copy_from_slice(&(base_high ^ stream).to_le_bytes());
        ReferenceAes128Ctr::new(key.into(), (&iv).into())
            .try_write_keystream(output)
            .unwrap();
    }

    #[test]
    fn parallel_stream_fills_match_serial_for_large_partial_blocks() {
        let expander = Aes128CtrExpander::new(&[0x42; 16], [0x24; 16]);
        let mut first = vec![0xa5; (1 << 18) + 7];
        let mut second = vec![0x5a; first.len()];
        let (a, b) = jolt_field::cfg_join!(|| expander.fill_stream(0, &mut first), || expander
            .fill_stream(1, &mut second));
        a.unwrap();
        b.unwrap();
        let mut reference = vec![0; first.len()];
        expander.fill_stream(0, &mut reference).unwrap();
        assert_eq!(first, reference);
        expander.fill_stream(1, &mut reference).unwrap();
        assert_eq!(second, reference);
        assert_ne!(first, second);
    }

    #[test]
    fn stream_and_counter_domains_are_disjoint_and_stable() {
        let expander = Aes128CtrExpander::new(&[0x42; 16], [0x24; 16]);
        let mut first = [0xa5u8; 37];
        let mut repeated = [0u8; 37];
        let mut other_stream = [0u8; 37];
        expander.fill_stream(0, &mut first).unwrap();
        expander.fill_stream(0, &mut repeated).unwrap();
        expander.fill_stream(1, &mut other_stream).unwrap();
        assert_eq!(first, repeated);
        assert_ne!(first, other_stream);
        assert_eq!(
            first,
            [
                91, 50, 126, 114, 212, 122, 94, 41, 249, 180, 5, 103, 85, 226, 166, 84, 214, 187,
                183, 151, 13, 161, 127, 52, 174, 197, 52, 95, 227, 254, 56, 184, 190, 250, 62, 118,
                78,
            ]
        );
    }

    #[test]
    fn expander_matches_rustcrypto_ctr_for_lengths_streams_and_nonzero_buffers() {
        let key = [0x42; 16];
        let base_block = [0x24; 16];
        let expander = Aes128CtrExpander::new(&key, base_block);
        for len in [0, 1, 15, 16, 17, 31, 32, 33, 511, 512, 513, 4099] {
            for stream in [0, 1, u64::MAX] {
                let mut actual = vec![0xa5; len];
                let mut expected = vec![0x5a; len];
                expander.fill_stream(stream, &mut actual).unwrap();
                reference_fill(&key, base_block, stream, &mut expected);
                assert_eq!(actual, expected, "length {len}, stream {stream}");
            }
        }
    }

    #[test]
    fn low_counter_wrap_matches_rustcrypto_ctr() {
        let key = [0x42; 16];
        let mut base_block = [0x24; 16];
        base_block[..8].copy_from_slice(&(u64::MAX - 1).to_le_bytes());
        let expander = Aes128CtrExpander::new(&key, base_block);
        let mut actual = [0xa5; 65];
        let mut expected = [0x5a; 65];
        expander.fill_stream(1, &mut actual).unwrap();
        reference_fill(&key, base_block, 1, &mut expected);
        assert_eq!(actual, expected);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn every_supported_x86_backend_matches_rustcrypto_ctr() {
        for (key, mut base_block) in [([0u8; 16], [0u8; 16]), ([0x42; 16], [0x24; 16])] {
            for base_low in [0, 1, u64::MAX - 1, u64::MAX] {
                base_block[..8].copy_from_slice(&base_low.to_le_bytes());
                let base_high = u64::from_le_bytes(base_block[8..].try_into().unwrap());
                for backend in x86::HardwareBackend::supported_for_tests(&key) {
                    for len in [0, 1, 15, 16, 17, 127, 128, 129, 1023, 1024, 1027] {
                        for stream in [0, 1, u64::MAX] {
                            let mut actual = vec![0xa5; len];
                            let mut expected = vec![0x5a; len];
                            backend.fill(base_low, base_high ^ stream, &mut actual);
                            reference_fill(&key, base_block, stream, &mut expected);
                            assert_eq!(
                                actual,
                                expected,
                                "{} length {len}, stream {stream}, base {base_low}",
                                backend.name()
                            );
                        }
                    }
                }
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    #[ignore = "manual x86 AES backend throughput comparison"]
    fn x86_aes_backend_microbench() {
        const SAMPLE_BYTES: usize = 1 << 30;
        let key = [0x42; 16];
        let base_block = [0x24; 16];

        for len in [128 << 10, 1 << 20, 8 << 20] {
            let iterations = SAMPLE_BYTES.div_ceil(len);
            let mut output = vec![0x5a; len];
            let started = Instant::now();
            for _ in 0..iterations {
                reference_fill(
                    black_box(&key),
                    black_box(base_block),
                    black_box(0),
                    black_box(&mut output),
                );
            }
            let rate = (iterations * len) as f64 / started.elapsed().as_secs_f64();
            black_box(&output);
            eprintln!("RustCrypto CTR {len} bytes: setup-included {rate:.0} B/s");
        }

        for backend in x86::HardwareBackend::supported_for_tests(&key) {
            for len in [128 << 10, 1 << 20, 8 << 20] {
                let iterations = SAMPLE_BYTES.div_ceil(len);
                let mut output = vec![0xa5; len];
                let started = Instant::now();
                for _ in 0..iterations {
                    backend.fill(
                        black_box(u64::from_le_bytes(base_block[..8].try_into().unwrap())),
                        black_box(u64::from_le_bytes(base_block[8..].try_into().unwrap())),
                        black_box(&mut output),
                    );
                }
                let elapsed = started.elapsed();
                black_box(&output);

                let started = Instant::now();
                for _ in 0..iterations {
                    backend.rebuild(black_box(&key)).fill(
                        black_box(u64::from_le_bytes(base_block[..8].try_into().unwrap())),
                        black_box(u64::from_le_bytes(base_block[8..].try_into().unwrap())),
                        black_box(&mut output),
                    );
                }
                let setup_elapsed = started.elapsed();
                black_box(&output);

                let mut reference = vec![0x5a; len];
                reference_fill(&key, base_block, 0, &mut reference);
                assert_eq!(output, reference);
                let rate = (iterations * len) as f64 / elapsed.as_secs_f64();
                let setup_rate = (iterations * len) as f64 / setup_elapsed.as_secs_f64();
                eprintln!(
                    "{} {len} bytes: steady {rate:.0} B/s, setup-included {setup_rate:.0} B/s",
                    backend.name()
                );
            }
        }
    }
}
