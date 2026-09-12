//! Canonical AES-128 counter expansion for JL matrix bitplanes.

use aes::Aes128;
use akita_error::AkitaError;
use ctr::cipher::{KeyIvInit, StreamCipher};

type Aes128Ctr = ctr::Ctr64LE<Aes128>;

pub(super) struct Aes128CtrExpander {
    key: [u8; 16],
    base_low: u64,
    base_high: u64,
}

impl Aes128CtrExpander {
    pub(super) fn new(key: &[u8; 16], base_block: [u8; 16]) -> Self {
        let base_low = u64::from_le_bytes(std::array::from_fn(|index| base_block[index]));
        let base_high = u64::from_le_bytes(std::array::from_fn(|index| base_block[index + 8]));
        Self {
            key: *key,
            base_low,
            base_high,
        }
    }

    pub(super) fn fill_stream(&self, stream: u64, output: &mut [u8]) -> Result<(), AkitaError> {
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
}
