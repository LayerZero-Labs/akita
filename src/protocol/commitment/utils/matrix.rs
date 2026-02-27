//! Matrix sampling helpers for setup.

use crate::algebra::ring::CyclotomicRing;
use crate::{FieldCore, FieldSampling};
use rand_core::{CryptoRng, RngCore};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;

/// Public seed used to derive commitment matrices.
pub(crate) type PublicMatrixSeed = [u8; 32];

const PUBLIC_MATRIX_DOMAIN: &[u8] = b"hachi/commitment/public-matrix";

/// Fixed public seed for deterministic, reproducible setup.
pub(crate) fn sample_public_matrix_seed() -> PublicMatrixSeed {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&0xDEAD_BEEF_CAFE_BABEu64.to_le_bytes());
    seed
}

/// Derive a public matrix from a seed using domain-separated SHAKE expansion.
///
/// This follows the same high-level pattern used in NIST lattice specs:
/// derive deterministic public structure from a seed + indices, then sample
/// coefficients via rejection-sampling at the field layer.
///
/// NOTE: Potential future hardening:
/// move toward stricter ML-KEM/ML-DSA-style byte layout and parsing rules
/// (fixed-format seed/index encoding and scheme-specific expansion details)
/// if we decide to maximize standards-shape interoperability.
pub(crate) fn derive_public_matrix<F: FieldCore + FieldSampling, const D: usize>(
    rows: usize,
    cols: usize,
    seed: &PublicMatrixSeed,
    matrix_label: &[u8],
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    let mut entry_rng = ShakeXofRng::new(seed, matrix_label, rows, cols, r, c);
                    CyclotomicRing::random(&mut entry_rng)
                })
                .collect()
        })
        .collect()
}

struct ShakeXofRng {
    reader: Box<dyn XofReader>,
}

impl ShakeXofRng {
    fn new(
        seed: &PublicMatrixSeed,
        matrix_label: &[u8],
        rows: usize,
        cols: usize,
        row: usize,
        col: usize,
    ) -> Self {
        let mut xof = Shake256::default();
        absorb_len_prefixed(&mut xof, b"domain", PUBLIC_MATRIX_DOMAIN);
        absorb_len_prefixed(&mut xof, b"seed", seed);
        absorb_len_prefixed(&mut xof, b"matrix", matrix_label);
        absorb_len_prefixed(&mut xof, b"rows", &(rows as u64).to_le_bytes());
        absorb_len_prefixed(&mut xof, b"cols", &(cols as u64).to_le_bytes());
        absorb_len_prefixed(&mut xof, b"row", &(row as u64).to_le_bytes());
        absorb_len_prefixed(&mut xof, b"col", &(col as u64).to_le_bytes());
        Self {
            reader: Box::new(xof.finalize_xof()),
        }
    }
}

impl RngCore for ShakeXofRng {
    fn next_u32(&mut self) -> u32 {
        let mut buf = [0u8; 4];
        self.fill_bytes(&mut buf);
        u32::from_le_bytes(buf)
    }

    fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.fill_bytes(&mut buf);
        u64::from_le_bytes(buf)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.reader.read(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for ShakeXofRng {}

fn absorb_len_prefixed(xof: &mut Shake256, label: &[u8], data: &[u8]) {
    xof.update(&(label.len() as u64).to_le_bytes());
    xof.update(label);
    xof.update(&(data.len() as u64).to_le_bytes());
    xof.update(data);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Fp64;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    #[test]
    fn matrix_derivation_is_deterministic_for_same_seed() {
        let seed = [42u8; 32];
        let m1 = derive_public_matrix::<F, D>(3, 5, &seed, b"A");
        let m2 = derive_public_matrix::<F, D>(3, 5, &seed, b"A");
        assert_eq!(m1, m2);
    }

    #[test]
    fn matrix_derivation_domain_separates_labels() {
        let seed = [7u8; 32];
        let a = derive_public_matrix::<F, D>(2, 3, &seed, b"A");
        let b = derive_public_matrix::<F, D>(2, 3, &seed, b"B");
        assert_ne!(a, b);
    }
}
