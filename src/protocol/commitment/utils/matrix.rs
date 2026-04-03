//! Matrix sampling helpers for setup.

use crate::algebra::ring::CyclotomicRing;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::{FieldCore, FieldSampling};
use rand_core::{CryptoRng, RngCore};
use sha3::digest::{ExtendableOutput, XofReader};
use sha3::Shake256;

use super::flat_matrix::FlatMatrix;
use crate::protocol::prg::absorb_len_prefixed;

/// Public seed used to derive commitment matrices.
pub(crate) type PublicMatrixSeed = [u8; 32];

const PUBLIC_MATRIX_DOMAIN: &[u8] = b"hachi/commitment/public-matrix";
const SHARED_MATRIX_LABEL: &[u8] = b"shared";

/// Fixed public seed for deterministic, reproducible setup.
pub(crate) fn sample_public_matrix_seed() -> PublicMatrixSeed {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&0xDEAD_BEEF_CAFE_BABEu64.to_le_bytes());
    seed
}

/// Derive a public matrix from a seed using domain-separated SHAKE expansion.
///
/// All role matrices (A, B, D) share one backing matrix with a fixed label
/// (`"shared"`). Each role is a row/column prefix of the shared matrix.
/// See `SHARED_PREFIX_BINDING.md` for the security argument.
#[cfg(test)]
pub(crate) fn derive_public_matrix<F: FieldCore + FieldSampling, const D: usize>(
    rows: usize,
    cols: usize,
    seed: &PublicMatrixSeed,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    cfg_into_iter!(0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    let mut entry_rng = ShakeXofRng::new(seed, r, c);
                    CyclotomicRing::random(&mut entry_rng)
                })
                .collect()
        })
        .collect()
}

/// Derive a public matrix directly into [`FlatMatrix`] storage, parallelized
/// at the individual entry level.
///
/// Avoids the intermediate `Vec<Vec<CyclotomicRing>>` allocation and
/// copy that [`derive_public_matrix`] + [`FlatMatrix::from_ring_matrix`]
/// would require.
#[tracing::instrument(skip_all, name = "derive_public_matrix_flat")]
pub(crate) fn derive_public_matrix_flat<F: FieldCore + FieldSampling, const D: usize>(
    rows: usize,
    cols: usize,
    seed: &PublicMatrixSeed,
) -> FlatMatrix<F> {
    let total = rows * cols;
    let ring_elements: Vec<CyclotomicRing<F, D>> = cfg_into_iter!(0..total)
        .map(|idx| {
            let r = idx / cols;
            let c = idx % cols;
            let mut entry_rng = ShakeXofRng::new(seed, r, c);
            CyclotomicRing::random(&mut entry_rng)
        })
        .collect();

    // SAFETY: CyclotomicRing<F, D> is #[repr(transparent)] over [F; D], so
    // Vec<CyclotomicRing<F, D>> and Vec<F> share the same backing allocation
    // layout (same element alignment, same total byte count).
    let data = unsafe {
        let ptr = ring_elements.as_ptr() as *mut F;
        let len = ring_elements.len() * D;
        let cap = ring_elements.capacity() * D;
        std::mem::forget(ring_elements);
        Vec::from_raw_parts(ptr, len, cap)
    };

    FlatMatrix::from_flat_data(data, rows, cols, D)
}

struct ShakeXofRng {
    reader: Box<dyn XofReader>,
}

impl ShakeXofRng {
    // Each entry is uniquely identified by `(seed, row, col)` with a fixed
    // matrix label, so a matrix derived at one size is a prefix of the same
    // matrix derived at a larger size.
    fn new(seed: &PublicMatrixSeed, row: usize, col: usize) -> Self {
        let mut xof = Shake256::default();
        absorb_len_prefixed(&mut xof, b"domain", PUBLIC_MATRIX_DOMAIN);
        absorb_len_prefixed(&mut xof, b"seed", seed);
        absorb_len_prefixed(&mut xof, b"matrix", SHARED_MATRIX_LABEL);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Fp64;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    #[test]
    fn matrix_derivation_is_deterministic_for_same_seed() {
        let seed = [42u8; 32];
        let m1 = derive_public_matrix::<F, D>(3, 5, &seed);
        let m2 = derive_public_matrix::<F, D>(3, 5, &seed);
        assert_eq!(m1, m2);
    }

    #[test]
    fn matrix_derivation_is_prefix_stable() {
        let seed = [7u8; 32];
        let small = derive_public_matrix::<F, D>(2, 3, &seed);
        let large = derive_public_matrix::<F, D>(4, 6, &seed);
        for r in 0..2 {
            for c in 0..3 {
                assert_eq!(small[r][c], large[r][c]);
            }
        }
    }
}
