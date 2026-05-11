//! Matrix sampling helpers for setup.

use akita_algebra::ring::CyclotomicRing;
#[allow(unused_imports)]
use akita_field::parallel::*;
use akita_field::{FieldCore, RandomSampling};
use rand_core::{CryptoRng, RngCore};
use sha3::digest::{ExtendableOutput, XofReader};
use sha3::Shake256;

use crate::protocol::prg::absorb_len_prefixed;
use akita_types::{FlatMatrix, PublicMatrixSeed};

const PUBLIC_MATRIX_DOMAIN: &[u8] = b"akita/commitment/public-matrix-1d";
const SHARED_MATRIX_LABEL: &[u8] = b"shared";

/// Fixed public seed for deterministic, reproducible setup.
pub fn sample_public_matrix_seed() -> PublicMatrixSeed {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&0xDEAD_BEEF_CAFE_BABEu64.to_le_bytes());
    seed
}

/// Derive a flat public vector of ring elements from a seed.
///
/// All role matrices (A, B, D) share one backing vector with a fixed label
/// (`"shared"`). Each role views a prefix of this vector reshaped with its
/// own `(num_rows, num_cols)` dimensions.
///
/// Domain separation uses a single flat index so that a vector of length N
/// is a prefix of any vector of length M > N derived from the same seed.
#[tracing::instrument(skip_all, name = "derive_public_matrix_flat")]
pub fn derive_public_matrix_flat<F: FieldCore + RandomSampling, const D: usize>(
    total_ring_elements: usize,
    seed: &PublicMatrixSeed,
) -> FlatMatrix<F> {
    let ring_elements: Vec<CyclotomicRing<F, D>> = cfg_into_iter!(0..total_ring_elements)
        .map(|idx| {
            let mut entry_rng = ShakeXofRng::new(seed, idx);
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

    FlatMatrix::from_flat_data(data, D)
}

struct ShakeXofRng {
    reader: Box<dyn XofReader>,
}

impl ShakeXofRng {
    fn new(seed: &PublicMatrixSeed, flat_index: usize) -> Self {
        let mut xof = Shake256::default();
        absorb_len_prefixed(&mut xof, b"domain", PUBLIC_MATRIX_DOMAIN);
        absorb_len_prefixed(&mut xof, b"seed", seed);
        absorb_len_prefixed(&mut xof, b"matrix", SHARED_MATRIX_LABEL);
        absorb_len_prefixed(&mut xof, b"index", &(flat_index as u64).to_le_bytes());
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

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;
    use akita_field::Fp64;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    #[test]
    fn flat_derivation_is_deterministic_for_same_seed() {
        let seed = [42u8; 32];
        let m1 = derive_public_matrix_flat::<F, D>(15, &seed);
        let m2 = derive_public_matrix_flat::<F, D>(15, &seed);
        assert_eq!(m1, m2);
    }

    #[test]
    fn flat_derivation_is_prefix_stable() {
        let seed = [7u8; 32];
        let small = derive_public_matrix_flat::<F, D>(6, &seed);
        let large = derive_public_matrix_flat::<F, D>(24, &seed);
        let small_view = small.ring_view::<D>(1, 6);
        let large_view = large.ring_view::<D>(1, 6);
        for c in 0..6 {
            assert_eq!(small_view.row(0)[c], large_view.row(0)[c]);
        }
    }

    #[test]
    fn different_shapes_from_same_flat() {
        let seed = [13u8; 32];
        let flat = derive_public_matrix_flat::<F, D>(12, &seed);
        let view_3x4 = flat.ring_view::<D>(3, 4);
        let view_2x6 = flat.ring_view::<D>(2, 6);

        assert_eq!(view_3x4.row(0)[0], view_2x6.row(0)[0]);
        assert_eq!(view_3x4.row(0)[3], view_2x6.row(0)[3]);
        assert_ne!(view_3x4.row(1)[0], view_2x6.row(1)[0]);
    }
}
