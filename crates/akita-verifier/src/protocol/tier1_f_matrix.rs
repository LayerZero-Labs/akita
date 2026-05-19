//! Verifier-side derivation of the tier-1 outer SIS matrix `F`.
//!
//! Mirrors `crates/akita-prover/src/kernels/matrix.rs::derive_tier1_f_matrix_flat`
//! byte-for-byte so prover and verifier produce identical `F` entries
//! from the same setup seed. The verifier needs an independent copy
//! because `akita-prover`'s helper depends on `akita_field::parallel`
//! and Rayon-iterator types that aren't carried by `akita-verifier`.
//!
//! Soundness: the domain-separation label `b"tier1-f"` matches the
//! prover's label exactly — see `specs/tiered_commit.md` §11 and the
//! prover commit `93a24fde`.

use akita_algebra::ring::CyclotomicRing;
use akita_field::{FieldCore, RandomSampling};
use akita_types::{FlatMatrix, PublicMatrixSeed};
use rand_core::{CryptoRng, RngCore};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;

const PUBLIC_MATRIX_DOMAIN: &[u8] = b"akita/commitment/public-matrix-1d";
const TIER1_F_MATRIX_LABEL: &[u8] = b"tier1-f";

/// Derive the tier-1 outer SIS matrix `F` from the setup seed.
///
/// Produces `total_ring_elements` ring elements laid out as a flat
/// `FlatMatrix<F>` (one ring element per output index). Identical
/// output to the prover-side
/// `crates/akita-prover/src/kernels/matrix.rs::derive_tier1_f_matrix_flat`.
pub(crate) fn derive_tier1_f_matrix_flat<F: FieldCore + RandomSampling, const D: usize>(
    total_ring_elements: usize,
    seed: &PublicMatrixSeed,
) -> FlatMatrix<F> {
    let ring_elements: Vec<CyclotomicRing<F, D>> = (0..total_ring_elements)
        .map(|idx| {
            let mut entry_rng = ShakeXofRng::new(seed, TIER1_F_MATRIX_LABEL, idx);
            CyclotomicRing::random(&mut entry_rng)
        })
        .collect();

    // SAFETY: `CyclotomicRing<F, D>` is `#[repr(transparent)]` over
    // `[F; D]`, so the underlying allocation lays out as a contiguous
    // `[F; total_ring_elements * D]` and can be reinterpreted as a
    // `Vec<F>` with the same byte capacity. Mirrors the prover-side
    // helper exactly.
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
    fn new(seed: &PublicMatrixSeed, matrix_label: &[u8], flat_index: usize) -> Self {
        let mut xof = Shake256::default();
        absorb_len_prefixed(&mut xof, b"domain", PUBLIC_MATRIX_DOMAIN);
        absorb_len_prefixed(&mut xof, b"seed", seed);
        absorb_len_prefixed(&mut xof, b"matrix", matrix_label);
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

fn absorb_len_prefixed(xof: &mut Shake256, label: &[u8], data: &[u8]) {
    xof.update(&(label.len() as u64).to_le_bytes());
    xof.update(label);
    xof.update(&(data.len() as u64).to_le_bytes());
    xof.update(data);
}
