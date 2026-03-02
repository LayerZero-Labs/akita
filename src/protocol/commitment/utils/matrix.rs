//! Matrix sampling helpers for setup.

use crate::algebra::ring::CyclotomicRing;
use crate::protocol::prg::{MatrixPrgBackendChoice, MatrixPrgContext};
use crate::{FieldCore, FieldSampling};

/// Public seed used to derive commitment matrices.
pub(crate) type PublicMatrixSeed = [u8; 32];

/// PRG backend used for commitment matrix derivation.
pub(crate) type PublicMatrixPrgBackend = MatrixPrgBackendChoice;

/// Fixed public seed for deterministic, reproducible setup.
pub(crate) fn sample_public_matrix_seed() -> PublicMatrixSeed {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&0xDEAD_BEEF_CAFE_BABEu64.to_le_bytes());
    seed
}

/// Fixed backend for deterministic, reproducible setup.
pub(crate) fn sample_public_matrix_prg_backend() -> PublicMatrixPrgBackend {
    // Full-cutover path: always use the backend abstraction.
    PublicMatrixPrgBackend::default()
}

/// Derive a public matrix from a seed using the selected PRG backend.
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
    backend: PublicMatrixPrgBackend,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    let context = MatrixPrgContext {
                        seed,
                        matrix_label,
                        rows,
                        cols,
                        row: r,
                        col: c,
                    };
                    let mut entry_rng = backend.entry_rng(&context);
                    CyclotomicRing::random(&mut entry_rng)
                })
                .collect()
        })
        .collect()
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
        let backend = PublicMatrixPrgBackend::default();
        let m1 = derive_public_matrix::<F, D>(3, 5, &seed, b"A", backend);
        let m2 = derive_public_matrix::<F, D>(3, 5, &seed, b"A", backend);
        assert_eq!(m1, m2);
    }

    #[test]
    fn matrix_derivation_domain_separates_labels() {
        let seed = [7u8; 32];
        let backend = PublicMatrixPrgBackend::default();
        let a = derive_public_matrix::<F, D>(2, 3, &seed, b"A", backend);
        let b = derive_public_matrix::<F, D>(2, 3, &seed, b"B", backend);
        assert_ne!(a, b);
    }
}
