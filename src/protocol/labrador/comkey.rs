//! Prefix-stable extendable commitment-key derivation for Labrador.
//!
//! Unlike setup matrices that bind full `(rows, cols)` shape, this derivation
//! binds only `(matrix_label, row, col)` so extending dimensions preserves the
//! previously derived prefix exactly.

use crate::algebra::ring::CyclotomicRing;
use crate::protocol::prg::{MatrixPrgBackendChoice, MatrixPrgContext};
use crate::{FieldCore, FieldSampling};

/// Public seed used to derive extendable Labrador commitment keys.
pub type LabradorComKeySeed = [u8; 32];

/// Derive a prefix-stable matrix for Labrador commitment keys.
///
/// Prefix-stable means: if `M_small = derive(rows, cols)` and
/// `M_large = derive(rows2, cols2)` with `rows2 >= rows`, `cols2 >= cols`,
/// then `M_large[r][c] == M_small[r][c]` for all `r < rows`, `c < cols`.
pub fn derive_extendable_comkey_matrix<F: FieldCore + FieldSampling, const D: usize>(
    rows: usize,
    cols: usize,
    seed: &LabradorComKeySeed,
    matrix_label: &[u8],
    backend: MatrixPrgBackendChoice,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    // Dedicated key path: keep shape fields constant, bind only
                    // entry indices and matrix label.
                    let context = MatrixPrgContext {
                        seed,
                        matrix_label,
                        rows: 0,
                        cols: 0,
                        row: r,
                        col: c,
                    };
                    let mut rng = backend.entry_rng(&context);
                    CyclotomicRing::random(&mut rng)
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
    fn extendable_derivation_has_prefix_stability() {
        let seed = [19u8; 32];
        let backend = MatrixPrgBackendChoice::Shake256;
        let small = derive_extendable_comkey_matrix::<F, D>(3, 4, &seed, b"comkey/A", backend);
        let large = derive_extendable_comkey_matrix::<F, D>(5, 7, &seed, b"comkey/A", backend);

        for r in 0..3 {
            for c in 0..4 {
                assert_eq!(small[r][c], large[r][c]);
            }
        }
    }

    #[test]
    fn extendable_derivation_domain_separates_labels() {
        let seed = [7u8; 32];
        let backend = MatrixPrgBackendChoice::Aes128Ctr;
        let a = derive_extendable_comkey_matrix::<F, D>(2, 3, &seed, b"comkey/A", backend);
        let b = derive_extendable_comkey_matrix::<F, D>(2, 3, &seed, b"comkey/B", backend);
        assert_ne!(a, b);
    }
}
