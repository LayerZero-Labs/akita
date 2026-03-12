//! Prefix-stable extendable commitment-key derivation for Labrador.
//!
//! Unlike setup matrices that bind full `(rows, cols)` shape, this derivation
//! binds only `(matrix_label, row, col)` so extending dimensions preserves the
//! previously derived prefix exactly.

use blake2::digest::consts::U32;
use blake2::digest::Digest;
use blake2::Blake2b;

use crate::algebra::ring::CyclotomicRing;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::prg::{MatrixPrgContext, Shake256Backend};
use crate::{FieldCore, FieldSampling};

/// Public seed used to derive extendable Labrador commitment keys.
pub type LabradorComKeySeed = [u8; 32];

/// Derive a Labrador commitment-key seed from the Hachi public-matrix seed.
///
/// Uses domain-separated BLAKE2b-256 so that the Labrador key space is
/// independent of the Hachi commitment-matrix key space.
pub fn derive_labrador_comkey_seed(hachi_public_matrix_seed: &[u8; 32]) -> LabradorComKeySeed {
    let mut hasher = Blake2b::<U32>::new();
    hasher.update(b"hachi/labrador/comkey-seed");
    hasher.update(hachi_public_matrix_seed);
    let hash = hasher.finalize();
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&hash);
    seed
}

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
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    use crate::protocol::prg::MatrixPrgBackend;

    cfg_into_iter!(0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    let context = MatrixPrgContext {
                        seed,
                        matrix_label,
                        rows: 0,
                        cols: 0,
                        row: r,
                        col: c,
                    };
                    let mut rng = Shake256Backend.entry_rng(&context);
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
        let small = derive_extendable_comkey_matrix::<F, D>(3, 4, &seed, b"comkey/A");
        let large = derive_extendable_comkey_matrix::<F, D>(5, 7, &seed, b"comkey/A");

        for r in 0..3 {
            for c in 0..4 {
                assert_eq!(small[r][c], large[r][c]);
            }
        }
    }

    #[test]
    fn extendable_derivation_domain_separates_labels() {
        let seed = [7u8; 32];
        let a = derive_extendable_comkey_matrix::<F, D>(2, 3, &seed, b"comkey/A");
        let b = derive_extendable_comkey_matrix::<F, D>(2, 3, &seed, b"comkey/B");
        assert_ne!(a, b);
    }
}
