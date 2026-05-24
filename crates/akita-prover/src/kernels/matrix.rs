//! Matrix sampling helpers for setup.

pub use akita_types::{derive_public_matrix_flat, sample_public_matrix_seed};

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
        let small_view = small.ring_view::<D>(1, 6).unwrap();
        let large_view = large.ring_view::<D>(1, 6).unwrap();
        for c in 0..6 {
            assert_eq!(small_view.row(0).unwrap()[c], large_view.row(0).unwrap()[c]);
        }
    }

    #[test]
    fn different_shapes_from_same_flat() {
        let seed = [13u8; 32];
        let flat = derive_public_matrix_flat::<F, D>(12, &seed);
        let view_3x4 = flat.ring_view::<D>(3, 4).unwrap();
        let view_2x6 = flat.ring_view::<D>(2, 6).unwrap();

        assert_eq!(view_3x4.row(0).unwrap()[0], view_2x6.row(0).unwrap()[0]);
        assert_eq!(view_3x4.row(0).unwrap()[3], view_2x6.row(0).unwrap()[3]);
        assert_ne!(view_3x4.row(1).unwrap()[0], view_2x6.row(1).unwrap()[0]);
    }
}
