use super::ProductNorm;
use jolt_field::{
    Ext2, Field, FpExt4, One, Prime128Offset275, Prime32Offset99, Prime64Offset59, Ring, Unreduced,
    Zero,
};

fn assert_product_norm_matches_direct_coefficients<E>(value: fn(usize) -> E)
where
    E: Field + Ring + Unreduced,
{
    for block_len in [0usize, 1, 2, 7, 8, 15, 16, 17, 255, 256, 257] {
        let mut expected = [E::zero(); 3];
        let mut full = ProductNorm::<E, false>::zero();
        let mut skip_linear = ProductNorm::<E, true>::zero();

        for pair in 0..block_len {
            let w0 = value(3 * pair);
            let dw = value(3 * pair + 1);
            let weight = value(3 * pair + 2);

            expected[0] += weight * (w0.square() + w0);
            expected[1] += weight * (dw * (w0 + w0 + E::one()));
            expected[2] += weight * dw.square();

            full.add(w0, dw, weight);
            skip_linear.add(w0, dw, weight);
        }

        assert_eq!(
            full.reduce(),
            expected,
            "full norm block length {block_len}"
        );
        assert_eq!(
            skip_linear.reduce(),
            [expected[0], E::zero(), expected[2]],
            "recovered-linear norm block length {block_len}"
        );
    }
}

fn prime128_value(index: usize) -> Prime128Offset275 {
    match index % 8 {
        0 => Prime128Offset275::zero(),
        1 => Prime128Offset275::one(),
        2 => -Prime128Offset275::one(),
        _ => -Prime128Offset275::from_u64((index as u64).wrapping_mul(37) + 11),
    }
}

fn ext2_value(index: usize) -> Ext2<Prime64Offset59> {
    type F = Prime64Offset59;

    match index % 8 {
        0 => Ext2::new(F::zero(), F::zero()),
        1 => Ext2::new(F::one(), F::zero()),
        2 => Ext2::new(-F::one(), -F::from_u64(3)),
        _ => Ext2::new(
            -F::from_u64((index as u64).wrapping_mul(19) + 7),
            -F::from_u64((index as u64).wrapping_mul(29) + 13),
        ),
    }
}

fn fp_ext4_value(index: usize) -> FpExt4<Prime32Offset99> {
    type F = Prime32Offset99;

    match index % 8 {
        0 => FpExt4::new([F::zero(); 4]),
        1 => FpExt4::new([F::one(), F::zero(), F::zero(), F::zero()]),
        2 => FpExt4::new([-F::one(), -F::from_u64(2), -F::from_u64(3), -F::from_u64(4)]),
        _ => FpExt4::new(std::array::from_fn(|coordinate| {
            -F::from_u64(
                (index as u64)
                    .wrapping_mul(23 + coordinate as u64 * 6)
                    .wrapping_add(5 + coordinate as u64),
            )
        })),
    }
}

#[test]
fn product_norm_matches_direct_formula_in_base_field() {
    assert_product_norm_matches_direct_coefficients(prime128_value);
}

#[test]
fn product_norm_matches_direct_formula_in_quadratic_extension() {
    assert_product_norm_matches_direct_coefficients(ext2_value);
}

#[test]
fn product_norm_matches_direct_formula_in_quartic_extension() {
    assert_product_norm_matches_direct_coefficients(fp_ext4_value);
}
