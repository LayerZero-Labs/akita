use super::*;
use crate::{
    tables::{q128_primes, Q128_NUM_PRIMES},
    CanonicalEncoding, Ring,
};
use jolt_field::Prime128OffsetA7F7;

/// `radix_sum_to_ring` against `F`'s own arithmetic: planes of products
/// `a·b` whose integers exceed `i128` (up to about `2^177.2`, the `P/8` the transposed product reaches, below `P/4`), with
/// both signs and the largest operands, summed at radix `2^32`.
#[test]
fn radix_sum_matches_field_arithmetic() {
    type F = Prime128OffsetA7F7;
    const D: usize = 64;
    let params = CrtNttParamSet::<i32, Q128_NUM_PRIMES, D>::new(q128_primes());
    let mut state = 0x243f_6a88_85a3_08d3_u128;
    let mut next = |bits: u32| -> i128 {
        state = state
            .wrapping_mul(0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645)
            .wrapping_add(0x5851_f42d_4c95_7f2d_1405_7b7e_f767_814f);
        (state as i128) >> (128 - bits)
    };
    let extreme = (3i128 << 84) - 1;
    for planes in 1..=RADIX_SUM_PLANES {
        let operands: Vec<([i128; D], [i128; D])> = (0..planes)
            .map(|plane| {
                let mut a = std::array::from_fn(|_| next(85));
                let mut b = std::array::from_fn(|_| next(85));
                // Every coefficient of a·b at its extreme, of either sign.
                if plane == 0 {
                    a = [extreme; D];
                    b = [-extreme; D];
                    b[0] = extreme;
                }
                (a, b)
            })
            .collect();
        let products: Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>> = operands
            .iter()
            .map(|(a, b)| {
                let mut product = CyclotomicCrtNtt::zero();
                product.add_assign_pointwise_mul(
                    &CyclotomicCrtNtt::from_centered_coefficients(a, &params),
                    &CyclotomicCrtNtt::from_centered_coefficients(b, &params),
                    &params,
                );
                product
            })
            .collect();
        let radix = F::from_u128_reduced(1 << RADIX_SUM_BITS);
        let expected = operands
            .iter()
            .rev()
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (a, b)| {
                acc.scale(&radix)
                    + CyclotomicRing::from_coefficients(a.map(F::from_i128))
                        * CyclotomicRing::from_coefficients(b.map(F::from_i128))
            });
        assert_eq!(
            CyclotomicCrtNtt::radix_sum_to_ring::<F>(&products, &params).unwrap(),
            expected,
            "{planes} planes"
        );
    }
    assert_eq!(
        CyclotomicCrtNtt::radix_sum_to_ring::<F>(&[], &params).unwrap(),
        CyclotomicRing::<F, D>::zero(),
    );
    let five = vec![CyclotomicCrtNtt::zero(); RADIX_SUM_PLANES + 1];
    assert!(CyclotomicCrtNtt::radix_sum_to_ring::<F>(&five, &params).is_err());
}
