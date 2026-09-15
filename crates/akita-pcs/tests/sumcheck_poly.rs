#![allow(missing_docs)]

use akita_sumcheck::UniPoly;
use jolt_field::{Field, Fp64, One, Ring, Zero};
use rand::rngs::StdRng;
use rand::SeedableRng;

type F = Fp64<4294967197>;

#[test]
fn compressed_unipoly_round_trip_and_eval() {
    let mut rng = StdRng::seed_from_u64(123);

    for degree in 0..8usize {
        let coeffs: Vec<F> = (0..=degree).map(|_| F::random(&mut rng)).collect();
        let poly = UniPoly::from_coeffs(coeffs);
        let hint = poly.evaluate(&F::zero()) + poly.evaluate(&F::one());
        let compressed = poly.compress();
        let decompressed = compressed.decompress(&hint);

        for x_u64 in [0u64, 1, 2, 3, 17] {
            let x = F::from_u64(x_u64);
            let direct = poly.evaluate(&x);
            assert_eq!(direct, decompressed.evaluate(&x));
            assert_eq!(direct, compressed.eval_from_hint(&hint, &x));
        }
    }
}
