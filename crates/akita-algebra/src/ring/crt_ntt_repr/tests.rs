use super::lut::{centered_prime_residue_i128, centered_prime_residue_i64};
use super::*;
use crate::ntt::prime::NttPrime;
use crate::ntt::tables::Q32_PRIMES;

const SYNTHETIC_I16_NUM_PRIMES: usize = 3;

fn synthetic_i16_primes() -> [NttPrime<i16>; SYNTHETIC_I16_NUM_PRIMES] {
    [
        NttPrime::compute(15361_i16),
        NttPrime::compute(13313_i16),
        NttPrime::compute(12289_i16),
    ]
}

#[test]
fn centered_prime_residue_keeps_positive_half_boundary() {
    let primes = synthetic_i16_primes();
    let prime16 = primes[0];
    let half16 = i64::from(prime16.p) / 2;
    assert_eq!(centered_prime_residue_i64(prime16, half16), half16 as i16);
    assert_eq!(
        centered_prime_residue_i64(prime16, half16 + 1),
        (half16 + 1 - i64::from(prime16.p)) as i16
    );

    let prime32 = Q32_PRIMES[0];
    let half32 = i64::from(prime32.p) / 2;
    assert_eq!(centered_prime_residue_i64(prime32, half32), half32 as i32);
    assert_eq!(
        centered_prime_residue_i64(prime32, half32 + 1),
        (half32 + 1 - i64::from(prime32.p)) as i32
    );
    assert_eq!(
        centered_prime_residue_i128(prime32, i128::from(half32)),
        half32 as i32
    );
    assert_eq!(
        centered_prime_residue_i128(prime32, i128::from(half32 + 1)),
        (half32 + 1 - i64::from(prime32.p)) as i32
    );
}

#[test]
fn centered_mont_lut_matches_centered_residue_boundary() {
    const D: usize = 64;
    let primes = synthetic_i16_primes();
    let params = CrtNttParamSet::<i16, SYNTHETIC_I16_NUM_PRIMES, D>::new(primes);
    let prime = params.primes[0];
    let half = i32::from(prime.p) / 2;
    let lut = CenteredMontLut::<i16, SYNTHETIC_I16_NUM_PRIMES>::new(&params, half + 1);

    let boundary = centered_prime_residue_i64(prime, i64::from(half));
    let past_boundary = centered_prime_residue_i64(prime, i64::from(half + 1));
    assert_eq!(boundary, half as i16);
    assert_eq!(past_boundary, (half + 1 - i32::from(prime.p)) as i16);
    assert_eq!(lut.get(0, half), Some(prime.from_canonical(boundary)));
    assert_eq!(
        lut.get(0, half + 1),
        Some(prime.from_canonical(past_boundary))
    );
}

#[test]
#[should_panic(expected = "lazy pointwise dot requires an i32 SIMD parameter set")]
fn lazy_pointwise_dot_rejects_non_i32_parameter_sets() {
    const D: usize = 64;
    let params = CrtNttParamSet::<i16, SYNTHETIC_I16_NUM_PRIMES, D>::new(synthetic_i16_primes());
    let lut = DigitMontLut::new_with_digit_bound(&params, 2);
    let mut accs = [CyclotomicCrtNtt::zero()];
    let matrix_row = [CyclotomicCrtNtt::zero()];
    let ntt_mat = [matrix_row.as_slice()];
    let digits = [[0i8; D]];
    let mut scratch = [[MontCoeff::from_raw(0i16); D]; I32_LAZY_DOT_BATCH];

    CyclotomicCrtNtt::add_assign_col_pointwise_dot_i8_multi_with_lut_scratch(
        &mut accs,
        &ntt_mat,
        0,
        &digits,
        &params,
        &lut,
        &mut scratch,
    );
}
