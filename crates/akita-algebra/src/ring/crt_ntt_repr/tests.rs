use super::lut::{centered_prime_residue_i128, centered_prime_residue_i64};
use super::*;
use crate::ntt::tables::{Q16_NUM_PRIMES, Q16_PRIMES, Q32_PRIMES};

#[test]
fn centered_prime_residue_keeps_positive_half_boundary() {
    let prime16 = Q16_PRIMES[0];
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
    let params = CrtNttParamSet::<i16, Q16_NUM_PRIMES, D>::new(Q16_PRIMES);
    let prime = params.primes[0];
    let half = i32::from(prime.p) / 2;
    let lut = CenteredMontLut::<i16, Q16_NUM_PRIMES>::new(&params, half + 1);

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
