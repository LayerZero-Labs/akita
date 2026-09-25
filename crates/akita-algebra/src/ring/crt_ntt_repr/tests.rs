use jolt_field::{Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};

use super::lut::centered_prime_residue_i64;
use super::*;
use crate::ntt::prime::NttPrime;
use crate::ntt::tables::{
    q128_primes, I16_TAIL_PRIME, Q128_NUM_PRIMES, Q32_NUM_PRIMES, Q32_PRIMES,
};
use crate::CyclotomicRing;

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
}

/// Canonical coefficients at every centering and 26-bit slice boundary below
/// `modulus`, followed by pseudo-random fill.
fn field_residue_probes<const D: usize>(modulus: u128) -> [u128; D] {
    let half = modulus / 2;
    let mut edges = vec![0, 1, 2, half - 1, half, half + 1, modulus - 2, modulus - 1];
    for bit in (26..128).step_by(26) {
        for value in [1u128 << bit, (1u128 << bit) - 1, half ^ (1u128 << bit)] {
            if value < modulus {
                edges.push(value);
                edges.push(modulus - 1 - value);
            }
        }
    }
    let mut state = 0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c834u128;
    std::array::from_fn(|index| {
        edges.get(index).copied().unwrap_or_else(|| {
            state = state
                .wrapping_mul(0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645)
                .wrapping_add(0x5851_f42d_4c95_7f2d_1405_7b7e_f767_814f);
            state % modulus
        })
    })
}

fn assert_field_residues_match<F, W, const K: usize, const D: usize>(
    params: &CrtNttParamSet<W, K, D>,
) where
    F: CrtNttConvertibleField,
    W: PrimeWidth,
{
    let modulus = (-F::one()).to_u128_checked().unwrap() + 1;
    let canonical = field_residue_probes::<D>(modulus);
    let ring = CyclotomicRing::<F, D>::from_coefficients(
        canonical.map(|value| F::from_u128_checked(value).unwrap()),
    );
    let mut residues = [[MontCoeff::from_raw(W::default()); D]; K];
    params.field_residues(&ring, &mut residues);
    for (prime, residues) in params.primes.iter().zip(&residues) {
        let p = i128::from(prime.p.to_i64());
        for (&value, residue) in canonical.iter().zip(residues) {
            let centered = if value > modulus / 2 {
                -((modulus - value) as i128)
            } else {
                value as i128
            };
            let expected =
                prime.from_canonical(prime.center(W::from_i64(centered.rem_euclid(p) as i64)));
            assert!(
                residue.raw().to_i64().unsigned_abs() < prime.p.to_i64() as u64,
                "residue of {value} outside (-p, p) for p = {p}"
            );
            assert_eq!(
                prime.normalize(*residue),
                prime.normalize(expected),
                "residue of {value} modulo {p}"
            );
        }
    }
}

#[test]
fn field_residues_match_centered_reduction() {
    fn check<F: CrtNttConvertibleField>() {
        // Several conversion chunks, so the SIMD kernels see nonzero offsets.
        assert_field_residues_match::<F, i32, Q128_NUM_PRIMES, 256>(&CrtNttParamSet::new(
            q128_primes(),
        ));
        assert_field_residues_match::<F, i32, Q32_NUM_PRIMES, 4>(&CrtNttParamSet::new(Q32_PRIMES));
        assert_field_residues_match::<F, i16, 1, 128>(&CrtNttParamSet::new([I16_TAIL_PRIME]));
        assert_field_residues_match::<F, i16, SYNTHETIC_I16_NUM_PRIMES, 64>(&CrtNttParamSet::new(
            synthetic_i16_primes(),
        ));
    }
    check::<Prime32Offset99>();
    check::<Prime64Offset59>();
    check::<Prime128OffsetA7F7>();
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
