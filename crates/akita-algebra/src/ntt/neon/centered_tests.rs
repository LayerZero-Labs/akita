use std::arch::aarch64::*;

use super::i32_kernels::centered_reduce_4x_i32;
use super::{forward_ntt_i32, forward_ntt_i8_i32};
use crate::ntt::butterfly::forward_ntt;
use crate::ntt::tables::{Q128_RAW_PRIMES, Q64_PRIMES};
use crate::ntt::{MontCoeff, NttKernelPlan, NttPrime, NttTwiddles};

#[test]
fn centered_reduction_range_residue_and_rounding_boundaries() {
    let moduli = [3, 5, 17, 257, 12289, (1 << 29) - 1, (1 << 30) - 1];
    for p in moduli
        .into_iter()
        .chain(Q128_RAW_PRIMES)
        .chain(Q64_PRIMES.map(|p| p.p))
    {
        let p64 = i64::from(p);
        let reciprocal = ((1i64 << 31) + p64 / 2) / p64;
        let mut inputs = vec![
            -2 * p64 + 1,
            -p64 - 1,
            -p64,
            -p64 + 1,
            -1,
            0,
            1,
            p64 - 1,
            p64,
            p64 + 1,
            2 * p64 - 1,
        ];
        // Include neighbors of every possible sqrdmulh rounding boundary.
        for quotient in -2..=2 {
            let threshold = ((2 * quotient + 1) * (1i64 << 30)) / reciprocal;
            inputs.extend(
                (-2..=2)
                    .map(|delta| threshold + delta)
                    .filter(|&x| x.abs() < 2 * p64),
            );
        }
        if p <= 257 {
            inputs.extend(-2 * p64 + 1..2 * p64);
        }
        let mut state = 0x483aef827u64;
        for _ in 0..8192 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            inputs.push((state % (4 * p64 - 1) as u64) as i64 - 2 * p64 + 1);
        }
        for chunk in inputs.chunks(4) {
            let values = std::array::from_fn::<_, 4, _>(|i| chunk[i % chunk.len()] as i32);
            let mut output = [0i32; 4];
            unsafe {
                vst1q_s32(
                    output.as_mut_ptr(),
                    centered_reduce_4x_i32(
                        vld1q_s32(values.as_ptr()),
                        vdupq_n_s32(p),
                        vdupq_n_s32(reciprocal as i32),
                    ),
                );
            }
            for (&x, &y) in values.iter().zip(&output) {
                assert!(i64::from(y).abs() < p64, "x={x}, y={y}, p={p}");
                assert_eq!(i64::from(y).rem_euclid(p64), i64::from(x).rem_euclid(p64));
                let q = (i128::from(x) * i128::from(reciprocal) + (1i128 << 30)) >> 31;
                assert_eq!(i128::from(y), i128::from(x) - q * i128::from(p));
            }
        }
    }
}

fn check_transform<const D: usize>(prime: NttPrime<i32>) {
    let tw = NttTwiddles::<i32, D>::compute(prime);
    let p = i64::from(prime.p);
    let mut state = 0xfa09c8e41u64;
    for case in 0..5 {
        let input = std::array::from_fn(|i| {
            let x = match case {
                0 => p - 1,
                1 => -p + 1,
                2 => {
                    if i & 1 == 0 {
                        p - 1
                    } else {
                        -p + 1
                    }
                }
                3 => [0, 1, -1, p / 2, -p / 2][i % 5],
                _ => {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state % (2 * p - 1) as u64) as i64 - p + 1
                }
            };
            MontCoeff::from_raw(x as i32)
        });
        let mut expected = input;
        forward_ntt(&mut expected, prime, &tw, NttKernelPlan::SCALAR);
        let mut actual = input;
        unsafe {
            forward_ntt_i32(&mut actual, prime, &tw);
        }
        assert_eq!(actual, expected, "D={D}, p={p}, case={case}");
    }
    let digits = std::array::from_fn(|i| (i as i8).wrapping_mul(37).wrapping_add(11));
    let mut expected = std::array::from_fn(|i| prime.from_canonical(i32::from(digits[i])));
    forward_ntt(&mut expected, prime, &tw, NttKernelPlan::SCALAR);
    let mut actual = [MontCoeff::from_raw(0); D];
    unsafe {
        forward_ntt_i8_i32(&mut actual, &digits, prime, &tw);
    }
    assert_eq!(actual, expected, "signed digits D={D}, p={p}");
}

#[test]
fn centered_forward_matches_scalar_for_general_and_signed_inputs() {
    for p in Q128_RAW_PRIMES.into_iter().chain(Q64_PRIMES.map(|p| p.p)) {
        let prime = NttPrime::compute(p);
        check_transform::<8>(prime);
        check_transform::<16>(prime);
        check_transform::<128>(prime);
        check_transform::<256>(prime);
        check_transform::<512>(prime);
        check_transform::<1024>(prime);
    }
    for prime in Q64_PRIMES {
        check_transform::<2048>(prime);
    }
}
