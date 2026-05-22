//! Cross-backend SIMD NTT parity tests.
//!
//! Currently only NEON is plugged into the `simd` alias; an earlier AVX2 /
//! AVX-512 NTT port was reverted (see `ntt/mod.rs` for the rationale).
//! This module is written against `super::simd::*` rather than `super::neon::*`
//! so the tests automatically extend to any future SIMD backend that
//! re-introduces the `simd` alias on x86.

#![cfg(target_arch = "aarch64")]

use super::butterfly::{
    forward_ntt as scalar_forward_ntt, forward_ntt_cyclic as scalar_forward_ntt_cyclic,
    inverse_ntt as scalar_inverse_ntt, inverse_ntt_cyclic as scalar_inverse_ntt_cyclic,
    NttTwiddles,
};
use super::prime::{MontCoeff, NttPrime};
use super::simd;

fn random_mont_array_i32<const D: usize>(prime: NttPrime<i32>, seed: u64) -> [MontCoeff<i32>; D] {
    let mut state = seed;
    std::array::from_fn(|_| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = ((state >> 33) as i64 % prime.p as i64) as i32;
        prime.from_canonical(val)
    })
}

fn random_mont_array_i16<const D: usize>(prime: NttPrime<i16>, seed: u64) -> [MontCoeff<i16>; D] {
    let mut state = seed;
    std::array::from_fn(|_| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = ((state >> 33) as i64 % prime.p as i64) as i16;
        prime.from_canonical(val)
    })
}

const TEST_PRIME_I32: i32 = 1073707009;
const TEST_PRIME_I16: i16 = 13697;

#[test]
fn simd_forward_ntt_i32_matches_scalar() {
    let prime = NttPrime::compute(TEST_PRIME_I32);
    let tw = NttTwiddles::<i32, 512>::compute(prime);
    let input = random_mont_array_i32::<512>(prime, 0xCAFE);

    let mut simd_result = input;
    unsafe { simd::forward_ntt_i32(&mut simd_result, prime, &tw) };

    let mut scalar_result = input;
    scalar_forward_ntt(&mut scalar_result, prime, &tw);

    for i in 0..512 {
        let n = prime.to_canonical(simd_result[i]);
        let s = prime.to_canonical(scalar_result[i]);
        assert_eq!(n, s, "mismatch at index {i}: simd={n}, scalar={s}");
    }
}

#[test]
fn simd_inverse_ntt_i32_matches_scalar() {
    let prime = NttPrime::compute(TEST_PRIME_I32);
    let tw = NttTwiddles::<i32, 512>::compute(prime);
    let input = random_mont_array_i32::<512>(prime, 0xBEEF);

    let mut simd_result = input;
    unsafe { simd::inverse_ntt_i32(&mut simd_result, prime, &tw) };

    let mut scalar_result = input;
    scalar_inverse_ntt(&mut scalar_result, prime, &tw);

    for i in 0..512 {
        let n = prime.to_canonical(simd_result[i]);
        let s = prime.to_canonical(scalar_result[i]);
        assert_eq!(n, s, "mismatch at index {i}: simd={n}, scalar={s}");
    }
}

#[test]
fn simd_forward_inverse_roundtrip_i32() {
    let prime = NttPrime::compute(TEST_PRIME_I32);
    let tw = NttTwiddles::<i32, 512>::compute(prime);
    let input = random_mont_array_i32::<512>(prime, 0xDEAD);
    let canonical_input: Vec<i32> = input.iter().map(|c| prime.to_canonical(*c)).collect();

    let mut a = input;
    unsafe {
        simd::forward_ntt_i32(&mut a, prime, &tw);
        simd::inverse_ntt_i32(&mut a, prime, &tw);
    }

    for i in 0..512 {
        let result = prime.to_canonical(a[i]);
        assert_eq!(
            result, canonical_input[i],
            "roundtrip mismatch at index {i}"
        );
    }
}

#[test]
fn simd_cyclic_ntt_i32_matches_scalar() {
    let prime = NttPrime::compute(TEST_PRIME_I32);
    let tw = NttTwiddles::<i32, 512>::compute(prime);
    let input = random_mont_array_i32::<512>(prime, 0xFACE);

    let mut simd_fwd = input;
    unsafe { simd::forward_ntt_cyclic_i32(&mut simd_fwd, prime, &tw) };

    let mut scalar_fwd = input;
    scalar_forward_ntt_cyclic(&mut scalar_fwd, prime, &tw);

    for i in 0..512 {
        let n = prime.to_canonical(simd_fwd[i]);
        let s = prime.to_canonical(scalar_fwd[i]);
        assert_eq!(n, s, "forward cyclic mismatch at {i}: simd={n}, scalar={s}");
    }

    let mut simd_inv = simd_fwd;
    unsafe { simd::inverse_ntt_cyclic_i32(&mut simd_inv, prime, &tw) };

    let mut scalar_inv = scalar_fwd;
    scalar_inverse_ntt_cyclic(&mut scalar_inv, prime, &tw);

    for i in 0..512 {
        let n = prime.to_canonical(simd_inv[i]);
        let s = prime.to_canonical(scalar_inv[i]);
        assert_eq!(n, s, "inverse cyclic mismatch at {i}: simd={n}, scalar={s}");
    }
}

#[test]
fn simd_pointwise_mul_acc_i32_matches_scalar() {
    let prime = NttPrime::compute(TEST_PRIME_I32);
    const D: usize = 512;
    let acc_init = random_mont_array_i32::<D>(prime, 0x1111);
    let lhs = random_mont_array_i32::<D>(prime, 0x2222);
    let rhs = random_mont_array_i32::<D>(prime, 0x3333);

    let mut simd_acc = acc_init;
    unsafe {
        simd::pointwise_mul_acc_i32(
            simd_acc.as_mut_ptr() as *mut i32,
            lhs.as_ptr() as *const i32,
            rhs.as_ptr() as *const i32,
            D,
            prime.p,
            prime.pinv,
        );
    }

    let mut scalar_acc = acc_init;
    for i in 0..D {
        let prod = prime.mul(lhs[i], rhs[i]);
        let sum = MontCoeff::from_raw(scalar_acc[i].raw().wrapping_add(prod.raw()));
        scalar_acc[i] = prime.reduce_range(sum);
    }

    for i in 0..D {
        let n = prime.to_canonical(simd_acc[i]);
        let s = prime.to_canonical(scalar_acc[i]);
        assert_eq!(n, s, "pointwise mul acc mismatch at {i}");
    }
}

#[test]
fn simd_pointwise_mul_acc_i32_handles_scalar_tail() {
    let prime = NttPrime::compute(TEST_PRIME_I32);
    const D: usize = 6;
    let acc_init = random_mont_array_i32::<D>(prime, 0x4444);
    let lhs = random_mont_array_i32::<D>(prime, 0x5555);
    let rhs = random_mont_array_i32::<D>(prime, 0x6666);

    let mut simd_acc = acc_init;
    unsafe {
        simd::pointwise_mul_acc_i32(
            simd_acc.as_mut_ptr() as *mut i32,
            lhs.as_ptr() as *const i32,
            rhs.as_ptr() as *const i32,
            D,
            prime.p,
            prime.pinv,
        );
    }

    let mut scalar_acc = acc_init;
    for i in 0..D {
        let prod = prime.mul(lhs[i], rhs[i]);
        let sum = MontCoeff::from_raw(scalar_acc[i].raw().wrapping_add(prod.raw()));
        scalar_acc[i] = prime.reduce_range(sum);
    }

    assert_eq!(simd_acc, scalar_acc);
}

#[cfg(feature = "parallel")]
#[test]
fn simd_add_reduce_i32_handles_scalar_tail() {
    let prime = NttPrime::compute(TEST_PRIME_I32);
    const D: usize = 6;
    let acc_init = random_mont_array_i32::<D>(prime, 0x7777);
    let other = random_mont_array_i32::<D>(prime, 0x8888);

    let mut simd_acc = acc_init;
    unsafe {
        simd::add_reduce_i32(
            simd_acc.as_mut_ptr() as *mut i32,
            other.as_ptr() as *const i32,
            D,
            prime.p,
        );
    }

    let mut scalar_acc = acc_init;
    for i in 0..D {
        let sum = MontCoeff::from_raw(scalar_acc[i].raw().wrapping_add(other[i].raw()));
        scalar_acc[i] = prime.reduce_range(sum);
    }

    assert_eq!(simd_acc, scalar_acc);
}

#[test]
fn simd_forward_ntt_i16_matches_scalar() {
    let prime = NttPrime::compute(TEST_PRIME_I16);
    let tw = NttTwiddles::<i16, 64>::compute(prime);
    let input = random_mont_array_i16::<64>(prime, 0xABCD);

    let mut simd_result = input;
    unsafe { simd::forward_ntt_i16(&mut simd_result, prime, &tw) };

    let mut scalar_result = input;
    scalar_forward_ntt(&mut scalar_result, prime, &tw);

    for i in 0..64 {
        let n = prime.to_canonical(simd_result[i]);
        let s = prime.to_canonical(scalar_result[i]);
        assert_eq!(n, s, "i16 forward mismatch at {i}: simd={n}, scalar={s}");
    }
}

#[test]
fn simd_inverse_ntt_i16_matches_scalar() {
    let prime = NttPrime::compute(TEST_PRIME_I16);
    let tw = NttTwiddles::<i16, 64>::compute(prime);
    let input = random_mont_array_i16::<64>(prime, 0xFEED);

    let mut simd_result = input;
    unsafe { simd::inverse_ntt_i16(&mut simd_result, prime, &tw) };

    let mut scalar_result = input;
    scalar_inverse_ntt(&mut scalar_result, prime, &tw);

    for i in 0..64 {
        let n = prime.to_canonical(simd_result[i]);
        let s = prime.to_canonical(scalar_result[i]);
        assert_eq!(n, s, "i16 inverse mismatch at {i}: simd={n}, scalar={s}");
    }
}

#[test]
fn simd_forward_inverse_roundtrip_i16() {
    let prime = NttPrime::compute(TEST_PRIME_I16);
    let tw = NttTwiddles::<i16, 64>::compute(prime);
    let input = random_mont_array_i16::<64>(prime, 0x7777);
    let canonical_input: Vec<i16> = input.iter().map(|c| prime.to_canonical(*c)).collect();

    let mut a = input;
    unsafe {
        simd::forward_ntt_i16(&mut a, prime, &tw);
        simd::inverse_ntt_i16(&mut a, prime, &tw);
    }

    for i in 0..64 {
        let result = prime.to_canonical(a[i]);
        assert_eq!(result, canonical_input[i], "i16 roundtrip mismatch at {i}");
    }
}

#[test]
fn simd_cyclic_i16_matches_scalar() {
    let prime = NttPrime::compute(TEST_PRIME_I16);
    let tw = NttTwiddles::<i16, 64>::compute(prime);
    let input = random_mont_array_i16::<64>(prime, 0x9999);

    let mut simd_fwd = input;
    unsafe { simd::forward_ntt_cyclic_i16(&mut simd_fwd, prime, &tw) };

    let mut scalar_fwd = input;
    scalar_forward_ntt_cyclic(&mut scalar_fwd, prime, &tw);

    for i in 0..64 {
        let n = prime.to_canonical(simd_fwd[i]);
        let s = prime.to_canonical(scalar_fwd[i]);
        assert_eq!(n, s, "i16 fwd cyclic mismatch at {i}");
    }

    let mut simd_inv = simd_fwd;
    unsafe { simd::inverse_ntt_cyclic_i16(&mut simd_inv, prime, &tw) };

    let mut scalar_inv = scalar_fwd;
    scalar_inverse_ntt_cyclic(&mut scalar_inv, prime, &tw);

    for i in 0..64 {
        let n = prime.to_canonical(simd_inv[i]);
        let s = prime.to_canonical(scalar_inv[i]);
        assert_eq!(n, s, "i16 inv cyclic mismatch at {i}");
    }
}

#[test]
fn simd_pointwise_mul_acc_i16_matches_scalar() {
    let prime = NttPrime::compute(TEST_PRIME_I16);
    const D: usize = 64;
    let acc_init = random_mont_array_i16::<D>(prime, 0xAAAA);
    let lhs = random_mont_array_i16::<D>(prime, 0xBBBB);
    let rhs = random_mont_array_i16::<D>(prime, 0xCCCC);

    let mut simd_acc = acc_init;
    unsafe {
        simd::pointwise_mul_acc_i16(
            simd_acc.as_mut_ptr() as *mut i16,
            lhs.as_ptr() as *const i16,
            rhs.as_ptr() as *const i16,
            D,
            prime.p,
            prime.pinv,
        );
    }

    let mut scalar_acc = acc_init;
    for i in 0..D {
        let prod = prime.mul(lhs[i], rhs[i]);
        let sum = MontCoeff::from_raw(scalar_acc[i].raw().wrapping_add(prod.raw()));
        scalar_acc[i] = prime.reduce_range(sum);
    }

    for i in 0..D {
        let n = prime.to_canonical(simd_acc[i]);
        let s = prime.to_canonical(scalar_acc[i]);
        assert_eq!(n, s, "i16 pointwise mul acc mismatch at {i}");
    }
}
