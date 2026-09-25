use super::*;
use crate::ntt::butterfly::NttTwiddles;
use crate::ntt::prime::{MontCoeff, NttPrime, I32_LAZY_DOT_BATCH};
use crate::ntt::tables::{I16_TAIL_PRIME, Q128_RAW_PRIMES};

const AVX2_ONLY: AvxCpuFeatures = AvxCpuFeatures {
    avx2: true,
    avx512: false,
};

const AVX512: AvxCpuFeatures = AvxCpuFeatures {
    avx2: true,
    avx512: true,
};

const NO_AVX2: AvxCpuFeatures = AvxCpuFeatures {
    avx2: false,
    avx512: false,
};

#[test]
fn avx_mode_defaults_to_avx2_when_supported() {
    assert_eq!(select_avx_ntt_mode(None, None, AVX2_ONLY), Some(AvxNttMode::Avx2));
    assert_eq!(select_avx_ntt_mode(None, None, AVX512), Some(AvxNttMode::Avx2));
}

#[test]
fn avx512_is_opt_in_and_requires_the_features() {
    assert_eq!(select_avx_ntt_mode(None, Some("1"), AVX512), Some(AvxNttMode::Avx512));
    assert_eq!(select_avx_ntt_mode(None, Some("1"), AVX2_ONLY), Some(AvxNttMode::Avx2));
    assert_eq!(select_avx_ntt_mode(None, Some("0"), AVX512), Some(AvxNttMode::Avx2));
}

#[test]
fn x86_ntt_requires_avx2() {
    assert_eq!(select_avx_ntt_mode(None, None, NO_AVX2), None);
    assert_eq!(select_avx_ntt_mode(None, Some("1"), NO_AVX2), None);
}

#[test]
fn scalar_kill_switch_disables_x86_ntt_simd() {
    assert_eq!(select_avx_ntt_mode(Some("1"), None, AVX2_ONLY), None);
    assert_eq!(select_avx_ntt_mode(Some("1"), Some("1"), AVX512), None);
}

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

fn edge_mont_array_i32<const D: usize>(prime: NttPrime<i32>) -> [MontCoeff<i32>; D] {
    let values = [
        0,
        1,
        -1,
        prime.p - 1,
        1 - prime.p,
        prime.p / 2,
        -(prime.p / 2),
        0x4000_1234_i32,
        -0x3fff_4321_i32,
    ];
    std::array::from_fn(|i| MontCoeff::from_raw(values[i % values.len()]))
}

fn edge_mont_array_i16<const D: usize>(prime: NttPrime<i16>) -> [MontCoeff<i16>; D] {
    let values = [
        0,
        1,
        -1,
        prime.p - 1,
        1 - prime.p,
        prime.p / 2,
        -(prime.p / 2),
        0x3a5a_i16,
        -0x3211_i16,
    ];
    std::array::from_fn(|i| MontCoeff::from_raw(values[i % values.len()]))
}

fn scalar_pointwise_i32<const D: usize>(
    acc: &mut [MontCoeff<i32>; D],
    lhs: &[MontCoeff<i32>; D],
    rhs: &[MontCoeff<i32>; D],
    prime: NttPrime<i32>,
) {
    for i in 0..D {
        let prod = prime.mul(lhs[i], rhs[i]);
        let sum = MontCoeff::from_raw(acc[i].raw().wrapping_add(prod.raw()));
        acc[i] = prime.reduce_range(sum);
    }
}

fn scalar_pointwise_i16<const D: usize>(
    acc: &mut [MontCoeff<i16>; D],
    lhs: &[MontCoeff<i16>; D],
    rhs: &[MontCoeff<i16>; D],
    prime: NttPrime<i16>,
) {
    for i in 0..D {
        let prod = prime.mul(lhs[i], rhs[i]);
        let sum = MontCoeff::from_raw(acc[i].raw().wrapping_add(prod.raw()));
        acc[i] = prime.reduce_range(sum);
    }
}

fn scalar_add_reduce_i32<const D: usize>(
    acc: &mut [MontCoeff<i32>; D],
    other: &[MontCoeff<i32>; D],
    prime: NttPrime<i32>,
) {
    for i in 0..D {
        let sum = MontCoeff::from_raw(acc[i].raw().wrapping_add(other[i].raw()));
        acc[i] = prime.reduce_range(sum);
    }
}

fn assert_i32_crt_ops<const D: usize>() {
    for (prime_index, raw_prime) in Q128_RAW_PRIMES.into_iter().enumerate() {
        let prime = NttPrime::compute(raw_prime);
        let lhs = edge_mont_array_i32::<D>(prime);
        let rhs = random_mont_array_i32::<D>(prime, 0x9173 + prime_index as u64);

        let mut expected_add = lhs;
        scalar_add_reduce_i32(&mut expected_add, &rhs, prime);

        if std::is_x86_feature_detected!("avx2") {
            let mut add = lhs;
            unsafe {
                add_reduce_i32(
                    add.as_mut_ptr().cast(),
                    add.as_ptr().cast(),
                    rhs.as_ptr().cast(),
                    D,
                    prime.p,
                );
            }
            assert_eq!(add, expected_add, "AVX2 add p={raw_prime} D={D}");
        }

        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512bw")
        {
            let mut add = lhs;
            unsafe {
                add_reduce_i32_avx512(
                    add.as_mut_ptr().cast(),
                    add.as_ptr().cast(),
                    rhs.as_ptr().cast(),
                    D,
                    prime.p,
                );
            }
            assert_eq!(add, expected_add, "AVX-512 add p={raw_prime} D={D}");
        }
    }
}

#[test]
fn q128_i32_crt_ops_match_scalar_at_vector_boundaries() {
    assert_i32_crt_ops::<7>();
    assert_i32_crt_ops::<8>();
    assert_i32_crt_ops::<9>();
    assert_i32_crt_ops::<15>();
    assert_i32_crt_ops::<16>();
    assert_i32_crt_ops::<17>();
    assert_i32_crt_ops::<31>();
    assert_i32_crt_ops::<32>();
    assert_i32_crt_ops::<33>();
}

fn scalar_add_reduce_i16<const D: usize>(
    acc: &mut [MontCoeff<i16>; D],
    other: &[MontCoeff<i16>; D],
    prime: NttPrime<i16>,
) {
    for i in 0..D {
        let sum = MontCoeff::from_raw(acc[i].raw().wrapping_add(other[i].raw()));
        acc[i] = prime.reduce_range(sum);
    }
}

fn scalar_forward_ntt_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    for (ai, psi) in a.iter_mut().zip(tw.psi_pows.iter()) {
        *ai = prime.mul(*ai, *psi);
    }
    scalar_forward_ntt_cyclic_i32(a, prime, tw);
}

fn scalar_inverse_ntt_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        let mut start = 0usize;
        while start < D {
            for j in 0..len {
                let w = tw.inv_twiddles[twiddle_base + j];
                let u = a[start + j];
                let v = prime.mul(a[start + j + len], w);
                let sum = u.raw().wrapping_add(v.raw());
                let diff = u.raw().wrapping_sub(v.raw());
                a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                a[start + j + len] = prime.reduce_range(MontCoeff::from_raw(diff));
            }
            start += 2 * len;
        }
        len *= 2;
    }
    for (ai, fused) in a.iter_mut().zip(tw.d_inv_psi_inv.iter()) {
        *ai = prime.mul(*ai, *fused);
    }
}

fn scalar_forward_ntt_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    for (coefficient, psi) in a.iter_mut().zip(tw.psi_pows.iter()) {
        *coefficient = prime.mul(*coefficient, *psi);
    }
    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        for start in (0..D).step_by(2 * len) {
            for j in 0..len {
                let u = a[start + j];
                let v = a[start + j + len];
                a[start + j] =
                    prime.reduce_range(MontCoeff::from_raw(u.raw().wrapping_add(v.raw())));
                a[start + j + len] = prime.mul(
                    MontCoeff::from_raw(u.raw().wrapping_sub(v.raw())),
                    tw.fwd_twiddles[twiddle_base + j],
                );
            }
        }
        len /= 2;
    }
    prime.reduce_range_in_place(a);
}

fn scalar_inverse_ntt_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        for start in (0..D).step_by(2 * len) {
            for j in 0..len {
                let u = a[start + j];
                let v = prime.mul(a[start + j + len], tw.inv_twiddles[twiddle_base + j]);
                a[start + j] =
                    prime.reduce_range(MontCoeff::from_raw(u.raw().wrapping_add(v.raw())));
                a[start + j + len] =
                    prime.reduce_range(MontCoeff::from_raw(u.raw().wrapping_sub(v.raw())));
            }
        }
        len *= 2;
    }
    for (coefficient, scale) in a.iter_mut().zip(tw.d_inv_psi_inv.iter()) {
        *coefficient = prime.mul(*coefficient, *scale);
    }
}

fn assert_i16_mont_arrays_eq_mod<const D: usize>(
    actual: &[MontCoeff<i16>; D],
    expected: &[MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    phase: &str,
) {
    // Montgomery coefficients are range-bounded residues, not unique raw
    // representatives. SIMD and scalar butterflies may differ by one modulus.
    for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.raw() > -prime.p && actual.raw() < prime.p,
            "{phase} AVX2 output outside (-p, p) at {i}: {actual:?}"
        );
        assert_eq!(
            prime.to_canonical(*actual),
            prime.to_canonical(*expected),
            "{phase} mismatch modulo p at {i}: avx2={actual:?}, scalar={expected:?}"
        );
    }
}

fn scalar_forward_ntt_cyclic_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let mut start = 0usize;
        while start < D {
            for j in 0..len {
                let w = tw.fwd_twiddles[twiddle_base + j];
                let u = a[start + j];
                let v = a[start + j + len];
                let sum = u.raw().wrapping_add(v.raw());
                let diff = u.raw().wrapping_sub(v.raw());
                a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                a[start + j + len] = prime.mul(MontCoeff::from_raw(diff), w);
            }
            start += 2 * len;
        }
        len /= 2;
    }
    prime.reduce_range_in_place(a);
}

fn scalar_inverse_ntt_cyclic_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        let mut start = 0usize;
        while start < D {
            for j in 0..len {
                let w = tw.inv_twiddles[twiddle_base + j];
                let u = a[start + j];
                let v = prime.mul(a[start + j + len], w);
                let sum = u.raw().wrapping_add(v.raw());
                let diff = u.raw().wrapping_sub(v.raw());
                a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                a[start + j + len] = prime.reduce_range(MontCoeff::from_raw(diff));
            }
            start += 2 * len;
        }
        len *= 2;
    }
    for c in a.iter_mut() {
        *c = prime.mul(*c, tw.d_inv);
    }
}

fn canonical_i32<const D: usize>(a: &[MontCoeff<i32>; D], p: i32) -> [i32; D] {
    a.map(|x| x.raw().rem_euclid(p))
}

/// Whether the host can run the AVX-512 `i32` transform instantiations.
fn avx512_transform_available() -> bool {
    runtime::detect_cpu_features().avx512
}

/// The x86 transforms agree with the scalar reference modulo `p`; forward
/// outputs are canonical and inverse outputs lie in `(-p, p)`.
fn assert_ntt_i32_transforms_match_scalar<const D: usize>(use_avx512: bool) {
    for raw_prime in Q128_RAW_PRIMES {
        let prime = NttPrime::compute(raw_prime);
        let p = prime.p;
        let tw = NttTwiddles::<i32, D>::compute(prime);
        let random = random_mont_array_i32::<D>(prime, 0x5150 ^ D as u64 ^ raw_prime as u64);
        let extremes: [MontCoeff<i32>; D] = std::array::from_fn(|index| {
            MontCoeff::from_raw([i32::MIN, i32::MAX, 1 - p, p - 1, 0][index % 5])
        });
        let in_range = extremes.map(|x| MontCoeff::from_raw(x.raw() % p));
        let in_range_inputs = [random, in_range];
        let context = format!("prime={raw_prime}, D={D}, avx512={use_avx512}");

        // The negacyclic forward transform accepts any `i32`.
        for input in [random, in_range, extremes] {
            let mut avx = input;
            let mut scalar = input.map(|x| MontCoeff::from_raw(x.raw().rem_euclid(p)));
            // SAFETY: the caller checks the target features.
            unsafe { forward_ntt_i32(&mut avx, prime, &tw, use_avx512) };
            scalar_forward_ntt_i32(&mut scalar, prime, &tw);
            assert!(avx.iter().all(|x| (0..p).contains(&x.raw())), "{context}");
            assert_eq!(
                canonical_i32(&avx, p),
                canonical_i32(&scalar, p),
                "{context}"
            );
        }

        for input in in_range_inputs {
            let mut avx = input;
            let mut scalar = input;
            // SAFETY: the caller checks the target features.
            unsafe { inverse_ntt_i32(&mut avx, prime, &tw, use_avx512) };
            scalar_inverse_ntt_i32(&mut scalar, prime, &tw);
            assert!(avx.iter().all(|x| x.raw().abs() < p), "{context}");
            assert_eq!(
                canonical_i32(&avx, p),
                canonical_i32(&scalar, p),
                "{context}"
            );

            let mut avx = input;
            let mut scalar = input;
            // SAFETY: the caller checks the target features.
            unsafe { forward_ntt_cyclic_i32(&mut avx, prime, &tw, use_avx512) };
            scalar_forward_ntt_cyclic_i32(&mut scalar, prime, &tw);
            assert!(avx.iter().all(|x| (0..p).contains(&x.raw())), "{context}");
            assert_eq!(
                canonical_i32(&avx, p),
                canonical_i32(&scalar, p),
                "{context}"
            );

            let mut avx = input;
            let mut scalar = input;
            // SAFETY: the caller checks the target features.
            unsafe { inverse_ntt_cyclic_i32(&mut avx, prime, &tw, use_avx512) };
            scalar_inverse_ntt_cyclic_i32(&mut scalar, prime, &tw);
            assert!(avx.iter().all(|x| x.raw().abs() < p), "{context}");
            assert_eq!(
                canonical_i32(&avx, p),
                canonical_i32(&scalar, p),
                "{context}"
            );
        }

        // Round trip through the negacyclic pair returns the input.
        let mut round_trip = random;
        // SAFETY: the caller checks the target features.
        unsafe {
            forward_ntt_i32(&mut round_trip, prime, &tw, use_avx512);
            inverse_ntt_i32(&mut round_trip, prime, &tw, use_avx512);
        }
        assert_eq!(
            canonical_i32(&round_trip, p),
            canonical_i32(&random, p),
            "{context}"
        );
    }
}

fn assert_ntt_i32_transforms_match_scalar_all_sizes(use_avx512: bool) {
    assert_ntt_i32_transforms_match_scalar::<64>(use_avx512);
    assert_ntt_i32_transforms_match_scalar::<128>(use_avx512);
    assert_ntt_i32_transforms_match_scalar::<256>(use_avx512);
    assert_ntt_i32_transforms_match_scalar::<512>(use_avx512);
    assert_ntt_i32_transforms_match_scalar::<1024>(use_avx512);
}

#[test]
fn avx2_ntt_i32_transforms_match_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    assert_ntt_i32_transforms_match_scalar_all_sizes(false);
}

#[test]
fn avx512_ntt_i32_transforms_match_scalar() {
    if !std::is_x86_feature_detected!("avx2") || !avx512_transform_available() {
        return;
    }
    assert_ntt_i32_transforms_match_scalar_all_sizes(true);
}

fn assert_fused_i8_ntt_i32_matches_scalar<const D: usize>(use_avx512: bool) {
    let digits: [i8; D] =
        std::array::from_fn(|index| [i8::MIN, -17, -1, 0, 1, 13, 63, i8::MAX][index % 8]);
    for raw_prime in Q128_RAW_PRIMES {
        let prime = NttPrime::compute(raw_prime);
        let tw = NttTwiddles::<i32, D>::compute(prime);
        let mut actual = [MontCoeff::from_raw(0_i32); D];
        // SAFETY: the caller checks the target features.
        unsafe { forward_ntt_i8_i32(&mut actual, &digits, prime, &tw, use_avx512) };

        let mut expected = digits.map(|digit| prime.from_canonical(i32::from(digit)));
        scalar_forward_ntt_i32(&mut expected, prime, &tw);
        assert_eq!(
            canonical_i32(&actual, prime.p),
            canonical_i32(&expected, prime.p),
            "prime={raw_prime}, D={D}, avx512={use_avx512}"
        );
        assert!(actual.iter().all(|x| (0..prime.p).contains(&x.raw())));
    }
}

#[test]
fn fused_i8_ntt_i32_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let mut widths = vec![false];
    if avx512_transform_available() {
        widths.push(true);
    }
    for use_avx512 in widths {
        assert_fused_i8_ntt_i32_matches_scalar::<64>(use_avx512);
        assert_fused_i8_ntt_i32_matches_scalar::<128>(use_avx512);
        assert_fused_i8_ntt_i32_matches_scalar::<256>(use_avx512);
        assert_fused_i8_ntt_i32_matches_scalar::<1024>(use_avx512);
    }
}

fn assert_avx2_ntt_i16_transforms_match_scalar<const D: usize>() {
    let prime = NttPrime::compute(12289_i16);
    let tw = NttTwiddles::<i16, D>::compute(prime);
    for input in [
        random_mont_array_i16::<D>(prime, 0x1616 ^ D as u64),
        edge_mont_array_i16::<D>(prime),
    ] {
        let mut avx = input;
        let mut scalar = input;
        // SAFETY: the caller checks AVX2 support.
        unsafe { forward_ntt_i16(&mut avx, prime, &tw) };
        scalar_forward_ntt_i16(&mut scalar, prime, &tw);
        assert_i16_mont_arrays_eq_mod(&avx, &scalar, prime, "forward");

        // SAFETY: the caller checks AVX2 support.
        unsafe { inverse_ntt_i16(&mut avx, prime, &tw) };
        scalar_inverse_ntt_i16(&mut scalar, prime, &tw);
        assert_i16_mont_arrays_eq_mod(&avx, &scalar, prime, "inverse");
        assert_i16_mont_arrays_eq_mod(&avx, &input, prime, "round-trip");
    }
}

#[test]
fn avx2_ntt_i16_transforms_match_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    assert_avx2_ntt_i16_transforms_match_scalar::<64>();
    assert_avx2_ntt_i16_transforms_match_scalar::<128>();
    assert_avx2_ntt_i16_transforms_match_scalar::<256>();
    assert_avx2_ntt_i16_transforms_match_scalar::<512>();
}

fn assert_avx2_fused_i8_ntt_i16_matches_scalar<const D: usize>() {
    let prime = I16_TAIL_PRIME;
    let tw = NttTwiddles::<i16, D>::compute(prime);
    let digits: [i8; D] =
        std::array::from_fn(|index| [i8::MIN, -17, -1, 0, 1, 13, 63, i8::MAX][index % 8]);
    let mut actual = [MontCoeff::from_raw(0_i16); D];
    // SAFETY: the caller checks AVX2 support.
    unsafe { forward_ntt_i8_i16(&mut actual, &digits, prime, &tw) };

    let mut expected = digits.map(|digit| prime.from_canonical(i16::from(digit)));
    scalar_forward_ntt_i16(&mut expected, prime, &tw);
    assert_i16_mont_arrays_eq_mod(&actual, &expected, prime, "fused i8 forward");
}

#[test]
fn avx2_fused_i8_ntt_i16_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    assert_avx2_fused_i8_ntt_i16_matches_scalar::<64>();
    assert_avx2_fused_i8_ntt_i16_matches_scalar::<256>();
}

#[test]
fn avx2_pointwise_mul_acc_i32_matches_scalar_with_tail() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let prime = NttPrime::compute(1073707009_i32);
    const D: usize = 19;
    let acc_init = random_mont_array_i32::<D>(prime, 0x1111);
    let lhs = edge_mont_array_i32::<D>(prime);
    let rhs = random_mont_array_i32::<D>(prime, 0x3333);

    let mut avx_acc = acc_init;
    // SAFETY: guarded by the runtime AVX2 detection above.
    unsafe {
        pointwise_mul_acc_i32(
            avx_acc.as_mut_ptr() as *mut i32,
            lhs.as_ptr() as *const i32,
            rhs.as_ptr() as *const i32,
            D,
            prime.p,
            prime.pinv,
        );
    }

    let mut scalar_acc = acc_init;
    scalar_pointwise_i32(&mut scalar_acc, &lhs, &rhs, prime);
    assert_eq!(avx_acc, scalar_acc);
}

#[test]
fn avx2_lazy_i32_dot_matches_repeated_reduction() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    const D: usize = 19;
    for raw_prime in Q128_RAW_PRIMES {
        let prime = NttPrime::compute(raw_prime);
        let lhs: [[MontCoeff<i32>; D]; I32_LAZY_DOT_BATCH] = std::array::from_fn(|index| {
            if index == 0 {
                edge_mont_array_i32(prime)
            } else {
                random_mont_array_i32(prime, 0x1000 + index as u64)
            }
        });
        let rhs: [[MontCoeff<i32>; D]; I32_LAZY_DOT_BATCH] = std::array::from_fn(|index| {
            if index + 1 == I32_LAZY_DOT_BATCH {
                edge_mont_array_i32(prime)
            } else {
                random_mont_array_i32(prime, 0x2000 + index as u64)
            }
        });
        let lhs_pointers: [*const i32; I32_LAZY_DOT_BATCH] =
            std::array::from_fn(|index| lhs[index].as_ptr().cast::<i32>());
        let rhs_pointers: [*const i32; I32_LAZY_DOT_BATCH] =
            std::array::from_fn(|index| rhs[index].as_ptr().cast::<i32>());

        for count in 1..=I32_LAZY_DOT_BATCH {
            let initial = random_mont_array_i32::<D>(prime, 0x3000 + count as u64);
            let mut actual = initial;
            // SAFETY: guarded by runtime AVX2 detection; every pointer covers D values.
            unsafe {
                pointwise_dot_acc_i32(
                    actual.as_mut_ptr().cast::<i32>(),
                    lhs_pointers.as_ptr(),
                    rhs_pointers.as_ptr(),
                    count,
                    D,
                    prime.p,
                    prime.pinv,
                );
            }
            let mut expected = initial;
            for product in 0..count {
                scalar_pointwise_i32(&mut expected, &lhs[product], &rhs[product], prime);
            }
            assert_eq!(actual, expected, "prime={raw_prime}, count={count}");
        }
    }
}

#[test]
fn avx2_add_reduce_i32_matches_scalar_with_tail() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let prime = NttPrime::compute(1073707009_i32);
    const D: usize = 19;
    let acc_init = random_mont_array_i32::<D>(prime, 0x4444);
    let other = edge_mont_array_i32::<D>(prime);

    let mut avx_acc = acc_init;
    // SAFETY: guarded by the runtime AVX2 detection above.
    unsafe {
        add_reduce_i32(
            avx_acc.as_mut_ptr() as *mut i32,
            avx_acc.as_ptr() as *const i32,
            other.as_ptr() as *const i32,
            D,
            prime.p,
        );
    }

    let mut scalar_acc = acc_init;
    scalar_add_reduce_i32(&mut scalar_acc, &other, prime);
    assert_eq!(avx_acc, scalar_acc);
}

#[test]
fn avx512_pointwise_mul_acc_i32_matches_scalar_with_tail() {
    if !(std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512dq")
        && std::is_x86_feature_detected!("avx512bw"))
    {
        return;
    }
    let prime = NttPrime::compute(1073707009_i32);
    const D: usize = 29;
    let acc_init = random_mont_array_i32::<D>(prime, 0x5151);
    let lhs = edge_mont_array_i32::<D>(prime);
    let rhs = random_mont_array_i32::<D>(prime, 0x7171);

    let mut avx_acc = acc_init;
    // SAFETY: guarded by runtime AVX-512 feature detection above.
    unsafe {
        pointwise_mul_acc_i32_avx512(
            avx_acc.as_mut_ptr() as *mut i32,
            lhs.as_ptr() as *const i32,
            rhs.as_ptr() as *const i32,
            D,
            prime.p,
            prime.pinv,
        );
    }

    let mut scalar_acc = acc_init;
    scalar_pointwise_i32(&mut scalar_acc, &lhs, &rhs, prime);
    assert_eq!(avx_acc, scalar_acc);
}

#[test]
fn avx512_add_reduce_i32_matches_scalar_with_tail() {
    if !(std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512dq")
        && std::is_x86_feature_detected!("avx512bw"))
    {
        return;
    }
    let prime = NttPrime::compute(1073707009_i32);
    const D: usize = 29;
    let acc_init = random_mont_array_i32::<D>(prime, 0x8181);
    let other = edge_mont_array_i32::<D>(prime);

    let mut avx_acc = acc_init;
    // SAFETY: guarded by runtime AVX-512 feature detection above.
    unsafe {
        add_reduce_i32_avx512(
            avx_acc.as_mut_ptr() as *mut i32,
            avx_acc.as_ptr() as *const i32,
            other.as_ptr() as *const i32,
            D,
            prime.p,
        );
    }

    let mut scalar_acc = acc_init;
    scalar_add_reduce_i32(&mut scalar_acc, &other, prime);
    assert_eq!(avx_acc, scalar_acc);
}

#[test]
fn avx2_pointwise_mul_acc_i16_matches_scalar_with_tail() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let prime = NttPrime::compute(15361_i16);
    const D: usize = 23;
    let acc_init = random_mont_array_i16::<D>(prime, 0xaaaa);
    let lhs = edge_mont_array_i16::<D>(prime);
    let rhs = random_mont_array_i16::<D>(prime, 0xcccc);

    let mut avx_acc = acc_init;
    // SAFETY: guarded by the runtime AVX2 detection above.
    unsafe {
        pointwise_mul_acc_i16(
            avx_acc.as_mut_ptr() as *mut i16,
            lhs.as_ptr() as *const i16,
            rhs.as_ptr() as *const i16,
            D,
            prime.p,
            prime.pinv,
        );
    }

    let mut scalar_acc = acc_init;
    scalar_pointwise_i16(&mut scalar_acc, &lhs, &rhs, prime);
    assert_eq!(avx_acc, scalar_acc);
}

#[test]
fn avx2_add_reduce_i16_matches_scalar_with_tail() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let prime = NttPrime::compute(15361_i16);
    const D: usize = 23;
    let acc_init = random_mont_array_i16::<D>(prime, 0xdddd);
    let other = edge_mont_array_i16::<D>(prime);

    let mut avx_acc = acc_init;
    // SAFETY: guarded by the runtime AVX2 detection above.
    unsafe {
        add_reduce_i16(
            avx_acc.as_mut_ptr() as *mut i16,
            other.as_ptr() as *const i16,
            D,
            prime.p,
        );
    }

    let mut scalar_acc = acc_init;
    scalar_add_reduce_i16(&mut scalar_acc, &other, prime);
    assert_eq!(avx_acc, scalar_acc);
}
