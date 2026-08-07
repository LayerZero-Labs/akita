use super::*;
use crate::ntt::butterfly::NttTwiddles;
use crate::ntt::prime::{MontCoeff, NttPrime, I16_VNNI_DOT_BATCH, I32_LAZY_DOT_BATCH};
use crate::ntt::tables::Q128_RAW_PRIMES;

const AVX2_ONLY: AvxCpuFeatures = AvxCpuFeatures {
    avx2: true,
    avx512f: false,
    avx512dq: false,
    avx512bw: false,
    avx512vnni: false,
};

const AVX512_CAPABLE: AvxCpuFeatures = AvxCpuFeatures {
    avx2: true,
    avx512f: true,
    avx512dq: true,
    avx512bw: true,
    avx512vnni: true,
};

const AVX512_WITHOUT_AVX2: AvxCpuFeatures = AvxCpuFeatures {
    avx2: false,
    avx512f: true,
    avx512dq: true,
    avx512bw: true,
    avx512vnni: true,
};

#[test]
fn avx_mode_defaults_to_avx2_when_supported() {
    assert_eq!(
        select_avx_ntt_mode(None, None, None, AVX2_ONLY),
        Some(AvxNttMode::Avx2)
    );
}

#[test]
fn avx512_is_default_pointwise_mode_when_available() {
    assert_eq!(
        select_avx_ntt_mode(None, None, None, AVX512_CAPABLE),
        Some(AvxNttMode::Avx512)
    );
    assert_eq!(
        select_avx_ntt_mode(None, None, Some("1"), AVX512_CAPABLE),
        Some(AvxNttMode::Avx512)
    );
}

#[test]
fn avx512_mode_rejects_a_feature_mask_without_avx2() {
    assert_eq!(
        select_avx_ntt_mode(None, None, None, AVX512_WITHOUT_AVX2),
        None
    );
    assert_eq!(
        select_avx_ntt_mode(None, None, Some("1"), AVX512_WITHOUT_AVX2),
        None
    );
}

#[test]
fn avx2_transform_is_default_and_avx512_is_explicit() {
    assert!(!select_avx512_transform_ntt(None, Some(AvxNttMode::Avx512)));
    assert!(select_avx512_transform_ntt(
        Some("1"),
        Some(AvxNttMode::Avx512)
    ));
    assert!(!select_avx512_transform_ntt(
        Some("1"),
        Some(AvxNttMode::Avx2)
    ));
}

#[test]
fn avx512_can_be_opted_out_to_avx2() {
    assert_eq!(
        select_avx_ntt_mode(None, None, Some("0"), AVX512_CAPABLE),
        Some(AvxNttMode::Avx2)
    );
}

#[test]
fn scalar_kill_switch_precedes_avx_flags() {
    assert_eq!(
        select_avx_ntt_mode(Some("1"), None, Some("1"), AVX512_CAPABLE),
        None
    );
}

#[test]
fn avx_kill_switch_disables_x86_ntt_simd() {
    assert_eq!(
        select_avx_ntt_mode(None, Some("0"), Some("1"), AVX512_CAPABLE),
        None
    );
}

#[test]
fn avx512_opt_in_falls_back_to_avx2_without_full_features() {
    let missing_bw = AvxCpuFeatures {
        avx512bw: false,
        ..AVX512_CAPABLE
    };
    assert_eq!(
        select_avx_ntt_mode(None, None, Some("1"), missing_bw),
        Some(AvxNttMode::Avx2)
    );
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

fn scalar_sub_reduce_i32<const D: usize>(
    acc: &mut [MontCoeff<i32>; D],
    other: &[MontCoeff<i32>; D],
    prime: NttPrime<i32>,
) {
    for i in 0..D {
        let diff = MontCoeff::from_raw(acc[i].raw().wrapping_sub(other[i].raw()));
        acc[i] = prime.reduce_range(diff);
    }
}

fn scalar_neg_reduce_i32<const D: usize>(acc: &mut [MontCoeff<i32>; D], prime: NttPrime<i32>) {
    for value in acc {
        *value = prime.reduce_range(MontCoeff::from_raw(value.raw().wrapping_neg()));
    }
}

fn scalar_mul_i32<const D: usize>(
    out: &mut [MontCoeff<i32>; D],
    lhs: &[MontCoeff<i32>; D],
    rhs: &[MontCoeff<i32>; D],
    prime: NttPrime<i32>,
) {
    for i in 0..D {
        out[i] = prime.reduce_range(prime.mul(lhs[i], rhs[i]));
    }
}

fn assert_i32_crt_ops<const D: usize>() {
    for (prime_index, raw_prime) in Q128_RAW_PRIMES.into_iter().enumerate() {
        let prime = NttPrime::compute(raw_prime);
        let lhs = edge_mont_array_i32::<D>(prime);
        let rhs = random_mont_array_i32::<D>(prime, 0x9173 + prime_index as u64);

        let mut expected_add = lhs;
        scalar_add_reduce_i32(&mut expected_add, &rhs, prime);
        let mut expected_sub = lhs;
        scalar_sub_reduce_i32(&mut expected_sub, &rhs, prime);
        let mut expected_neg = lhs;
        scalar_neg_reduce_i32(&mut expected_neg, prime);
        let mut expected_mul = [MontCoeff::from_raw(0); D];
        scalar_mul_i32(&mut expected_mul, &lhs, &rhs, prime);

        if std::is_x86_feature_detected!("avx2") {
            let mut add = lhs;
            let mut sub = lhs;
            let mut neg = lhs;
            let mut mul = [MontCoeff::from_raw(0); D];
            unsafe {
                add_reduce_i32(
                    add.as_mut_ptr().cast(),
                    add.as_ptr().cast(),
                    rhs.as_ptr().cast(),
                    D,
                    prime.p,
                );
                sub_reduce_i32(
                    sub.as_mut_ptr().cast(),
                    sub.as_ptr().cast(),
                    rhs.as_ptr().cast(),
                    D,
                    prime.p,
                );
                neg_reduce_i32(neg.as_mut_ptr().cast(), neg.as_ptr().cast(), D, prime.p);
                pointwise_mul_i32(
                    mul.as_mut_ptr().cast(),
                    lhs.as_ptr().cast(),
                    rhs.as_ptr().cast(),
                    D,
                    prime.p,
                    prime.pinv,
                );
            }
            assert_eq!(add, expected_add, "AVX2 add p={raw_prime} D={D}");
            assert_eq!(sub, expected_sub, "AVX2 sub p={raw_prime} D={D}");
            assert_eq!(neg, expected_neg, "AVX2 neg p={raw_prime} D={D}");
            assert_eq!(mul, expected_mul, "AVX2 mul p={raw_prime} D={D}");
        }

        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512bw")
        {
            let mut add = lhs;
            let mut sub = lhs;
            let mut neg = lhs;
            let mut mul = [MontCoeff::from_raw(0); D];
            unsafe {
                add_reduce_i32_avx512(
                    add.as_mut_ptr().cast(),
                    add.as_ptr().cast(),
                    rhs.as_ptr().cast(),
                    D,
                    prime.p,
                );
                sub_reduce_i32_avx512(
                    sub.as_mut_ptr().cast(),
                    sub.as_ptr().cast(),
                    rhs.as_ptr().cast(),
                    D,
                    prime.p,
                );
                neg_reduce_i32_avx512(neg.as_mut_ptr().cast(), neg.as_ptr().cast(), D, prime.p);
                pointwise_mul_i32_avx512(
                    mul.as_mut_ptr().cast(),
                    lhs.as_ptr().cast(),
                    rhs.as_ptr().cast(),
                    D,
                    prime.p,
                    prime.pinv,
                );
            }
            assert_eq!(add, expected_add, "AVX-512 add p={raw_prime} D={D}");
            assert_eq!(sub, expected_sub, "AVX-512 sub p={raw_prime} D={D}");
            assert_eq!(neg, expected_neg, "AVX-512 neg p={raw_prime} D={D}");
            assert_eq!(mul, expected_mul, "AVX-512 mul p={raw_prime} D={D}");
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

fn assert_avx2_ntt_i32_transforms_match_scalar<const D: usize>() {
    let prime = NttPrime::compute(1073707009_i32);
    let tw = NttTwiddles::<i32, D>::compute(prime);
    let input = random_mont_array_i32::<D>(prime, 0x5150 ^ D as u64);

    let mut avx_fwd = input;
    let mut scalar_fwd = input;
    // SAFETY: guarded by runtime AVX2 detection above.
    unsafe { forward_ntt_i32(&mut avx_fwd, prime, &tw, false) };
    scalar_forward_ntt_i32(&mut scalar_fwd, prime, &tw);
    assert_eq!(avx_fwd, scalar_fwd);

    let mut avx_inv = avx_fwd;
    let mut scalar_inv = scalar_fwd;
    // SAFETY: guarded by runtime AVX2 detection above.
    unsafe { inverse_ntt_i32(&mut avx_inv, prime, &tw, false) };
    scalar_inverse_ntt_i32(&mut scalar_inv, prime, &tw);
    assert_eq!(avx_inv, scalar_inv);

    let mut avx_cyclic = input;
    let mut scalar_cyclic = input;
    // SAFETY: guarded by runtime AVX2 detection above.
    unsafe { forward_ntt_cyclic_i32(&mut avx_cyclic, prime, &tw, false) };
    scalar_forward_ntt_cyclic_i32(&mut scalar_cyclic, prime, &tw);
    assert_eq!(avx_cyclic, scalar_cyclic);

    // SAFETY: guarded by runtime AVX2 detection above.
    unsafe { inverse_ntt_cyclic_i32(&mut avx_cyclic, prime, &tw, false) };
    scalar_inverse_ntt_cyclic_i32(&mut scalar_cyclic, prime, &tw);
    assert_eq!(avx_cyclic, scalar_cyclic);
}

#[test]
fn avx2_ntt_i32_transforms_match_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    assert_avx2_ntt_i32_transforms_match_scalar::<64>();
    assert_avx2_ntt_i32_transforms_match_scalar::<128>();
    assert_avx2_ntt_i32_transforms_match_scalar::<256>();
    assert_avx2_ntt_i32_transforms_match_scalar::<512>();
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
    if !(std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("avx512f")
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
fn avx512vnni_six_way_i16_dot_matches_scalar_with_tail() {
    let available = std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512dq")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vnni");
    if !available {
        assert_ne!(
            std::env::var("AKITA_REQUIRE_AVX512VNNI").ok().as_deref(),
            Some("1"),
            "required AVX-512VNNI test backend is unavailable"
        );
        return;
    }
    let prime = NttPrime::compute(15361_i16);
    const D: usize = 47;
    let acc_init = random_mont_array_i16::<D>(prime, 0xd0d0);
    let lhs = std::array::from_fn::<_, I16_VNNI_DOT_BATCH, _>(|column| {
        if column == 0 {
            edge_mont_array_i16::<D>(prime)
        } else {
            random_mont_array_i16::<D>(prime, 0x1100 + column as u64)
        }
    });
    let rhs = std::array::from_fn::<_, I16_VNNI_DOT_BATCH, _>(|column| {
        random_mont_array_i16::<D>(prime, 0x2200 + column as u64)
    });
    let lhs_ptrs = std::array::from_fn::<_, I16_VNNI_DOT_BATCH, _>(|column| {
        lhs[column].as_ptr().cast::<i16>()
    });
    let rhs_ptrs = std::array::from_fn::<_, I16_VNNI_DOT_BATCH, _>(|column| {
        rhs[column].as_ptr().cast::<i16>()
    });

    let mut avx_acc = acc_init;
    // SAFETY: guarded by runtime AVX2 and AVX-512F/DQ/BW/VNNI detection above.
    unsafe {
        pointwise_dot_acc_6_i16_avx512vnni(
            avx_acc.as_mut_ptr().cast::<i16>(),
            lhs_ptrs.as_ptr(),
            rhs_ptrs.as_ptr(),
            D,
            prime.p,
            prime.pinv,
        );
    }

    let mut scalar_acc = acc_init;
    for column in 0..I16_VNNI_DOT_BATCH {
        scalar_pointwise_i16(&mut scalar_acc, &lhs[column], &rhs[column], prime);
    }
    for (index, (actual, expected)) in avx_acc.into_iter().zip(scalar_acc).enumerate() {
        assert_eq!(
            prime.normalize(actual),
            prime.normalize(expected),
            "coefficient {index}"
        );
    }
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
