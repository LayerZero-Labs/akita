use super::*;
use crate::ntt::butterfly::NttTwiddles;
use crate::ntt::prime::{MontCoeff, NttPrime};

const AVX2_ONLY: AvxCpuFeatures = AvxCpuFeatures {
    avx2: true,
    avx512f: false,
    avx512dq: false,
    avx512bw: false,
};

const AVX512_CAPABLE: AvxCpuFeatures = AvxCpuFeatures {
    avx2: true,
    avx512f: true,
    avx512dq: true,
    avx512bw: true,
};

#[test]
fn avx_mode_defaults_to_avx2_when_supported() {
    assert_eq!(
        select_avx_ntt_mode(None, None, None, AVX2_ONLY),
        Some(AvxNttMode::Avx2)
    );
}

#[test]
fn avx512_is_default_when_available() {
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

#[test]
fn avx2_ntt_i32_transforms_match_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let prime = NttPrime::compute(1073707009_i32);
    let tw = NttTwiddles::<i32, 64>::compute(prime);
    let input = random_mont_array_i32::<64>(prime, 0x5150);

    let mut avx_fwd = input;
    let mut scalar_fwd = input;
    // SAFETY: guarded by runtime AVX2 detection above.
    unsafe { forward_ntt_i32(&mut avx_fwd, prime, &tw) };
    scalar_forward_ntt_i32(&mut scalar_fwd, prime, &tw);
    assert_eq!(avx_fwd, scalar_fwd);

    let mut avx_inv = avx_fwd;
    let mut scalar_inv = scalar_fwd;
    // SAFETY: guarded by runtime AVX2 detection above.
    unsafe { inverse_ntt_i32(&mut avx_inv, prime, &tw) };
    scalar_inverse_ntt_i32(&mut scalar_inv, prime, &tw);
    assert_eq!(avx_inv, scalar_inv);

    let mut avx_cyclic = input;
    let mut scalar_cyclic = input;
    // SAFETY: guarded by runtime AVX2 detection above.
    unsafe { forward_ntt_cyclic_i32(&mut avx_cyclic, prime, &tw) };
    scalar_forward_ntt_cyclic_i32(&mut scalar_cyclic, prime, &tw);
    assert_eq!(avx_cyclic, scalar_cyclic);

    // SAFETY: guarded by runtime AVX2 detection above.
    unsafe { inverse_ntt_cyclic_i32(&mut avx_cyclic, prime, &tw) };
    scalar_inverse_ntt_cyclic_i32(&mut scalar_cyclic, prime, &tw);
    assert_eq!(avx_cyclic, scalar_cyclic);
}

#[test]
fn avx2_ntt_i16_transforms_match_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let prime = NttPrime::compute(12289_i16);
    let tw = NttTwiddles::<i16, 64>::compute(prime);
    let input = random_mont_array_i16::<64>(prime, 0x1616);

    let mut avx = input;
    let mut scalar = input;
    unsafe { forward_ntt_i16(&mut avx, prime, &tw) };
    scalar_forward_ntt_i16(&mut scalar, prime, &tw);
    assert_eq!(avx, scalar);

    unsafe { inverse_ntt_i16(&mut avx, prime, &tw) };
    scalar_inverse_ntt_i16(&mut scalar, prime, &tw);
    assert_eq!(avx, scalar);
    assert_eq!(avx, input);
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
