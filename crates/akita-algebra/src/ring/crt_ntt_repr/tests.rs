use super::lut::{balanced_limbs, CenteredMontReducer};
use super::*;
use crate::ntt::prime::NttPrime;
use crate::ntt::tables::{
    q128_primes, I16_TAIL_PRIME, Q128_NUM_PRIMES, Q128_RAW_PRIMES, Q32_PRIMES, Q64_PRIMES,
};

const SYNTHETIC_I16_NUM_PRIMES: usize = 3;

fn synthetic_i16_primes() -> [NttPrime<i16>; SYNTHETIC_I16_NUM_PRIMES] {
    [
        NttPrime::compute(15361_i16),
        NttPrime::compute(13313_i16),
        NttPrime::compute(12289_i16),
    ]
}

fn reducer_cases() -> Vec<i128> {
    let mut cases = vec![0, 1, -1, i128::MAX, i128::MIN + 1, i128::MIN];
    for shift in [15, 16, 31, 32, 33, 63, 64, 95, 96, 97, 126] {
        for offset in [-1i128, 0, 1] {
            cases.push((1i128 << shift) + offset);
            cases.push(-(1i128 << shift) + offset);
        }
    }
    let mut state = 0x9e37_79b9_7f4a_7c15_u128;
    for _ in 0..512 {
        state = state
            .wrapping_mul(0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645)
            .wrapping_add(0x5851_f42d_4c95_7f2d_1405_7b7e_f767_814f);
        cases.push((state as i128) >> (state % 97));
    }
    cases
}

fn check_reducer<W: PrimeWidth>(prime: NttPrime<W>) {
    let p = prime.p.to_i64();
    let reducer = CenteredMontReducer::new(prime);
    for value in reducer_cases() {
        let limbs = balanced_limbs(value);
        let rebuilt = limbs.iter().rev().fold(0i128, |acc, &limb| {
            (acc << 32).wrapping_add(i128::from(limb))
        });
        assert_eq!(rebuilt, value);
        assert!(limbs[..3].iter().all(|&limb| i32::try_from(limb).is_ok()));
        assert!(limbs[3].abs() <= 1 << 31);

        let mont = reducer.reduce_limbs(limbs);
        assert!(mont.raw().to_i64().abs() < p);
        let expected = value.rem_euclid(i128::from(p)) as i64;
        assert_eq!(prime.to_canonical(mont).to_i64(), expected, "{value}");

        let narrow = value as i32;
        let mont = reducer.reduce_i32(narrow);
        assert!(mont.raw().to_i64().abs() < p);
        let expected = i64::from(narrow).rem_euclid(p);
        assert_eq!(prime.to_canonical(mont).to_i64(), expected, "{narrow}");
    }
}

#[test]
fn centered_mont_reducer_matches_euclidean_residues() {
    for prime in synthetic_i16_primes() {
        check_reducer(prime);
    }
    check_reducer(I16_TAIL_PRIME);
    for prime in Q32_PRIMES.into_iter().chain(Q64_PRIMES) {
        check_reducer(prime);
    }
    for p in Q128_RAW_PRIMES.into_iter().chain([1_073_707_009]) {
        check_reducer(NttPrime::compute(p));
    }
}

#[test]
fn centered_mont_lut_matches_centered_residue_boundary() {
    const D: usize = 64;
    let primes = synthetic_i16_primes();
    let params = CrtNttParamSet::<i16, SYNTHETIC_I16_NUM_PRIMES, D>::new(primes);
    let prime = params.primes[0];
    let half = i32::from(prime.p) / 2;
    let lut = CenteredMontLut::<i16, SYNTHETIC_I16_NUM_PRIMES>::new(&params, half + 1);

    let canonical = |value| lut.get(0, value).map(|mont| prime.to_canonical(mont));
    assert_eq!(canonical(half), Some(half as i16));
    assert_eq!(canonical(half + 1), Some((half + 1) as i16));
    assert_eq!(
        canonical(-half - 1),
        Some((i32::from(prime.p) - half - 1) as i16)
    );
    assert_eq!(lut.get(0, half + 2), None);
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

/// Only i32 parameters carry the NEON Barrett tables.
#[cfg(target_arch = "aarch64")]
#[test]
fn i16_twiddles_hold_only_montgomery_tables() {
    const D: usize = 1024;
    assert_eq!(size_of::<<i16 as PrimeWidth>::NeonTables<D>>(), 0);
    // Eight D-entry tables, then D^-1 and the stage count (16 bytes with
    // padding), rounded up to the 64-byte struct alignment.
    let tables = 8 * D * size_of::<i16>() + 2 * size_of::<usize>();
    assert_eq!(
        size_of::<crate::ntt::butterfly::NttTwiddles<i16, D>>(),
        tables.next_multiple_of(64)
    );
}

/// Embedding applications and rayon workers may run on 2 MiB stacks, which
/// the workspace `RUST_MIN_STACK` does not raise for an explicit size.
#[test]
fn q128_parameters_build_on_a_2_mib_stack() {
    std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(|| CrtNttParamSet::<i32, Q128_NUM_PRIMES, 1024>::new(q128_primes()))
        .expect("spawn")
        .join()
        .expect("build Q128 parameters");
}
