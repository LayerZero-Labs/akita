//! Cyclotomic ring arithmetic, CRT+NTT transforms, and SIMD kernels.
//!
//! The oracle is `CyclotomicRing`'s schoolbook negacyclic `Mul` (and a cyclic
//! schoolbook product below). Exactness is claimed only when the public
//! `CrtCapacity` bound admits the accumulation, which is the same predicate
//! production uses to select a CRT profile. Scalar and SIMD kernels are both
//! reached: the runner starts some workers with `AKITA_SCALAR_NTT=1`.

use crate::gen::{self, Domain};
use crate::input::Reader;
use crate::stats;
use akita_algebra::ntt::tables::{q128_primes, I16_TAIL_PRIME, Q32_PRIMES, Q64_PRIMES};
use akita_algebra::{
    CenteredMontLut, CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing, DigitMontLut, I16TailParams,
};
use akita_config::proof_optimized::{fp128, fp32, fp64};
use jolt_field::{CanonicalEncoding, Field};
use std::sync::OnceLock;

pub fn run(data: &[u8]) {
    let mut reader = Reader::new(data);
    let field = reader.u8() % 3;
    let dimension = reader.u8();
    macro_rules! dispatch {
        ($field:ty, $primes:expr, $k:literal, [$($d:literal),+]) => {{
            const DIMENSIONS: &[usize] = &[$($d),+];
            match DIMENSIONS[usize::from(dimension) % DIMENSIONS.len()] {
                $($d => {
                    static PARAMS: OnceLock<(CrtNttParamSet<i32, $k, $d>, I16TailParams<$k, $d>)> =
                        OnceLock::new();
                    let (params, tail) = PARAMS.get_or_init(|| {
                        let wide = CrtNttParamSet::<i32, $k, $d>::new($primes);
                        let tail = I16TailParams::new(
                            wide.clone(),
                            CrtNttParamSet::<i16, 1, $d>::new([I16_TAIL_PRIME]),
                        );
                        (wide, tail)
                    });
                    case::<$field, $k, $d>(&mut reader, params, tail)
                })+
                _ => unreachable!(),
            }
        }};
    }
    match field {
        0 => dispatch!(fp32::Field, Q32_PRIMES, 2, [64, 128, 256, 512, 1024, 2048]),
        1 => dispatch!(fp64::Field, Q64_PRIMES, 3, [64, 128, 256, 512, 1024, 2048]),
        _ => dispatch!(fp128::Field, q128_primes(), 6, [64, 128, 256, 512, 1024]),
    }
}

fn ring<F: Field + CanonicalEncoding, const D: usize>(
    reader: &mut Reader<'_>,
    domain: Domain,
) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_slice(&gen::table::<F>(reader, D, domain))
}

fn lift<F: Field, const D: usize>(values: &[i64; D]) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|i| F::from_i64(values[i])))
}

/// Product in `F[X]/(X^D - 1)`.
fn cyclic_schoolbook<F: Field, const D: usize>(
    a: &CyclotomicRing<F, D>,
    b: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let mut out = [F::zero(); D];
    for (i, &ai) in a.coefficients().iter().enumerate() {
        for (j, &bj) in b.coefficients().iter().enumerate() {
            out[(i + j) % D] += ai * bj;
        }
    }
    CyclotomicRing::from_coefficients(out)
}

/// Balanced digits in `[-bound, bound)` with a power-of-two bound.
fn digits<const D: usize>(reader: &mut Reader<'_>, bound: i64) -> [i8; D] {
    let pattern = reader.u8();
    let seed = reader.u64();
    let mut rng = crate::input::SplitMix64::new(seed);
    std::array::from_fn(|i| {
        let raw = match pattern % 4 {
            0 => -bound,
            1 => bound - 1,
            2 => {
                if i % 2 == 0 {
                    -bound
                } else {
                    bound - 1
                }
            }
            _ => (rng.next_u64() % (2 * bound as u64)) as i64 - bound,
        };
        raw as i8
    })
}

fn centered_values<const D: usize>(reader: &mut Reader<'_>, max_abs: i64) -> [i64; D] {
    let pattern = reader.u8();
    let seed = reader.u64();
    let mut rng = crate::input::SplitMix64::new(seed);
    std::array::from_fn(|i| match pattern % 4 {
        0 => max_abs,
        1 => -max_abs,
        2 => {
            if i % 2 == 0 {
                max_abs
            } else {
                -max_abs
            }
        }
        _ => (rng.next_u64() % (2 * max_abs as u64 + 1)) as i64 - max_abs,
    })
}

fn case<F, const K: usize, const D: usize>(
    reader: &mut Reader<'_>,
    params: &CrtNttParamSet<i32, K, D>,
    tail: &I16TailParams<K, D>,
) where
    F: Field + CanonicalEncoding,
{
    let capacity = params.crt_capacity();
    match reader.u8() % 7 {
        0 => {
            // Negacyclic sum of `width` products of full ring elements by digits.
            let bound = 1i64 << (1 + reader.u8() % 7);
            let width = 1 + usize::from(reader.u8() % 8);
            if !capacity.supports::<F, D>(width, bound as u64) {
                stats::count("ring_outside_capacity");
                return;
            }
            let lut = DigitMontLut::new_with_digit_bound(params, bound as u64);
            let mut accumulator = CyclotomicCrtNtt::<i32, K, D>::zero();
            let mut expected = CyclotomicRing::<F, D>::zero();
            for _ in 0..width {
                let a = ring::<F, D>(reader, Domain::Full);
                let b = digits::<D>(reader, bound);
                let b_ntt = CyclotomicCrtNtt::from_i8_with_params(&b, params);
                assert_eq!(
                    CyclotomicCrtNtt::from_i8_with_lut(&b, params, &lut),
                    b_ntt,
                    "digit LUT transform differs from direct transform"
                );
                accumulator.add_assign_pointwise_mul(
                    &CyclotomicCrtNtt::from_ring(&a, params),
                    &b_ntt,
                    params,
                );
                expected += a * lift::<F, D>(&b.map(i64::from));
            }
            assert_eq!(
                accumulator.to_ring::<F>(params),
                expected,
                "negacyclic CRT-NTT product"
            );
            stats::count("ring_negacyclic");
        }
        1 => {
            let bound = 1i64 << (1 + reader.u8() % 7);
            if !capacity.supports::<F, D>(1, bound as u64) {
                return;
            }
            let lut = DigitMontLut::new_with_digit_bound(params, bound as u64);
            let a = ring::<F, D>(reader, Domain::Full);
            let b = digits::<D>(reader, bound);
            let (negacyclic, cyclic) = CyclotomicCrtNtt::from_ring_pair_with_params(&a, params);
            assert_eq!(
                negacyclic,
                CyclotomicCrtNtt::from_ring(&a, params),
                "pair negacyclic half"
            );
            assert_eq!(
                cyclic,
                CyclotomicCrtNtt::from_ring_cyclic(&a, params),
                "pair cyclic half"
            );
            let mut product = CyclotomicCrtNtt::<i32, K, D>::zero();
            product.add_assign_pointwise_mul(
                &cyclic,
                &CyclotomicCrtNtt::from_i8_cyclic_with_lut(&b, params, &lut),
                params,
            );
            assert_eq!(
                product.to_ring_cyclic::<F>(params),
                cyclic_schoolbook(&a, &lift::<F, D>(&b.map(i64::from))),
                "cyclic CRT-NTT product"
            );
            stats::count("ring_cyclic");
        }
        2 => {
            // Centered i32 inputs through the LUT, including LUT misses.
            let max_abs = 1 + i64::from(reader.u32() % (1 << 20));
            let lut_abs = i32::try_from(i64::from(reader.u32()) % (max_abs + 1)).expect("fits i32");
            let values = centered_values::<D>(reader, max_abs);
            let coefficients: [i32; D] = values.map(|value| value as i32);
            let lut = CenteredMontLut::new(params, lut_abs);
            let expected = CyclotomicCrtNtt::from_ring(&lift::<F, D>(&values), params);
            assert_eq!(
                CyclotomicCrtNtt::from_centered_i32_with_lut(&coefficients, params, &lut),
                expected,
                "centered i32 LUT transform"
            );
            let (negacyclic, cyclic) =
                CyclotomicCrtNtt::from_centered_i32_pair_with_params(&coefficients, params);
            assert_eq!(negacyclic, expected, "centered i32 pair negacyclic half");
            assert_eq!(
                cyclic,
                CyclotomicCrtNtt::from_ring_cyclic(&lift::<F, D>(&values), params),
                "centered i32 pair cyclic half"
            );
            stats::count("ring_centered_i32");
        }
        3 | 4 => {
            let with_tail = reader.u8().is_multiple_of(2);
            let rows = 1 + usize::from(reader.u8() % 3);
            let cols = 1 + usize::from(reader.u8() % 6);
            let bound = 1 + i64::from(reader.u16() % i16::MAX as u16);
            let admitted = if with_tail {
                capacity
                    .clone()
                    .with_prime_modulus(I16_TAIL_PRIME.p as u128)
                    .supports::<F, D>(cols, bound as u64)
            } else {
                capacity.supports::<F, D>(cols, bound as u64)
            };
            if !admitted {
                stats::count("ring_outside_capacity");
                return;
            }
            let matrix: Vec<CyclotomicRing<F, D>> = (0..rows * cols)
                .map(|_| ring::<F, D>(reader, Domain::Full))
                .collect();
            let rhs: Vec<[i16; D]> = (0..cols)
                .map(|_| centered_values::<D>(reader, bound).map(|value| value as i16))
                .collect();
            let expected: Vec<CyclotomicRing<F, D>> = (0..rows)
                .map(|row| {
                    (0..cols).fold(CyclotomicRing::zero(), |acc, col| {
                        acc + matrix[row * cols + col] * lift::<F, D>(&rhs[col].map(i64::from))
                    })
                })
                .collect();
            let wide: Vec<_> = matrix
                .iter()
                .map(|entry| CyclotomicCrtNtt::from_ring(entry, params))
                .collect();
            let actual = if with_tail {
                let narrow: Vec<_> = matrix
                    .iter()
                    .map(|entry| CyclotomicCrtNtt::<i16, 1, D>::from_ring(entry, &tail.tail))
                    .collect();
                akita_algebra::mat_vec_i16_with_tail::<F, K, D>(
                    &wide, &narrow, rows, cols, &rhs, tail,
                )
            } else {
                CyclotomicCrtNtt::mat_vec_i16::<F>(&wide, rows, cols, &rhs, params)
            }
            .expect("admitted mat-vec shape");
            assert_eq!(actual, expected, "i16 mat-vec (tail={with_tail})");
            stats::count(if with_tail {
                "ring_matvec_tail"
            } else {
                "ring_matvec"
            });
        }
        5 => {
            let a = ring::<F, D>(reader, Domain::Full);
            assert_eq!(
                CyclotomicCrtNtt::from_ring(&a, params).to_ring::<F>(params),
                a,
                "negacyclic round trip"
            );
            assert_eq!(
                CyclotomicCrtNtt::from_ring_cyclic(&a, params).to_ring_cyclic::<F>(params),
                a,
                "cyclic round trip"
            );
            stats::count("ring_round_trip");
        }
        _ => ring_identities::<F, D>(reader),
    }
}

fn ring_identities<F: Field + CanonicalEncoding, const D: usize>(reader: &mut Reader<'_>) {
    let a = ring::<F, D>(reader, Domain::Full);
    let b = ring::<F, D>(reader, Domain::Full);
    let product = a * b;
    let mut accumulated = CyclotomicRing::<F, D>::zero();
    a.mul_accumulate_into(&b, &mut accumulated);
    assert_eq!(accumulated, product, "mul_accumulate_into");
    let sparse = ring::<F, D>(reader, Domain::symmetric(2));
    let mut sparse_accumulated = CyclotomicRing::<F, D>::zero();
    a.mul_accumulate_sparse_rhs_into(&sparse, &mut sparse_accumulated);
    assert_eq!(
        sparse_accumulated,
        a * sparse,
        "mul_accumulate_sparse_rhs_into"
    );
    let shift = usize::from(reader.u16()) % (2 * D);
    let mut monomial = CyclotomicRing::<F, D>::zero();
    let sign = if shift >= D { -F::one() } else { F::one() };
    monomial.coefficients_mut()[shift % D] = sign;
    assert_eq!(
        a.negacyclic_shift(shift),
        a * monomial,
        "negacyclic shift is multiplication by X^k"
    );
    let k = 2 * (usize::from(reader.u16()) % D) + 1;
    assert_eq!(
        (a * b).sigma(k),
        a.sigma(k) * b.sigma(k),
        "sigma_k is a ring automorphism"
    );
    let scalar: F = gen::scalar(reader, Domain::Full);
    let mut scalar_ring = CyclotomicRing::<F, D>::zero();
    scalar_ring.coefficients_mut()[0] = scalar;
    assert_eq!(a.scale(&scalar), a * scalar_ring, "scale");
    stats::count("ring_identities");
}
