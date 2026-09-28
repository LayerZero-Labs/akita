//! Device NTT matvec against the CPU's prepared-cache matvecs and against
//! schoolbook ring arithmetic.

#![cfg(target_os = "macos")]

mod support;

use akita_algebra::{CanonicalEncoding, CyclotomicRing, Field, Zero};
use akita_cpu_backend::benchmark_support::mat_vec_mul_ntt_digits_i8;
use akita_metal::decompose::decompose;
use akita_metal::matvec::DeviceNttMatrix;
use akita_metal::{AkitaMetal, AkitaMetalError, DeviceDigitPlanes};
use akita_types::{prepare_ntt_cache, FlatMatrix, NttCacheMode, PreparedNttCache};
use jolt_field::{Prime128OffsetA7F7, Prime64Offset59};
use jolt_metal::runtime::DeviceBuffer;
use jolt_metal::MetalField;
use support::{gpu, SplitMix64};

/// A random matrix whose first entry has every coefficient at `-1`, an
/// adversarial canonical value near the modulus.
fn matrix<F: Field + CanonicalEncoding, const D: usize>(
    rows: usize,
    cols: usize,
    rng: &mut SplitMix64,
) -> Vec<CyclotomicRing<F, D>> {
    (0..rows * cols)
        .map(|entry| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|_| {
                if entry == 0 {
                    -F::one()
                } else {
                    F::from_u128_reduced(
                        u128::from(rng.next_u64()) << 64 | u128::from(rng.next_u64()),
                    )
                }
            }))
        })
        .collect()
}

fn prepare<F: Field + CanonicalEncoding, const D: usize>(
    matrix: &[CyclotomicRing<F, D>],
    rows: usize,
    cols: usize,
) -> PreparedNttCache<D> {
    prepare_with(matrix, rows, cols, NttCacheMode::Negacyclic)
}

fn prepare_with<F: Field + CanonicalEncoding, const D: usize>(
    matrix: &[CyclotomicRing<F, D>],
    rows: usize,
    cols: usize,
    mode: NttCacheMode,
) -> PreparedNttCache<D> {
    let flat = FlatMatrix::from_ring_slice(matrix);
    prepare_ntt_cache(flat.ring_view::<D>(rows, cols).expect("view"), mode).expect("prepare")
}

/// Balanced digits in `[-2^(b-1), 2^(b-1))`; the first plane of each block
/// holds the extremes.
fn planes<const D: usize>(
    blocks: usize,
    cols: usize,
    log_basis: u32,
    rng: &mut SplitMix64,
) -> Vec<[i16; D]> {
    let half = 1i32 << (log_basis - 1);
    (0..blocks * cols)
        .map(|plane| {
            std::array::from_fn(|coefficient| {
                let digit = if plane % cols == 0 {
                    if coefficient % 2 == 0 {
                        -half
                    } else {
                        half - 1
                    }
                } else {
                    (rng.next_u64() % (2 * half as u64)) as i32 - half
                };
                digit as i16
            })
        })
        .collect()
}

/// `sum_c A[r][c] x[b][c]` by schoolbook ring arithmetic.
fn schoolbook<F: Field + CanonicalEncoding, const D: usize>(
    matrix: &[CyclotomicRing<F, D>],
    rows: usize,
    cols: usize,
    planes: &[[i16; D]],
) -> Vec<F> {
    planes
        .chunks_exact(cols)
        .flat_map(|block| {
            (0..rows).flat_map(move |row| {
                let mut sum = CyclotomicRing::<F, D>::zero();
                for (col, plane) in block.iter().enumerate() {
                    let x = CyclotomicRing::from_coefficients(
                        plane.map(|digit| F::from_i64(i64::from(digit))),
                    );
                    sum += matrix[row * cols + col] * x;
                }
                *sum.coefficients()
            })
        })
        .collect()
}

fn device_i8<F: MetalField + CanonicalEncoding, const K: usize, const D: usize>(
    metal: &AkitaMetal,
    device: &DeviceNttMatrix<F, K, D>,
    planes: &[[i16; D]],
    log_basis: u32,
) -> Vec<F> {
    let digits = planes
        .iter()
        .flatten()
        .map(|&digit| digit as i8)
        .collect::<Vec<_>>();
    let planes = DeviceDigitPlanes::from_slice(metal, &digits, log_basis).expect("upload");
    let blocks = planes.len() / (device.cols() * D);
    let mut out =
        DeviceBuffer::<F>::zeroed(metal.device(), blocks * device.rows() * D).expect("output");
    device.mat_vec(metal, &planes, &mut out).expect("matvec");
    out.read().expect("canonical read-back").to_vec()
}

fn device_i16<F: MetalField + CanonicalEncoding, const K: usize, const D: usize>(
    metal: &AkitaMetal,
    device: &DeviceNttMatrix<F, K, D>,
    planes: &[[i16; D]],
    log_basis: u32,
) -> Vec<F> {
    let digits = planes.iter().flatten().copied().collect::<Vec<_>>();
    let planes = DeviceDigitPlanes::from_slice(metal, &digits, log_basis).expect("upload");
    let blocks = planes.len() / (device.cols() * D);
    let mut out =
        DeviceBuffer::<F>::zeroed(metal.device(), blocks * device.rows() * D).expect("output");
    device.mat_vec(metal, &planes, &mut out).expect("matvec");
    out.read().expect("canonical read-back").to_vec()
}

fn cpu_i8<F: Field + CanonicalEncoding, const D: usize>(
    cache: &PreparedNttCache<D>,
    rows: usize,
    cols: usize,
    planes: &[[i16; D]],
    log_basis: u32,
) -> Vec<F> {
    let digits = planes
        .iter()
        .map(|plane| plane.map(|digit| digit as i8))
        .collect::<Vec<_>>();
    let blocks = digits.chunks_exact(cols).collect::<Vec<_>>();
    mat_vec_mul_ntt_digits_i8::<F, D>(cache, rows, cols, &blocks, log_basis)
        .expect("cpu matvec")
        .into_iter()
        .flatten()
        .flat_map(|ring| *ring.coefficients())
        .collect()
}

macro_rules! device_matrix {
    ($metal:expr, $cache:expr, $base:ident, $rows:expr, $cols:expr) => {
        DeviceNttMatrix::new($metal, $cache.$base().expect("base profile"), $rows, $cols)
            .expect("upload matrix")
    };
}

/// Shapes past each tile: 5 blocks > BLOCK_TILE, 5 rows > ROW_TILE.
const SHAPES: [(usize, usize, usize); 4] = [(1, 1, 1), (2, 3, 7), (5, 1, 43), (3, 5, 9)];

#[test]
fn fp128_i8_matches_cpu_and_schoolbook() {
    let test = gpu();
    let mut rng = SplitMix64::new(0x128);
    for (blocks, rows, cols) in SHAPES {
        for log_basis in [3, 8] {
            let a = matrix::<Prime128OffsetA7F7, 64>(rows, cols, &mut rng);
            let cache = prepare(&a, rows, cols);
            let device = device_matrix!(&test.metal, cache, q128_base, rows, cols);
            let x = planes::<64>(blocks, cols, log_basis, &mut rng);
            let got = device_i8::<Prime128OffsetA7F7, 6, 64>(&test.metal, &device, &x, log_basis);
            assert_eq!(
                got,
                schoolbook(&a, rows, cols, &x),
                "schoolbook {blocks}x{rows}x{cols} b={log_basis}"
            );
            assert_eq!(
                got,
                cpu_i8::<Prime128OffsetA7F7, 64>(&cache, rows, cols, &x, log_basis)
            );
        }
    }
}

fn fp128_i8_degree<const D: usize>(metal: &AkitaMetal, rng: &mut SplitMix64) {
    for (blocks, rows, cols) in SHAPES {
        let a = matrix::<Prime128OffsetA7F7, D>(rows, cols, rng);
        let cache = prepare(&a, rows, cols);
        let device = device_matrix!(metal, cache, q128_base, rows, cols);
        let x = planes::<D>(blocks, cols, 8, rng);
        assert_eq!(
            device_i8::<Prime128OffsetA7F7, 6, D>(metal, &device, &x, 8),
            cpu_i8::<Prime128OffsetA7F7, D>(&cache, rows, cols, &x, 8),
            "D={D} {blocks}x{rows}x{cols}"
        );
    }
}

#[test]
fn fp128_i8_matches_cpu_at_every_degree() {
    let test = gpu();
    let mut rng = SplitMix64::new(0x1280);
    fp128_i8_degree::<128>(&test.metal, &mut rng);
    fp128_i8_degree::<256>(&test.metal, &mut rng);
    fp128_i8_degree::<512>(&test.metal, &mut rng);
    fp128_i8_degree::<1024>(&test.metal, &mut rng);
}

#[test]
fn fp128_i16_matches_cpu() {
    let test = gpu();
    let mut rng = SplitMix64::new(0x16);
    for (rows, cols) in [(1, 1), (3, 17), (5, 64)] {
        let a = matrix::<Prime128OffsetA7F7, 256>(rows, cols, &mut rng);
        // The CPU's i16 matvec reads the exact negacyclic cache.
        let cache = prepare_with(
            &a,
            rows,
            cols,
            NttCacheMode::ExactNegacyclic {
                width: cols,
                rhs_abs_bound: 1 << 15,
            },
        );
        let device = device_matrix!(&test.metal, cache, q128_base, rows, cols);
        let x = planes::<256>(1, cols, 16, &mut rng);
        let expected = cache
            .mat_vec_i16::<Prime128OffsetA7F7>(16, rows, &x)
            .expect("cpu matvec")
            .into_iter()
            .flat_map(|ring| *ring.coefficients())
            .collect::<Vec<_>>();
        assert_eq!(
            device_i16::<Prime128OffsetA7F7, 6, 256>(&test.metal, &device, &x, 16),
            expected
        );
    }
}

#[test]
fn fp64_i8_matches_cpu_and_schoolbook() {
    let test = gpu();
    let mut rng = SplitMix64::new(0x64);
    for (blocks, rows, cols) in SHAPES {
        let a = matrix::<Prime64Offset59, 64>(rows, cols, &mut rng);
        let cache = prepare(&a, rows, cols);
        let device = device_matrix!(&test.metal, cache, q64_base, rows, cols);
        let x = planes::<64>(blocks, cols, 8, &mut rng);
        let got = device_i8::<Prime64Offset59, 3, 64>(&test.metal, &device, &x, 8);
        assert_eq!(got, schoolbook(&a, rows, cols, &x));
        assert_eq!(
            got,
            cpu_i8::<Prime64Offset59, 64>(&cache, rows, cols, &x, 8)
        );
    }
}

#[test]
fn fp64_i16_matches_cpu_and_schoolbook() {
    let test = gpu();
    let mut rng = SplitMix64::new(0x6416);
    let (blocks, rows, cols) = (2, 3, 17);
    let a = matrix::<Prime64Offset59, 64>(rows, cols, &mut rng);
    let cache = prepare_with(
        &a,
        rows,
        cols,
        NttCacheMode::ExactNegacyclic {
            width: cols,
            rhs_abs_bound: 1 << 15,
        },
    );
    let device = device_matrix!(&test.metal, cache, q64_base, rows, cols);
    let x = planes::<64>(blocks, cols, 16, &mut rng);
    let expected = x
        .chunks_exact(cols)
        .flat_map(|block| {
            cache
                .mat_vec_i16::<Prime64Offset59>(16, rows, block)
                .expect("cpu matvec")
                .into_iter()
                .flat_map(|ring| *ring.coefficients())
        })
        .collect::<Vec<_>>();
    let got = device_i16::<Prime64Offset59, 3, 64>(&test.metal, &device, &x, 16);
    assert_eq!(got, expected);
    assert_eq!(got, schoolbook(&a, rows, cols, &x));
}

#[test]
fn decomposed_planes_feed_matvec_without_relabeling() {
    const DEGREE: usize = 64;
    const RINGS: usize = 3;
    const LEVELS: usize = 4;
    const COLS: usize = RINGS * LEVELS;

    let test = gpu();
    let mut rng = SplitMix64::new(0xdeca_fbad);
    let source = matrix::<Prime64Offset59, DEGREE>(1, RINGS, &mut rng);
    let coefficients = source
        .iter()
        .flat_map(|ring| ring.coefficients().iter().copied())
        .collect::<Vec<_>>();
    let input = DeviceBuffer::from_slice(test.metal.device(), &coefficients).expect("input");
    let (planes, _) = decompose::<Prime64Offset59, i16>(&test.metal, &input, DEGREE, LEVELS, 16)
        .expect("decompose");
    assert_eq!(planes.log_basis(), 16);

    let expected_planes = source
        .iter()
        .flat_map(|ring| {
            let mut digits = vec![[0i16; DEGREE]; LEVELS];
            ring.balanced_decompose_pow2_i16_into(&mut digits, 16);
            digits
        })
        .collect::<Vec<_>>();
    let matrix = matrix::<Prime64Offset59, DEGREE>(1, COLS, &mut rng);
    let cache = prepare_with(
        &matrix,
        1,
        COLS,
        NttCacheMode::ExactNegacyclic {
            width: COLS,
            rhs_abs_bound: 1 << 15,
        },
    );
    let device = device_matrix!(&test.metal, cache, q64_base, 1, COLS);
    let mut out =
        DeviceBuffer::<Prime64Offset59>::zeroed(test.metal.device(), DEGREE).expect("output");
    device
        .mat_vec(&test.metal, &planes, &mut out)
        .expect("matvec");
    assert_eq!(
        out.read().expect("read"),
        schoolbook(&matrix, 1, COLS, &expected_planes).as_slice(),
    );
}

/// Past the CRT capacity the columns run in segments that add in the
/// field: Q64 holds about 2000 columns of 8-bit digits at degree 256, so
/// 4500 columns take three segments.
#[test]
fn fp64_segments_past_crt_capacity_match_cpu() {
    let test = gpu();
    let mut rng = SplitMix64::new(0xcafe);
    let (blocks, rows, cols) = (2, 1, 4500);
    let a = matrix::<Prime64Offset59, 256>(rows, cols, &mut rng);
    let cache = prepare(&a, rows, cols);
    let device = device_matrix!(&test.metal, cache, q64_base, rows, cols);
    let x = planes::<256>(blocks, cols, 8, &mut rng);
    assert_eq!(
        device_i8::<Prime64Offset59, 3, 256>(&test.metal, &device, &x, 8),
        cpu_i8::<Prime64Offset59, 256>(&cache, rows, cols, &x, 8)
    );
}

#[test]
fn digit_upload_rejects_values_outside_the_claimed_basis() {
    let test = gpu();
    let i8_error = match DeviceDigitPlanes::from_slice(&test.metal, &[-5i8, 3], 3) {
        Ok(_) => panic!("-5 is outside the balanced base-8 interval"),
        Err(error) => error,
    };
    assert!(matches!(&i8_error, AkitaMetalError::Shape(_)), "{i8_error}");

    let i16_error = match DeviceDigitPlanes::from_slice(&test.metal, &[-2i16, 1], 1) {
        Ok(_) => panic!("-2 is outside the balanced base-2 interval"),
        Err(error) => error,
    };
    assert!(
        matches!(&i16_error, AkitaMetalError::Shape(_)),
        "{i16_error}"
    );
}

/// Reproduces the trust-boundary failure from the review: the same i16
/// storage cannot be labeled as base-2 (which would underprice CRT capacity),
/// while its truthful base-2^16 label segments the columns and is exact.
#[test]
fn fp64_i16_wrong_basis_rejects_and_full_basis_segments_exactly() {
    use akita_algebra::tables::Q64_PRIMES;
    use akita_algebra::One;

    const DEGREE: usize = 64;
    const COLS: usize = 64;
    let test = gpu();
    let q = (-Prime64Offset59::one())
        .to_u128_checked()
        .expect("fp64 modulus")
        + 1;
    let half = Prime64Offset59::from_u128_reduced(q / 2);
    let entry = CyclotomicRing::from_coefficients([half; DEGREE]);
    let matrix = vec![entry; COLS];
    let plane = std::array::from_fn(|index| if index == 0 { 32767 } else { -32768 });
    let planes = vec![plane; COLS];
    let flat = planes.iter().flatten().copied().collect::<Vec<_>>();
    let expected = schoolbook(&matrix, 1, COLS, &planes);
    let crt_product = Q64_PRIMES
        .iter()
        .map(|prime| u128::from(prime.p.unsigned_abs()))
        .product::<u128>();
    let coefficient_zero_magnitude = (q / 2)
        .checked_mul(32767 + 63 * 32768)
        .and_then(|value| value.checked_mul(COLS as u128))
        .expect("reproduction magnitude");
    assert!(coefficient_zero_magnitude > crt_product);

    let cache = prepare_with(
        &matrix,
        1,
        COLS,
        NttCacheMode::ExactNegacyclic {
            width: COLS,
            rhs_abs_bound: 1,
        },
    );
    assert!(cache.mat_vec_i16::<Prime64Offset59>(1, 1, &planes).is_err());
    let device = device_matrix!(&test.metal, cache, q64_base, 1, COLS);
    let invalid = DeviceDigitPlanes::from_slice(&test.metal, &flat, 1);
    assert!(matches!(invalid, Err(AkitaMetalError::Shape(_))));

    let valid = DeviceDigitPlanes::from_slice(&test.metal, &flat, 16).expect("base-2^16 planes");
    let mut out =
        DeviceBuffer::<Prime64Offset59>::zeroed(test.metal.device(), DEGREE).expect("output");
    device
        .mat_vec(&test.metal, &valid, &mut out)
        .expect("segmented matvec");
    assert_eq!(out.read().expect("read"), expected.as_slice());
}

#[test]
fn unsupported_field_and_crt_profile_reject_at_matrix_construction() {
    use akita_algebra::tables::q128_primes;
    use akita_algebra::CrtNttParamSet;

    const DEGREE: usize = 64;
    let test = gpu();
    let params = CrtNttParamSet::<i32, 6, DEGREE>::new(q128_primes());
    let matrix = [CyclotomicRing::<Prime64Offset59, DEGREE>::zero()];
    let error = match DeviceNttMatrix::from_rings(&test.metal, &params, &matrix, 1, 1, 1) {
        Ok(_) => panic!("fp64 has no six-prime CRT reconstruction pipeline"),
        Err(error) => error,
    };
    assert!(matches!(&error, AkitaMetalError::Shape(_)), "{error}");
}

/// Device-prepared limb matrices over prefixes of the Q128 primes: every
/// limb count the kernels support, against the CPU's own prepared matvec on
/// the same matrix, for i8 and i16 digits.
/// Whether K primes hold one column of `limbs`-limb entries and `log_basis`
/// digits exactly, computed independently of the crate: a limb of
/// ceil(129 / limbs) bits, or the centered entry itself for one limb.
fn one_column_fits<const K: usize>(limbs: usize, degree: usize, log_basis: u32) -> bool {
    use akita_algebra::tables::q128_primes;
    use akita_algebra::{CrtCapacity, One};
    let q = (-Prime128OffsetA7F7::one())
        .to_u128_checked()
        .expect("u128")
        + 1;
    let bound = if limbs == 1 {
        q
    } else {
        1u128 << 129usize.div_ceil(limbs)
    };
    CrtCapacity::from_prime_moduli(q128_primes()[..K].iter().map(|prime| prime.p as u128))
        .supports_modulus(1, degree, bound, 1 << (log_basis - 1))
}

fn limb_matrix<const K: usize, const D: usize>(
    metal: &AkitaMetal,
    limbs: usize,
    rng: &mut SplitMix64,
) {
    use akita_algebra::tables::q128_primes;
    use akita_algebra::CrtNttParamSet;
    let primes: [akita_algebra::NttPrime<i32>; K] =
        std::array::from_fn(|index| q128_primes()[index]);
    let params = CrtNttParamSet::<i32, K, D>::new(primes);
    for (blocks, cols, log_basis) in [(3, 5, 8), (2, 9, 3)] {
        if !one_column_fits::<K>(limbs, D, log_basis) {
            continue;
        }
        let a = matrix::<Prime128OffsetA7F7, D>(1, cols, rng);
        let cache = prepare(&a, 1, cols);
        let device =
            DeviceNttMatrix::from_rings(metal, &params, &a, 1, cols, limbs).expect("limb matrix");
        let x = planes::<D>(blocks, cols, log_basis, rng);
        assert_eq!(
            device_i8::<Prime128OffsetA7F7, K, D>(metal, &device, &x, log_basis),
            cpu_i8::<Prime128OffsetA7F7, D>(&cache, 1, cols, &x, log_basis),
            "i8 K={K} limbs={limbs} D={D} {blocks}x{cols}"
        );
    }
    // 16-bit digits need more headroom: exercise them where it fits.
    if !one_column_fits::<K>(limbs, D, 16) {
        return;
    }
    let cols = 4;
    let a = matrix::<Prime128OffsetA7F7, D>(1, cols, rng);
    let cache = prepare_with(
        &a,
        1,
        cols,
        NttCacheMode::ExactNegacyclic {
            width: cols,
            rhs_abs_bound: 1 << 15,
        },
    );
    let device =
        DeviceNttMatrix::from_rings(metal, &params, &a, 1, cols, limbs).expect("limb matrix");
    let x = planes::<D>(1, cols, 16, rng);
    let expected = cache
        .mat_vec_i16::<Prime128OffsetA7F7>(16, 1, &x)
        .expect("cpu matvec")
        .into_iter()
        .flat_map(|ring| *ring.coefficients())
        .collect::<Vec<_>>();
    assert_eq!(
        device_i16::<Prime128OffsetA7F7, K, D>(metal, &device, &x, 16),
        expected,
        "i16 K={K} limbs={limbs} D={D}"
    );
}

/// The configurations checked below that fit (the rest are skipped by
/// `one_column_fits`) cover every limb count and prime prefix.
#[test]
fn limb_matrices_match_cpu() {
    assert!(one_column_fits::<3>(2, 256, 8) && one_column_fits::<2>(4, 128, 16));
    let test = gpu();
    let mut rng = SplitMix64::new(0x11b);
    for limbs in [1, 2, 3, 4] {
        limb_matrix::<6, 64>(&test.metal, limbs, &mut rng);
        limb_matrix::<3, 256>(&test.metal, limbs, &mut rng);
        limb_matrix::<4, 1024>(&test.metal, limbs, &mut rng);
    }
    // Two primes hold only narrow limbs: four limbs of 33 bits.
    limb_matrix::<2, 128>(&test.metal, 4, &mut rng);
}

#[test]
fn fp64_two_limb_i16_matrix_segments_match_schoolbook() {
    use akita_algebra::tables::Q64_PRIMES;
    use akita_algebra::{CrtCapacity, CrtNttParamSet};

    const DEGREE: usize = 64;
    const COLS: usize = 96;
    let test = gpu();
    let mut rng = SplitMix64::new(0x64216);
    let primes = [Q64_PRIMES[0], Q64_PRIMES[1]];
    let capacity = CrtCapacity::from_prime_moduli(
        primes
            .iter()
            .map(|prime| u128::from(prime.p.unsigned_abs())),
    );
    let segment_cols = capacity
        .max_safe_width_for_modulus(DEGREE, 1u128 << 33, 1 << 15)
        .expect("one two-limb column fits two primes");
    assert!(segment_cols < COLS, "test must cross a CRT segment");

    let params = CrtNttParamSet::<i32, 2, DEGREE>::new(primes);
    let matrix = matrix::<Prime64Offset59, DEGREE>(1, COLS, &mut rng);
    let planes = planes::<DEGREE>(1, COLS, 16, &mut rng);
    let device = DeviceNttMatrix::from_rings(&test.metal, &params, &matrix, 1, COLS, 2)
        .expect("two-limb matrix");
    assert_eq!(
        device_i16::<Prime64Offset59, 2, DEGREE>(&test.metal, &device, &planes, 16),
        schoolbook(&matrix, 1, COLS, &planes),
    );
}

/// The positive fp128 centering boundary has a negative low balanced limb.
/// The old `(centered - digit) >> width` recurrence overflowed at `q / 2`
/// before producing the mathematically representable quotient.
#[test]
fn fp128_centering_boundary_limb_matrices_match_constant_multiplication() {
    use akita_algebra::tables::q128_primes;
    use akita_algebra::{CrtNttParamSet, One};

    const DEGREE: usize = 64;
    let test = gpu();
    let modulus = (-Prime128OffsetA7F7::one())
        .to_u128_checked()
        .expect("u128")
        + 1;
    let matrix = [CyclotomicRing::from_coefficients(std::array::from_fn(
        |coefficient| match coefficient {
            0 => Prime128OffsetA7F7::from_u128_reduced(modulus / 2),
            1 => Prime128OffsetA7F7::from_u128_reduced(modulus / 2 - 1),
            2 => Prime128OffsetA7F7::from_u128_reduced(modulus / 2 + 1),
            _ => Prime128OffsetA7F7::zero(),
        },
    ))];
    let primes = std::array::from_fn(|index| q128_primes()[index]);
    let params = CrtNttParamSet::<i32, 6, DEGREE>::new(primes);
    let constant_one = [std::array::from_fn(|index| i16::from(index == 0))];

    for limbs in [2, 3, 4] {
        let device = DeviceNttMatrix::from_rings(&test.metal, &params, &matrix, 1, 1, limbs)
            .expect("centering-boundary matrix");
        assert_eq!(
            device_i8::<Prime128OffsetA7F7, 6, DEGREE>(&test.metal, &device, &constant_one, 2,),
            matrix[0].coefficients(),
            "limbs={limbs}",
        );
    }
}

/// The planner halves the primes on the production shapes.
#[test]
fn plans_fewer_primes_with_limbs() {
    use akita_algebra::tables::q128_primes;
    use akita_algebra::One;
    use akita_metal::matvec::{plan_matvec, MatvecPlan};
    let q = (-Prime128OffsetA7F7::one())
        .to_u128_checked()
        .expect("u128")
        + 1;
    let primes = q128_primes();
    // Dense nv26 inner: 2048 columns of 16-bit digits at D = 1024.
    let dense = plan_matvec(q, &primes, 2048, 1024, 16);
    // One-hot nv32 outer: 44032 columns of base-8 digits at D = 64.
    let outer = plan_matvec(q, &primes, 44032, 64, 3);
    assert!(
        dense.primes <= 3 && outer.primes <= 3,
        "{dense:?} {outer:?}"
    );
    assert_ne!(
        dense,
        MatvecPlan {
            primes: 6,
            limbs: 1
        }
    );
}
