//! Device NTT matvec against the CPU's prepared-cache matvecs and against
//! schoolbook ring arithmetic.

#![cfg(target_os = "macos")]

mod support;

use akita_algebra::{CanonicalEncoding, CyclotomicRing, Field};
use akita_cpu_backend::benchmark_support::mat_vec_mul_ntt_digits_i8;
use akita_metal::matvec::DeviceNttMatrix;
use akita_metal::AkitaMetal;
use akita_types::{prepare_ntt_cache, FlatMatrix, NttCacheMode, PreparedNttCache};
use jolt_field::{Prime128OffsetA7F7, Prime64Offset59};
use jolt_metal::runtime::DeviceBuffer;
use jolt_metal::MetalField;
use support::{gpu, SplitMix64};

/// A random matrix whose first entry has every coefficient at `-1`, the
/// largest centered magnitude.
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
    device: &DeviceNttMatrix<K, D>,
    planes: &[[i16; D]],
    log_basis: u32,
) -> Vec<F> {
    let digits = planes
        .iter()
        .flatten()
        .map(|&digit| digit as i8)
        .collect::<Vec<_>>();
    let buffer = DeviceBuffer::from_slice(metal.device(), &digits).expect("upload");
    let blocks = planes.len() / device.cols();
    let mut out =
        DeviceBuffer::<F>::zeroed(metal.device(), blocks * device.rows() * D).expect("output");
    device
        .mat_vec(metal, &buffer, log_basis, &mut out)
        .expect("matvec");
    out.read().expect("canonical read-back").to_vec()
}

fn device_i16<F: MetalField + CanonicalEncoding, const K: usize, const D: usize>(
    metal: &AkitaMetal,
    device: &DeviceNttMatrix<K, D>,
    planes: &[[i16; D]],
    log_basis: u32,
) -> Vec<F> {
    let digits = planes.iter().flatten().copied().collect::<Vec<_>>();
    let buffer = DeviceBuffer::from_slice(metal.device(), &digits).expect("upload");
    let blocks = planes.len() / device.cols();
    let mut out =
        DeviceBuffer::<F>::zeroed(metal.device(), blocks * device.rows() * D).expect("output");
    device
        .mat_vec(metal, &buffer, log_basis, &mut out)
        .expect("matvec");
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
